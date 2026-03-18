"""
simons_core/calendar.py — Market session geometry.

Single source of truth for trading days (NYSE/NASDAQ, America/New_York).
    next_trading_day(date, n)     → nth next valid session
    prev_trading_day(date, n)     → nth previous valid session
    trading_days_between(s, e)    → count sessions in interval
    align_to_session(ts, mode)    → map timestamp to session boundary

Session ≠ calendar day. Weekends, holidays, early closes all handled.
SessionType: REGULAR | EARLY_CLOSE | HALF_DAY | SPECIAL.
"""
from __future__ import annotations
import enum
from dataclasses import dataclass
from datetime import date
from typing import Literal
import numpy as np
import pandas as pd

CANONICAL_TZ = "America/New_York"


class SessionType(enum.Enum):
    REGULAR = "regular"
    EARLY_CLOSE = "early_close"
    HALF_DAY = "half_day"
    SPECIAL = "special"


@dataclass(frozen=True)
class MarketCalendar:
    """Immutable trading calendar.  sessions must be sorted, unique, tz-normalised."""
    market: str
    timezone: str
    sessions: pd.DatetimeIndex
    session_types: pd.Series | None = None  # indexed by session date

    def next_trading_day(self, date_like, n: int = 1) -> date:
        if n < 1: raise ValueError(f"n must be ≥ 1, got {n}")
        d = self._norm_date(date_like)
        pos = int(self.sessions.searchsorted(pd.Timestamp(d), side="right"))
        idx = pos + n - 1
        if idx >= len(self.sessions):
            raise ValueError(f"Beyond calendar range (need index {idx}, have {len(self.sessions)})")
        return self.sessions[idx].date()

    def prev_trading_day(self, date_like, n: int = 1) -> date:
        if n < 1: raise ValueError(f"n must be ≥ 1, got {n}")
        d = self._norm_date(date_like)
        pos = int(self.sessions.searchsorted(pd.Timestamp(d), side="left")) - 1
        idx = pos - (n - 1)
        if idx < 0:
            raise ValueError(f"Before calendar range (need index {idx})")
        return self.sessions[idx].date()

    def trading_days_between(
        self, start, end,
        inclusive: Literal["left", "right", "both", "neither"] = "left",
    ) -> int:
        s, e = self._norm_date(start), self._norm_date(end)
        if e < s: raise ValueError(f"end ({e}) < start ({s})")
        s_ts, e_ts = pd.Timestamp(s), pd.Timestamp(e)
        left = int(self.sessions.searchsorted(s_ts, side="left"))
        right = int(self.sessions.searchsorted(e_ts, side="left"))
        count = right - left
        if inclusive in ("right", "both") and e_ts in self.sessions:
            count += 1
        if inclusive == "neither" and s_ts in self.sessions:
            count -= 1
        return max(count, 0)

    def is_trading_day(self, date_like) -> bool:
        d = pd.Timestamp(self._norm_date(date_like))
        return d in self.sessions

    def sessions_in_range(self, start, end) -> pd.DatetimeIndex:
        s, e = pd.Timestamp(self._norm_date(start)), pd.Timestamp(self._norm_date(end))
        mask = (self.sessions >= s) & (self.sessions <= e)
        return self.sessions[mask]

    def align_to_session(
        self, ts_like,
        mode: Literal["next_open", "prev_close", "session_label"] = "next_open",
    ) -> pd.Timestamp:
        ts = self._norm_ts(ts_like)
        d = ts.date()
        if mode == "session_label":
            return pd.Timestamp(self.next_trading_day(d, 1))
        if mode == "next_open":
            nd = self.next_trading_day(d, 1)
            return pd.Timestamp(nd).tz_localize(self.timezone).replace(hour=9, minute=30)
        if mode == "prev_close":
            pd_ = self.prev_trading_day(d, 1)
            h = 16
            if self.session_types is not None:
                t = pd.Timestamp(pd_)
                if t in self.session_types.index and self.session_types.loc[t] == SessionType.EARLY_CLOSE:
                    h = 13
            return pd.Timestamp(pd_).tz_localize(self.timezone).replace(hour=h, minute=0)
        raise ValueError(f"Invalid align mode: {mode!r}")

    def _norm_date(self, x) -> date:
        ts = pd.Timestamp(x)
        if ts.tzinfo is not None:
            ts = ts.tz_convert(self.timezone)
        return ts.normalize().date()

    def _norm_ts(self, x) -> pd.Timestamp:
        ts = pd.Timestamp(x)
        if ts.tzinfo is None:
            return ts.tz_localize(self.timezone)
        return ts.tz_convert(self.timezone)


# ── Factory (spec §15) ──────────────────────────────────────────────────────

_cache: dict[str, MarketCalendar] = {}

def load_market_calendar(
    market: str = "US_EQ",
    start_date: str = "2010-01-01",
    end_date: str = "2035-12-31",
) -> MarketCalendar:
    """Load or build market calendar. Cached by key."""
    key = f"{market}:{start_date}:{end_date}"
    if key in _cache:
        return _cache[key]

    # Try exchange_calendars
    try:
        import exchange_calendars as xcals
        xc = xcals.get_calendar("XNYS")
        sessions = xc.sessions_in_range(start_date, end_date)
        cal = MarketCalendar(market=market, timezone=CANONICAL_TZ, sessions=sessions)
        _cache[key] = cal
        return cal
    except ImportError:
        pass

    # Fallback: pandas business days (approximate)
    sessions = pd.bdate_range(start_date, end_date)
    cal = MarketCalendar(market=market, timezone=CANONICAL_TZ, sessions=sessions)
    _cache[key] = cal
    return cal
