
from __future__ import annotations

"""
simons_core.calendar
====================

Single source of truth for market-session geometry in the stack.

This module intentionally models *trading sessions* rather than civil business
days. For US equities the canonical timezone is America/New_York and the
calendar backend is encapsulated behind a small public API so research,
validation, backtests and execution all share the same temporal semantics.

Key design choices
------------------
- Fail-closed on unsupported markets and out-of-range requests.
- Explicit inclusive semantics for session counting.
- Strict next/previous trading-day semantics.
- Timezone normalization never depends on the host locale.
- Session alignment works on real opens/closes, not on naive date offsets.

The public entry points are:

    load_market_calendar(...)
    next_trading_day(...)
    prev_trading_day(...)
    trading_days_between(...)
    align_to_session(...)

The preferred way to use the module in the rest of the stack is through the
`MarketCalendar` object returned by `load_market_calendar`.
"""

from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from functools import lru_cache
from typing import Any, Literal, Mapping

import pandas as pd

try:
    import exchange_calendars as xcals
except ImportError as exc:  # pragma: no cover - environment dependent
    raise ImportError(
        "simons_core.calendar requires the optional dependency "
        "'exchange_calendars'. Install it to use market-session geometry."
    ) from exc


CANONICAL_TIMEZONE = "America/New_York"
DEFAULT_START_DATE = "2010-01-01"
DEFAULT_END_DATE = "2027-03-15"

InclusiveMode = Literal["left", "right", "both", "neither"]
AlignMode = Literal["next_open", "prev_close", "session_label"]


class CalendarError(ValueError):
    """Base class for calendar-related failures."""


class UnsupportedMarketError(CalendarError):
    """Raised when the requested market alias/backend is not supported."""


class CalendarRangeError(CalendarError):
    """Raised when a request falls outside the loaded calendar coverage."""


class TimestampNormalizationError(CalendarError):
    """Raised when a timestamp cannot be normalized safely."""


class SessionType(str, Enum):
    """Coarse geometry of a session."""

    REGULAR = "regular"
    EARLY_CLOSE = "early_close"
    HALF_DAY = "half_day"
    SPECIAL = "special"


@dataclass(frozen=True, slots=True)
class MarketCalendar:
    """
    Canonical market-session geometry.

    Parameters
    ----------
    market
        Public market identifier used by the stack, e.g. ``"US_EQ"``.
    timezone
        Canonical timezone for normalization and returned timestamps.
    sessions
        Ordered unique session labels as normalized local dates stored as a
        tz-naive ``DatetimeIndex``. These are session *labels*, not opens/closes.
    opens
        Session opens aligned 1:1 with ``sessions`` and stored as tz-aware
        timestamps in ``timezone``.
    closes
        Session closes aligned 1:1 with ``sessions`` and stored as tz-aware
        timestamps in ``timezone``.
    session_types
        Optional per-session coarse geometry. At minimum distinguishes regular
        sessions from early closes when the backend exposes them.
    backend_name
        Name of the underlying provider/backend used to build the schedule.

    Notes
    -----
    - ``next_trading_day`` is *strictly posterior* to the given logical date.
    - ``prev_trading_day`` is *strictly anterior* to the given logical date.
    - ``trading_days_between`` counts sessions according to the explicit
      interval convention requested via ``inclusive``.
    """

    market: str
    timezone: str
    sessions: pd.DatetimeIndex
    opens: pd.DatetimeIndex
    closes: pd.DatetimeIndex
    session_types: pd.Series | None = None
    backend_name: str = "exchange_calendars"

    def __post_init__(self) -> None:
        if not isinstance(self.sessions, pd.DatetimeIndex):
            raise TypeError("sessions must be a pandas.DatetimeIndex")
        if not isinstance(self.opens, pd.DatetimeIndex):
            raise TypeError("opens must be a pandas.DatetimeIndex")
        if not isinstance(self.closes, pd.DatetimeIndex):
            raise TypeError("closes must be a pandas.DatetimeIndex")
        if len(self.sessions) == 0:
            raise ValueError("sessions cannot be empty")
        if len(self.opens) != len(self.sessions) or len(self.closes) != len(self.sessions):
            raise ValueError("sessions, opens and closes must have identical lengths")
        if not self.sessions.is_monotonic_increasing or not self.sessions.is_unique:
            raise ValueError("sessions must be sorted and unique")
        if not self.opens.is_monotonic_increasing or not self.opens.is_unique:
            raise ValueError("opens must be sorted and unique")
        if not self.closes.is_monotonic_increasing or not self.closes.is_unique:
            raise ValueError("closes must be sorted and unique")
        if self.opens.tz is None or self.closes.tz is None:
            raise ValueError("opens and closes must be timezone-aware")
        if str(self.opens.tz) != self.timezone or str(self.closes.tz) != self.timezone:
            raise ValueError("opens and closes must be expressed in the canonical timezone")
        if self.session_types is not None and len(self.session_types) != len(self.sessions):
            raise ValueError("session_types must be aligned 1:1 with sessions")

    @property
    def first_session(self) -> date:
        return self.sessions[0].date()

    @property
    def last_session(self) -> date:
        return self.sessions[-1].date()

    @property
    def coverage(self) -> tuple[date, date]:
        return self.first_session, self.last_session

    @property
    def schedule(self) -> pd.DataFrame:
        """
        Canonical schedule view indexed by session label.

        Returns
        -------
        pd.DataFrame
            Columns: ``open``, ``close``, ``session_type``.
        """
        data: dict[str, Any] = {
            "open": self.opens,
            "close": self.closes,
        }
        if self.session_types is not None:
            data["session_type"] = self.session_types.to_numpy()
        else:
            data["session_type"] = pd.Index(
                [SessionType.REGULAR] * len(self.sessions), dtype=object
            )
        return pd.DataFrame(data, index=self.sessions.copy())

    def is_trading_day(self, date_like: Any) -> bool:
        """Return True iff the normalized local date is a valid session label."""
        label = self._normalize_session_label(date_like)
        pos = self.sessions.searchsorted(label, side="left")
        return bool(pos < len(self.sessions) and self.sessions[pos] == label)

    def session_open(self, date_like: Any) -> pd.Timestamp:
        """Return the canonical open timestamp for a valid session label."""
        idx = self._locate_exact_session(date_like)
        return self.opens[idx]

    def session_close(self, date_like: Any) -> pd.Timestamp:
        """Return the canonical close timestamp for a valid session label."""
        idx = self._locate_exact_session(date_like)
        return self.closes[idx]

    def session_type(self, date_like: Any) -> SessionType:
        """Return the coarse geometry for a valid session label."""
        idx = self._locate_exact_session(date_like)
        if self.session_types is None:
            return SessionType.REGULAR
        value = self.session_types.iloc[idx]
        return value if isinstance(value, SessionType) else SessionType(str(value))

    def next_trading_day(self, date_like: Any, n: int = 1) -> date:
        """
        Return the n-th valid session *strictly after* the given logical date.

        Examples
        --------
        - Friday session label -> Monday (or next valid session)
        - Saturday -> Monday
        - Monday with n=2 -> Wednesday, assuming no holiday in between
        """
        if n < 1:
            raise ValueError("n must be >= 1")

        label = self._normalize_session_label(date_like)
        self._require_label_in_extended_coverage(label)

        pos = self.sessions.searchsorted(label, side="right")
        target = pos + (n - 1)
        if target >= len(self.sessions):
            raise CalendarRangeError(
                f"Requested next_trading_day beyond loaded calendar range. "
                f"Last loaded session is {self.last_session}."
            )
        return self.sessions[target].date()

    def prev_trading_day(self, date_like: Any, n: int = 1) -> date:
        """
        Return the n-th valid session *strictly before* the given logical date.

        Examples
        --------
        - Monday session label -> previous Friday (or previous valid session)
        - Sunday -> previous Friday
        """
        if n < 1:
            raise ValueError("n must be >= 1")

        label = self._normalize_session_label(date_like)
        self._require_label_in_extended_coverage(label)

        pos = self.sessions.searchsorted(label, side="left") - 1
        target = pos - (n - 1)
        if target < 0:
            raise CalendarRangeError(
                f"Requested prev_trading_day before loaded calendar range. "
                f"First loaded session is {self.first_session}."
            )
        return self.sessions[target].date()

    def trading_days_between(
        self,
        start: Any,
        end: Any,
        inclusive: InclusiveMode = "left",
    ) -> int:
        """
        Count valid sessions between two logical dates under explicit interval semantics.

        Interval semantics
        ------------------
        - ``left``    -> [start, end)
        - ``right``   -> (start, end]
        - ``both``    -> [start, end]
        - ``neither`` -> (start, end)

        Parameters
        ----------
        start, end
            Date-like or timestamp-like inputs that are normalized to local
            session labels in the canonical timezone.
        inclusive
            One of ``"left"``, ``"right"``, ``"both"``, ``"neither"``.

        Returns
        -------
        int
            Non-negative count of valid sessions.

        Raises
        ------
        ValueError
            If ``end < start`` or ``inclusive`` is invalid.
        CalendarRangeError
            If either endpoint falls outside loaded calendar coverage.
        """
        start_label = self._normalize_session_label(start)
        end_label = self._normalize_session_label(end)

        if end_label < start_label:
            raise ValueError("end must be >= start")

        self._require_label_in_extended_coverage(start_label)
        self._require_label_in_extended_coverage(end_label)

        if inclusive == "left":
            left = self.sessions.searchsorted(start_label, side="left")
            right = self.sessions.searchsorted(end_label, side="left")
        elif inclusive == "right":
            left = self.sessions.searchsorted(start_label, side="right")
            right = self.sessions.searchsorted(end_label, side="right")
        elif inclusive == "both":
            left = self.sessions.searchsorted(start_label, side="left")
            right = self.sessions.searchsorted(end_label, side="right")
        elif inclusive == "neither":
            left = self.sessions.searchsorted(start_label, side="right")
            right = self.sessions.searchsorted(end_label, side="left")
        else:
            raise ValueError(
                "inclusive must be one of: 'left', 'right', 'both', 'neither'"
            )

        return max(int(right - left), 0)

    def align_to_session(self, ts_like: Any, mode: AlignMode = "next_open") -> pd.Timestamp:
        """
        Align a timestamp-like input to a canonical session-derived timestamp.

        Modes
        -----
        ``next_open``
            First market open greater than or equal to the normalized timestamp.
            If the timestamp is during an active session, this returns the next
            session's open because the current day's open has already passed.

        ``prev_close``
            Last market close less than or equal to the normalized timestamp.
            If the timestamp is during an active session before that day's close,
            this returns the previous session's close.

        ``session_label``
            Canonical session label as a timezone-aware midnight timestamp in the
            canonical timezone. If the local civil date is itself a valid session,
            that label is returned. Otherwise the next valid session label is used.
        """
        ts = self._normalize_timestamp(ts_like)

        if mode == "next_open":
            idx = self.opens.searchsorted(ts, side="left")
            if idx >= len(self.opens):
                raise CalendarRangeError(
                    f"Cannot align to next_open beyond loaded range. "
                    f"Last known open is {self.opens[-1]!s}."
                )
            return self.opens[idx]

        if mode == "prev_close":
            idx = self.closes.searchsorted(ts, side="right") - 1
            if idx < 0:
                raise CalendarRangeError(
                    f"Cannot align to prev_close before loaded range. "
                    f"First known close is {self.closes[0]!s}."
                )
            return self.closes[idx]

        if mode == "session_label":
            label = self._normalize_session_label(ts)
            pos = self.sessions.searchsorted(label, side="left")

            if pos < len(self.sessions) and self.sessions[pos] == label:
                chosen = label
            else:
                if pos >= len(self.sessions):
                    raise CalendarRangeError(
                        f"Cannot align to session_label beyond loaded range. "
                        f"Last loaded session is {self.last_session}."
                    )
                chosen = self.sessions[pos]

            return chosen.tz_localize(self.timezone)

        raise ValueError("mode must be one of: 'next_open', 'prev_close', 'session_label'")

    def _locate_exact_session(self, date_like: Any) -> int:
        label = self._normalize_session_label(date_like)
        pos = self.sessions.searchsorted(label, side="left")
        if pos >= len(self.sessions) or self.sessions[pos] != label:
            raise CalendarRangeError(
                f"{label.date()} is not a valid loaded session for market {self.market!r}."
            )
        return int(pos)

    def _normalize_session_label(self, value: Any) -> pd.Timestamp:
        """
        Normalize a date/timestamp-like input to a canonical local session label.

        Returned value is tz-naive midnight in the canonical timezone. Keeping
        session labels tz-naive avoids the cognitive overhead of mixing "a date
        that labels a session" with "the actual open/close instants", which are
        stored separately as tz-aware timestamps.
        """
        ts = self._normalize_timestamp(value)
        return ts.normalize().tz_localize(None)

    def _normalize_timestamp(self, value: Any) -> pd.Timestamp:
        """
        Parse and normalize to a tz-aware timestamp in the canonical timezone.

        Naive inputs are localized explicitly to the canonical timezone and
        ambiguous/nonexistent local times around DST transitions raise
        ``TimestampNormalizationError``.
        """
        try:
            ts = pd.Timestamp(value)
        except Exception as exc:  # pragma: no cover - pandas parsing surface
            raise TimestampNormalizationError(
                f"Could not parse timestamp/date value: {value!r}"
            ) from exc

        if ts.tzinfo is None:
            try:
                return ts.tz_localize(
                    self.timezone,
                    ambiguous="raise",
                    nonexistent="raise",
                )
            except Exception as exc:
                raise TimestampNormalizationError(
                    f"Naive timestamp {value!r} cannot be localized safely to "
                    f"{self.timezone}."
                ) from exc

        try:
            return ts.tz_convert(self.timezone)
        except Exception as exc:
            raise TimestampNormalizationError(
                f"Timestamp {value!r} could not be converted to {self.timezone}."
            ) from exc

    def _require_label_in_extended_coverage(self, label: pd.Timestamp) -> None:
        """
        Fail closed if the logical query is materially outside the loaded calendar.

        We allow equality with boundaries so that a query on the last loaded
        session can still fail meaningfully at the operation level (for example,
        asking the "next" day from the last session).
        """
        first = self.sessions[0]
        last = self.sessions[-1]
        if label < first or label > last:
            raise CalendarRangeError(
                f"Date {label.date()} is outside loaded calendar coverage "
                f"[{first.date()}, {last.date()}] for market {self.market!r}."
            )


_MARKET_ALIAS_TO_BACKEND: Mapping[str, str] = {
    "US_EQ": "XNYS",
    "NYSE": "XNYS",
    "NASDAQ": "XNYS",  # same session geometry for the intended use in this stack
    "XNYS": "XNYS",
}


def _normalize_market_alias(market: str) -> str:
    key = str(market).strip().upper()
    if key not in _MARKET_ALIAS_TO_BACKEND:
        supported = ", ".join(sorted(_MARKET_ALIAS_TO_BACKEND))
        raise UnsupportedMarketError(
            f"Unsupported market {market!r}. Supported aliases: {supported}"
        )
    return key


def _coerce_date_string(value: Any, *, timezone: str = CANONICAL_TIMEZONE) -> str:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize(timezone, ambiguous="raise", nonexistent="raise")
    else:
        ts = ts.tz_convert(timezone)
    return str(ts.normalize().date())


def _classify_session_types(
    sessions: pd.DatetimeIndex,
    opens: pd.DatetimeIndex,
    closes: pd.DatetimeIndex,
    *,
    backend_calendar: Any,
) -> pd.Series:
    """
    Classify sessions conservatively.

    For XNYS via ``exchange_calendars`` we explicitly label backend-declared
    early closes. Anything else defaults to ``REGULAR``. We keep the enum rich
    enough for future extension without over-claiming unsupported geometry.
    """
    types = pd.Series(
        [SessionType.REGULAR] * len(sessions),
        index=sessions,
        dtype=object,
        name="session_type",
    )

    early_close_idx = getattr(backend_calendar, "early_closes", None)
    if isinstance(early_close_idx, pd.DatetimeIndex) and len(early_close_idx) > 0:
        early_close_idx = early_close_idx.intersection(sessions)
        if len(early_close_idx) > 0:
            types.loc[early_close_idx] = SessionType.EARLY_CLOSE

    # Optional refinement: if a session is shorter than the regular modal
    # duration and was not explicitly marked, classify as SPECIAL rather than
    # silently treating it as a normal session.
    durations = closes.tz_convert("UTC") - opens.tz_convert("UTC")
    if len(durations) > 0:
        regular_duration = pd.Series(durations).mode().iloc[0]
        shorter = sessions[durations < regular_duration]
        shorter = shorter.difference(types.index[types == SessionType.EARLY_CLOSE])
        if len(shorter) > 0:
            types.loc[shorter] = SessionType.SPECIAL

    return types


def _build_calendar(
    market: str,
    start_date: str,
    end_date: str,
) -> MarketCalendar:
    market_alias = _normalize_market_alias(market)
    backend_symbol = _MARKET_ALIAS_TO_BACKEND[market_alias]

    backend = xcals.get_calendar(backend_symbol)

    requested_start = pd.Timestamp(start_date)
    requested_end = pd.Timestamp(end_date)
    if requested_end < requested_start:
        raise ValueError("end_date must be >= start_date")

    backend_first = pd.Timestamp(getattr(backend, "first_session"))
    backend_last = pd.Timestamp(getattr(backend, "last_session"))

    if requested_start < backend_first or requested_end > backend_last:
        raise CalendarRangeError(
            "Requested calendar range exceeds backend coverage. "
            f"Requested=[{requested_start.date()}, {requested_end.date()}], "
            f"backend=[{backend_first.date()}, {backend_last.date()}]."
        )

    schedule = backend.schedule.loc[start_date:end_date, ["open", "close"]].copy()
    if schedule.empty:
        raise CalendarRangeError(
            f"No sessions available for requested range [{start_date}, {end_date}] "
            f"and market {market_alias!r}."
        )

    sessions = pd.DatetimeIndex(schedule.index).tz_localize(None)
    opens = pd.DatetimeIndex(schedule["open"].dt.tz_convert(CANONICAL_TIMEZONE))
    closes = pd.DatetimeIndex(schedule["close"].dt.tz_convert(CANONICAL_TIMEZONE))
    session_types = _classify_session_types(
        sessions=sessions,
        opens=opens,
        closes=closes,
        backend_calendar=backend,
    )

    return MarketCalendar(
        market=market_alias,
        timezone=CANONICAL_TIMEZONE,
        sessions=sessions,
        opens=opens,
        closes=closes,
        session_types=session_types,
        backend_name="exchange_calendars",
    )


@lru_cache(maxsize=64)
def load_market_calendar(
    market: str = "US_EQ",
    start_date: Any = DEFAULT_START_DATE,
    end_date: Any = DEFAULT_END_DATE,
) -> MarketCalendar:
    """
    Load or retrieve a cached market calendar with deterministic coverage.

    Parameters
    ----------
    market
        Public market alias, e.g. ``"US_EQ"``.
    start_date, end_date
        Range boundaries used to materialize the loaded schedule. Inputs may be
        date-like or timestamp-like and are normalized in the canonical timezone.

    Returns
    -------
    MarketCalendar
        Cached immutable calendar object suitable for reuse across the stack.
    """
    market_alias = _normalize_market_alias(market)
    start_str = _coerce_date_string(start_date)
    end_str = _coerce_date_string(end_date)
    return _build_calendar(market_alias, start_str, end_str)


def next_trading_day(
    date_like: Any,
    n: int = 1,
    market: str = "US_EQ",
    start_date: Any = DEFAULT_START_DATE,
    end_date: Any = DEFAULT_END_DATE,
) -> date:
    """Convenience wrapper over ``MarketCalendar.next_trading_day``."""
    cal = load_market_calendar(market=market, start_date=start_date, end_date=end_date)
    return cal.next_trading_day(date_like, n=n)


def prev_trading_day(
    date_like: Any,
    n: int = 1,
    market: str = "US_EQ",
    start_date: Any = DEFAULT_START_DATE,
    end_date: Any = DEFAULT_END_DATE,
) -> date:
    """Convenience wrapper over ``MarketCalendar.prev_trading_day``."""
    cal = load_market_calendar(market=market, start_date=start_date, end_date=end_date)
    return cal.prev_trading_day(date_like, n=n)


def trading_days_between(
    start: Any,
    end: Any,
    market: str = "US_EQ",
    inclusive: InclusiveMode = "left",
    start_date: Any = DEFAULT_START_DATE,
    end_date: Any = DEFAULT_END_DATE,
) -> int:
    """Convenience wrapper over ``MarketCalendar.trading_days_between``."""
    cal = load_market_calendar(market=market, start_date=start_date, end_date=end_date)
    return cal.trading_days_between(start, end, inclusive=inclusive)


def align_to_session(
    ts_like: Any,
    mode: AlignMode = "next_open",
    market: str = "US_EQ",
    start_date: Any = DEFAULT_START_DATE,
    end_date: Any = DEFAULT_END_DATE,
) -> pd.Timestamp:
    """Convenience wrapper over ``MarketCalendar.align_to_session``."""
    cal = load_market_calendar(market=market, start_date=start_date, end_date=end_date)
    return cal.align_to_session(ts_like, mode=mode)


__all__ = [
    "AlignMode",
    "CalendarError",
    "CalendarRangeError",
    "CANONICAL_TIMEZONE",
    "DEFAULT_END_DATE",
    "DEFAULT_START_DATE",
    "InclusiveMode",
    "MarketCalendar",
    "SessionType",
    "TimestampNormalizationError",
    "UnsupportedMarketError",
    "align_to_session",
    "load_market_calendar",
    "next_trading_day",
    "prev_trading_day",
    "trading_days_between",
]
