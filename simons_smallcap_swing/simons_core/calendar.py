from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

from .io import paths as path_utils


class CalendarError(Exception):
    """Base calendar exception."""


class CalendarNotFoundError(CalendarError):
    """Raised when no calendar artifact can be found."""


class CalendarCorruptError(CalendarError):
    """Raised when a calendar artifact cannot be parsed safely."""


class CalendarBoundsError(CalendarError):
    """Raised when requesting dates outside loaded calendar bounds."""


def _to_session_date(value: object) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        raise ValueError("Invalid date-like value.")
    if ts.tz is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.normalize()


@dataclass(frozen=True)
class MarketCalendar:
    sessions: pd.DatetimeIndex

    def __post_init__(self) -> None:
        if not isinstance(self.sessions, pd.DatetimeIndex):
            raise TypeError("MarketCalendar.sessions must be a pandas.DatetimeIndex.")
        if self.sessions.empty:
            raise ValueError("MarketCalendar.sessions must be non-empty.")
        normalized = pd.DatetimeIndex(self.sessions).sort_values().unique()
        object.__setattr__(self, "sessions", normalized)

    def is_session(self, date_like: object) -> bool:
        date = _to_session_date(date_like)
        return bool(date in self.sessions)

    def next_trading_day(self, date_like: object, n: int = 1) -> pd.Timestamp:
        if n < 1:
            raise ValueError("n must be >= 1.")
        date = _to_session_date(date_like)
        idx = int(self.sessions.searchsorted(date, side="right"))
        target = idx + n - 1
        if target >= len(self.sessions):
            raise CalendarBoundsError("Requested next session is outside calendar bounds.")
        return self.sessions[target]

    def prev_trading_day(self, date_like: object, n: int = 1) -> pd.Timestamp:
        if n < 1:
            raise ValueError("n must be >= 1.")
        date = _to_session_date(date_like)
        idx = int(self.sessions.searchsorted(date, side="left")) - 1
        target = idx - (n - 1)
        if target < 0:
            raise CalendarBoundsError("Requested previous session is outside calendar bounds.")
        return self.sessions[target]

    def sessions_between(
        self,
        start: object,
        end: object,
        *,
        inclusive: str = "both",
    ) -> pd.DatetimeIndex:
        start_ts = _to_session_date(start)
        end_ts = _to_session_date(end)
        if start_ts > end_ts:
            return pd.DatetimeIndex([], dtype="datetime64[ns]")

        include_left = inclusive in {"both", "left"}
        include_right = inclusive in {"both", "right"}

        left = int(self.sessions.searchsorted(start_ts, side="left" if include_left else "right"))
        right = int(self.sessions.searchsorted(end_ts, side="right" if include_right else "left"))
        return self.sessions[left:right]

    def trading_days_between(
        self,
        start: object,
        end: object,
        *,
        inclusive: str = "left",
    ) -> int:
        return int(len(self.sessions_between(start, end, inclusive=inclusive)))


def _resolve_default_calendar_path() -> Path:
    ref_dir = path_utils.reference_dir()
    candidates = (
        ref_dir / "trading_calendar.parquet",
        ref_dir / "market_calendar.parquet",
        ref_dir / "us_market_calendar.parquet",
    )
    for candidate in candidates:
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    names = ", ".join(str(path) for path in candidates)
    raise CalendarNotFoundError(f"No usable calendar artifact found. Checked: {names}")


@lru_cache(maxsize=8)
def _load_calendar_cached(path_str: str) -> MarketCalendar:
    path = Path(path_str)
    if not path.exists():
        raise CalendarNotFoundError(f"Calendar artifact does not exist: {path}")
    if path.stat().st_size == 0:
        raise CalendarCorruptError(f"Calendar artifact is empty: {path}")

    try:
        frame = pd.read_parquet(path)
    except Exception as exc:  # pragma: no cover - branch depends on parquet backend details
        raise CalendarCorruptError(f"Failed to read calendar parquet: {path}") from exc

    if "date" in frame.columns:
        date_values = frame["date"]
    elif isinstance(frame.index, pd.DatetimeIndex):
        date_values = frame.index.to_series(index=frame.index)
    else:
        raise CalendarCorruptError(
            "Calendar parquet must contain a 'date' column or datetime index."
        )

    if "is_session" in frame.columns:
        mask = frame["is_session"].fillna(False).astype(bool)
        date_values = date_values[mask]

    parsed = pd.to_datetime(date_values, errors="coerce", utc=True)
    if parsed.isna().any():
        raise CalendarCorruptError("Calendar contains non-parseable dates.")

    sessions = pd.DatetimeIndex(parsed.dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize())
    sessions = sessions.sort_values().unique()
    if len(sessions) == 0:
        raise CalendarCorruptError("Calendar artifact has no valid sessions.")

    return MarketCalendar(sessions=sessions)


def load_market_calendar(calendar_path: str | Path | None = None) -> MarketCalendar:
    path = Path(calendar_path) if calendar_path is not None else _resolve_default_calendar_path()
    if not path.is_absolute():
        path = path_utils.safe_join(path_utils.app_root(), str(path))
    return _load_calendar_cached(str(path.resolve()))


def next_trading_day(
    date_like: object,
    n: int = 1,
    *,
    calendar: MarketCalendar | None = None,
) -> pd.Timestamp:
    cal = calendar or load_market_calendar()
    return cal.next_trading_day(date_like, n=n)


def prev_trading_day(
    date_like: object,
    n: int = 1,
    *,
    calendar: MarketCalendar | None = None,
) -> pd.Timestamp:
    cal = calendar or load_market_calendar()
    return cal.prev_trading_day(date_like, n=n)


def trading_days_between(
    start: object,
    end: object,
    *,
    inclusive: str = "left",
    calendar: MarketCalendar | None = None,
) -> int:
    cal = calendar or load_market_calendar()
    return cal.trading_days_between(start, end, inclusive=inclusive)


__all__ = [
    "CalendarBoundsError",
    "CalendarCorruptError",
    "CalendarError",
    "CalendarNotFoundError",
    "MarketCalendar",
    "load_market_calendar",
    "next_trading_day",
    "prev_trading_day",
    "trading_days_between",
]
