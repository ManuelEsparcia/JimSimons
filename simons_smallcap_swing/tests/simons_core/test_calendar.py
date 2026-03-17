from __future__ import annotations

from datetime import date

import pandas as pd
import pytest


pytestmark = [pytest.mark.simons_core, pytest.mark.calendar]


def test_load_market_calendar_basic_properties(calendar_mod, market_calendar):
    assert market_calendar.market == "US_EQ"
    assert market_calendar.timezone == calendar_mod.CANONICAL_TIMEZONE
    assert market_calendar.backend_name == "exchange_calendars"
    assert isinstance(market_calendar.sessions, pd.DatetimeIndex)
    assert isinstance(market_calendar.opens, pd.DatetimeIndex)
    assert isinstance(market_calendar.closes, pd.DatetimeIndex)
    assert len(market_calendar.sessions) == len(market_calendar.opens) == len(market_calendar.closes)
    assert market_calendar.sessions.is_monotonic_increasing
    assert market_calendar.sessions.is_unique
    assert str(market_calendar.opens.tz) == calendar_mod.CANONICAL_TIMEZONE
    assert str(market_calendar.closes.tz) == calendar_mod.CANONICAL_TIMEZONE


def test_market_calendar_coverage_and_boundaries(market_calendar):
    assert market_calendar.first_session == date(2024, 7, 1)
    assert market_calendar.last_session == date(2024, 7, 10)
    assert market_calendar.coverage == (date(2024, 7, 1), date(2024, 7, 10))


def test_schedule_property_contains_expected_columns(calendar_mod, market_calendar):
    schedule = market_calendar.schedule
    assert list(schedule.columns) == ["open", "close", "session_type"]
    assert schedule.index.equals(market_calendar.sessions)
    assert schedule.loc[pd.Timestamp("2024-07-03"), "session_type"] == calendar_mod.SessionType.EARLY_CLOSE


def test_is_trading_day_for_session_weekend_and_holiday(market_calendar):
    assert market_calendar.is_trading_day("2024-07-03") is True
    assert market_calendar.is_trading_day("2024-07-06") is False
    assert market_calendar.is_trading_day("2024-07-04") is False


def test_session_open_close_and_type_for_early_close(calendar_mod, market_calendar):
    open_ts = market_calendar.session_open("2024-07-03")
    close_ts = market_calendar.session_close("2024-07-03")
    stype = market_calendar.session_type("2024-07-03")
    assert open_ts == pd.Timestamp("2024-07-03 09:30:00", tz="America/New_York")
    assert close_ts == pd.Timestamp("2024-07-03 13:00:00", tz="America/New_York")
    assert stype == calendar_mod.SessionType.EARLY_CLOSE


def test_session_accessors_raise_on_non_session(calendar_mod, market_calendar):
    with pytest.raises(calendar_mod.CalendarRangeError):
        market_calendar.session_open("2024-07-04")
    with pytest.raises(calendar_mod.CalendarRangeError):
        market_calendar.session_close("2024-07-06")
    with pytest.raises(calendar_mod.CalendarRangeError):
        market_calendar.session_type("2024-07-04")


def test_next_trading_day_strictly_posterior_on_session_and_weekend(market_calendar):
    assert market_calendar.next_trading_day("2024-07-03") == date(2024, 7, 5)
    assert market_calendar.next_trading_day("2024-07-06") == date(2024, 7, 8)


def test_prev_trading_day_strictly_anterior_on_session_and_weekend(market_calendar):
    assert market_calendar.prev_trading_day("2024-07-08") == date(2024, 7, 5)
    assert market_calendar.prev_trading_day("2024-07-07") == date(2024, 7, 5)


def test_next_prev_trading_day_support_n_parameter(market_calendar):
    assert market_calendar.next_trading_day("2024-07-01", n=3) == date(2024, 7, 5)
    assert market_calendar.prev_trading_day("2024-07-10", n=2) == date(2024, 7, 8)


def test_next_prev_trading_day_validate_n(calendar_mod, market_calendar):
    with pytest.raises(ValueError, match="n must be >= 1"):
        market_calendar.next_trading_day("2024-07-03", n=0)
    with pytest.raises(ValueError, match="n must be >= 1"):
        market_calendar.prev_trading_day("2024-07-03", n=0)


def test_next_prev_trading_day_raise_on_range_exhaustion(calendar_mod, market_calendar):
    with pytest.raises(calendar_mod.CalendarRangeError):
        market_calendar.next_trading_day("2024-07-10")
    with pytest.raises(calendar_mod.CalendarRangeError):
        market_calendar.prev_trading_day("2024-07-01")


@pytest.mark.parametrize(
    ("inclusive", "expected"),
    [
        ("left", 6),
        ("right", 6),
        ("both", 7),
        ("neither", 5),
    ],
)
def test_trading_days_between_interval_semantics(market_calendar, inclusive, expected):
    observed = market_calendar.trading_days_between("2024-07-01", "2024-07-10", inclusive=inclusive)
    assert observed == expected


def test_trading_days_between_same_day_semantics(market_calendar):
    assert market_calendar.trading_days_between("2024-07-03", "2024-07-03", inclusive="both") == 1
    assert market_calendar.trading_days_between("2024-07-03", "2024-07-03", inclusive="left") == 0
    assert market_calendar.trading_days_between("2024-07-03", "2024-07-03", inclusive="right") == 0
    assert market_calendar.trading_days_between("2024-07-03", "2024-07-03", inclusive="neither") == 0


def test_trading_days_between_errors(calendar_mod, market_calendar):
    with pytest.raises(ValueError, match="end must be >= start"):
        market_calendar.trading_days_between("2024-07-10", "2024-07-01")
    with pytest.raises(ValueError, match="inclusive must be one of"):
        market_calendar.trading_days_between("2024-07-01", "2024-07-10", inclusive="bad")  # type: ignore[arg-type]


def test_align_to_session_next_open_prev_close_and_session_label(market_calendar):
    assert market_calendar.align_to_session("2024-07-03 12:15", mode="next_open") == pd.Timestamp(
        "2024-07-05 09:30:00", tz="America/New_York"
    )
    assert market_calendar.align_to_session("2024-07-03 12:15", mode="prev_close") == pd.Timestamp(
        "2024-07-02 16:00:00", tz="America/New_York"
    )
    assert market_calendar.align_to_session("2024-07-03 12:15", mode="session_label") == pd.Timestamp(
        "2024-07-03 00:00:00", tz="America/New_York"
    )


def test_align_to_session_session_label_on_holiday_chooses_next_session(market_calendar):
    observed = market_calendar.align_to_session("2024-07-04 11:00", mode="session_label")
    assert observed == pd.Timestamp("2024-07-05 00:00:00", tz="America/New_York")


def test_align_to_session_validate_mode_and_range(calendar_mod, market_calendar):
    with pytest.raises(ValueError, match="mode must be one of"):
        market_calendar.align_to_session("2024-07-03 12:00", mode="bad")  # type: ignore[arg-type]
    with pytest.raises(calendar_mod.CalendarRangeError):
        market_calendar.align_to_session("2024-07-10 18:00", mode="next_open")
    with pytest.raises(calendar_mod.CalendarRangeError):
        market_calendar.align_to_session("2024-07-01 08:00", mode="prev_close")
    with pytest.raises(calendar_mod.CalendarRangeError):
        market_calendar.align_to_session("2024-07-11 10:00", mode="session_label")


def test_normalize_timestamp_accepts_naive_and_aware_inputs(market_calendar):
    naive = market_calendar._normalize_timestamp("2024-07-03 09:30")
    aware = market_calendar._normalize_timestamp(pd.Timestamp("2024-07-03 13:30:00", tz="UTC"))
    assert naive == pd.Timestamp("2024-07-03 09:30:00", tz="America/New_York")
    assert aware == pd.Timestamp("2024-07-03 09:30:00", tz="America/New_York")


def test_normalize_timestamp_dst_nonexistent_raises(calendar_mod, market_calendar):
    with pytest.raises(calendar_mod.TimestampNormalizationError):
        market_calendar._normalize_timestamp("2024-03-10 02:30:00")


def test_normalize_session_label_returns_tz_naive_midnight(market_calendar):
    label = market_calendar._normalize_session_label("2024-07-03 12:15")
    assert label == pd.Timestamp("2024-07-03")
    assert label.tzinfo is None


def test_load_market_calendar_uses_cache(calendar_mod):
    a = calendar_mod.load_market_calendar("US_EQ", "2024-07-01", "2024-07-10")
    b = calendar_mod.load_market_calendar("US_EQ", "2024-07-01", "2024-07-10")
    assert a is b


def test_load_market_calendar_rejects_unsupported_market(calendar_mod):
    with pytest.raises(calendar_mod.UnsupportedMarketError):
        calendar_mod.load_market_calendar("LSE", "2024-07-01", "2024-07-10")


def test_load_market_calendar_rejects_bad_range(calendar_mod):
    with pytest.raises(ValueError, match="end_date must be >= start_date"):
        calendar_mod.load_market_calendar("US_EQ", "2024-07-10", "2024-07-01")


def test_load_market_calendar_rejects_empty_session_range(calendar_mod):
    with pytest.raises(calendar_mod.CalendarRangeError, match="No sessions available"):
        calendar_mod.load_market_calendar("US_EQ", "2024-07-04", "2024-07-04")


def test_top_level_wrappers_match_instance_methods(calendar_mod, market_calendar):
    kwargs = {"market": "US_EQ", "start_date": "2024-07-01", "end_date": "2024-07-10"}
    assert calendar_mod.next_trading_day("2024-07-03", **kwargs) == market_calendar.next_trading_day("2024-07-03")
    assert calendar_mod.prev_trading_day("2024-07-08", **kwargs) == market_calendar.prev_trading_day("2024-07-08")
    assert calendar_mod.trading_days_between("2024-07-01", "2024-07-10", inclusive="both", **kwargs) == \
        market_calendar.trading_days_between("2024-07-01", "2024-07-10", inclusive="both")
    assert calendar_mod.align_to_session("2024-07-03 12:15", mode="session_label", **kwargs) == \
        market_calendar.align_to_session("2024-07-03 12:15", mode="session_label")


def test_market_alias_normalization(calendar_mod):
    assert calendar_mod._normalize_market_alias("us_eq") == "US_EQ"
    assert calendar_mod._normalize_market_alias("NYSE") == "NYSE"
    assert calendar_mod._normalize_market_alias("nasdaq") == "NASDAQ"


def test_coerce_date_string_normalizes_naive_and_aware(calendar_mod):
    naive = calendar_mod._coerce_date_string("2024-07-03 12:00")
    aware = calendar_mod._coerce_date_string(pd.Timestamp("2024-07-03 16:00:00", tz="UTC"))
    assert naive == "2024-07-03"
    assert aware == "2024-07-03"


def test_market_calendar_dataclass_post_init_validates_lengths_and_timezones(calendar_mod):
    sessions = pd.DatetimeIndex(pd.to_datetime(["2024-07-01", "2024-07-02"]))
    opens = pd.DatetimeIndex(pd.to_datetime(["2024-07-01 09:30", "2024-07-02 09:30"]).tz_localize("America/New_York"))
    closes = pd.DatetimeIndex(pd.to_datetime(["2024-07-01 16:00"]).tz_localize("America/New_York"))
    with pytest.raises(ValueError, match="identical lengths"):
        calendar_mod.MarketCalendar(
            market="US_EQ",
            timezone="America/New_York",
            sessions=sessions,
            opens=opens,
            closes=closes,
        )


def test_market_calendar_dataclass_post_init_requires_aware_timestamps(calendar_mod):
    sessions = pd.DatetimeIndex(pd.to_datetime(["2024-07-01"]))
    opens = pd.DatetimeIndex(pd.to_datetime(["2024-07-01 09:30"]))
    closes = pd.DatetimeIndex(pd.to_datetime(["2024-07-01 16:00"]))
    with pytest.raises(ValueError, match="timezone-aware"):
        calendar_mod.MarketCalendar(
            market="US_EQ",
            timezone="America/New_York",
            sessions=sessions,
            opens=opens,
            closes=closes,
        )
