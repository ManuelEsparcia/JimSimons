from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest


pytestmark = pytest.mark.math


def test_nanmean1d_omits_nonfinite(stats_mod, vector_with_nonfinite):
    value = stats_mod.nanmean1d(vector_with_nonfinite)
    assert math.isclose(value, 2.0, rel_tol=0.0, abs_tol=1e-12)


def test_nanvar1d_matches_numpy_on_finite_subset(stats_mod, vector_with_nonfinite):
    value = stats_mod.nanvar1d(vector_with_nonfinite, ddof=1)
    expected = float(np.var(np.array([1.0, 2.0, 3.0]), ddof=1))
    assert math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-12)


def test_nanstd1d_matches_numpy_on_clean_vector(stats_mod, clean_vector):
    value = stats_mod.nanstd1d(clean_vector, ddof=1)
    expected = float(np.std(clean_vector, ddof=1))
    assert math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-12)


def test_sem1d_matches_std_over_sqrt_n(stats_mod, clean_vector):
    value = stats_mod.sem1d(clean_vector, ddof=1)
    expected = float(np.std(clean_vector, ddof=1) / math.sqrt(clean_vector.size))
    assert math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-12)


def test_sem1d_returns_nan_when_insufficient_data(stats_mod):
    value = stats_mod.sem1d(np.array([5.0]), ddof=1)
    assert math.isnan(value)


def test_rolling_mean_preserves_index_and_values(stats_mod, return_series, assert_allclose):
    out = stats_mod.rolling_mean(return_series, window=3)
    expected = return_series.rolling(window=3, min_periods=3).mean()

    assert out.index.equals(return_series.index)
    assert out.name == return_series.name
    assert_allclose(out.to_numpy(), expected.to_numpy())


def test_rolling_std_matches_pandas(stats_mod, return_series, assert_allclose):
    out = stats_mod.rolling_std(return_series, window=3, ddof=1)
    expected = return_series.rolling(window=3, min_periods=3).std(ddof=1)
    assert_allclose(out.to_numpy(), expected.to_numpy())


def test_rolling_zscore_constant_window_returns_zero_after_warmup(stats_mod, assert_allclose):
    s = pd.Series([5.0, 5.0, 5.0, 5.0], dtype=float)
    out = stats_mod.rolling_zscore(s, window=2)
    expected = np.array([np.nan, 0.0, 0.0, 0.0], dtype=float)
    assert_allclose(out.to_numpy(), expected)


def test_rolling_zscore_rejects_nonpositive_eps(stats_mod, return_series):
    with pytest.raises(stats_mod.InvalidInputError, match="eps must be > 0"):
        stats_mod.rolling_zscore(return_series, window=3, eps=0.0)


def test_pearson_corr_perfect_positive(stats_mod, pearson_pair_perfect_pos):
    x, y = pearson_pair_perfect_pos
    value = stats_mod.pearson_corr(x, y)
    assert math.isclose(value, 1.0, rel_tol=0.0, abs_tol=1e-12)


def test_pearson_corr_perfect_negative(stats_mod, pearson_pair_perfect_neg):
    x, y = pearson_pair_perfect_neg
    value = stats_mod.pearson_corr(x, y)
    assert math.isclose(value, -1.0, rel_tol=0.0, abs_tol=1e-12)


def test_pearson_corr_pairwise_finite_handling(stats_mod):
    x = np.array([1.0, 2.0, np.nan, 4.0, 5.0], dtype=float)
    y = np.array([2.0, 4.0, 6.0, np.nan, 10.0], dtype=float)
    value = stats_mod.pearson_corr(x, y)
    assert math.isclose(value, 1.0, rel_tol=0.0, abs_tol=1e-12)


def test_pearson_corr_constant_series_returns_nan(stats_mod):
    x = np.array([1.0, 1.0, 1.0, 1.0], dtype=float)
    y = np.array([2.0, 3.0, 4.0, 5.0], dtype=float)
    value = stats_mod.pearson_corr(x, y)
    assert math.isnan(value)


def test_spearman_corr_handles_ties_correctly(stats_mod, spearman_pair_with_ties):
    x, y = spearman_pair_with_ties
    value = stats_mod.spearman_corr(x, y)

    rx = np.array([1.5, 1.5, 3.0, 4.5, 4.5], dtype=float)
    ry = np.array([1.5, 1.5, 3.0, 4.0, 5.0], dtype=float)
    expected = float(np.corrcoef(rx, ry)[0, 1])
    assert math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-12)


def test_spearman_ic_is_alias_of_spearman_corr(stats_mod, spearman_pair_with_ties):
    x, y = spearman_pair_with_ties
    assert math.isclose(
        stats_mod.spearman_ic(x, y),
        stats_mod.spearman_corr(x, y),
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def test_winsorize_preserves_nonfinite_positions_and_clips(stats_mod):
    x = np.array([1.0, 2.0, 100.0, np.nan, np.inf, -np.inf], dtype=float)
    out = stats_mod.winsorize(x, lower_q=0.0, upper_q=0.5)

    assert out.shape == x.shape
    assert np.isnan(out[3])
    assert np.isposinf(out[4])
    assert np.isneginf(out[5])
    assert math.isclose(float(out[0]), 1.0, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(float(out[1]), 2.0, rel_tol=0.0, abs_tol=1e-12)
    assert out[2] <= 2.0 + 1e-12


def test_winsorize_rejects_invalid_quantile_order(stats_mod, clean_vector):
    with pytest.raises(stats_mod.InvalidInputError, match="lower_q < upper_q"):
        stats_mod.winsorize(clean_vector, lower_q=0.9, upper_q=0.1)


def test_winsorize_series_preserves_index_and_name(stats_mod, return_series):
    out = stats_mod.winsorize_series(return_series, lower_q=0.1, upper_q=0.9)
    assert isinstance(out, pd.Series)
    assert out.index.equals(return_series.index)
    assert out.name == return_series.name


def test_sharpe_ratio_matches_manual_formula_with_scalar_rf(stats_mod, return_series):
    r = return_series.to_numpy(dtype=float)
    value = stats_mod.sharpe_ratio(r, rf=0.0, periods_per_year=252, ddof=1)
    expected = float(np.mean(r) / np.std(r, ddof=1) * math.sqrt(252))
    assert math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-12)


def test_sharpe_ratio_supports_array_like_benchmark(stats_mod, return_series, benchmark_series):
    r = return_series.to_numpy(dtype=float)
    rf = benchmark_series.to_numpy(dtype=float)
    value = stats_mod.sharpe_ratio(r, rf=rf, periods_per_year=252, ddof=1)

    excess = r - rf
    expected = float(np.mean(excess) / np.std(excess, ddof=1) * math.sqrt(252))
    assert math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-12)


def test_sharpe_ratio_zero_volatility_returns_nan(stats_mod):
    r = np.array([0.01, 0.01, 0.01, 0.01], dtype=float)
    value = stats_mod.sharpe_ratio(r, rf=0.0)
    assert math.isnan(value)


def test_sortino_ratio_matches_manual_formula(stats_mod):
    r = np.array([0.01, -0.02, 0.03, -0.01], dtype=float)
    value = stats_mod.sortino_ratio(r, target=0.0, periods_per_year=252)

    downside = np.minimum(r, 0.0)
    downside_dev = float(np.sqrt(np.mean(np.square(downside))))
    expected = float(np.mean(r) / downside_dev * math.sqrt(252))
    assert math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-12)


def test_sortino_ratio_no_downside_returns_nan(stats_mod):
    r = np.array([0.01, 0.02, 0.03], dtype=float)
    value = stats_mod.sortino_ratio(r, target=0.0)
    assert math.isnan(value)


def test_drawdown_series_matches_known_path(stats_mod, drawdown_returns, assert_allclose):
    out = stats_mod.drawdown_series(drawdown_returns)
    expected = np.array([0.0, -0.1, -0.055, -0.1495], dtype=float)
    assert_allclose(out, expected, atol=1e-10)


def test_drawdown_series_preserves_nan_positions(stats_mod, assert_allclose):
    r = np.array([0.10, np.nan, -0.10, 0.05], dtype=float)
    out = stats_mod.drawdown_series(r)
    expected = np.array([0.0, np.nan, -0.1, -0.055], dtype=float)
    assert_allclose(out, expected, atol=1e-10)


def test_drawdown_series_rejects_returns_less_than_minus_one(stats_mod):
    with pytest.raises(stats_mod.InvalidInputError, match="Simple returns < -1.0"):
        stats_mod.drawdown_series(np.array([0.1, -1.1], dtype=float))


def test_max_drawdown_matches_known_value(stats_mod, drawdown_returns):
    value = stats_mod.max_drawdown(drawdown_returns)
    assert math.isclose(value, -0.1495, rel_tol=0.0, abs_tol=1e-12)


def test_drawdown_details_detects_unrecovered_drawdown(stats_mod):
    r = np.array([0.10, -0.20, 0.05, -0.10, 0.20], dtype=float)
    details = stats_mod.drawdown_details(r)

    assert math.isclose(details.max_drawdown, -0.244, rel_tol=0.0, abs_tol=1e-12)
    assert details.peak_index == 0
    assert details.trough_index == 3
    assert details.recovery_index is None
    assert details.duration == 3
    assert details.recovery_duration is None


def test_drawdown_details_detects_recovery(stats_mod):
    r = np.array([0.10, -0.20, 0.30], dtype=float)
    details = stats_mod.drawdown_details(r)

    assert details.peak_index == 0
    assert details.trough_index == 1
    assert details.recovery_index == 2
    assert details.duration == 1
    assert details.recovery_duration == 1


def test_cagr_matches_one_year_compounding(stats_mod, monotone_gain_returns):
    value = stats_mod.cagr(monotone_gain_returns, periods_per_year=4)
    expected = float(np.prod(1.0 + monotone_gain_returns) - 1.0)
    assert math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-12)


def test_cagr_terminal_zero_wealth_returns_minus_one(stats_mod):
    value = stats_mod.cagr(np.array([-1.0], dtype=float), periods_per_year=252)
    assert math.isclose(value, -1.0, rel_tol=0.0, abs_tol=0.0)


def test_calmar_ratio_annual_mean_matches_manual_formula(stats_mod, drawdown_returns):
    value = stats_mod.calmar_ratio(drawdown_returns, periods_per_year=252, method="annual_mean")
    expected = float(np.mean(drawdown_returns) * 252 / abs(stats_mod.max_drawdown(drawdown_returns)))
    assert math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-12)


def test_calmar_ratio_cagr_matches_manual_formula(stats_mod, drawdown_returns):
    value = stats_mod.calmar_ratio(drawdown_returns, periods_per_year=252, method="cagr")
    expected = float(stats_mod.cagr(drawdown_returns, periods_per_year=252) / abs(stats_mod.max_drawdown(drawdown_returns)))
    assert math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-12)


def test_calmar_ratio_rejects_invalid_method(stats_mod, drawdown_returns):
    with pytest.raises(stats_mod.InvalidInputError, match="Unsupported Calmar method"):
        stats_mod.calmar_ratio(drawdown_returns, method="mean_over_mdd")


def test_self_test_returns_ok_payload(stats_mod):
    payload = stats_mod._self_test()
    assert payload["status"] == "ok"
    assert payload["tests"] >= 10
    assert "drawdown_details_example" in payload
