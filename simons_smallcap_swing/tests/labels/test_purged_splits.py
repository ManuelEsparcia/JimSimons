from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_module():
    candidates = [
        "simons_smallcap_swing.labels.purged_splits",
        "labels.purged_splits",
        "purged_splits",
    ]
    last_exc = None
    for name in candidates:
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError as exc:  # pragma: no cover - import fallback path
            last_exc = exc
    raise last_exc  # type: ignore[misc]


ps = _load_module()

SplitConfig = ps.SplitConfig
WalkForwardSpec = ps.WalkForwardSpec
CPCVSpec = ps.CPCVSpec
ConfigError = ps.ConfigError
MetadataError = ps.MetadataError
build_purged_splits = ps.build_purged_splits


@pytest.fixture()
def calendar_11() -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-02", periods=11)
    return pd.DataFrame({"date": dates})


@pytest.fixture()
def labels_10(calendar_11: pd.DataFrame) -> pd.DataFrame:
    dates = calendar_11["date"].iloc[:10].tolist()
    rows = []
    sample_index = 0
    for date in dates:
        for symbol in ("AAA", "BBB"):
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "sample_index": sample_index,
                    "label_valid_flag": True,
                    "horizon_days": 1,
                }
            )
            sample_index += 1
    return pd.DataFrame(rows)


@pytest.fixture()
def labels_manifest_h1() -> dict[str, object]:
    return {
        "decision_lag": 0,
        "horizons": [1],
        "primary_horizon": 1,
        "labels_version": "unit-test-v1",
    }


@pytest.fixture()
def feature_manifest_memory_3() -> dict[str, object]:
    return {
        "primary_feature_memory": 3,
    }


@pytest.fixture()
def features_index_from_labels(labels_10: pd.DataFrame) -> pd.DataFrame:
    frame = labels_10[["date", "symbol", "sample_index"]].copy()
    frame["inclusion_flag"] = True
    return frame


def test_purged_kfold_purges_overlap_and_applies_embargo(
    calendar_11: pd.DataFrame,
    labels_10: pd.DataFrame,
    labels_manifest_h1: dict[str, object],
) -> None:
    config = SplitConfig(
        split_mode="purged_kfold",
        k_folds=2,
        decision_lag=0,
        horizon_policy="max",
        embargo_policy="fixed_days",
        fixed_embargo_days=2,
        min_train_obs=1,
        min_test_obs=1,
    )

    result = build_purged_splits(
        labels_df=labels_10,
        calendar_df=calendar_11,
        config=config,
        run_id="unit_test_run",
        labels_manifest=labels_manifest_h1,
    )

    fold_01_summary = result.fold_summary.loc[result.fold_summary["fold_id"] == "fold_01"].iloc[0]
    assert int(fold_01_summary["n_train_raw"]) == 10
    assert int(fold_01_summary["n_train_purged"]) == 8
    assert int(fold_01_summary["n_train_final"]) == 6
    assert int(fold_01_summary["effective_embargo_days"]) == 2
    assert float(fold_01_summary["pct_removed_by_purge"]) == pytest.approx(0.2)
    assert float(fold_01_summary["pct_removed_by_embargo"]) == pytest.approx(0.2)

    fold_01 = result.splits.loc[result.splits["fold_id"] == "fold_01"].copy()
    included_train = fold_01.loc[(fold_01["split_role"] == "train") & (fold_01["inclusion_flag"])]
    purged_train = fold_01.loc[(fold_01["split_role"] == "train") & (fold_01["is_purged"])]
    embargoed_train = fold_01.loc[
        (fold_01["split_role"] == "train") & (~fold_01["is_purged"]) & (fold_01["is_embargoed"])
    ]

    unique_dates = sorted(pd.to_datetime(labels_10["date"].unique()))
    assert set(pd.to_datetime(included_train["date"])) == set(unique_dates[7:10])
    assert set(pd.to_datetime(purged_train["date"])) == {unique_dates[5]}
    assert set(pd.to_datetime(embargoed_train["date"])) == {unique_dates[6]}

    leakage = result.leakage_validation.loc[result.leakage_validation["fold_id"] == "fold_01"]
    assert set(leakage["status"].unique()) == {"PASS"}
    assert result.manifest["run_id"] == "unit_test_run"
    assert result.manifest["h_eff"] == 1
    assert result.manifest["b_eff"] == 2


def test_walkforward_builds_expected_fold_geometry(
    labels_manifest_h1: dict[str, object],
) -> None:
    calendar = pd.DataFrame({"date": pd.bdate_range("2025-02-03", periods=12)})
    label_dates = calendar["date"].iloc[:11].tolist()
    labels = pd.DataFrame(
        [
            {
                "date": date,
                "symbol": "AAA",
                "label_valid_flag": True,
                "sample_index": i,
                "horizon_days": 1,
            }
            for i, date in enumerate(label_dates)
        ]
    )

    config = SplitConfig(
        split_mode="walkforward",
        decision_lag=0,
        horizon_policy="max",
        embargo_policy="fixed_days",
        fixed_embargo_days=0,
        walkforward=WalkForwardSpec(
            test_size_dates=2,
            step_size_dates=2,
            valid_size_dates=1,
            min_train_size_dates=4,
            expanding_train=True,
        ),
        min_train_obs=1,
        min_test_obs=1,
    )

    result = build_purged_splits(
        labels_df=labels,
        calendar_df=calendar,
        config=config,
        run_id="wf_run",
        labels_manifest=labels_manifest_h1,
    )

    assert list(result.fold_summary["fold_id"]) == ["fold_01", "fold_02", "fold_03"]
    assert list(result.fold_summary["n_valid"]) == [1, 1, 1]
    assert list(result.fold_summary["n_test"]) == [2, 2, 2]

    fold_01 = result.splits.loc[result.splits["fold_id"] == "fold_01"]
    train_dates = sorted(pd.to_datetime(fold_01.loc[fold_01["split_role"] == "train", "date"]).unique())
    valid_dates = sorted(pd.to_datetime(fold_01.loc[fold_01["split_role"] == "valid", "date"]).unique())
    test_dates = sorted(pd.to_datetime(fold_01.loc[fold_01["split_role"] == "test", "date"]).unique())

    assert len(train_dates) == 4
    assert len(valid_dates) == 1
    assert len(test_dates) == 2
    assert max(train_dates) < min(valid_dates) < min(test_dates)


def test_cpcv_builds_expected_number_of_combinations(
    labels_manifest_h1: dict[str, object],
) -> None:
    calendar = pd.DataFrame({"date": pd.bdate_range("2025-03-03", periods=9)})
    label_dates = calendar["date"].iloc[:8].tolist()
    labels = pd.DataFrame(
        [
            {
                "date": date,
                "symbol": "AAA",
                "label_valid_flag": True,
                "sample_index": i,
                "horizon_days": 1,
            }
            for i, date in enumerate(label_dates)
        ]
    )

    config = SplitConfig(
        split_mode="cpcv",
        decision_lag=0,
        horizon_policy="max",
        embargo_policy="fixed_days",
        fixed_embargo_days=0,
        cpcv=CPCVSpec(n_blocks=4, n_test_blocks=2),
        min_train_obs=1,
        min_test_obs=1,
    )

    result = build_purged_splits(
        labels_df=labels,
        calendar_df=calendar,
        config=config,
        run_id="cpcv_run",
        labels_manifest=labels_manifest_h1,
    )

    assert len(result.fold_summary) == 6
    assert set(result.fold_summary["n_test"].tolist()) == {4}
    assert set(result.fold_summary["split_mode"].tolist()) == {"cpcv"}


def test_per_label_horizon_policy_preserves_row_level_effective_horizon() -> None:
    calendar = pd.DataFrame({"date": pd.bdate_range("2025-04-01", periods=9)})
    label_dates = calendar["date"].iloc[:6].tolist()
    rows = []
    horizons = [1, 2, 3, 1, 2, 3]
    for i, (date, horizon) in enumerate(zip(label_dates, horizons)):
        rows.append(
            {
                "date": date,
                "symbol": "AAA",
                "sample_index": i,
                "label_valid_flag": True,
                "horizon_days": horizon,
            }
        )
    labels = pd.DataFrame(rows)

    config = SplitConfig(
        split_mode="purged_kfold",
        k_folds=2,
        decision_lag=0,
        horizon_policy="per_label",
        horizon_col="horizon_days",
        embargo_policy="fixed_days",
        fixed_embargo_days=0,
        min_train_obs=0,
        min_test_obs=1,
    )

    result = build_purged_splits(
        labels_df=labels,
        calendar_df=calendar,
        config=config,
        run_id="per_label_run",
        labels_manifest={"decision_lag": 0, "labels_version": "unit-test-v2"},
    )

    observed = (
        result.splits[["date", "symbol", "effective_horizon_days"]]
        .drop_duplicates()
        .sort_values(["date", "symbol"])
        .reset_index(drop=True)
    )
    expected = (
        labels[["date", "symbol", "horizon_days"]]
        .rename(columns={"horizon_days": "effective_horizon_days"})
        .sort_values(["date", "symbol"])
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(observed, expected)


def test_reconcile_with_features_index_filters_to_allowed_rows(
    calendar_11: pd.DataFrame,
    labels_10: pd.DataFrame,
    labels_manifest_h1: dict[str, object],
    features_index_from_labels: pd.DataFrame,
) -> None:
    allowed = features_index_from_labels.copy()
    allowed["inclusion_flag"] = False
    allowed.loc[allowed["sample_index"] < 6, "inclusion_flag"] = True

    config = SplitConfig(
        split_mode="purged_kfold",
        k_folds=2,
        decision_lag=0,
        horizon_policy="max",
        embargo_policy="fixed_days",
        fixed_embargo_days=0,
        min_train_obs=0,
        min_test_obs=1,
    )

    result = build_purged_splits(
        labels_df=labels_10,
        calendar_df=calendar_11,
        config=config,
        run_id="features_index_run",
        labels_manifest=labels_manifest_h1,
        features_index_df=allowed,
    )

    surviving_sample_indices = set(
        result.splits["sample_index"].dropna().astype(int).unique().tolist()
    )
    assert surviving_sample_indices == set(range(6))


def test_missing_label_valid_flag_raises(calendar_11: pd.DataFrame) -> None:
    labels = pd.DataFrame(
        {
            "date": calendar_11["date"].iloc[:8],
            "symbol": ["AAA"] * 8,
            "sample_index": list(range(8)),
            "horizon_days": [1] * 8,
        }
    )
    config = SplitConfig(
        split_mode="purged_kfold",
        k_folds=2,
        decision_lag=0,
        horizon_policy="max",
        embargo_policy="fixed_days",
        fixed_embargo_days=0,
        require_label_valid_flag=True,
    )

    with pytest.raises(ConfigError, match="label_valid_flag"):
        build_purged_splits(
            labels_df=labels,
            calendar_df=calendar_11,
            config=config,
            run_id="missing_flag",
            labels_manifest={"decision_lag": 0, "horizons": [1]},
        )


def test_interval_beyond_calendar_raises_metadata_error() -> None:
    calendar = pd.DataFrame({"date": pd.bdate_range("2025-05-05", periods=5)})
    labels = pd.DataFrame(
        {
            "date": calendar["date"],
            "symbol": ["AAA"] * 5,
            "sample_index": list(range(5)),
            "label_valid_flag": [True] * 5,
            "horizon_days": [2] * 5,
        }
    )
    config = SplitConfig(
        split_mode="purged_kfold",
        k_folds=2,
        decision_lag=0,
        horizon_policy="max",
        embargo_policy="fixed_days",
        fixed_embargo_days=0,
    )

    with pytest.raises(MetadataError, match="beyond available master calendar coverage"):
        build_purged_splits(
            labels_df=labels,
            calendar_df=calendar,
            config=config,
            run_id="bad_calendar",
            labels_manifest={"decision_lag": 0, "horizons": [2]},
        )
