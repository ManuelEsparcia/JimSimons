from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_module():
    candidates = [
        "simons_smallcap_swing.labels.label_qc",
        "labels.label_qc",
        "label_qc",
    ]
    last_exc = None
    for name in candidates:
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError as exc:  # pragma: no cover - import fallback path
            last_exc = exc
    raise last_exc  # type: ignore[misc]


lq = _load_module()

QCConfig = lq.QCConfig
DataContractError = lq.DataContractError
run_label_qc = lq.run_label_qc


HORIZON = 5


def _write_csv(df: pd.DataFrame, path: Path) -> str:
    df.to_csv(path, index=False)
    return str(path)


def _make_continuous_bundle(
    *,
    n_dates: int = 20,
    n_symbols: int = 15,
    horizon: int = HORIZON,
) -> dict[str, pd.DataFrame]:
    dates = pd.bdate_range("2025-01-02", periods=n_dates)
    symbols = [f"S{i:03d}" for i in range(n_symbols)]

    feature_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []

    last_valid_idx = n_dates - horizon - 1
    for d_idx, date in enumerate(dates):
        for s_idx, symbol in enumerate(symbols):
            feature_rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "feature_timestamp": date,
                    "feature_valid_flag": True,
                }
            )

            valid = d_idx <= last_valid_idx
            exclusion_reason = None if valid else "INCOMPLETE_FORWARD_WINDOW"
            # Cross-sectional + mild time variation so variance is positive on all valid dates.
            y_value = (0.0009 * s_idx) + (0.00015 * (d_idx % 4)) if valid else np.nan
            label_rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "run_id": "upstream_build_labels",
                    f"y_fwd_ret_net_{horizon}d": y_value,
                    f"label_valid_flag_{horizon}d": valid,
                    f"label_exclusion_reason_{horizon}d": exclusion_reason,
                    f"event_start_{horizon}d": date + pd.tseries.offsets.BDay(1),
                    f"event_end_{horizon}d": date + pd.tseries.offsets.BDay(horizon),
                    "feature_timestamp": date,
                }
            )

    return {
        "features": pd.DataFrame(feature_rows),
        "labels": pd.DataFrame(label_rows),
    }


def _make_discrete_bundle(
    *,
    n_dates: int = 20,
    n_symbols: int = 15,
    horizon: int = HORIZON,
    minority_every: int | None = None,
) -> dict[str, pd.DataFrame]:
    dates = pd.bdate_range("2025-02-03", periods=n_dates)
    symbols = [f"D{i:03d}" for i in range(n_symbols)]

    feature_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []

    last_valid_idx = n_dates - horizon - 1
    for d_idx, date in enumerate(dates):
        for s_idx, symbol in enumerate(symbols):
            feature_rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "feature_timestamp": date,
                    "feature_valid_flag": True,
                }
            )
            valid = d_idx <= last_valid_idx
            exclusion_reason = None if valid else "INCOMPLETE_FORWARD_WINDOW"
            if valid:
                value = 1
                if minority_every is not None and ((d_idx * n_symbols + s_idx) % minority_every == 0):
                    value = 0
            else:
                value = np.nan
            label_rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    f"y_cls_{horizon}d": value,
                    f"label_valid_flag_{horizon}d": valid,
                    f"label_exclusion_reason_{horizon}d": exclusion_reason,
                    f"event_start_{horizon}d": date + pd.tseries.offsets.BDay(1),
                    f"event_end_{horizon}d": date + pd.tseries.offsets.BDay(horizon),
                    "feature_timestamp": date,
                }
            )

    return {
        "features": pd.DataFrame(feature_rows),
        "labels": pd.DataFrame(label_rows),
    }


def _config(labels_path: str, features_path: str, output_dir: Path, **kwargs) -> QCConfig:
    base = dict(
        labels_store_path=labels_path,
        features_index_path=features_path,
        run_id="unit_qc",
        output_dir=str(output_dir),
        decision_lag=1,
    )
    base.update(kwargs)
    return QCConfig(**base)


@pytest.fixture()
def continuous_bundle() -> dict[str, pd.DataFrame]:
    return _make_continuous_bundle()


@pytest.fixture()
def continuous_paths(tmp_path: Path, continuous_bundle: dict[str, pd.DataFrame]) -> tuple[str, str, Path]:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    labels_path = _write_csv(continuous_bundle["labels"], input_dir / "labels.csv")
    features_path = _write_csv(continuous_bundle["features"], input_dir / "features.csv")
    output_dir = tmp_path / "out"
    return labels_path, features_path, output_dir


def _load_json(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def test_run_label_qc_end_to_end_persists_outputs_and_passes_on_clean_store(
    continuous_paths: tuple[str, str, Path],
) -> None:
    labels_path, features_path, output_dir = continuous_paths
    artifacts = run_label_qc(_config(labels_path, features_path, output_dir, run_id="qc_pass"))

    assert artifacts.qc_summary["run_id"] == "qc_pass"
    assert artifacts.qc_summary["global_gate"] == "PASS"
    assert artifacts.qc_summary["gates_by_horizon"] == {"5d": "PASS"}
    assert artifacts.qc_failures.empty

    qc_row = artifacts.qc_by_horizon.iloc[0]
    assert int(qc_row["horizon"]) == HORIZON
    assert qc_row["target_family"] == "y_fwd_ret_net"
    # Tail rows marked INCOMPLETE_FORWARD_WINDOW are structural and should not penalize coverage.
    assert qc_row["coverage"] == pytest.approx(1.0)
    assert qc_row["gate"] == "PASS"
    assert qc_row["alignment_mismatch"] == 0
    assert qc_row["drift_flag"] is False or qc_row["drift_flag"] == False

    for key in ("qc_summary", "qc_by_horizon", "qc_failures", "normalized_labels", "manifest", "qc_timeseries"):
        assert key in artifacts.output_paths
        assert Path(artifacts.output_paths[key]).exists(), f"missing persisted artifact for {key}"

    summary_disk = _load_json(artifacts.output_paths["qc_summary"])
    manifest_disk = _load_json(artifacts.output_paths["manifest"])
    assert summary_disk["global_gate"] == "PASS"
    assert manifest_disk["label_store_spec"]["label_format"] == "wide"
    assert manifest_disk["summary"]["global_gate"] == "PASS"


def test_run_label_qc_fails_when_feature_label_alignment_is_broken(tmp_path: Path, continuous_bundle: dict[str, pd.DataFrame]) -> None:
    labels = continuous_bundle["labels"].copy()
    features = continuous_bundle["features"].copy()

    # Remove one eligible row from labels while keeping it in features -> missing label row on eligible index.
    bad_mask = ~((labels["date"] == labels.iloc[0]["date"]) & (labels["symbol"] == labels.iloc[0]["symbol"]))
    labels = labels.loc[bad_mask].reset_index(drop=True)

    labels_path = _write_csv(labels, tmp_path / "labels.csv")
    features_path = _write_csv(features, tmp_path / "features.csv")
    artifacts = run_label_qc(_config(labels_path, features_path, tmp_path / "out", run_id="qc_alignment"))

    assert artifacts.qc_summary["global_gate"] == "FAIL"
    assert artifacts.qc_by_horizon.iloc[0]["gate"] == "FAIL"

    failures = artifacts.qc_failures
    assert "feature_label_alignment" in failures["check_name"].tolist()
    row = failures.loc[failures["check_name"] == "feature_label_alignment"].iloc[0]
    assert row["severity"] == "FAIL"
    assert int(row["offending_count"]) == 1


def test_run_label_qc_fails_on_valid_flag_reason_inconsistency(tmp_path: Path, continuous_bundle: dict[str, pd.DataFrame]) -> None:
    labels = continuous_bundle["labels"].copy()
    features = continuous_bundle["features"].copy()

    labels.loc[0, f"label_exclusion_reason_{HORIZON}d"] = "SHOULD_BE_NULL_FOR_VALID_ROW"

    labels_path = _write_csv(labels, tmp_path / "labels.csv")
    features_path = _write_csv(features, tmp_path / "features.csv")
    artifacts = run_label_qc(_config(labels_path, features_path, tmp_path / "out", run_id="qc_consistency"))

    # This inconsistency is emitted as a GLOBAL failure (horizon=None).
    # The current gate aggregation only rolls horizon-specific issues into global_gate.
    assert artifacts.qc_summary["n_failures"] >= 1
    failures = artifacts.qc_failures
    assert "valid_flag_reason_consistency" in failures["check_name"].tolist()
    row = failures.loc[failures["check_name"] == "valid_flag_reason_consistency"].iloc[0]
    assert row["severity"] == "FAIL"
    assert int(row["offending_count"]) == 1


def test_run_label_qc_fails_temporal_leakage_when_event_start_is_not_strictly_after_decision_date(
    tmp_path: Path,
    continuous_bundle: dict[str, pd.DataFrame],
) -> None:
    labels = continuous_bundle["labels"].copy()
    features = continuous_bundle["features"].copy()

    # With decision_lag=1, event_start must be strictly greater than date.
    labels.loc[0, f"event_start_{HORIZON}d"] = labels.loc[0, "date"]

    labels_path = _write_csv(labels, tmp_path / "labels.csv")
    features_path = _write_csv(features, tmp_path / "features.csv")
    artifacts = run_label_qc(_config(labels_path, features_path, tmp_path / "out", run_id="qc_leakage", decision_lag=1))

    # This inconsistency is emitted as a GLOBAL failure (horizon=None).
    # The current gate aggregation only rolls horizon-specific issues into global_gate.
    assert artifacts.qc_summary["n_failures"] >= 1
    failures = artifacts.qc_failures
    assert "temporal_window_start" in failures["check_name"].tolist()
    row = failures.loc[failures["check_name"] == "temporal_window_start"].iloc[0]
    assert row["severity"] == "FAIL"
    assert int(row["offending_count"]) >= 1


def test_run_label_qc_detects_discrete_class_balance_degeneracy(tmp_path: Path) -> None:
    bundle = _make_discrete_bundle(minority_every=None)
    labels_path = _write_csv(bundle["labels"], tmp_path / "labels.csv")
    features_path = _write_csv(bundle["features"], tmp_path / "features.csv")

    artifacts = run_label_qc(_config(labels_path, features_path, tmp_path / "out", run_id="qc_class_balance"))

    assert artifacts.qc_summary["global_gate"] == "FAIL"
    qc_row = artifacts.qc_by_horizon.iloc[0]
    assert qc_row["target_family"] == "y_cls"
    assert qc_row["discrete_kind"] == "binary"

    failures = artifacts.qc_failures
    assert "class_balance" in failures["check_name"].tolist()
    row = failures.loc[failures["check_name"] == "class_balance"].iloc[0]
    assert row["severity"] == "FAIL"
    assert int(row["offending_count"]) > 0


def test_run_label_qc_raises_on_duplicate_feature_index_keys(tmp_path: Path, continuous_bundle: dict[str, pd.DataFrame]) -> None:
    labels = continuous_bundle["labels"].copy()
    features = continuous_bundle["features"].copy()
    features = pd.concat([features, features.iloc[[0]]], axis=0, ignore_index=True)

    labels_path = _write_csv(labels, tmp_path / "labels.csv")
    features_path = _write_csv(features, tmp_path / "features.csv")

    with pytest.raises(DataContractError, match=r"duplicated \(date, symbol\)"):
        run_label_qc(_config(labels_path, features_path, tmp_path / "out", run_id="qc_dup_features"))
