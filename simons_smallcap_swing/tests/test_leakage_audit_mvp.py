from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from simons_core.io.parquet_store import read_parquet
from validation.leakage_audit import run_leakage_audit


def _seed_leakage_inputs(tmp_workspace: dict[str, Path]) -> dict[str, Path | list[pd.Timestamp]]:
    base = tmp_workspace["data"] / "leakage_audit_case"
    labels_dir = base / "labels"
    features_dir = base / "features"
    datasets_dir = base / "datasets"
    reference_dir = base / "reference"
    edgar_dir = base / "edgar"
    validation_dir = base / "validation"
    for directory in (labels_dir, features_dir, datasets_dir, reference_dir, edgar_dir, validation_dir):
        directory.mkdir(parents=True, exist_ok=True)

    sessions = pd.bdate_range("2026-01-05", periods=10, freq="B")
    decision_dates = sessions[:7]
    instruments = [("SIMA", "AAA"), ("SIMB", "BBB")]
    role_by_idx = {
        0: "train",
        1: "train",
        2: "train",
        3: "dropped_by_purge",
        4: "valid",
        5: "test",
        6: "dropped_by_embargo",
    }
    cv_role_by_idx = {
        0: "train",
        1: "train",
        2: "train",
        3: "dropped_by_purge",
        4: "valid",
        5: "dropped_by_embargo",
        6: "dropped_by_embargo",
    }

    calendar = pd.DataFrame({"date": sessions, "is_session": True})
    calendar_path = reference_dir / "trading_calendar.parquet"
    calendar.to_parquet(calendar_path, index=False)

    labels_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    dataset_rows: list[dict[str, object]] = []
    splits_rows: list[dict[str, object]] = []
    cv_rows: list[dict[str, object]] = []

    for idx, date in enumerate(decision_dates):
        entry_date = sessions[idx + 1]
        exit_date = sessions[idx + 2]
        split_role = role_by_idx[idx]
        cv_role = cv_role_by_idx[idx]
        for instrument_id, ticker in instruments:
            sign = 1.0 if instrument_id == "SIMA" else -1.0
            label_value = sign * 0.01 * float(idx + 1)
            ret_1d_lag1 = sign * 0.001 * float(idx + 1)
            ret_5d_lag1 = sign * 0.002 * float(idx + 1)
            mkt_breadth = 0.45 + 0.01 * float(idx)
            log_assets = 10.0 + (0.2 if instrument_id == "SIMB" else 0.0)

            labels_rows.append(
                {
                    "date": date,
                    "instrument_id": instrument_id,
                    "ticker": ticker,
                    "horizon_days": 1,
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "label_name": "fwd_ret_1d",
                    "label_value": label_value,
                    "price_entry": 100.0 + idx,
                    "price_exit": 101.0 + idx,
                    "source_price_field": "close_adj",
                }
            )

            feature_rows.append(
                {
                    "date": date,
                    "instrument_id": instrument_id,
                    "ticker": ticker,
                    "ret_1d_lag1": ret_1d_lag1,
                    "ret_5d_lag1": ret_5d_lag1,
                    "mkt_breadth_up_lag1": mkt_breadth,
                    "log_total_assets": log_assets,
                }
            )

            dataset_rows.append(
                {
                    "date": date,
                    "instrument_id": instrument_id,
                    "ticker": ticker,
                    "horizon_days": 1,
                    "label_name": "fwd_ret_1d",
                    "split_name": "holdout_temporal_purged",
                    "split_role": split_role,
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "target_value": label_value,
                    "target_type": "continuous_forward_return",
                    "ret_1d_lag1": ret_1d_lag1,
                    "ret_5d_lag1": ret_5d_lag1,
                    "mkt_breadth_up_lag1": mkt_breadth,
                    "log_total_assets": log_assets,
                }
            )

            splits_rows.append(
                {
                    "date": date,
                    "instrument_id": instrument_id,
                    "horizon_days": 1,
                    "label_name": "fwd_ret_1d",
                    "split_name": "holdout_temporal_purged",
                    "split_role": split_role,
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                }
            )

            cv_rows.append(
                {
                    "fold_id": 1,
                    "date": date,
                    "instrument_id": instrument_id,
                    "horizon_days": 1,
                    "label_name": "fwd_ret_1d",
                    "split_role": cv_role,
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                }
            )

    labels_path = labels_dir / "labels_forward.parquet"
    features_path = features_dir / "features_matrix.parquet"
    dataset_path = datasets_dir / "model_dataset.parquet"
    splits_path = labels_dir / "purged_splits.parquet"
    cv_path = labels_dir / "purged_cv_folds.parquet"

    pd.DataFrame(labels_rows).to_parquet(labels_path, index=False)
    pd.DataFrame(feature_rows).to_parquet(features_path, index=False)
    pd.DataFrame(dataset_rows).to_parquet(dataset_path, index=False)
    pd.DataFrame(splits_rows).to_parquet(splits_path, index=False)
    pd.DataFrame(cv_rows).to_parquet(cv_path, index=False)

    (labels_dir / "purged_splits.summary.json").write_text(
        json.dumps({"embargo_sessions": 1}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (labels_dir / "purged_cv_folds.summary.json").write_text(
        json.dumps({"embargo_sessions": 1}, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    fundamentals_rows: list[dict[str, object]] = []
    for instrument_id, _ticker in instruments:
        for asof in (sessions[2], sessions[5]):
            fundamentals_rows.append(
                {
                    "instrument_id": instrument_id,
                    "asof_date": pd.Timestamp(asof).tz_localize("UTC") + pd.Timedelta(hours=23, minutes=59),
                    "acceptance_ts": pd.Timestamp(asof).tz_localize("UTC") + pd.Timedelta(hours=12),
                    "metric_name": "Revenues",
                    "metric_value": 100_000_000.0,
                }
            )
    fundamentals_path = edgar_dir / "fundamentals_pit.parquet"
    pd.DataFrame(fundamentals_rows).to_parquet(fundamentals_path, index=False)

    return {
        "labels_path": labels_path,
        "features_path": features_path,
        "dataset_path": dataset_path,
        "calendar_path": calendar_path,
        "splits_path": splits_path,
        "cv_path": cv_path,
        "fundamentals_path": fundamentals_path,
        "validation_dir": validation_dir,
        "decision_dates": list(decision_dates),
    }


def _run_seeded_audit(paths: dict[str, Path | list[pd.Timestamp]], run_id: str) -> tuple[object, dict[str, object]]:
    result = run_leakage_audit(
        labels_path=paths["labels_path"],
        features_path=paths["features_path"],
        model_dataset_path=paths["dataset_path"],
        trading_calendar_path=paths["calendar_path"],
        purged_splits_path=paths["splits_path"],
        purged_cv_folds_path=paths["cv_path"],
        fundamentals_pit_path=paths["fundamentals_path"],
        output_dir=paths["validation_dir"],
        run_id=run_id,
    )
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    return result, summary


def test_leakage_audit_generates_artifacts_and_summary(tmp_workspace: dict[str, Path]) -> None:
    paths = _seed_leakage_inputs(tmp_workspace)
    result, summary = _run_seeded_audit(paths, run_id="test_leakage_audit_clean")

    assert result.findings_path.exists()
    assert result.metrics_path.exists()
    assert result.summary_path.exists()
    assert (result.findings_path.with_suffix(".parquet.manifest.json")).exists()
    assert (result.metrics_path.with_suffix(".parquet.manifest.json")).exists()

    findings = read_parquet(result.findings_path)
    metrics = read_parquet(result.metrics_path)
    assert len(findings) > 0
    assert len(metrics) > 0

    expected_summary_keys = {
        "overall_status",
        "n_checks_run",
        "n_fail",
        "n_warn",
        "n_pass",
        "failed_checks",
        "warning_checks",
        "key_findings",
        "input_paths",
    }
    assert expected_summary_keys.issubset(summary.keys())
    assert summary["overall_status"] in {"PASS", "WARN"}
    assert summary["n_checks_run"] > 0


def test_leakage_audit_flags_entry_date_leakage(tmp_workspace: dict[str, Path]) -> None:
    paths = _seed_leakage_inputs(tmp_workspace)
    labels = pd.read_parquet(paths["labels_path"]).copy()
    labels.loc[0, "entry_date"] = labels.loc[0, "date"]
    labels.to_parquet(paths["labels_path"], index=False)

    _result, summary = _run_seeded_audit(paths, run_id="test_leakage_audit_entry_leak")
    assert summary["overall_status"] == "FAIL"
    assert "labels_entry_after_decision" in summary["failed_checks"]


def test_leakage_audit_flags_train_overlap_in_splits(tmp_workspace: dict[str, Path]) -> None:
    paths = _seed_leakage_inputs(tmp_workspace)
    splits = pd.read_parquet(paths["splits_path"]).copy()
    target_date = paths["decision_dates"][3]
    mask = (pd.to_datetime(splits["date"]) == pd.Timestamp(target_date)) & (
        splits["instrument_id"].astype(str) == "SIMA"
    )
    assert mask.any()
    splits.loc[mask, "split_role"] = "train"
    splits.to_parquet(paths["splits_path"], index=False)

    _result, summary = _run_seeded_audit(paths, run_id="test_leakage_audit_split_overlap")
    assert summary["overall_status"] == "FAIL"
    assert "splits_train_eval_no_overlap" in set(summary["failed_checks"] + summary["warning_checks"])


def test_leakage_audit_flags_dataset_pk_duplicates(tmp_workspace: dict[str, Path]) -> None:
    paths = _seed_leakage_inputs(tmp_workspace)
    dataset = pd.read_parquet(paths["dataset_path"]).copy()
    dataset = pd.concat([dataset, dataset.iloc[[0]].copy()], ignore_index=True)
    dataset.to_parquet(paths["dataset_path"], index=False)

    result, summary = _run_seeded_audit(paths, run_id="test_leakage_audit_dataset_duplicates")
    findings = read_parquet(result.findings_path)
    dup_fail = findings[
        (findings["check_name"].astype(str) == "dataset_logical_pk_unique")
        & (findings["severity"].astype(str) == "FAIL")
    ]
    assert len(dup_fail) >= 1
    assert "dataset_logical_pk_unique" in summary["failed_checks"]

