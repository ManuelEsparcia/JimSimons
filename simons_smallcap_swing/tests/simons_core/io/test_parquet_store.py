from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


pytestmark = [pytest.mark.io, pytest.mark.filesystem, pytest.mark.parquet]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dataset_path(datasets_root: Path, name: str = "prices_adjusted") -> Path:
    path = datasets_root / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _current_pointer(path: Path) -> Path:
    return path / "CURRENT.json"


def _versions_dir(path: Path) -> Path:
    return path / "_versions"


def _version_dir(path: Path, version: str) -> Path:
    return _versions_dir(path) / version


def _parquet_files_under(path: Path) -> list[Path]:
    return sorted(path.rglob("*.parquet"))


def _first_data_file(dataset_root: Path, dataset_version: str) -> Path:
    files = _parquet_files_under(_version_dir(dataset_root, dataset_version) / "data")
    assert files, "Expected at least one parquet data file"
    return files[0]


def _flip_last_byte(path: Path) -> None:
    blob = path.read_bytes()
    if not blob:
        raise AssertionError(f"Cannot corrupt empty file: {path}")
    mutated = blob[:-1] + bytes([blob[-1] ^ 0x01])
    path.write_bytes(mutated)


# ---------------------------------------------------------------------------
# Compatibility matrix
# ---------------------------------------------------------------------------


def test_assess_schema_compatibility_pass(parquet_store_mod, stored_schema_meta):
    report = parquet_store_mod.assess_schema_compatibility(stored_schema_meta, stored_schema_meta)
    assert report.status == "PASS"
    assert report.issues == ()


def test_assess_schema_compatibility_warn_on_dtype_widen(parquet_store_mod):
    stored = {
        "columns": [
            {"name": "date", "dtype": "datetime64[ns]", "nullable": False, "required": True},
            {"name": "id", "dtype": "int64", "nullable": False, "required": True},
            {"name": "score", "dtype": "float64", "nullable": False, "required": True},
        ]
    }
    expected = {
        "columns": [
            {"name": "date", "dtype": "datetime64[ns]", "nullable": False, "required": True},
            {"name": "id", "dtype": "float64", "nullable": False, "required": True},
            {"name": "score", "dtype": "float64", "nullable": False, "required": True},
        ]
    }
    report = parquet_store_mod.assess_schema_compatibility(stored, expected)
    assert report.status == "WARN"
    assert any(issue.category == "type_widen" for issue in report.issues)


def test_assess_schema_compatibility_fail_on_breaking_change(parquet_store_mod, stored_schema_meta, breaking_expected_schema_meta):
    report = parquet_store_mod.assess_schema_compatibility(stored_schema_meta, breaking_expected_schema_meta)
    assert report.status == "FAIL"
    cats = {issue.category for issue in report.issues}
    assert "type_narrow_or_incompatible" in cats or "drop_column" in cats


# ---------------------------------------------------------------------------
# Publish / inspect / read flows
# ---------------------------------------------------------------------------


def test_write_parquet_publishes_snapshot_and_current_pointer(
    parquet_store_mod,
    require_parquet,
    install_fake_schema_registry,
    datasets_root,
    prices_frame_unsorted,
    schema_ref_prices,
    partition_cols,
):
    install_fake_schema_registry(schema_ref=schema_ref_prices)
    dataset_root = _dataset_path(datasets_root, "publish_case")

    result = parquet_store_mod.write_parquet(
        prices_frame_unsorted,
        dataset_root,
        schema_ref=schema_ref_prices,
        partition_cols=partition_cols,
        write_mode="fail_if_exists",
        deterministic=True,
        allowed_roots=[datasets_root],
        metadata={"owner": "unit", "run_id": "t1"},
    )

    assert result["path"] == str(dataset_root.resolve())
    assert "dataset_version" in result and result["dataset_version"]
    assert "manifest" in result and "schema" in result
    assert result["manifest"]["row_count"] == len(prices_frame_unsorted)
    assert result["manifest"]["partition_cols"] == list(partition_cols)
    assert _current_pointer(dataset_root).exists()
    assert _version_dir(dataset_root, result["dataset_version"]).exists()


def test_inspect_snapshot_returns_manifest_and_schema(
    parquet_store_mod,
    require_parquet,
    install_fake_schema_registry,
    datasets_root,
    prices_frame_unsorted,
    schema_ref_prices,
    partition_cols,
):
    install_fake_schema_registry(schema_ref=schema_ref_prices)
    dataset_root = _dataset_path(datasets_root, "inspect_case")

    write_result = parquet_store_mod.write_parquet(
        prices_frame_unsorted,
        dataset_root,
        schema_ref=schema_ref_prices,
        partition_cols=partition_cols,
        allowed_roots=[datasets_root],
    )

    snapshot = parquet_store_mod.inspect_snapshot(
        dataset_root,
        dataset_version=write_result["dataset_version"],
        allowed_roots=[datasets_root],
    )
    assert snapshot["root"] == str(dataset_root.resolve())
    assert snapshot["dataset_version"] == write_result["dataset_version"]
    assert snapshot["manifest"]["schema_ref"] == schema_ref_prices
    assert snapshot["schema"]["schema_ref"] == schema_ref_prices
    assert snapshot["manifest"]["checksums"]


def test_read_parquet_roundtrip_current_version_is_deterministically_sorted(
    parquet_store_mod,
    require_parquet,
    install_fake_schema_registry,
    datasets_root,
    prices_frame_unsorted,
    prices_frame_sorted_expected,
    schema_ref_prices,
    partition_cols,
    assert_frame_equal,
):
    install_fake_schema_registry(schema_ref=schema_ref_prices)
    dataset_root = _dataset_path(datasets_root, "roundtrip_case")

    write_result = parquet_store_mod.write_parquet(
        prices_frame_unsorted,
        dataset_root,
        schema_ref=schema_ref_prices,
        partition_cols=partition_cols,
        deterministic=True,
        allowed_roots=[datasets_root],
    )

    observed = parquet_store_mod.read_parquet(
        dataset_root,
        schema_ref=schema_ref_prices,
        allowed_roots=[datasets_root],
    )

    assert observed.attrs["snapshot_manifest"]["dataset_version"] == write_result["dataset_version"]
    assert observed.attrs["read_metrics"]["rows_read"] == len(prices_frame_unsorted)
    assert_frame_equal(observed.reset_index(drop=True), prices_frame_sorted_expected.reset_index(drop=True))


def test_read_parquet_supports_projection_and_partition_filter(
    parquet_store_mod,
    require_parquet,
    install_fake_schema_registry,
    datasets_root,
    prices_frame_unsorted,
    schema_ref_prices,
    partition_cols,
):
    install_fake_schema_registry(schema_ref=schema_ref_prices)
    dataset_root = _dataset_path(datasets_root, "projection_case")

    parquet_store_mod.write_parquet(
        prices_frame_unsorted,
        dataset_root,
        schema_ref=schema_ref_prices,
        partition_cols=partition_cols,
        allowed_roots=[datasets_root],
    )

    observed = parquet_store_mod.read_parquet(
        dataset_root,
        columns=["date", "symbol", "close"],
        filters=[("exchange", "==", "NASDAQ")],
        allowed_roots=[datasets_root],
    )

    assert list(observed.columns) == ["date", "symbol", "close"]
    assert len(observed) == int((prices_frame_unsorted["exchange"] == "NASDAQ").sum())
    assert observed.attrs["read_metrics"]["pruned_partitions"] >= 1
    assert set(observed["symbol"]) == {"AAA"}


def test_read_parquet_returns_empty_dataframe_with_projection_for_no_match(
    parquet_store_mod,
    require_parquet,
    install_fake_schema_registry,
    datasets_root,
    prices_frame_unsorted,
    schema_ref_prices,
    partition_cols,
):
    install_fake_schema_registry(schema_ref=schema_ref_prices)
    dataset_root = _dataset_path(datasets_root, "empty_filter_case")

    parquet_store_mod.write_parquet(
        prices_frame_unsorted,
        dataset_root,
        schema_ref=schema_ref_prices,
        partition_cols=partition_cols,
        allowed_roots=[datasets_root],
    )

    observed = parquet_store_mod.read_parquet(
        dataset_root,
        columns=["date", "symbol", "close"],
        filters=[("exchange", "==", "NOT_REAL")],
        allowed_roots=[datasets_root],
    )

    assert observed.empty
    assert list(observed.columns) == ["date", "symbol", "close"]
    assert "snapshot_manifest" in observed.attrs


# ---------------------------------------------------------------------------
# Versioning / write modes
# ---------------------------------------------------------------------------


def test_fail_if_exists_rejects_second_publish_to_same_logical_destination(
    parquet_store_mod,
    require_parquet,
    install_fake_schema_registry,
    datasets_root,
    prices_frame_unsorted,
    schema_ref_prices,
):
    install_fake_schema_registry(schema_ref=schema_ref_prices)
    dataset_root = _dataset_path(datasets_root, "conflict_case")

    parquet_store_mod.write_parquet(
        prices_frame_unsorted,
        dataset_root,
        schema_ref=schema_ref_prices,
        allowed_roots=[datasets_root],
    )

    with pytest.raises(parquet_store_mod.WriteConflictError):
        parquet_store_mod.write_parquet(
            prices_frame_unsorted,
            dataset_root,
            schema_ref=schema_ref_prices,
            write_mode="fail_if_exists",
            allowed_roots=[datasets_root],
        )


def test_append_versioned_creates_new_current_version_but_old_version_remains_readable(
    parquet_store_mod,
    require_parquet,
    install_fake_schema_registry,
    datasets_root,
    prices_frame_unsorted,
    schema_ref_prices,
):
    install_fake_schema_registry(schema_ref=schema_ref_prices)
    dataset_root = _dataset_path(datasets_root, "append_case")

    first = parquet_store_mod.write_parquet(
        prices_frame_unsorted,
        dataset_root,
        schema_ref=schema_ref_prices,
        write_mode="fail_if_exists",
        allowed_roots=[datasets_root],
    )
    second_frame = prices_frame_unsorted.copy()
    second_frame.loc[:, "close"] = second_frame["close"] + 1.0

    second = parquet_store_mod.write_parquet(
        second_frame,
        dataset_root,
        schema_ref=schema_ref_prices,
        write_mode="append_versioned",
        allowed_roots=[datasets_root],
    )

    assert first["dataset_version"] != second["dataset_version"]

    current = parquet_store_mod.read_parquet(dataset_root, allowed_roots=[datasets_root])
    old = parquet_store_mod.read_parquet(
        dataset_root,
        dataset_version=first["dataset_version"],
        allowed_roots=[datasets_root],
    )

    assert current["close"].mean() > old["close"].mean()
    assert current.attrs["snapshot_manifest"]["dataset_version"] == second["dataset_version"]
    assert old.attrs["snapshot_manifest"]["dataset_version"] == first["dataset_version"]


def test_overwrite_partition_replaces_only_target_partition_in_new_version(
    parquet_store_mod,
    require_parquet,
    install_fake_schema_registry,
    datasets_root,
    prices_frame_unsorted,
    schema_ref_prices,
    partition_cols,
):
    install_fake_schema_registry(schema_ref=schema_ref_prices)
    dataset_root = _dataset_path(datasets_root, "overwrite_partition_case")

    base = parquet_store_mod.write_parquet(
        prices_frame_unsorted,
        dataset_root,
        schema_ref=schema_ref_prices,
        partition_cols=partition_cols,
        allowed_roots=[datasets_root],
    )

    incoming = prices_frame_unsorted.loc[prices_frame_unsorted["exchange"] == "NASDAQ"].copy()
    incoming.loc[:, "close"] = incoming["close"] + 10.0

    updated = parquet_store_mod.write_parquet(
        incoming,
        dataset_root,
        schema_ref=schema_ref_prices,
        partition_cols=partition_cols,
        write_mode="overwrite_partition",
        allowed_roots=[datasets_root],
    )

    assert base["dataset_version"] != updated["dataset_version"]

    latest = parquet_store_mod.read_parquet(dataset_root, allowed_roots=[datasets_root])
    latest_nasdaq = latest.loc[latest["exchange"] == "NASDAQ", "close"]
    latest_nyse = latest.loc[latest["exchange"] == "NYSE", "close"]

    original_nasdaq = prices_frame_unsorted.loc[prices_frame_unsorted["exchange"] == "NASDAQ", "close"]
    original_nyse = prices_frame_unsorted.loc[prices_frame_unsorted["exchange"] == "NYSE", "close"]

    assert latest_nasdaq.mean() > original_nasdaq.mean()
    assert sorted(latest_nyse.tolist()) == sorted(original_nyse.tolist())
    assert len(latest) == len(prices_frame_unsorted)


# ---------------------------------------------------------------------------
# Validation / corruption / strict-vs-tolerant reads
# ---------------------------------------------------------------------------


def test_validate_dataset_success(
    parquet_store_mod,
    require_parquet,
    install_fake_schema_registry,
    datasets_root,
    prices_frame_unsorted,
    schema_ref_prices,
    partition_cols,
):
    install_fake_schema_registry(schema_ref=schema_ref_prices)
    dataset_root = _dataset_path(datasets_root, "validate_ok_case")

    write_result = parquet_store_mod.write_parquet(
        prices_frame_unsorted,
        dataset_root,
        schema_ref=schema_ref_prices,
        partition_cols=partition_cols,
        allowed_roots=[datasets_root],
    )

    summary = parquet_store_mod.validate_dataset(
        dataset_root,
        schema_ref=schema_ref_prices,
        dataset_version=write_result["dataset_version"],
        strict=True,
        allowed_roots=[datasets_root],
    )

    assert summary["valid"] is True
    assert summary["row_count_manifest"] == len(prices_frame_unsorted)
    assert summary["row_count_observed"] == len(prices_frame_unsorted)
    assert summary["schema_compatibility"]["status"] in {"PASS", "WARN"}


def test_validate_dataset_strict_false_reports_corruption_and_strict_true_raises(
    parquet_store_mod,
    require_parquet,
    install_fake_schema_registry,
    datasets_root,
    prices_frame_unsorted,
    schema_ref_prices,
    partition_cols,
):
    install_fake_schema_registry(schema_ref=schema_ref_prices)
    dataset_root = _dataset_path(datasets_root, "validate_corrupt_case")

    write_result = parquet_store_mod.write_parquet(
        prices_frame_unsorted,
        dataset_root,
        schema_ref=schema_ref_prices,
        partition_cols=partition_cols,
        allowed_roots=[datasets_root],
    )

    data_file = _first_data_file(dataset_root, write_result["dataset_version"])
    _flip_last_byte(data_file)

    summary = parquet_store_mod.validate_dataset(
        dataset_root,
        schema_ref=schema_ref_prices,
        dataset_version=write_result["dataset_version"],
        strict=False,
        allowed_roots=[datasets_root],
    )
    assert summary["valid"] is False
    assert summary["corrupt_partitions"] or summary["checksum_mismatches"]
    assert summary["issues"]

    with pytest.raises(parquet_store_mod.ManifestError):
        parquet_store_mod.validate_dataset(
            dataset_root,
            schema_ref=schema_ref_prices,
            dataset_version=write_result["dataset_version"],
            strict=True,
            allowed_roots=[datasets_root],
        )


def test_validate_dataset_can_quarantine_corrupt_files(
    parquet_store_mod,
    require_parquet,
    install_fake_schema_registry,
    datasets_root,
    prices_frame_unsorted,
    schema_ref_prices,
    partition_cols,
):
    install_fake_schema_registry(schema_ref=schema_ref_prices)
    dataset_root = _dataset_path(datasets_root, "quarantine_case")

    write_result = parquet_store_mod.write_parquet(
        prices_frame_unsorted,
        dataset_root,
        schema_ref=schema_ref_prices,
        partition_cols=partition_cols,
        allowed_roots=[datasets_root],
    )

    data_file = _first_data_file(dataset_root, write_result["dataset_version"])
    rel = data_file.relative_to(_version_dir(dataset_root, write_result["dataset_version"]))
    _flip_last_byte(data_file)

    summary = parquet_store_mod.validate_dataset(
        dataset_root,
        schema_ref=schema_ref_prices,
        dataset_version=write_result["dataset_version"],
        strict=False,
        quarantine_corrupt=True,
        allowed_roots=[datasets_root],
    )

    quarantine_target = _version_dir(dataset_root, write_result["dataset_version"]) / "_quarantine" / rel
    assert quarantine_target.exists()
    assert summary["valid"] is False


def test_read_parquet_strict_false_skips_corrupt_partitions_when_others_are_readable(
    parquet_store_mod,
    require_parquet,
    install_fake_schema_registry,
    datasets_root,
    prices_frame_unsorted,
    schema_ref_prices,
    partition_cols,
):
    install_fake_schema_registry(schema_ref=schema_ref_prices)
    dataset_root = _dataset_path(datasets_root, "tolerant_read_case")

    write_result = parquet_store_mod.write_parquet(
        prices_frame_unsorted,
        dataset_root,
        schema_ref=schema_ref_prices,
        partition_cols=partition_cols,
        allowed_roots=[datasets_root],
    )

    version_data_root = _version_dir(dataset_root, write_result["dataset_version"]) / "data"
    nyse_file = next(version_data_root.rglob("exchange=NYSE/*.parquet"))
    _flip_last_byte(nyse_file)

    observed = parquet_store_mod.read_parquet(
        dataset_root,
        strict_schema=False,
        allowed_roots=[datasets_root],
    )

    assert len(observed) == int((prices_frame_unsorted["exchange"] == "NASDAQ").sum())
    assert set(observed["exchange"]) == {"NASDAQ"}
    assert observed.attrs["read_metrics"]["corrupt_partitions"] == 1

    with pytest.raises(parquet_store_mod.CorruptPartitionError):
        parquet_store_mod.read_parquet(
            dataset_root,
            strict_schema=True,
            allowed_roots=[datasets_root],
        )


# ---------------------------------------------------------------------------
# Root confinement / path semantics
# ---------------------------------------------------------------------------


def test_public_api_rejects_paths_outside_allowed_roots(
    parquet_store_mod,
    require_parquet,
    install_fake_schema_registry,
    tmp_path,
    prices_frame_unsorted,
    schema_ref_prices,
):
    install_fake_schema_registry(schema_ref=schema_ref_prices)
    outside = tmp_path / "outside_dataset"
    outside.mkdir(parents=True, exist_ok=True)

    with pytest.raises(parquet_store_mod.InvalidPathError):
        parquet_store_mod.write_parquet(
            prices_frame_unsorted,
            outside,
            schema_ref=schema_ref_prices,
            allowed_roots=[tmp_path / "allowed_only"],
        )


def test_public_api_rejects_non_file_uris(parquet_store_mod):
    with pytest.raises(parquet_store_mod.InvalidPathError):
        parquet_store_mod.inspect_snapshot("s3://bucket/path")
