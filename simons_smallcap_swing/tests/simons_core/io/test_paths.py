from __future__ import annotations

from pathlib import Path
import shutil

import pytest


pytestmark = [pytest.mark.io, pytest.mark.filesystem]


def test_canonical_descriptor_digest_is_deterministic_and_order_invariant(paths_mod):
    left = {"b": 2, "a": 1, "nested": {"x": [3, 2, 1]}}
    right = {"nested": {"x": [3, 2, 1]}, "a": 1, "b": 2}
    d1 = paths_mod.canonical_descriptor_digest(left)
    d2 = paths_mod.canonical_descriptor_digest(right)
    assert d1 == d2
    assert len(d1) == paths_mod.DIGEST_DEFAULT_LEN


def test_canonical_descriptor_digest_length_validation(paths_mod):
    with pytest.raises(ValueError):
        paths_mod.canonical_descriptor_digest({"a": 1}, length=7)
    with pytest.raises(ValueError):
        paths_mod.canonical_descriptor_digest({"a": 1}, length=65)


def test_normalize_segment_generic_date_version_and_run_id(paths_mod):
    assert paths_mod.normalize_segment("  Foo Bar  ") == "foo-bar"
    assert paths_mod.normalize_segment("2024-01-03", kind="date") == "2024-01-03"
    assert paths_mod.normalize_segment("V2.1", kind="version") == "v2.1"
    assert paths_mod.normalize_segment("Nightly Alpha", kind="run_id") == "run_nightly-alpha"


def test_normalize_segment_reserved_and_traversal_rejected(paths_mod):
    with pytest.raises(paths_mod.InvalidSegmentError):
        paths_mod.normalize_segment("latest")
    with pytest.raises(paths_mod.PathTraversalError):
        paths_mod.normalize_segment("../secret")
    with pytest.raises(paths_mod.PathTraversalError):
        paths_mod.normalize_segment("..")


def test_normalize_segment_artifact_name_partition_and_digest(paths_mod):
    assert paths_mod.normalize_segment("Part-000.parquet", kind="artifact_name") == "part-000.parquet"
    assert paths_mod.normalize_segment("exchange=NASDAQ", kind="partition") == "exchange=nasdaq"
    assert paths_mod.normalize_segment("abcdef123456", kind="digest") == "abcdef123456"


@pytest.mark.parametrize(
    ("kind", "value", "exc_type"),
    [
        ("date", "20240103", "InvalidDateFormatError"),
        ("version", "release-1", "InvalidVersionError"),
        ("env", "qa", "UnknownEnvironmentError"),
        ("artifact_class", "dataset", "InvalidSegmentError"),
        ("digest", "xyz", "InvalidSegmentError"),
    ],
)
def test_normalize_segment_invalid_cases(paths_mod, kind, value, exc_type):
    exc = getattr(paths_mod, exc_type)
    with pytest.raises(exc):
        paths_mod.normalize_segment(value, kind=kind)


def test_get_root_returns_normalized_roots(paths_mod, env_roots_file):
    local_root = paths_mod.get_root("local", env_roots=env_roots_file)
    dev_root = paths_mod.get_root("dev", env_roots=env_roots_file)
    assert local_root.startswith("file://")
    assert dev_root == "s3://simons-dev/"


def test_get_root_unknown_environment_raises(paths_mod, env_roots_file):
    with pytest.raises(paths_mod.UnknownEnvironmentError):
        paths_mod.get_root("qa", env_roots=env_roots_file)


def test_build_locator_for_local_data_success(paths_mod, env_roots_file, sample_data_locator):
    locator = sample_data_locator
    assert locator.backend == "file"
    assert locator.env == "local"
    assert locator.artifact_class == "data"
    assert locator.layout_version == "v1"
    assert locator.segments[:4] == ("data", "v1", "prices", "adjusted")
    assert locator.segments[-1] == "part-000.parquet"
    assert locator.uri.startswith(env_roots_file["local"])
    assert locator.local_path().name == "part-000.parquet"


def test_build_locator_for_s3_cache_success(paths_mod, env_roots_file):
    digest = paths_mod.canonical_descriptor_digest({"namespace": "features", "window": 20})
    locator = paths_mod.build_locator(
        artifact_class="cache",
        env="dev",
        domain="features",
        entity="daily_momentum",
        digest=digest,
        artifact_name="payload.cache",
        env_roots=env_roots_file,
    )
    assert locator.backend == "s3"
    assert locator.env == "dev"
    assert locator.artifact_class == "cache"
    assert locator.uri.startswith("s3://simons-dev/")


def test_build_locator_requires_semantic_fields(paths_mod, env_roots_file):
    with pytest.raises(paths_mod.LocatorSemanticError):
        paths_mod.build_locator(
            artifact_class="data",
            env="local",
            domain="prices",
            env_roots=env_roots_file,
        )


def test_build_tmp_locator_has_tmp_class_and_run_id(paths_mod, env_roots_file, sample_tmp_locator):
    locator = sample_tmp_locator
    assert locator.artifact_class == "tmp"
    assert locator.is_temporary is True
    assert "staging" in locator.segments
    assert any(seg.startswith("run_") for seg in locator.segments)
    assert locator.segments[0] == "tmp"
    assert locator.uri.startswith(env_roots_file["local"])


def test_locator_from_uri_roundtrip_local(paths_mod, env_roots_file, sample_data_locator):
    parsed = paths_mod.locator_from_uri(sample_data_locator.uri, env_roots=env_roots_file)
    assert parsed.uri == sample_data_locator.uri
    assert parsed.backend == sample_data_locator.backend
    assert parsed.env == sample_data_locator.env
    assert parsed.artifact_class == sample_data_locator.artifact_class
    assert parsed.layout_version == sample_data_locator.layout_version
    assert parsed.segments[:6] == sample_data_locator.segments[:6]
    assert parsed.segments[-1] == sample_data_locator.segments[-1]
    assert parsed.segments[-2] in {sample_data_locator.segments[-2], sample_data_locator.segments[-2].replace('=', '%3D')}


def test_locator_from_uri_roundtrip_s3(paths_mod, env_roots_file, sample_cache_locator):
    parsed = paths_mod.locator_from_uri(sample_cache_locator.uri, env_roots=env_roots_file)
    assert parsed == sample_cache_locator


def test_locator_from_uri_rejects_unauthorized_root(paths_mod, env_roots_file):
    with pytest.raises(paths_mod.UnauthorizedRootError):
        paths_mod.locator_from_uri("file:///outside/root/data/v1/foo/bar", env_roots=env_roots_file)


def test_validate_locator_rejects_wrong_root(paths_mod, env_roots_file, sample_data_locator):
    bad = paths_mod.Locator(
        uri=sample_data_locator.uri,
        backend=sample_data_locator.backend,
        env=sample_data_locator.env,
        artifact_class=sample_data_locator.artifact_class,
        segments=sample_data_locator.segments,
        root=env_roots_file["test"],
        layout_version=sample_data_locator.layout_version,
    )
    with pytest.raises(paths_mod.UnauthorizedRootError):
        paths_mod.validate_locator(bad, env_roots=env_roots_file)


def test_validate_locator_rejects_artifact_class_mismatch(paths_mod, env_roots_file, sample_data_locator):
    bad = paths_mod.Locator(
        uri=sample_data_locator.uri,
        backend=sample_data_locator.backend,
        env=sample_data_locator.env,
        artifact_class="cache",
        segments=sample_data_locator.segments,
        root=sample_data_locator.root,
        layout_version=sample_data_locator.layout_version,
    )
    with pytest.raises(paths_mod.LocatorSemanticError):
        paths_mod.validate_locator(bad, env_roots=env_roots_file)


def test_confined_to_root_true_for_nested_local_uri(paths_mod, env_roots_file, sample_data_locator):
    assert paths_mod.confined_to_root(sample_data_locator.uri, env_roots_file["local"]) is True


def test_confined_to_root_false_for_escape_local_uri(paths_mod, env_roots_file):
    candidate = (Path(env_roots_file["local"].removeprefix("file://")).parent / "escape" / "x").resolve().as_uri()
    assert paths_mod.confined_to_root(candidate, env_roots_file["local"]) is False


def test_confined_to_root_for_s3(paths_mod):
    assert paths_mod.confined_to_root("s3://bucket/root/a/b", "s3://bucket/root") is True
    assert paths_mod.confined_to_root("s3://bucket/other/a/b", "s3://bucket/root") is False
    assert paths_mod.confined_to_root("s3://other/root/a/b", "s3://bucket/root") is False


def test_ensure_parent_dirs_creates_local_directory(paths_mod, env_roots_file, sample_data_locator):
    target_parent = sample_data_locator.local_path().parent
    if target_parent.exists():
        shutil.rmtree(target_parent)
    assert not target_parent.exists()
    paths_mod.ensure_parent_dirs(sample_data_locator)
    assert target_parent.exists()
    assert target_parent.is_dir()


def test_ensure_parent_dirs_noop_for_s3(paths_mod, env_roots_file):
    digest = paths_mod.canonical_descriptor_digest({'namespace': 'features', 'window': 20})
    locator = paths_mod.build_locator(
        artifact_class='cache',
        env='dev',
        domain='features',
        entity='daily_momentum',
        digest=digest,
        artifact_name='payload.cache',
        env_roots=env_roots_file,
    )
    paths_mod.ensure_parent_dirs(locator)
    assert locator.backend == 's3'


def test_locator_local_path_rejected_for_non_file(paths_mod, env_roots_file):
    digest = paths_mod.canonical_descriptor_digest({'namespace': 'features', 'window': 20})
    locator = paths_mod.build_locator(
        artifact_class='cache',
        env='dev',
        domain='features',
        entity='daily_momentum',
        digest=digest,
        artifact_name='payload.cache',
        env_roots=env_roots_file,
    )
    with pytest.raises(paths_mod.LocatorSemanticError):
        locator.local_path()


def test_metrics_snapshot_counts_built_and_unknown_env(paths_mod, env_roots_file):
    paths_mod.reset_metrics()
    paths_mod.build_locator(
        artifact_class="data",
        env="local",
        domain="prices",
        entity="adjusted",
        env_roots=env_roots_file,
    )
    with pytest.raises(paths_mod.UnknownEnvironmentError):
        paths_mod.get_root("qa", env_roots=env_roots_file)
    snap = paths_mod.get_metrics_snapshot()
    assert snap.n_locators_built == 1
    assert snap.n_unknown_env_errors >= 1
    assert snap.distribution_by_env.get("local") == 1
    assert snap.distribution_by_artifact_class.get("data") == 1


def test_build_arg_parser_and_main_print(paths_mod, monkeypatch, capsys, env_roots_file):
    monkeypatch.setattr(paths_mod, "DEFAULT_ENV_ROOTS", env_roots_file)
    rc = paths_mod.main(
        [
            "print",
            "--artifact-class",
            "data",
            "--env",
            "local",
            "--domain",
            "prices",
            "--entity",
            "adjusted",
            "--artifact-name",
            "part-001.parquet",
        ]
    )
    out = capsys.readouterr().out.strip()
    assert rc == 0
    assert out.startswith(env_roots_file["local"])


def test_main_validate_command_roundtrip(paths_mod, monkeypatch, capsys, env_roots_file, sample_data_locator):
    monkeypatch.setattr(paths_mod, "DEFAULT_ENV_ROOTS", env_roots_file)
    rc = paths_mod.main(["validate", "--uri", sample_data_locator.uri])
    out = capsys.readouterr().out
    assert rc == 0
    assert '"artifact_class": "data"' in out
    assert '"env": "local"' in out


def test_main_roots_and_tmp_commands(paths_mod, monkeypatch, capsys, env_roots_file):
    monkeypatch.setattr(paths_mod, "DEFAULT_ENV_ROOTS", env_roots_file)
    rc_roots = paths_mod.main(["roots"])
    out_roots = capsys.readouterr().out
    assert rc_roots == 0
    assert '"local"' in out_roots

    rc_tmp = paths_mod.main(
        [
            "tmp",
            "--env",
            "local",
            "--scope",
            "research",
            "--run-id",
            "run_batch_001",
            "--artifact-name",
            "scratch.json",
            "--date",
            "2024-01-03",
        ]
    )
    out_tmp = capsys.readouterr().out.strip()
    assert rc_tmp == 0
    assert "/tmp/" in out_tmp or out_tmp.startswith(env_roots_file["local"])
    assert "/run_batch_001/" in out_tmp


def test_locator_to_dict_contains_expected_keys(sample_data_locator):
    payload = sample_data_locator.to_dict()
    assert payload["env"] == "local"
    assert payload["artifact_class"] == "data"
    assert payload["layout_version"] == "v1"
    assert isinstance(payload["segments"], tuple)
