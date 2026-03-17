from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


DEFAULT_TAXONOMY_PRIORITY = {
    "us-gaap": 1,
    "dei": 2,
    "ifrs-full": 3,
    "issuer-extension": 4,
}

DEFAULT_MAPPING_RULES: list[dict[str, Any]] = [
    {
        "taxonomy": "dei",
        "tag": "EntityCommonStockSharesOutstanding",
        "metric_name": "shares_outstanding",
        "mapping_status": "exact",
        "mapping_priority": 1,
        "metric_family": "per_share_inputs",
        "unit_base": "shares",
        "allowed_period_types": ["Instant"],
        "value_min": 0.0,
    },
    {
        "taxonomy": "us-gaap",
        "tag": "Assets",
        "metric_name": "total_assets",
        "mapping_status": "exact",
        "mapping_priority": 1,
        "metric_family": "balance_sheet",
        "unit_base": "USD",
        "allowed_period_types": ["Instant"],
    },
    {
        "taxonomy": "us-gaap",
        "tag": "Liabilities",
        "metric_name": "total_liabilities",
        "mapping_status": "exact",
        "mapping_priority": 1,
        "metric_family": "balance_sheet",
        "unit_base": "USD",
        "allowed_period_types": ["Instant"],
    },
    {
        "taxonomy": "us-gaap",
        "tag": "StockholdersEquity",
        "metric_name": "stockholders_equity",
        "mapping_status": "exact",
        "mapping_priority": 1,
        "metric_family": "balance_sheet",
        "unit_base": "USD",
        "allowed_period_types": ["Instant"],
    },
    {
        "taxonomy": "us-gaap",
        "tag": "Revenues",
        "metric_name": "revenue",
        "mapping_status": "exact",
        "mapping_priority": 1,
        "metric_family": "income_statement",
        "unit_base": "USD",
        "allowed_period_types": ["Q", "Y", "H1", "NineM", "TTM"],
    },
    {
        "taxonomy": "us-gaap",
        "tag": "RevenueFromContractWithCustomerExcludingAssessedTax",
        "metric_name": "revenue",
        "mapping_status": "alias",
        "mapping_priority": 2,
        "metric_family": "income_statement",
        "unit_base": "USD",
        "allowed_period_types": ["Q", "Y", "H1", "NineM", "TTM"],
    },
    {
        "taxonomy": "us-gaap",
        "tag": "NetIncomeLoss",
        "metric_name": "net_income",
        "mapping_status": "exact",
        "mapping_priority": 1,
        "metric_family": "income_statement",
        "unit_base": "USD",
        "allowed_period_types": ["Q", "Y", "H1", "NineM", "TTM"],
    },
    {
        "taxonomy": "us-gaap",
        "tag": "NetCashProvidedByUsedInOperatingActivities",
        "metric_name": "operating_cash_flow",
        "mapping_status": "exact",
        "mapping_priority": 1,
        "metric_family": "cash_flow",
        "unit_base": "USD",
        "allowed_period_types": ["Q", "Y", "H1", "NineM", "TTM"],
    },
    {
        "taxonomy": "us-gaap",
        "tag": "EarningsPerShareBasic",
        "metric_name": "eps_basic",
        "mapping_status": "exact",
        "mapping_priority": 1,
        "metric_family": "per_share",
        "unit_base": "USD/share",
        "allowed_period_types": ["Q", "Y", "H1", "NineM", "TTM"],
    },
    {
        "taxonomy": "us-gaap",
        "tag": "EarningsPerShareDiluted",
        "metric_name": "eps_diluted",
        "mapping_status": "exact",
        "mapping_priority": 1,
        "metric_family": "per_share",
        "unit_base": "USD/share",
        "allowed_period_types": ["Q", "Y", "H1", "NineM", "TTM"],
    },
]

DEFAULT_UNIT_ALIASES = {
    "usd": "USD",
    "usdollars": "USD",
    "usd/shares": "USD/share",
    "usd/share": "USD/share",
    "usdper share": "USD/share",
    "usdper shares": "USD/share",
    "shares": "shares",
    "sharesoutstanding": "shares",
    "pure": "pure",
    "ratio": "ratio",
    "percent": "ratio",
    "percentage": "ratio",
}

DEFAULT_PERIOD_WINDOWS = {
    "Instant": {"min_days": 0, "max_days": 10},
    "Q": {"min_days": 70, "max_days": 105},
    "H1": {"min_days": 160, "max_days": 200},
    "NineM": {"min_days": 250, "max_days": 290},
    "Y": {"min_days": 330, "max_days": 380},
    "TTM": {"min_days": 330, "max_days": 380},
}

DEFAULT_VALUE_BOUNDS = {
    "ratio": {"min": -1e6, "max": 1e6},
    "pure": {"min": -1e12, "max": 1e12},
    "shares": {"min": 0.0, "max": 1e15},
    "USD": {"min": -1e16, "max": 1e16},
    "USD/share": {"min": -1e8, "max": 1e8},
}


class ParseXBRLError(RuntimeError):
    """Base exception for parse_xbrl failures."""


class InputValidationError(ParseXBRLError):
    """Raised for invalid inputs or configuration."""


@dataclass(slots=True)
class ParseConfig:
    output_dir: str = "data/edgar/parsed"
    allow_csv_fallback: bool = True
    table_format: str = "parquet"
    unknown_taxonomy_policy: str = "reject"
    unknown_unit_policy: str = "reject"
    conflict_policy: str = "reject"
    unresolved_tie_policy: str = "reject"
    systemic_mapping_failure_ratio: float = 0.50
    max_schema_failure_ratio: float = 0.05
    warn_reject_ratio: float = 0.25
    accepted_mapping_statuses: list[str] = field(default_factory=lambda: ["exact", "alias", "heuristic"])
    taxonomy_priority: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_TAXONOMY_PRIORITY))
    period_windows: dict[str, dict[str, int]] = field(default_factory=lambda: dict(DEFAULT_PERIOD_WINDOWS))
    unit_aliases: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_UNIT_ALIASES))
    value_bounds_by_unit: dict[str, dict[str, float]] = field(default_factory=lambda: dict(DEFAULT_VALUE_BOUNDS))
    mapping_rules: list[dict[str, Any]] = field(default_factory=lambda: list(DEFAULT_MAPPING_RULES))
    quality_weights: dict[str, float] = field(
        default_factory=lambda: {
            "taxonomy": 0.25,
            "mapping": 0.25,
            "unit": 0.20,
            "context": 0.15,
            "dominance": 0.15,
        }
    )
    metric_bounds: dict[str, dict[str, float]] = field(default_factory=dict)
    metric_period_overrides: dict[str, list[str]] = field(default_factory=dict)
    metric_taxonomy_overrides: dict[str, dict[str, int]] = field(default_factory=dict)

    def validate(self) -> None:
        self.table_format = str(self.table_format).strip().lower()
        if self.table_format not in {"parquet", "csv"}:
            raise InputValidationError("config.table_format must be 'parquet' or 'csv'")
        for attr_name in ["unknown_taxonomy_policy", "unknown_unit_policy", "conflict_policy", "unresolved_tie_policy"]:
            value = getattr(self, attr_name)
            if str(value).strip().lower() not in {"reject", "warn", "accept"}:
                raise InputValidationError(f"config.{attr_name} must be one of reject/warn/accept")
            setattr(self, attr_name, str(value).strip().lower())
        for attr_name in ["systemic_mapping_failure_ratio", "max_schema_failure_ratio", "warn_reject_ratio"]:
            value = float(getattr(self, attr_name))
            if not (0 <= value <= 1):
                raise InputValidationError(f"config.{attr_name} must be in [0, 1]")
            setattr(self, attr_name, value)
        if not isinstance(self.accepted_mapping_statuses, list) or not self.accepted_mapping_statuses:
            raise InputValidationError("config.accepted_mapping_statuses must be a non-empty list")
        self.accepted_mapping_statuses = [str(x).strip().lower() for x in self.accepted_mapping_statuses]
        if not isinstance(self.taxonomy_priority, dict) or not self.taxonomy_priority:
            raise InputValidationError("config.taxonomy_priority must be a non-empty mapping")
        self.taxonomy_priority = {str(k).strip().lower(): int(v) for k, v in self.taxonomy_priority.items()}
        if not isinstance(self.period_windows, dict) or not self.period_windows:
            raise InputValidationError("config.period_windows must be a non-empty mapping")
        normalized_windows: dict[str, dict[str, int]] = {}
        for k, v in self.period_windows.items():
            if not isinstance(v, dict) or "min_days" not in v or "max_days" not in v:
                raise InputValidationError(f"config.period_windows[{k!r}] must contain min_days and max_days")
            min_days = int(v["min_days"])
            max_days = int(v["max_days"])
            if min_days < 0 or max_days < min_days:
                raise InputValidationError(f"invalid period window for {k!r}")
            normalized_windows[str(k)] = {"min_days": min_days, "max_days": max_days}
        self.period_windows = normalized_windows
        self.unit_aliases = {normalize_unit_token(k): str(v) for k, v in self.unit_aliases.items()}
        if not isinstance(self.mapping_rules, list) or not self.mapping_rules:
            raise InputValidationError("config.mapping_rules must be a non-empty list")
        validated_rules: list[dict[str, Any]] = []
        for idx, rule in enumerate(self.mapping_rules):
            if not isinstance(rule, dict):
                raise InputValidationError(f"mapping_rules[{idx}] must be a dict")
            taxonomy = str(rule.get("taxonomy", "")).strip().lower()
            tag = str(rule.get("tag", "")).strip()
            metric_name = str(rule.get("metric_name", "")).strip()
            if not taxonomy or not tag or not metric_name:
                raise InputValidationError(f"mapping_rules[{idx}] missing taxonomy/tag/metric_name")
            vrule = dict(rule)
            vrule["taxonomy"] = taxonomy
            vrule["tag"] = tag
            vrule["metric_name"] = metric_name
            vrule["mapping_status"] = str(rule.get("mapping_status", "exact")).strip().lower()
            vrule["mapping_priority"] = int(rule.get("mapping_priority", 100))
            vrule["metric_family"] = str(rule.get("metric_family", "unknown")).strip()
            vrule["unit_base"] = str(rule.get("unit_base", "")).strip() or None
            if vrule["unit_base"] is not None:
                vrule["unit_base"] = canonicalize_unit(vrule["unit_base"], self.unit_aliases)
            allowed_period_types = rule.get("allowed_period_types") or []
            if allowed_period_types:
                vrule["allowed_period_types"] = [str(x).strip() for x in allowed_period_types]
            else:
                vrule["allowed_period_types"] = []
            for bound_field in ["value_min", "value_max"]:
                if rule.get(bound_field) is not None:
                    vrule[bound_field] = float(rule.get(bound_field))
                else:
                    vrule[bound_field] = None
            validated_rules.append(vrule)
        self.mapping_rules = validated_rules
        qweights = {str(k).strip().lower(): float(v) for k, v in self.quality_weights.items()}
        if not qweights or any(v < 0 for v in qweights.values()):
            raise InputValidationError("config.quality_weights must be non-negative")
        total_w = sum(qweights.values())
        if total_w <= 0:
            raise InputValidationError("config.quality_weights must sum to > 0")
        self.quality_weights = {k: v / total_w for k, v in qweights.items()}
        self.metric_bounds = {
            str(metric): {str(k): float(v) for k, v in bounds.items() if v is not None}
            for metric, bounds in self.metric_bounds.items()
            if isinstance(bounds, dict)
        }
        self.metric_period_overrides = {
            str(metric): [str(x) for x in values] for metric, values in self.metric_period_overrides.items()
        }
        self.metric_taxonomy_overrides = {
            str(metric): {str(k).strip().lower(): int(v) for k, v in values.items()}
            for metric, values in self.metric_taxonomy_overrides.items()
            if isinstance(values, dict)
        }


@dataclass(slots=True)
class MappingDecision:
    metric_name: Optional[str]
    mapping_status: str
    mapping_priority: int
    metric_family: Optional[str]
    unit_base: Optional[str]
    allowed_period_types: list[str]
    value_min: Optional[float]
    value_max: Optional[float]
    mapping_version: str
    rule_index: Optional[int]


@dataclass(slots=True)
class UnitDecision:
    unit_base: Optional[str]
    source_unit: str
    canonical_source_unit: str
    unit_scale_factor: Optional[float]
    unit_compatibility: str
    normalized_value: Optional[float]
    rejection_class: Optional[str]
    rejection_detail: Optional[str]


@dataclass(slots=True)
class PeriodDecision:
    period_type: Optional[str]
    duration_days: Optional[int]
    context_quality: float
    rejection_class: Optional[str]
    rejection_detail: Optional[str]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def load_config(config_path: Path) -> tuple[ParseConfig, dict[str, Any], str]:
    if not config_path.exists() or not config_path.is_file():
        raise InputValidationError(f"config path does not exist or is not a file: {config_path}")
    raw_text = config_path.read_text(encoding="utf-8")
    suffix = config_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise InputValidationError("PyYAML is required to read YAML configs")
        raw_obj = yaml.safe_load(raw_text) or {}
    elif suffix == ".json":
        raw_obj = json.loads(raw_text)
    else:
        raise InputValidationError("config path must be YAML or JSON")
    if not isinstance(raw_obj, dict):
        raise InputValidationError("config root must be a mapping")
    cfg = ParseConfig(**raw_obj)
    cfg.validate()
    return cfg, raw_obj, sha256_text(stable_json_dumps(raw_obj))


def parse_date(value: Any) -> Optional[pd.Timestamp]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, pd.Timestamp):
        ts = value
    else:
        text = str(value).strip()
        if not text or text.lower() in {"nan", "nat", "none"}:
            return None
        try:
            ts = pd.Timestamp(text)
        except Exception:
            return None
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.normalize()


def parse_datetime_utc(value: Any) -> Optional[pd.Timestamp]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, pd.Timestamp):
        ts = value
    else:
        text = str(value).strip()
        if not text or text.lower() in {"nan", "nat", "none"}:
            return None
        try:
            ts = pd.Timestamp(text)
        except Exception:
            return None
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def canonicalize_cik(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = re.sub(r"\D", "", str(value))
    if not text:
        return None
    return text.zfill(10)


def normalize_taxonomy(value: Any) -> str:
    return str(value or "").strip().lower()


def normalize_unit_token(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    text = text.replace("iso4217:", "")
    text = text.replace("xbrli:", "")
    text = text.replace("per", "/")
    text = text.replace(" ", "")
    text = text.replace("-", "")
    return text


def canonicalize_unit(value: Any, aliases: dict[str, str]) -> str:
    token = normalize_unit_token(value)
    if token in aliases:
        return aliases[token]
    if not token:
        return ""
    if token in {"usd/share", "usd/shares"}:
        return "USD/share"
    if token == "usd":
        return "USD"
    return str(value).strip()


def to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return float(value)
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    try:
        val = float(text)
    except Exception:
        return None
    if not math.isfinite(val):
        return None
    return val


def asof_date_token(asof: Optional[str]) -> str:
    if asof is None:
        return "na"
    return str(asof).replace(":", "-")


def config_mapping_version(config_hash: str) -> str:
    return config_hash[:12]


def table_write(path_no_suffix: Path, df: pd.DataFrame, table_format: str, allow_csv_fallback: bool) -> str:
    ensure_directory(path_no_suffix.parent)
    fmt = table_format.lower().strip()
    if fmt == "parquet":
        out = f"{path_no_suffix}.parquet"
        try:
            df.to_parquet(out, index=False)
            return out
        except Exception as exc:
            if not allow_csv_fallback:
                raise RuntimeError(
                    "Failed to write parquet. Install pyarrow or fastparquet, or enable CSV fallback. "
                    f"Original error: {exc}"
                ) from exc
            out = f"{path_no_suffix}.csv"
            df.to_csv(out, index=False)
            return out
    if fmt == "csv":
        out = f"{path_no_suffix}.csv"
        df.to_csv(out, index=False)
        return out
    raise ValueError(f"Unsupported table format: {table_format}")


def list_json_files(base_path: Path) -> list[Path]:
    if base_path.is_file() and base_path.suffix.lower() == ".json":
        return [base_path]
    if not base_path.exists():
        return []
    return sorted(p for p in base_path.rglob("*.json") if p.is_file())


def load_submissions_metadata(submissions_path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in list_json_files(submissions_path):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        cik = canonicalize_cik(payload.get("cik") or payload.get("cik_str") or payload.get("entityCik"))
        if cik is None:
            continue
        symbol = None
        tickers = payload.get("tickers")
        if isinstance(tickers, list) and tickers:
            symbol = str(tickers[0]).strip().upper() or None
        name = payload.get("name")
        exchanges = payload.get("exchanges") if isinstance(payload.get("exchanges"), list) else None
        filings = payload.get("filings") or {}
        recent = filings.get("recent") or {}
        recent_rows = _extract_submission_block_rows(cik, symbol, name, exchanges, recent, path, history_file=None)
        rows.extend(recent_rows)
        for history in filings.get("files") or []:
            if not isinstance(history, dict):
                continue
            history_name = history.get("name")
            if not history_name:
                continue
            hist_path = path.parent / str(history_name)
            if not hist_path.exists():
                continue
            try:
                hist_payload = json.loads(hist_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(hist_payload, dict):
                continue
            rows.extend(_extract_submission_block_rows(cik, symbol, name, exchanges, hist_payload, hist_path, history_file=str(history_name)))
    if not rows:
        return pd.DataFrame(
            columns=[
                "cik",
                "symbol",
                "issuer_name",
                "accession_number",
                "form_type",
                "filed_date",
                "acceptance_datetime",
                "submission_file_path",
                "submission_history_file",
                "exchange",
            ]
        )
    df = pd.DataFrame(rows).drop_duplicates(subset=["cik", "accession_number", "filed_date", "form_type"], keep="last")
    return df.sort_values(["cik", "filed_date", "acceptance_datetime", "accession_number"], kind="stable").reset_index(drop=True)


def _extract_submission_block_rows(
    cik: str,
    symbol: Optional[str],
    issuer_name: Optional[str],
    exchanges: Optional[list[Any]],
    block: dict[str, Any],
    path: Path,
    history_file: Optional[str],
) -> list[dict[str, Any]]:
    accession_numbers = block.get("accessionNumber") or block.get("accessionNumbers") or []
    forms = block.get("form") or []
    filed_dates = block.get("filingDate") or []
    acceptance_datetimes = block.get("acceptanceDateTime") or []
    if not isinstance(accession_numbers, list):
        return []
    n = len(accession_numbers)

    def _coerce_list(raw: Any) -> list[Any]:
        if isinstance(raw, list):
            if len(raw) == n:
                return raw
            if len(raw) == 0:
                return [None] * n
            return list(raw) + [None] * max(0, n - len(raw))
        return [None] * n

    forms = _coerce_list(forms)
    filed_dates = _coerce_list(filed_dates)
    acceptance_datetimes = _coerce_list(acceptance_datetimes)
    exchange = str(exchanges[0]).strip().upper() if exchanges else None
    rows: list[dict[str, Any]] = []
    for accn, form_type, filed_date, acceptance_dt in zip(accession_numbers, forms, filed_dates, acceptance_datetimes):
        accn_text = str(accn).strip() if accn is not None else ""
        if not accn_text:
            continue
        rows.append(
            {
                "cik": cik,
                "symbol": symbol,
                "issuer_name": issuer_name,
                "accession_number": accn_text,
                "form_type": str(form_type).strip() if form_type is not None else None,
                "filed_date": parse_date(filed_date),
                "acceptance_datetime": parse_datetime_utc(acceptance_dt),
                "submission_file_path": str(path),
                "submission_history_file": history_file,
                "exchange": exchange,
            }
        )
    return rows


def load_companyfacts_candidates(companyfacts_path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in list_json_files(companyfacts_path):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        cik = canonicalize_cik(payload.get("cik") or payload.get("cik_str") or payload.get("entityName"))
        entity_name = payload.get("entityName")
        facts_root = payload.get("facts") or {}
        if cik is None or not isinstance(facts_root, dict):
            continue
        for taxonomy, tag_map in facts_root.items():
            taxonomy_norm = normalize_taxonomy(taxonomy)
            if not isinstance(tag_map, dict):
                continue
            for tag, node in tag_map.items():
                if not isinstance(node, dict):
                    continue
                label = node.get("label")
                description = node.get("description")
                units = node.get("units") or {}
                if not isinstance(units, dict):
                    continue
                for unit, facts_list in units.items():
                    if not isinstance(facts_list, list):
                        continue
                    for fact in facts_list:
                        if not isinstance(fact, dict):
                            continue
                        rows.append(
                            {
                                "cik": cik,
                                "taxonomy": taxonomy_norm,
                                "tag": str(tag).strip(),
                                "label": label,
                                "description": description,
                                "source_unit": str(unit).strip(),
                                "value_raw": fact.get("val"),
                                "period_end": parse_date(fact.get("end")),
                                "period_start": parse_date(fact.get("start")),
                                "filed_date": parse_date(fact.get("filed")),
                                "acceptance_datetime": parse_datetime_utc(fact.get("accepted")),
                                "fy": fact.get("fy"),
                                "fp": fact.get("fp"),
                                "frame": fact.get("frame"),
                                "form_type": fact.get("form"),
                                "accession_number": fact.get("accn"),
                                "source_payload_path": str(path),
                                "entity_name": entity_name,
                            }
                        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "cik",
                "taxonomy",
                "tag",
                "source_unit",
                "value_raw",
                "period_end",
                "period_start",
                "filed_date",
                "acceptance_datetime",
                "fy",
                "fp",
                "frame",
                "form_type",
                "accession_number",
                "source_payload_path",
                "entity_name",
            ]
        )
    df = pd.DataFrame(rows)
    return df.reset_index(drop=True)


def build_accession_lookup(submissions_df: pd.DataFrame) -> pd.DataFrame:
    if submissions_df.empty:
        return pd.DataFrame(
            columns=["cik", "accession_number", "symbol", "issuer_name", "form_type", "filed_date", "acceptance_datetime", "exchange"]
        )
    sort_cols = ["cik", "accession_number", "filed_date", "acceptance_datetime"]
    df = submissions_df.sort_values(sort_cols, kind="stable").copy()
    return df.groupby(["cik", "accession_number"], as_index=False).last()


def resolve_mapping(config: ParseConfig, taxonomy: str, tag: str) -> MappingDecision:
    for idx, rule in enumerate(config.mapping_rules):
        if rule["taxonomy"] == taxonomy and rule["tag"] == tag:
            return MappingDecision(
                metric_name=rule["metric_name"],
                mapping_status=rule["mapping_status"],
                mapping_priority=rule["mapping_priority"],
                metric_family=rule.get("metric_family"),
                unit_base=rule.get("unit_base"),
                allowed_period_types=list(rule.get("allowed_period_types") or []),
                value_min=rule.get("value_min"),
                value_max=rule.get("value_max"),
                mapping_version="config_rule",
                rule_index=idx,
            )
    return MappingDecision(
        metric_name=None,
        mapping_status="unmapped",
        mapping_priority=10_000,
        metric_family=None,
        unit_base=None,
        allowed_period_types=[],
        value_min=None,
        value_max=None,
        mapping_version="config_rule",
        rule_index=None,
    )


def taxonomy_priority(config: ParseConfig, taxonomy: str, metric_name: Optional[str]) -> tuple[int, str]:
    taxonomy_norm = normalize_taxonomy(taxonomy)
    if metric_name and metric_name in config.metric_taxonomy_overrides:
        overrides = config.metric_taxonomy_overrides[metric_name]
        if taxonomy_norm in overrides:
            return int(overrides[taxonomy_norm]), "metric_override"
    if taxonomy_norm in config.taxonomy_priority:
        return int(config.taxonomy_priority[taxonomy_norm]), "global"
    return 10_000, "unknown"


def resolve_period_type(
    config: ParseConfig,
    period_start: Optional[pd.Timestamp],
    period_end: Optional[pd.Timestamp],
    frame: Any,
    fy: Any,
    fp: Any,
) -> PeriodDecision:
    if period_end is None or pd.isna(period_end):
        return PeriodDecision(None, None, 0.0, "invalid_date", "missing_period_end")
    if period_start is None or pd.isna(period_start):
        return PeriodDecision("Instant", 0, 1.0, None, None)
    duration_days = int((period_end - period_start).days)
    if duration_days < 0:
        return PeriodDecision(None, duration_days, 0.0, "invalid_date", "period_start_after_period_end")
    for period_type, bounds in config.period_windows.items():
        if bounds["min_days"] <= duration_days <= bounds["max_days"]:
            window_mid = 0.5 * (bounds["min_days"] + bounds["max_days"])
            spread = max(1.0, 0.5 * (bounds["max_days"] - bounds["min_days"]))
            quality = max(0.5, 1.0 - abs(duration_days - window_mid) / (spread + 1.0))
            if fp is not None and str(fp).strip().upper() in {"FY", "Q1", "Q2", "Q3"}:
                quality = min(1.0, quality + 0.05)
            if frame:
                quality = min(1.0, quality + 0.05)
            return PeriodDecision(period_type, duration_days, float(quality), None, None)
    fp_text = str(fp).strip().upper() if fp is not None else ""
    if fp_text == "FY" and duration_days >= 300:
        return PeriodDecision("Y", duration_days, 0.75, None, None)
    if fp_text in {"Q1", "Q2", "Q3", "Q4"} and 60 <= duration_days <= 120:
        return PeriodDecision("Q", duration_days, 0.70, None, None)
    frame_text = str(frame).strip().upper() if frame is not None else ""
    if frame_text.startswith("CY") and duration_days >= 300:
        return PeriodDecision("Y", duration_days, 0.65, None, None)
    return PeriodDecision(None, duration_days, 0.0, "ambiguous_context", f"unresolved_duration_days={duration_days}")


def unit_conversion_factor(source_unit: str, unit_base: str) -> Optional[float]:
    if source_unit == unit_base:
        return 1.0
    src = source_unit.upper()
    dst = unit_base.upper()
    if src == dst:
        return 1.0
    if src in {"USD", "USDOLLARS"} and dst == "USD":
        return 1.0
    if src in {"USD/SHARE", "USD/SHARES"} and dst == "USD/SHARE":
        return 1.0
    if src == "SHARES" and dst == "SHARES":
        return 1.0
    return None


def resolve_unit(
    config: ParseConfig,
    metric_name: Optional[str],
    source_unit: Any,
    source_value: Optional[float],
    requested_unit_base: Optional[str],
) -> UnitDecision:
    raw_source_unit = str(source_unit).strip() if source_unit is not None else ""
    canonical_source_unit = canonicalize_unit(raw_source_unit, config.unit_aliases)
    if source_value is None:
        return UnitDecision(None, raw_source_unit, canonical_source_unit, None, None, None, "invalid_value", "value_not_parseable")
    if requested_unit_base is None:
        requested_unit_base = canonical_source_unit or None
    if requested_unit_base is None:
        if config.unknown_unit_policy == "accept":
            return UnitDecision(None, raw_source_unit, canonical_source_unit, 1.0, "unknown_accepted", source_value, None, None)
        return UnitDecision(None, raw_source_unit, canonical_source_unit, None, None, None, "incompatible_unit", "missing_unit_base_for_metric")
    factor = unit_conversion_factor(canonical_source_unit, requested_unit_base)
    if factor is None:
        if config.unknown_unit_policy == "accept":
            return UnitDecision(requested_unit_base, raw_source_unit, canonical_source_unit, 1.0, "unknown_accepted", source_value, None, None)
        return UnitDecision(
            requested_unit_base,
            raw_source_unit,
            canonical_source_unit,
            None,
            "incompatible",
            None,
            "incompatible_unit",
            f"source_unit={canonical_source_unit!r} not compatible with unit_base={requested_unit_base!r}",
        )
    normalized_value = float(source_value) * factor
    compatibility = "exact" if factor == 1.0 and canonical_source_unit == requested_unit_base else "converted"
    return UnitDecision(
        unit_base=requested_unit_base,
        source_unit=raw_source_unit,
        canonical_source_unit=canonical_source_unit,
        unit_scale_factor=factor,
        unit_compatibility=compatibility,
        normalized_value=normalized_value,
        rejection_class=None,
        rejection_detail=None,
    )


def allowed_period_types_for_metric(config: ParseConfig, metric_name: Optional[str], mapping_allowed: list[str]) -> list[str]:
    if metric_name and metric_name in config.metric_period_overrides:
        return list(config.metric_period_overrides[metric_name])
    return list(mapping_allowed)


def check_value_bounds(
    config: ParseConfig,
    metric_name: Optional[str],
    unit_base: Optional[str],
    value: Optional[float],
    mapping: MappingDecision,
) -> Optional[tuple[str, str]]:
    if value is None:
        return ("invalid_value", "value_not_parseable")
    if not math.isfinite(value):
        return ("invalid_value", "value_non_finite")
    bounds: dict[str, float] = {}
    if unit_base and unit_base in config.value_bounds_by_unit:
        bounds.update(config.value_bounds_by_unit[unit_base])
    if metric_name and metric_name in config.metric_bounds:
        bounds.update(config.metric_bounds[metric_name])
    if mapping.value_min is not None:
        bounds["min"] = mapping.value_min
    if mapping.value_max is not None:
        bounds["max"] = mapping.value_max
    min_v = bounds.get("min")
    max_v = bounds.get("max")
    if min_v is not None and value < min_v:
        return ("invalid_value", f"value_below_min:{value}<{min_v}")
    if max_v is not None and value > max_v:
        return ("invalid_value", f"value_above_max:{value}>{max_v}")
    return None


def compute_quality_score(
    config: ParseConfig,
    tax_priority: int,
    mapping_status: str,
    unit_compatibility: Optional[str],
    context_quality: float,
    dominance_score: float,
) -> float:
    max_rank = max(config.taxonomy_priority.values()) if config.taxonomy_priority else 5
    q_tax = 1.0 if tax_priority <= 1 else max(0.0, 1.0 - ((tax_priority - 1) / max(1, max_rank)))
    q_map = {"exact": 1.0, "alias": 0.90, "heuristic": 0.75}.get(mapping_status, 0.0)
    q_unit = {"exact": 1.0, "converted": 0.90, "unknown_accepted": 0.50, None: 0.0}.get(unit_compatibility, 0.0)
    q_context = float(max(0.0, min(1.0, context_quality)))
    q_dom = float(max(0.0, min(1.0, dominance_score)))
    w = config.quality_weights
    score = (
        w.get("taxonomy", 0.25) * q_tax
        + w.get("mapping", 0.25) * q_map
        + w.get("unit", 0.20) * q_unit
        + w.get("context", 0.15) * q_context
        + w.get("dominance", 0.15) * q_dom
    )
    return round(float(score), 6)


def build_candidate_rows(
    facts_df: pd.DataFrame,
    submissions_lookup: pd.DataFrame,
    config: ParseConfig,
    mapping_version: str,
    run_id: str,
    asof: Optional[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if facts_df.empty:
        accepted_cols = [
            "cik",
            "symbol",
            "metric_name",
            "value",
            "unit_base",
            "period_end",
            "period_type",
            "filed_date",
            "acceptance_datetime",
            "source_taxonomy",
            "source_tag",
            "source_unit",
            "accession_number",
            "form_type",
            "mapping_version",
            "run_id",
        ]
        rejected_cols = [
            "cik",
            "taxonomy",
            "tag",
            "unit",
            "period_end",
            "filed_date",
            "acceptance_datetime",
            "rejection_class",
            "rejection_detail",
            "run_id",
        ]
        return pd.DataFrame(columns=accepted_cols), pd.DataFrame(columns=rejected_cols)

    sub_cols = ["cik", "accession_number", "symbol", "issuer_name", "form_type", "filed_date", "acceptance_datetime", "exchange"]
    submissions_lookup = submissions_lookup[sub_cols].copy() if not submissions_lookup.empty else pd.DataFrame(columns=sub_cols)
    merged = facts_df.merge(submissions_lookup, on=["cik", "accession_number"], how="left", suffixes=("", "_sub"))

    accepted_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    for row in merged.to_dict(orient="records"):
        raw_value = to_float(row.get("value_raw"))
        mapping = resolve_mapping(config, normalize_taxonomy(row.get("taxonomy")), str(row.get("tag") or "").strip())
        tax_rank, tax_rank_source = taxonomy_priority(config, row.get("taxonomy"), mapping.metric_name)
        if mapping.metric_name is None:
            rejected_rows.append(
                _make_reject_row(row, run_id, "unmapped_tag", f"taxonomy={row.get('taxonomy')} tag={row.get('tag')}")
            )
            continue
        if normalize_taxonomy(row.get("taxonomy")) not in config.taxonomy_priority and config.unknown_taxonomy_policy != "accept":
            rejected_rows.append(
                _make_reject_row(row, run_id, "unsupported_taxonomy", f"taxonomy={row.get('taxonomy')}")
            )
            continue
        period = resolve_period_type(
            config,
            row.get("period_start"),
            row.get("period_end"),
            row.get("frame"),
            row.get("fy"),
            row.get("fp"),
        )
        if period.rejection_class is not None:
            rejected_rows.append(_make_reject_row(row, run_id, period.rejection_class, period.rejection_detail))
            continue
        allowed_pt = allowed_period_types_for_metric(config, mapping.metric_name, mapping.allowed_period_types)
        if allowed_pt and period.period_type not in allowed_pt:
            rejected_rows.append(
                _make_reject_row(
                    row,
                    run_id,
                    "ambiguous_context",
                    f"period_type={period.period_type} not allowed for metric={mapping.metric_name}",
                )
            )
            continue
        unit = resolve_unit(config, mapping.metric_name, row.get("source_unit"), raw_value, mapping.unit_base)
        if unit.rejection_class is not None:
            rejected_rows.append(_make_reject_row(row, run_id, unit.rejection_class, unit.rejection_detail))
            continue
        if row.get("period_end") is None or pd.isna(row.get("period_end")) or row.get("filed_date") is None or pd.isna(row.get("filed_date")):
            rejected_rows.append(_make_reject_row(row, run_id, "invalid_date", "missing_period_end_or_filed_date"))
            continue
        if row["filed_date"] < row["period_end"]:
            rejected_rows.append(_make_reject_row(row, run_id, "invalid_date", "filed_date_before_period_end"))
            continue
        acceptance_dt = row.get("acceptance_datetime")
        if acceptance_dt is not None and not pd.isna(acceptance_dt) and acceptance_dt.tz_convert("UTC").tz_localize(None) < row["period_end"]:
            rejected_rows.append(_make_reject_row(row, run_id, "invalid_date", "acceptance_datetime_before_period_end"))
            continue
        bounds_error = check_value_bounds(config, mapping.metric_name, unit.unit_base, unit.normalized_value, mapping)
        if bounds_error is not None:
            rejected_rows.append(_make_reject_row(row, run_id, bounds_error[0], bounds_error[1]))
            continue
        source_record_hash = sha256_text(
            stable_json_dumps(
                {
                    "cik": row.get("cik"),
                    "taxonomy": row.get("taxonomy"),
                    "tag": row.get("tag"),
                    "source_unit": row.get("source_unit"),
                    "value": unit.normalized_value,
                    "period_start": str(row.get("period_start")),
                    "period_end": str(row.get("period_end")),
                    "filed_date": str(row.get("filed_date")),
                    "acceptance_datetime": str(row.get("acceptance_datetime")),
                    "accession_number": row.get("accession_number"),
                }
            )
        )
        accepted_rows.append(
            {
                "cik": row.get("cik"),
                "symbol": first_not_null(row.get("symbol"), row.get("symbol_sub")),
                "issuer_name": first_not_null(row.get("entity_name"), row.get("issuer_name")),
                "metric_name": mapping.metric_name,
                "metric_family": mapping.metric_family,
                "value": unit.normalized_value,
                "unit_base": unit.unit_base,
                "period_end": row.get("period_end"),
                "period_start": row.get("period_start"),
                "period_type": period.period_type,
                "duration_days": period.duration_days,
                "filed_date": row.get("filed_date"),
                "acceptance_datetime": row.get("acceptance_datetime"),
                "source_taxonomy": row.get("taxonomy"),
                "source_tag": row.get("tag"),
                "source_unit": unit.source_unit,
                "canonical_source_unit": unit.canonical_source_unit,
                "accession_number": row.get("accession_number"),
                "form_type": first_not_null(row.get("form_type"), row.get("form_type_sub")),
                "fy": row.get("fy"),
                "fp": row.get("fp"),
                "frame": row.get("frame"),
                "mapping_status": mapping.mapping_status,
                "mapping_priority": mapping.mapping_priority,
                "mapping_version": mapping_version,
                "source_taxonomy_priority": tax_rank,
                "source_taxonomy_priority_source": tax_rank_source,
                "unit_scale_factor": unit.unit_scale_factor,
                "unit_compatibility": unit.unit_compatibility,
                "context_quality": period.context_quality,
                "quality_score": None,
                "selection_reason": None,
                "canonical_priority_rank": None,
                "run_id": run_id,
                "asof": asof,
                "source_payload_path": row.get("source_payload_path"),
                "source_record_hash": source_record_hash,
                "label": row.get("label"),
                "description": row.get("description"),
            }
        )
    accepted_df = pd.DataFrame(accepted_rows)
    rejected_df = pd.DataFrame(rejected_rows)
    return accepted_df, rejected_df


def first_not_null(*values: Any) -> Any:
    for value in values:
        if value is not None and not (isinstance(value, float) and math.isnan(value)):
            return value
    return None


def _make_reject_row(row: dict[str, Any], run_id: str, rejection_class: str, rejection_detail: Optional[str]) -> dict[str, Any]:
    return {
        "cik": row.get("cik"),
        "taxonomy": row.get("taxonomy"),
        "tag": row.get("tag"),
        "unit": row.get("source_unit"),
        "period_end": row.get("period_end"),
        "filed_date": row.get("filed_date"),
        "acceptance_datetime": row.get("acceptance_datetime"),
        "rejection_class": rejection_class,
        "rejection_detail": rejection_detail,
        "accession_number": row.get("accession_number"),
        "form_type": row.get("form_type"),
        "value_raw": row.get("value_raw"),
        "run_id": run_id,
    }


def select_canonical_facts(config: ParseConfig, candidates_df: pd.DataFrame, run_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if candidates_df.empty:
        return candidates_df.copy(), pd.DataFrame(columns=["cik", "taxonomy", "tag", "unit", "period_end", "filed_date", "acceptance_datetime", "rejection_class", "rejection_detail", "run_id"])
    accepted_groups: list[pd.DataFrame] = []
    rejected_rows: list[dict[str, Any]] = []
    key_cols = ["cik", "metric_name", "period_end", "period_type"]
    work = candidates_df.copy()
    work["_filed_sort"] = pd.to_datetime(work["filed_date"], errors="coerce")
    work["_acc_sort"] = pd.to_datetime(work["acceptance_datetime"], errors="coerce", utc=True)
    work["_acc_sort"] = work["_acc_sort"].dt.tz_convert("UTC").dt.tz_localize(None)
    work["_unit_comp_rank"] = work["unit_compatibility"].map({"exact": 0, "converted": 1, "unknown_accepted": 2}).fillna(9)
    work["_context_rank"] = -work["context_quality"].fillna(0.0)
    for _, group in work.groupby(key_cols, dropna=False, sort=False):
        group = group.copy().sort_values(
            [
                "source_taxonomy_priority",
                "mapping_priority",
                "_unit_comp_rank",
                "_context_rank",
                "_filed_sort",
                "_acc_sort",
                "source_record_hash",
            ],
            ascending=[True, True, True, True, False, False, True],
            kind="stable",
        )
        if group.empty:
            continue
        top = group.iloc[0].copy()
        dominance_score = 1.0
        if len(group) > 1:
            comp_cols = [
                "source_taxonomy_priority",
                "mapping_priority",
                "_unit_comp_rank",
                "_context_rank",
                "_filed_sort",
                "_acc_sort",
            ]
            first_signature = tuple(_normalize_signature_value(top[col]) for col in comp_cols)
            tied = group[group.apply(lambda r: tuple(_normalize_signature_value(r[c]) for c in comp_cols) == first_signature, axis=1)].copy()
            if len(tied) > 1:
                tied_distinct_values = tied["value"].round(12).nunique(dropna=False)
                tied_distinct_hashes = tied["source_record_hash"].nunique(dropna=False)
                if tied_distinct_values > 1 or tied_distinct_hashes > 1:
                    if config.unresolved_tie_policy == "accept":
                        top["selection_reason"] = "tie_accepted_by_policy"
                        dominance_score = 0.6
                        dominated = group.iloc[1:].copy()
                    else:
                        for _, row in group.iterrows():
                            rejected_rows.append(
                                {
                                    "cik": row.get("cik"),
                                    "taxonomy": row.get("source_taxonomy"),
                                    "tag": row.get("source_tag"),
                                    "unit": row.get("source_unit"),
                                    "period_end": row.get("period_end"),
                                    "filed_date": row.get("filed_date"),
                                    "acceptance_datetime": row.get("acceptance_datetime"),
                                    "rejection_class": "conflict_unresolved",
                                    "rejection_detail": "unresolved_tie_after_selection_order",
                                    "accession_number": row.get("accession_number"),
                                    "form_type": row.get("form_type"),
                                    "value_raw": row.get("value"),
                                    "run_id": run_id,
                                }
                            )
                        continue
                else:
                    top["selection_reason"] = "tie_broken_by_source_hash"
                    dominance_score = 0.7
                    dominated = group.iloc[1:].copy()
            else:
                dominated = group.iloc[1:].copy()
                dominance_score = 0.85 if len(group) == 2 else 0.75
            if top.get("selection_reason") is None:
                top["selection_reason"] = "dominant_candidate"
            for _, row in dominated.iterrows():
                rejected_rows.append(
                    {
                        "cik": row.get("cik"),
                        "taxonomy": row.get("source_taxonomy"),
                        "tag": row.get("source_tag"),
                        "unit": row.get("source_unit"),
                        "period_end": row.get("period_end"),
                        "filed_date": row.get("filed_date"),
                        "acceptance_datetime": row.get("acceptance_datetime"),
                        "rejection_class": "duplicate_dominated",
                        "rejection_detail": f"dominated_by_source_record_hash={top.get('source_record_hash')}",
                        "accession_number": row.get("accession_number"),
                        "form_type": row.get("form_type"),
                        "value_raw": row.get("value"),
                        "run_id": run_id,
                    }
                )
        else:
            top["selection_reason"] = "single_candidate"
        top["quality_score"] = compute_quality_score(
            config=config,
            tax_priority=int(top.get("source_taxonomy_priority") or 10_000),
            mapping_status=str(top.get("mapping_status") or "unmapped"),
            unit_compatibility=top.get("unit_compatibility"),
            context_quality=float(top.get("context_quality") or 0.0),
            dominance_score=dominance_score,
        )
        top["canonical_priority_rank"] = 1
        accepted_groups.append(pd.DataFrame([top]))
    accepted_df = pd.concat(accepted_groups, ignore_index=True) if accepted_groups else pd.DataFrame(columns=candidates_df.columns)
    rejected_df = pd.DataFrame(rejected_rows)
    drop_cols = [c for c in ["_filed_sort", "_acc_sort", "_unit_comp_rank", "_context_rank"] if c in accepted_df.columns]
    if drop_cols:
        accepted_df = accepted_df.drop(columns=drop_cols)
    return accepted_df, rejected_df


def _normalize_signature_value(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def build_metrics_table(candidates_df: pd.DataFrame, canonical_df: pd.DataFrame, rejected_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rows.append({"metric_group": "pipeline", "metric_name": "candidates_processed", "metric_value": len(candidates_df)})
    rows.append({"metric_group": "pipeline", "metric_name": "accepted_canonical", "metric_value": len(canonical_df)})
    rows.append({"metric_group": "pipeline", "metric_name": "rejected_total", "metric_value": len(rejected_df)})
    if not canonical_df.empty:
        by_tax = canonical_df.groupby("source_taxonomy").size().reset_index(name="metric_value")
        rows.extend(
            {"metric_group": "accepted_by_taxonomy", "metric_name": str(r["source_taxonomy"]), "metric_value": int(r["metric_value"])}
            for _, r in by_tax.iterrows()
        )
        by_mapping = canonical_df.groupby("mapping_status").size().reset_index(name="metric_value")
        rows.extend(
            {"metric_group": "accepted_by_mapping_status", "metric_name": str(r["mapping_status"]), "metric_value": int(r["metric_value"])}
            for _, r in by_mapping.iterrows()
        )
        by_metric = canonical_df.groupby("metric_name").size().reset_index(name="metric_value")
        rows.extend(
            {"metric_group": "accepted_by_metric", "metric_name": str(r["metric_name"]), "metric_value": int(r["metric_value"])}
            for _, r in by_metric.iterrows()
        )
    if not rejected_df.empty:
        by_rej = rejected_df.groupby("rejection_class").size().reset_index(name="metric_value")
        rows.extend(
            {"metric_group": "rejected_by_class", "metric_name": str(r["rejection_class"]), "metric_value": int(r["metric_value"])}
            for _, r in by_rej.iterrows()
        )
    return pd.DataFrame(rows)


def determine_gate(config: ParseConfig, candidates_df: pd.DataFrame, canonical_df: pd.DataFrame, rejected_df: pd.DataFrame) -> tuple[str, list[str]]:
    reasons: list[str] = []
    total = max(1, len(candidates_df))
    reject_ratio = len(rejected_df) / total
    unmapped_rejects = 0
    if not rejected_df.empty and "rejection_class" in rejected_df.columns:
        unmapped_rejects = int((rejected_df["rejection_class"] == "unmapped_tag").sum())
    mapping_fail_ratio = unmapped_rejects / total
    if mapping_fail_ratio > config.systemic_mapping_failure_ratio:
        reasons.append(f"systemic_mapping_failure_ratio={mapping_fail_ratio:.4f}")
    if reject_ratio > max(config.warn_reject_ratio, 0.75):
        reasons.append(f"excessive_reject_ratio={reject_ratio:.4f}")
    if len(canonical_df) == 0 and len(candidates_df) > 0:
        reasons.append("no_canonical_facts_selected")
    if reasons:
        return "FAIL", reasons
    if reject_ratio > config.warn_reject_ratio:
        return "WARN", [f"warn_reject_ratio={reject_ratio:.4f}"]
    return "PASS", []


def coerce_output_columns(canonical_df: pd.DataFrame, rejected_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    canonical_cols = [
        "cik",
        "symbol",
        "issuer_name",
        "metric_name",
        "metric_family",
        "value",
        "unit_base",
        "period_end",
        "period_type",
        "filed_date",
        "acceptance_datetime",
        "source_taxonomy",
        "source_tag",
        "source_unit",
        "canonical_source_unit",
        "accession_number",
        "form_type",
        "mapping_version",
        "mapping_status",
        "unit_scale_factor",
        "canonical_priority_rank",
        "selection_reason",
        "quality_score",
        "source_record_hash",
        "run_id",
        "asof",
    ]
    rejected_cols = [
        "cik",
        "taxonomy",
        "tag",
        "unit",
        "period_end",
        "filed_date",
        "acceptance_datetime",
        "rejection_class",
        "rejection_detail",
        "accession_number",
        "form_type",
        "value_raw",
        "run_id",
    ]
    for col in canonical_cols:
        if col not in canonical_df.columns:
            canonical_df[col] = None
    for col in rejected_cols:
        if col not in rejected_df.columns:
            rejected_df[col] = None
    return canonical_df[canonical_cols].copy(), rejected_df[rejected_cols].copy()


def parse_xbrl_run(
    submissions_raw_path: str,
    companyfacts_raw_path: str,
    config_path: str,
    run_id: str,
    asof: Optional[str] = None,
) -> dict[str, Any]:
    started_at = utc_now_iso()
    submissions_path = Path(submissions_raw_path)
    companyfacts_path = Path(companyfacts_raw_path)
    if not submissions_path.exists():
        raise InputValidationError(f"submissions_raw_path does not exist: {submissions_path}")
    if not companyfacts_path.exists():
        raise InputValidationError(f"companyfacts_raw_path does not exist: {companyfacts_path}")
    config, raw_config, config_hash = load_config(Path(config_path))
    mapping_version = config_mapping_version(config_hash)
    output_dir = Path(config.output_dir)
    ensure_directory(output_dir)

    submissions_df = load_submissions_metadata(submissions_path)
    facts_df = load_companyfacts_candidates(companyfacts_path)
    accession_lookup = build_accession_lookup(submissions_df)
    candidates_df, preselection_rejected_df = build_candidate_rows(
        facts_df=facts_df,
        submissions_lookup=accession_lookup,
        config=config,
        mapping_version=mapping_version,
        run_id=run_id,
        asof=asof,
    )
    canonical_df, selection_rejected_df = select_canonical_facts(config, candidates_df, run_id)
    rejected_df = pd.concat([preselection_rejected_df, selection_rejected_df], ignore_index=True) if not preselection_rejected_df.empty or not selection_rejected_df.empty else pd.DataFrame(columns=preselection_rejected_df.columns if not preselection_rejected_df.empty else selection_rejected_df.columns)
    canonical_df, rejected_df = coerce_output_columns(canonical_df, rejected_df)
    metrics_df = build_metrics_table(candidates_df, canonical_df, rejected_df)
    gate, gate_reasons = determine_gate(config, candidates_df, canonical_df, rejected_df)

    canonical_path = table_write(output_dir / f"facts_canonical_{run_id}", canonical_df, config.table_format, config.allow_csv_fallback)
    rejected_path = table_write(output_dir / f"facts_rejected_{run_id}", rejected_df, config.table_format, config.allow_csv_fallback)
    metrics_path = table_write(output_dir / f"facts_metrics_{run_id}", metrics_df, config.table_format, config.allow_csv_fallback)

    ended_at = utc_now_iso()
    manifest = {
        "module": "data.edgar.parse_xbrl",
        "run_id": run_id,
        "asof": asof,
        "started_at_utc": started_at,
        "ended_at_utc": ended_at,
        "config_path": str(Path(config_path).resolve()),
        "config_hash": config_hash,
        "mapping_version": mapping_version,
        "submissions_raw_path": str(submissions_path.resolve()),
        "companyfacts_raw_path": str(companyfacts_path.resolve()),
        "inputs": {
            "submissions_rows": int(len(submissions_df)),
            "raw_fact_candidates": int(len(facts_df)),
            "preselection_candidates": int(len(candidates_df)),
        },
        "outputs": {
            "facts_canonical": canonical_path,
            "facts_rejected": rejected_path,
            "facts_metrics": metrics_path,
        },
        "counts": {
            "accepted": int(len(canonical_df)),
            "rejected": int(len(rejected_df)),
            "rejected_by_class": rejected_df["rejection_class"].value_counts(dropna=False).to_dict() if not rejected_df.empty else {},
            "accepted_by_taxonomy": canonical_df["source_taxonomy"].value_counts(dropna=False).to_dict() if not canonical_df.empty else {},
            "accepted_by_mapping_status": canonical_df["mapping_status"].value_counts(dropna=False).to_dict() if not canonical_df.empty else {},
        },
        "gate": gate,
        "gate_reasons": gate_reasons,
        "raw_config": raw_config,
    }
    manifest_path = output_dir / f"manifest_{run_id}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    return manifest


def _json_default(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        if value.tzinfo is not None:
            return value.isoformat()
        return value.isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Canonicalize raw SEC companyfacts XBRL into semantically resolved internal metrics."
    )
    parser.add_argument("--submissions-path", required=True, help="Path to raw submissions repository (JSON tree).")
    parser.add_argument("--companyfacts-path", required=True, help="Path to raw companyfacts repository (JSON tree).")
    parser.add_argument("--config-path", required=True, help="YAML/JSON config for mappings, priorities and rules.")
    parser.add_argument("--run-id", required=True, help="Auditable run identifier.")
    parser.add_argument("--asof", required=False, default=None, help="Logical asof snapshot date, e.g. 2026-03-05.")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    manifest = parse_xbrl_run(
        submissions_raw_path=args.submissions_path,
        companyfacts_raw_path=args.companyfacts_path,
        config_path=args.config_path,
        run_id=args.run_id,
        asof=args.asof,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
