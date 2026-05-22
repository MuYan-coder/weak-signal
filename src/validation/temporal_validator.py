"""Temporal validation for v2.7 key-core potential scoring.

This module keeps the validation deliberately lightweight: it builds candidate
level time-window features from already extracted evidence, records why a
candidate did or did not pass, and never drops candidates because dates are
missing.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd


TEMPORAL_VALIDATION_COLUMNS = [
    "candidate_id",
    "candidate_name",
    "first_seen_date",
    "last_seen_date",
    "observation_window_start",
    "observation_window_end",
    "validation_window_start",
    "validation_window_end",
    "mentions_observation",
    "mentions_validation",
    "high_quality_mentions_observation",
    "high_quality_mentions_validation",
    "source_count_observation",
    "source_count_validation",
    "org_count_observation",
    "org_count_validation",
    "asset_count_observation",
    "asset_count_validation",
    "growth_rate",
    "source_growth_rate",
    "org_growth_rate",
    "high_quality_growth_rate",
    "cagr",
    "temporal_momentum_score",
    "temporal_validation_passed",
    "temporal_validation_tier",
    "temporal_validation_status",
    "earlyness_years",
    "date_coverage_ratio",
    "temporal_validation_reason",
]

TEMPORAL_CANDIDATE_COLUMNS = [
    column
    for column in TEMPORAL_VALIDATION_COLUMNS
    if column not in {"candidate_name"}
]

DATE_FIELDS = [
    "date",
    "event_date",
    "publication_date",
    "application_date",
    "grant_date",
    "time",
    "updated_at",
]

CANDIDATE_ID_FIELDS = ["candidate_id", "candidate_cluster_id", "id"]
CANDIDATE_NAME_FIELDS = [
    "final_research_object_name",
    "topic_summary_name",
    "display_candidate_name",
    "tech_name",
    "technology",
    "canonical_candidate_name_en",
    "normalized_candidate_text",
    "raw_candidate_text",
]

UNKNOWN_VALUES = {"", "nan", "none", "null", "nat", "unknown", "未知", "无"}


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    return "" if text.lower() in UNKNOWN_VALUES else text


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return default


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = _safe_text(value).lower()
    return text in {"1", "true", "yes", "y", "是", "通过"}


def _parse_structured(value: Any) -> Any:
    if isinstance(value, (list, dict, tuple, set)):
        return value
    text = _safe_text(value)
    if not text:
        return []
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        return ast.literal_eval(text)
    except Exception:
        return text


def _safe_list(value: Any) -> List[Any]:
    parsed = _parse_structured(value)
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, tuple):
        return list(parsed)
    if isinstance(parsed, set):
        return list(parsed)
    if isinstance(parsed, dict):
        return [parsed]
    text = _safe_text(parsed)
    if not text:
        return []
    if "|" in text:
        return [part.strip() for part in text.split("|") if part.strip()]
    if "," in text:
        return [part.strip() for part in text.split(",") if part.strip()]
    return [text]


def _parse_date(value: Any) -> pd.Timestamp | None:
    text = _safe_text(value)
    if not text:
        return None
    text = text.strip()
    chinese_match = re.search(r"(20\d{2}|19\d{2})年(?:\s*(\d{1,2})月)?(?:\s*(\d{1,2})日)?", text)
    if chinese_match:
        year = int(chinese_match.group(1))
        month = int(chinese_match.group(2) or 1)
        day = int(chinese_match.group(3) or 1)
        try:
            return pd.Timestamp(year=year, month=month, day=day).normalize()
        except Exception:
            pass
    slash_match = re.search(r"\b(20\d{2}|19\d{2})[./-](\d{1,2})(?:[./-](\d{1,2}))?\b", text)
    if slash_match:
        year = int(slash_match.group(1))
        month = int(slash_match.group(2) or 1)
        day = int(slash_match.group(3) or 1)
        try:
            return pd.Timestamp(year=year, month=month, day=day).normalize()
        except Exception:
            pass
    year_match = re.search(r"\b(20\d{2}|19\d{2})\b", text)
    if year_match and len(text) <= 16:
        try:
            return pd.Timestamp(year=int(year_match.group(1)), month=1, day=1).normalize()
        except Exception:
            pass
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed).normalize()


def _date_to_text(value: pd.Timestamp | None) -> str:
    if value is None or pd.isna(value):
        return ""
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _candidate_id(row: pd.Series, index: int) -> str:
    for field in CANDIDATE_ID_FIELDS:
        text = _safe_text(row.get(field))
        if text:
            return text
    name = _candidate_name(row)
    return f"candidate::{name}" if name else f"candidate_row_{index + 1}"


def _candidate_name(row: pd.Series) -> str:
    for field in CANDIDATE_NAME_FIELDS:
        text = _safe_text(row.get(field))
        if text:
            return text
    return ""


def _normalize_candidate_key(value: Any) -> str:
    text = _safe_text(value).lower()
    text = re.sub(r"[\s_\-]+", "", text)
    text = re.sub(r"[^\w\u4e00-\u9fff]+", "", text)
    return text


def _temporal_input_key(row: pd.Series, index: int) -> str:
    parts = [
        _candidate_id(row, index),
        _candidate_name(row),
        row.get("mechanism_core"),
        row.get("relation_target"),
        row.get("relation_task"),
        row.get("stable_object_label"),
    ]
    key = "::".join(_normalize_candidate_key(part) for part in parts if _normalize_candidate_key(part))
    return key or f"candidate_row_{index + 1}"


def _combine_group_rows(rows: List[pd.Series]) -> pd.Series:
    if not rows:
        return pd.Series(dtype="object")
    combined = rows[0].copy()
    list_fields = [
        "evidence_items",
        "mention_ids",
        "mention_dates",
        "source_types",
        "orgs",
        "display_candidate_aliases",
    ]
    for field in list_fields:
        values: List[Any] = []
        for row in rows:
            values.extend(_safe_list(row.get(field)))
        if values:
            combined[field] = values

    for field in [
        "source_count",
        "cluster_evidence_count",
        "total_mentions",
        "candidate_evidence_quality",
        "candidate_core_evidence_quality",
        "high_quality_evidence_count",
        "weak_signal_score",
        "quality_adjusted_rank_score",
    ]:
        values = [_safe_float(row.get(field), 0.0) for row in rows]
        if values and any(value > 0 for value in values):
            combined[field] = max(values)
    return combined


def _record_key_values(record: Dict[str, Any]) -> List[str]:
    keys = []
    for field in ["id", "event_id", "source_id", "title"]:
        text = _safe_text(record.get(field))
        if text:
            keys.append(text)
    return keys


def _record_from_mapping(record: Dict[str, Any]) -> Dict[str, Any]:
    date = None
    for field in DATE_FIELDS:
        date = _parse_date(record.get(field))
        if date is not None:
            break
    return {
        "id": _safe_text(record.get("id") or record.get("event_id") or record.get("source_id")),
        "title": _safe_text(record.get("title")),
        "date": date,
        "source_type": _safe_text(record.get("source_type")),
        "org": _safe_text(record.get("org") or record.get("organization") or record.get("source")),
        "event_quality_score": _safe_float(record.get("event_quality_score"), 0.0),
    }


def _build_record_lookup(*frames: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
    lookup: Dict[str, List[Dict[str, Any]]] = {}
    for frame in frames:
        if frame is None or frame.empty:
            continue
        for _, row in frame.iterrows():
            raw = row.to_dict()
            record = _record_from_mapping(raw)
            for key in _record_key_values(raw):
                lookup.setdefault(key, []).append(record)
    return lookup


def _records_from_evidence_items(items: Iterable[Any]) -> List[Dict[str, Any]]:
    records = []
    for item in items:
        if isinstance(item, dict):
            records.append(_record_from_mapping(item))
    return records


def _dedupe_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    deduped = []
    for record in records:
        key = (
            _safe_text(record.get("id")),
            _safe_text(record.get("title")),
            _date_to_text(record.get("date")),
            _safe_text(record.get("source_type")),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    return deduped


def _collect_candidate_records(row: pd.Series, lookup: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    records.extend(_records_from_evidence_items(_safe_list(row.get("evidence_items"))))

    for mention_id in _safe_list(row.get("mention_ids")):
        text = _safe_text(mention_id)
        if text and text in lookup:
            records.extend(lookup[text])

    mention_dates = [_parse_date(item) for item in _safe_list(row.get("mention_dates"))]
    for idx, date in enumerate([item for item in mention_dates if item is not None]):
        records.append(
            {
                "id": f"mention_date_{idx + 1}",
                "title": "",
                "date": date,
                "source_type": "",
                "org": "",
                "event_quality_score": 0.0,
            }
        )

    return _dedupe_records(records)


def _growth_rate(new_value: int, old_value: int) -> float:
    return round((new_value - old_value) / max(old_value, 1), 3)


def _count_window(
    records: List[Dict[str, Any]],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> Tuple[int, int, int, int, int]:
    subset = [
        record
        for record in records
        if record.get("date") is not None and start <= record["date"] < end
    ]
    high_quality = [
        record
        for record in subset
        if _safe_float(record.get("event_quality_score"), 0.0) >= 7.0
    ]
    sources = {_safe_text(record.get("source_type")) for record in subset if _safe_text(record.get("source_type"))}
    orgs = {_safe_text(record.get("org")) for record in subset if _safe_text(record.get("org"))}
    assets = [
        record
        for record in subset
        if _safe_text(record.get("source_type")).lower() in {"patent", "standard", "asset", "software"}
    ]
    return len(subset), len(high_quality), len(sources), len(orgs), len(assets)


def _mainstream_hint(row: pd.Series) -> bool:
    signal_type = _safe_text(row.get("signal_type"))
    candidate_stage = _safe_text(row.get("candidate_stage"))
    total_mentions = _safe_float(row.get("total_mentions"), 0.0)
    evidence_count = _safe_float(row.get("cluster_evidence_count"), 0.0)
    source_count = _safe_float(row.get("source_count"), 0.0)
    return (
        signal_type in {"hotspot", "scope_overview"}
        or candidate_stage == "scope_overview"
        or (source_count >= 4 and (total_mentions >= 80 or evidence_count >= 50))
    )


def _tier_score(tier: str, growth_rate: float, source_growth_rate: float, org_growth_rate: float) -> float:
    base = {
        "rising_validated": 90.0,
        "quality_rising": 78.0,
        "single_source_high_quality_watch": 62.0,
        "stable_no_growth": 48.0,
        "insufficient_date": 40.0,
        "mainstream_or_hotspot": 52.0,
        "declining_or_one_off": 25.0,
    }.get(tier, 40.0)
    lift = max(growth_rate, source_growth_rate, org_growth_rate, 0.0) * 10.0
    return round(max(0.0, min(base + lift, 100.0)), 2)


def _temporal_row(
    row: pd.Series,
    index: int,
    lookup: Dict[str, List[Dict[str, Any]]],
    observation_months: int,
    validation_months: int,
    growth_threshold: float,
    source_growth_threshold: float,
    org_growth_threshold: float,
    high_quality_growth_threshold: float,
    min_dated_evidence: int,
) -> Dict[str, Any]:
    candidate_id = _candidate_id(row, index)
    candidate_name = _candidate_name(row)
    records = _collect_candidate_records(row, lookup)
    total_record_count = len(records)
    dated_records = [record for record in records if record.get("date") is not None]
    date_coverage_ratio = round(len(dated_records) / max(total_record_count, 1), 3) if total_record_count else 0.0

    empty = {
        "candidate_id": candidate_id,
        "candidate_name": candidate_name,
        "first_seen_date": "",
        "last_seen_date": "",
        "observation_window_start": "",
        "observation_window_end": "",
        "validation_window_start": "",
        "validation_window_end": "",
        "mentions_observation": 0,
        "mentions_validation": 0,
        "high_quality_mentions_observation": 0,
        "high_quality_mentions_validation": 0,
        "source_count_observation": 0,
        "source_count_validation": 0,
        "org_count_observation": 0,
        "org_count_validation": 0,
        "asset_count_observation": 0,
        "asset_count_validation": 0,
        "growth_rate": 0.0,
        "source_growth_rate": 0.0,
        "org_growth_rate": 0.0,
        "high_quality_growth_rate": 0.0,
        "cagr": 0.0,
        "temporal_momentum_score": 0.0,
        "temporal_validation_passed": False,
        "temporal_validation_tier": "insufficient_date",
        "temporal_validation_status": "insufficient_date",
        "earlyness_years": 0.0,
        "date_coverage_ratio": date_coverage_ratio,
        "temporal_validation_reason": "缺少可解析日期，无法划分观察期和验证期。",
    }

    if _mainstream_hint(row) and _safe_float(row.get("cluster_evidence_count"), 0.0) >= 30:
        empty.update(
            {
                "temporal_momentum_score": _tier_score("mainstream_or_hotspot", 0.0, 0.0, 0.0),
                "temporal_validation_tier": "mainstream_or_hotspot",
                "temporal_validation_status": "mainstream_or_hotspot",
                "temporal_validation_reason": "候选提及量或来源覆盖已经较高，更像主流热点或观察范围概览，不宜仅凭增长判为早期关键核心候选。",
            }
        )
        if not dated_records:
            return empty

    if len(dated_records) < min_dated_evidence:
        core_quality = _safe_float(row.get("candidate_core_evidence_quality"), 0.0)
        source_count = _safe_float(row.get("source_count"), 0.0)
        if core_quality >= 7.5 and source_count <= 1:
            empty.update(
                {
                    "temporal_momentum_score": _tier_score("single_source_high_quality_watch", 0.0, 0.0, 0.0),
                    "temporal_validation_tier": "single_source_high_quality_watch",
                    "temporal_validation_status": "insufficient_date",
                    "temporal_validation_reason": "日期证据不足，但核心证据质量较高且来源较少，保留为单源高质量观察对象。",
                }
            )
        return empty

    dates = sorted(record["date"] for record in dated_records)
    first_seen = dates[0]
    last_seen = dates[-1]
    observation_start = first_seen
    observation_end = observation_start + pd.DateOffset(months=observation_months)
    validation_start = observation_end
    validation_end = validation_start + pd.DateOffset(months=validation_months)

    (
        mentions_observation,
        high_quality_observation,
        source_count_observation,
        org_count_observation,
        asset_count_observation,
    ) = _count_window(dated_records, observation_start, observation_end)
    (
        mentions_validation,
        high_quality_validation,
        source_count_validation,
        org_count_validation,
        asset_count_validation,
    ) = _count_window(dated_records, validation_start, validation_end)

    growth_rate = _growth_rate(mentions_validation, mentions_observation)
    source_growth_rate = _growth_rate(source_count_validation, source_count_observation)
    org_growth_rate = _growth_rate(org_count_validation, org_count_observation)
    high_quality_growth_rate = _growth_rate(high_quality_validation, high_quality_observation)
    years = max(validation_months / 12.0, 0.01)
    cagr = round(((mentions_validation + 1) / max(mentions_observation, 1)) ** (1 / years) - 1, 3)

    passed = (
        growth_rate >= growth_threshold
        or source_growth_rate >= source_growth_threshold
        or org_growth_rate >= org_growth_threshold
        or high_quality_growth_rate >= high_quality_growth_threshold
    )

    if _mainstream_hint(row):
        tier = "mainstream_or_hotspot"
        passed = False
        status = "mainstream_or_hotspot"
    elif passed and high_quality_growth_rate >= high_quality_growth_threshold:
        tier = "quality_rising"
        status = "passed"
    elif passed:
        tier = "rising_validated"
        status = "passed"
    elif mentions_validation > 0:
        tier = "stable_no_growth"
        status = "not_passed"
    else:
        tier = "declining_or_one_off"
        status = "not_passed"

    earlyness_years = round((pd.Timestamp.now().normalize() - first_seen).days / 365.25, 2)
    momentum_score = _tier_score(tier, growth_rate, source_growth_rate, org_growth_rate)

    reasons = [
        f"观察期证据={mentions_observation}，验证期证据={mentions_validation}",
        f"提及增长={growth_rate:.2f}",
        f"来源增长={source_growth_rate:.2f}",
        f"主体增长={org_growth_rate:.2f}",
        f"高质量证据增长={high_quality_growth_rate:.2f}",
        f"日期覆盖率={date_coverage_ratio:.2f}",
    ]
    if tier == "mainstream_or_hotspot":
        reasons.append("候选已呈现热点或观察范围概览特征，关键核心潜力需结合早期性降权。")
    elif passed:
        reasons.append("后续窗口存在增长或扩散，时间验证通过。")
    else:
        reasons.append("后续窗口增长不足，暂不判为持续成长信号。")

    return {
        "candidate_id": candidate_id,
        "candidate_name": candidate_name,
        "first_seen_date": _date_to_text(first_seen),
        "last_seen_date": _date_to_text(last_seen),
        "observation_window_start": _date_to_text(observation_start),
        "observation_window_end": _date_to_text(observation_end),
        "validation_window_start": _date_to_text(validation_start),
        "validation_window_end": _date_to_text(validation_end),
        "mentions_observation": mentions_observation,
        "mentions_validation": mentions_validation,
        "high_quality_mentions_observation": high_quality_observation,
        "high_quality_mentions_validation": high_quality_validation,
        "source_count_observation": source_count_observation,
        "source_count_validation": source_count_validation,
        "org_count_observation": org_count_observation,
        "org_count_validation": org_count_validation,
        "asset_count_observation": asset_count_observation,
        "asset_count_validation": asset_count_validation,
        "growth_rate": growth_rate,
        "source_growth_rate": source_growth_rate,
        "org_growth_rate": org_growth_rate,
        "high_quality_growth_rate": high_quality_growth_rate,
        "cagr": cagr,
        "temporal_momentum_score": momentum_score,
        "temporal_validation_passed": bool(passed),
        "temporal_validation_tier": tier,
        "temporal_validation_status": status,
        "earlyness_years": earlyness_years,
        "date_coverage_ratio": date_coverage_ratio,
        "temporal_validation_reason": "；".join(reasons),
    }


def build_temporal_validation_table(
    candidates_df: pd.DataFrame,
    events_df: pd.DataFrame | None = None,
    raw_data: pd.DataFrame | None = None,
    *,
    observation_months: int = 12,
    validation_months: int = 12,
    growth_threshold: float = 0.5,
    source_growth_threshold: float = 0.5,
    org_growth_threshold: float = 0.5,
    high_quality_growth_threshold: float = 0.3,
    min_dated_evidence: int = 2,
) -> pd.DataFrame:
    """Build candidate-level temporal validation rows."""
    if candidates_df is None or candidates_df.empty:
        return pd.DataFrame(columns=TEMPORAL_VALIDATION_COLUMNS)

    lookup = _build_record_lookup(
        events_df if isinstance(events_df, pd.DataFrame) else pd.DataFrame(),
        raw_data if isinstance(raw_data, pd.DataFrame) else pd.DataFrame(),
    )
    grouped_rows: Dict[str, List[pd.Series]] = {}
    for index, row in candidates_df.reset_index(drop=True).iterrows():
        grouped_rows.setdefault(_temporal_input_key(row, index), []).append(row)

    rows = [
        _temporal_row(
            _combine_group_rows(group_rows),
            index,
            lookup,
            observation_months,
            validation_months,
            growth_threshold,
            source_growth_threshold,
            org_growth_threshold,
            high_quality_growth_threshold,
            min_dated_evidence,
        )
        for index, group_rows in enumerate(grouped_rows.values())
    ]
    return pd.DataFrame(rows, columns=TEMPORAL_VALIDATION_COLUMNS)


def merge_temporal_validation_into_candidates(
    candidates_df: pd.DataFrame,
    temporal_validation_df: pd.DataFrame,
) -> pd.DataFrame:
    """Attach temporal validation fields to candidate rows."""
    if candidates_df is None or candidates_df.empty:
        return candidates_df.copy() if isinstance(candidates_df, pd.DataFrame) else pd.DataFrame()

    merged = candidates_df.reset_index(drop=True).copy()
    merged["candidate_id"] = [
        _safe_text(row.get("candidate_id")) or _candidate_id(row, index)
        for index, row in merged.iterrows()
    ]

    if temporal_validation_df is None or temporal_validation_df.empty:
        for column in TEMPORAL_CANDIDATE_COLUMNS:
            if column not in merged.columns:
                merged[column] = False if column == "temporal_validation_passed" else 0.0 if column in {
                    "mentions_observation",
                    "mentions_validation",
                    "growth_rate",
                    "source_growth_rate",
                    "org_growth_rate",
                    "high_quality_growth_rate",
                    "cagr",
                    "temporal_momentum_score",
                    "earlyness_years",
                    "date_coverage_ratio",
                } else ""
        return merged

    mapping = temporal_validation_df.reset_index(drop=True).copy()
    temporal_columns = [
        column for column in TEMPORAL_VALIDATION_COLUMNS if column not in {"candidate_id", "candidate_name"}
    ]
    if len(mapping) == len(merged):
        joined = merged.drop(
            columns=[
                column
                for column in TEMPORAL_CANDIDATE_COLUMNS
                if column in merged.columns and column != "candidate_id"
            ]
        )
        for column in temporal_columns:
            if column in mapping.columns:
                joined[column] = mapping[column].values
    else:
        mapping["_temporal_join_id"] = mapping["candidate_id"].astype(str)
        mapping = mapping.drop_duplicates(subset=["_temporal_join_id"], keep="first")
        working = merged.drop(
            columns=[
                column
                for column in TEMPORAL_CANDIDATE_COLUMNS
                if column in merged.columns and column != "candidate_id"
            ]
        )
        working["_temporal_join_id"] = working["candidate_id"].astype(str)
        columns = ["_temporal_join_id"] + temporal_columns
        joined = working.merge(mapping[columns], on="_temporal_join_id", how="left")
        joined = joined.drop(columns=["_temporal_join_id"])

    for column in TEMPORAL_CANDIDATE_COLUMNS:
        if column not in joined.columns:
            joined[column] = False if column == "temporal_validation_passed" else ""
    joined["temporal_validation_passed"] = joined["temporal_validation_passed"].map(_safe_bool)
    for column in [
        "mentions_observation",
        "mentions_validation",
        "high_quality_mentions_observation",
        "high_quality_mentions_validation",
        "source_count_observation",
        "source_count_validation",
        "org_count_observation",
        "org_count_validation",
        "asset_count_observation",
        "asset_count_validation",
        "growth_rate",
        "source_growth_rate",
        "org_growth_rate",
        "high_quality_growth_rate",
        "cagr",
        "temporal_momentum_score",
        "earlyness_years",
        "date_coverage_ratio",
    ]:
        if column in joined.columns:
            joined[column] = pd.to_numeric(joined[column], errors="coerce").fillna(0.0)
    return joined


__all__ = [
    "TEMPORAL_VALIDATION_COLUMNS",
    "TEMPORAL_CANDIDATE_COLUMNS",
    "build_temporal_validation_table",
    "merge_temporal_validation_into_candidates",
]
