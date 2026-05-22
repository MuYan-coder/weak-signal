"""Key-core technology potential scoring for v2.7."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ..utils.config import Config


KEY_CORE_SCORE_COLUMNS = [
    "candidate_id",
    "candidate_name",
    "final_research_object_name",
    "tech_chain_node_id",
    "tech_chain_name",
    "mapping_relation",
    "mapping_confidence",
    "bottleneck_level",
    "strategic_importance_level",
    "weak_signal_score",
    "key_core_score",
    "key_core_tier",
    "weak_signal_component",
    "growth_validation_component",
    "tech_chain_bottleneck_component",
    "strategic_importance_component",
    "evidence_confidence_component",
    "asset_support_component",
    "reverse_validation_component",
    "quality_gate_passed",
    "mapping_gate_passed",
    "temporal_gate_passed",
    "object_gate_passed",
    "key_core_gate_passed",
    "key_core_reason",
    "key_core_risk",
    "recommended_action",
    "top_evidence_ids",
]

KEY_CORE_CANDIDATE_COLUMNS = [
    "key_core_rank",
    "candidate_id",
    "candidate_name",
    "final_research_object_name",
    "key_core_score",
    "key_core_tier",
    "tech_chain_name",
    "bottleneck_level",
    "strategic_importance_level",
    "temporal_validation_tier",
    "candidate_core_evidence_quality",
    "source_count",
    "cluster_evidence_count",
    "key_core_reason",
    "key_core_risk",
    "recommended_action",
    "top_evidence_ids",
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


def _clamp(value: float, lower: float = 0.0, upper: float = 100.0) -> float:
    return round(max(lower, min(float(value), upper)), 2)


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


def _load_assets(base_dir: Optional[Path | str] = None) -> pd.DataFrame:
    path = Path(base_dir) if base_dir else Config.DATA_DIR / "tech_chain"
    asset_path = path / "technology_assets.csv"
    if not asset_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(asset_path, encoding="utf-8-sig")
    except Exception:
        return pd.DataFrame()


def _asset_index(assets_df: Optional[pd.DataFrame]) -> Dict[str, List[Dict[str, Any]]]:
    if assets_df is None or assets_df.empty:
        return {}
    index: Dict[str, List[Dict[str, Any]]] = {}
    for _, row in assets_df.iterrows():
        record = row.to_dict()
        for field in ["belongs_to_tech_node_id", "asset_name", "related_product"]:
            key = _safe_text(record.get(field))
            if key:
                index.setdefault(key.lower(), []).append(record)
    return index


def _matching_assets(row: pd.Series, assets: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    matches: List[Dict[str, Any]] = []
    node_id = _safe_text(row.get("tech_chain_node_id"))
    if node_id:
        matches.extend(assets.get(node_id.lower(), []))
    name = _candidate_name(row)
    if name:
        name_lower = name.lower()
        for key, records in assets.items():
            if name_lower and (name_lower in key or key in name_lower):
                matches.extend(records)
    seen = set()
    deduped = []
    for record in matches:
        key = (_safe_text(record.get("asset_id")), _safe_text(record.get("asset_name")))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    return deduped


def _weak_signal_component(row: pd.Series) -> float:
    score = _safe_float(row.get("weak_signal_score"), _safe_float(row.get("score"), 0.0))
    component = min(score / 12.0 * 100.0, 100.0)
    signal_type = _safe_text(row.get("signal_type"))
    if signal_type == "weak_signal":
        component += 10
    elif signal_type == "near_strong":
        component += 5
    elif signal_type == "hotspot":
        component -= 15
    elif signal_type == "scope_overview":
        component -= 40
    if _safe_bool(row.get("weak_signal_focus_candidate")):
        component += 5
    if _safe_float(row.get("source_count")) >= 2:
        component += 3
    return _clamp(component)


def _growth_validation_component(row: pd.Series) -> float:
    tier = _safe_text(row.get("temporal_validation_tier")) or "insufficient_date"
    base = {
        "rising_validated": 92,
        "quality_rising": 82,
        "single_source_high_quality_watch": 62,
        "stable_no_growth": 48,
        "insufficient_date": 40,
        "mainstream_or_hotspot": 50,
        "declining_or_one_off": 25,
    }.get(tier, 40)
    growth_lift = max(
        _safe_float(row.get("growth_rate")),
        _safe_float(row.get("source_growth_rate")),
        _safe_float(row.get("org_growth_rate")),
        _safe_float(row.get("high_quality_growth_rate")),
        0.0,
    ) * 10
    return _clamp(base + growth_lift)


def _tech_chain_bottleneck_component(row: pd.Series) -> float:
    relation = _safe_text(row.get("mapping_relation"))
    if relation == "no_match":
        return 0.0
    base = {
        "high": 100,
        "medium": 65,
        "low": 30,
        "unknown": 45,
    }.get(_safe_text(row.get("bottleneck_level")).lower() or "unknown", 45)
    method = _safe_text(row.get("mapping_method"))
    confidence = _safe_float(row.get("tech_chain_mapping_confidence"), _safe_float(row.get("mapping_confidence"), 0.0))
    if relation in {"exact_match", "close_match", "manual_override"} or method in {"exact_match", "manual_override"}:
        base += 5
    if relation in {"related_match"} or method == "semantic_match":
        base -= 10
    if relation == "broader_match":
        base -= 20
    if confidence < 0.6:
        base -= 20
    return _clamp(base)


def _strategic_importance_component(row: pd.Series) -> float:
    base = {
        "high": 100,
        "medium": 65,
        "low": 30,
        "unknown": 45,
    }.get(_safe_text(row.get("strategic_importance_level")).lower() or "unknown", 45)
    context = " ".join(
        _safe_text(row.get(field))
        for field in ["mapping_reason", "explanation", "tech_chain_name", "mechanism_core", "relation_summary"]
    )
    policy_terms = ["自主", "可控", "安全", "卡点", "瓶颈", "核心", "platform", "chip", "control"]
    if any(term.lower() in context.lower() for term in policy_terms):
        base += 5
    return _clamp(base)


def _evidence_confidence_component(row: pd.Series) -> float:
    base = _safe_float(row.get("candidate_core_evidence_quality"), _safe_float(row.get("candidate_evidence_quality"), 0.0)) * 10.0
    if _safe_float(row.get("source_count")) >= 3:
        base += 5
    if _safe_float(row.get("high_quality_evidence_count")) >= 3:
        base += 5
    if _safe_float(row.get("low_quality_evidence_ratio")) > 0.3:
        base -= 20
    if _safe_text(row.get("source_semantic_consistency")) == "low":
        base -= 15
    if _safe_text(row.get("reverse_validation_status")) in {"mixed_or_unclear", "only_upper_topic_in_source"}:
        base -= 20
    if _safe_bool(row.get("event_doc_semantic_validated")):
        base += 5
    return _clamp(base)


def _asset_support_component(row: pd.Series, asset_matches: List[Dict[str, Any]]) -> float:
    if asset_matches:
        confidence_values = [_safe_float(record.get("asset_confidence"), 0.0) for record in asset_matches]
        if any(value >= 0.8 for value in confidence_values):
            return 95.0
        return 82.0
    if _safe_bool(row.get("patent_validated")) or _safe_float(row.get("patent_count")) > 0:
        return 72.0
    if _safe_bool(row.get("literature_validated")):
        return 55.0
    if _safe_float(row.get("candidate_core_evidence_quality")) >= 7.5:
        return 42.0
    return 25.0


def _reverse_validation_component(row: pd.Series) -> float:
    status = _safe_text(row.get("reverse_validation_status"))
    consistency = _safe_text(row.get("source_semantic_consistency"))
    if status == "natural_topic_in_source" or consistency == "high":
        return 90.0
    if consistency == "medium":
        return 70.0
    if status in {"mixed_or_unclear", "only_upper_topic_in_source"} or consistency == "low":
        return 25.0
    return 50.0


def _top_evidence_ids(row: pd.Series, limit: int = 5) -> str:
    ids: List[str] = []
    for item in _safe_list(row.get("evidence_items")):
        if isinstance(item, dict):
            text = _safe_text(item.get("id") or item.get("source_id") or item.get("title"))
            if text:
                ids.append(text)
    if not ids:
        ids.extend(_safe_text(item) for item in _safe_list(row.get("mention_ids")) if _safe_text(item))
    return "|".join(ids[:limit])


def _manual_overrides(manual_review_df: Optional[pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
    overrides: Dict[str, Dict[str, Any]] = {}
    if manual_review_df is None or manual_review_df.empty:
        return overrides
    for _, row in manual_review_df.iterrows():
        record = row.to_dict()
        for field in ["candidate_id", "candidate_name"]:
            key = _safe_text(record.get(field))
            if key:
                overrides[key.lower()] = record
    return overrides


def _score_row(
    row: pd.Series,
    index: int,
    assets: Dict[str, List[Dict[str, Any]]],
    overrides: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    candidate_id = _candidate_id(row, index)
    candidate_name = _candidate_name(row)
    asset_matches = _matching_assets(row, assets)

    weak_component = _weak_signal_component(row)
    growth_component = _growth_validation_component(row)
    bottleneck_component = _tech_chain_bottleneck_component(row)
    strategic_component = _strategic_importance_component(row)
    evidence_component = _evidence_confidence_component(row)
    asset_component = _asset_support_component(row, asset_matches)
    reverse_component = _reverse_validation_component(row)

    key_core_score = _clamp(
        weak_component * 0.20
        + growth_component * 0.22
        + bottleneck_component * 0.20
        + strategic_component * 0.15
        + evidence_component * 0.13
        + asset_component * 0.10
    )

    relation = _safe_text(row.get("mapping_relation"))
    mapping_confidence = _safe_float(row.get("tech_chain_mapping_confidence"), _safe_float(row.get("mapping_confidence"), 0.0))
    temporal_tier = _safe_text(row.get("temporal_validation_tier"))
    reverse_status = _safe_text(row.get("reverse_validation_status"))
    quality_gate = _safe_float(row.get("candidate_core_evidence_quality"), _safe_float(row.get("candidate_evidence_quality"), 0.0)) >= 6.0
    mapping_gate = relation != "no_match" and mapping_confidence >= 0.6 and relation != "broader_match"
    temporal_gate = temporal_tier in {"rising_validated", "quality_rising"}
    object_gate = (
        _safe_text(row.get("topic_granularity")) != "generic_or_failed"
        and _safe_text(row.get("candidate_stage")) != "scope_overview"
        and not _safe_bool(row.get("is_observation_scope"))
        and bool(candidate_name)
        and reverse_status not in {"mixed_or_unclear", "only_upper_topic_in_source"}
    )

    override = overrides.get(candidate_id.lower()) or overrides.get(candidate_name.lower()) or {}
    manual_tier = _safe_text(override.get("manual_key_core_tier"))
    score_adjustment = _safe_float(override.get("manual_score_adjustment"), 0.0)
    if score_adjustment:
        key_core_score = _clamp(key_core_score + score_adjustment)

    gate_passed = bool(quality_gate and mapping_gate and temporal_gate and object_gate)
    if manual_tier:
        tier = manual_tier
    elif key_core_score >= 80 and gate_passed:
        tier = "core_key_candidate"
    elif key_core_score >= 65 and quality_gate and mapping_gate and object_gate:
        tier = "strong_key_potential"
    elif key_core_score >= 50 or temporal_tier in {"single_source_high_quality_watch", "insufficient_date"}:
        tier = "watchlist_key_potential"
    elif _safe_float(row.get("weak_signal_score")) >= 6 or _safe_text(row.get("signal_type")) == "weak_signal":
        tier = "weak_signal_only"
    elif not quality_gate or not mapping_gate or not object_gate:
        tier = "insufficient_evidence"
    else:
        tier = "not_key_core_candidate"

    risks = []
    if not quality_gate:
        risks.append("quality_gate_fail")
    if not mapping_gate:
        risks.append("mapping_gate_fail")
    if not temporal_gate:
        risks.append(f"temporal:{temporal_tier or 'unknown'}")
    if not object_gate:
        risks.append("object_gate_fail")
    if relation in {"related_match", "broader_match"} or _safe_text(row.get("mapping_method")) == "semantic_match":
        risks.append("mapping_review_needed")
    if manual_tier:
        risks.append("manual_override_applied")

    if tier == "core_key_candidate":
        action = "纳入关键核心潜力候选，优先人工复核并持续跟踪。"
    elif tier == "strong_key_potential":
        action = "纳入强候选，补充时间序列和资产证据。"
    elif tier == "watchlist_key_potential":
        action = "进入观察清单，重点补充日期、来源扩散和专家判断。"
    elif tier == "weak_signal_only":
        action = "保留为弱信号，暂不提升为关键核心候选。"
    else:
        action = "暂不作为关键核心候选，优先修正证据或映射问题。"

    reason_parts = [
        f"弱信号分项={weak_component:.1f}",
        f"时间验证分项={growth_component:.1f}({temporal_tier or 'unknown'})",
        f"技术链卡点分项={bottleneck_component:.1f}({ _safe_text(row.get('bottleneck_level')) or 'unknown'})",
        f"战略重要性分项={strategic_component:.1f}({ _safe_text(row.get('strategic_importance_level')) or 'unknown'})",
        f"证据可信分项={evidence_component:.1f}",
        f"资产支撑分项={asset_component:.1f}",
    ]
    if override and _safe_text(override.get("manual_reason")):
        reason_parts.append(f"人工修正={_safe_text(override.get('manual_reason'))}")

    return {
        "candidate_id": candidate_id,
        "candidate_name": candidate_name,
        "final_research_object_name": _safe_text(row.get("final_research_object_name")) or candidate_name,
        "tech_chain_node_id": _safe_text(row.get("tech_chain_node_id")),
        "tech_chain_name": _safe_text(row.get("tech_chain_name")),
        "mapping_relation": relation,
        "mapping_confidence": mapping_confidence,
        "bottleneck_level": _safe_text(row.get("bottleneck_level")) or "unknown",
        "strategic_importance_level": _safe_text(row.get("strategic_importance_level")) or "unknown",
        "weak_signal_score": _safe_float(row.get("weak_signal_score"), _safe_float(row.get("score"), 0.0)),
        "key_core_score": key_core_score,
        "key_core_tier": tier,
        "weak_signal_component": weak_component,
        "growth_validation_component": growth_component,
        "tech_chain_bottleneck_component": bottleneck_component,
        "strategic_importance_component": strategic_component,
        "evidence_confidence_component": evidence_component,
        "asset_support_component": asset_component,
        "reverse_validation_component": reverse_component,
        "quality_gate_passed": bool(quality_gate),
        "mapping_gate_passed": bool(mapping_gate),
        "temporal_gate_passed": bool(temporal_gate),
        "object_gate_passed": bool(object_gate),
        "key_core_gate_passed": bool(gate_passed),
        "key_core_reason": "；".join(reason_parts),
        "key_core_risk": "; ".join(risks),
        "recommended_action": action,
        "top_evidence_ids": _top_evidence_ids(row),
    }


def _merge_temporal_if_needed(
    candidates_df: pd.DataFrame,
    temporal_validation_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    if temporal_validation_df is None or temporal_validation_df.empty or candidates_df is None or candidates_df.empty:
        return candidates_df.copy() if isinstance(candidates_df, pd.DataFrame) else pd.DataFrame()
    if "temporal_validation_tier" in candidates_df.columns:
        working = candidates_df.copy().reset_index(drop=True)
        working["candidate_id"] = [
            _safe_text(row.get("candidate_id")) or _candidate_id(row, index)
            for index, row in working.iterrows()
        ]
        return working
    working = candidates_df.copy().reset_index(drop=True)
    working["candidate_id"] = [
        _safe_text(row.get("candidate_id")) or _candidate_id(row, index)
        for index, row in working.iterrows()
    ]
    temporal = temporal_validation_df.copy()
    columns = [column for column in temporal.columns if column not in {"candidate_id", "candidate_name"}]
    if len(temporal) == len(working):
        for column in columns:
            working[column] = temporal.reset_index(drop=True)[column].values
        return working
    temporal["_join_id"] = temporal["candidate_id"].astype(str)
    temporal = temporal.drop_duplicates(subset=["_join_id"], keep="first")
    working["_join_id"] = working["candidate_id"].astype(str)
    columns = [
        "_join_id",
    ] + [
        column
        for column in temporal.columns
        if column not in {"candidate_id", "candidate_name", "_join_id"}
    ]
    return working.merge(temporal[columns], on="_join_id", how="left").drop(columns=["_join_id"])


def score_key_core_candidates(
    candidates_df: pd.DataFrame,
    temporal_validation_df: Optional[pd.DataFrame] = None,
    assets_df: Optional[pd.DataFrame] = None,
    manual_review_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Score candidates for key-core technology potential."""
    if candidates_df is None or candidates_df.empty:
        return pd.DataFrame(columns=KEY_CORE_SCORE_COLUMNS)
    working = _merge_temporal_if_needed(candidates_df, temporal_validation_df)
    assets = _asset_index(assets_df if assets_df is not None else _load_assets())
    overrides = _manual_overrides(manual_review_df)
    rows = [
        _score_row(row, index, assets, overrides)
        for index, row in working.reset_index(drop=True).iterrows()
    ]
    scored = working.reset_index(drop=True).copy()
    score_df = pd.DataFrame(rows, columns=KEY_CORE_SCORE_COLUMNS)
    for column in KEY_CORE_SCORE_COLUMNS:
        if column in scored.columns:
            scored = scored.drop(columns=[column])
    return pd.concat([scored, score_df], axis=1)


def build_key_core_candidate_table(key_core_scored_df: pd.DataFrame, top_k: int = 20) -> pd.DataFrame:
    """Build the top key-core potential candidate table."""
    if key_core_scored_df is None or key_core_scored_df.empty:
        return pd.DataFrame(columns=KEY_CORE_CANDIDATE_COLUMNS)
    df = key_core_scored_df.copy()
    if "key_core_tier" not in df.columns:
        return pd.DataFrame(columns=KEY_CORE_CANDIDATE_COLUMNS)
    preferred_tiers = {"core_key_candidate", "strong_key_potential", "watchlist_key_potential"}
    subset = df[df["key_core_tier"].astype(str).isin(preferred_tiers)].copy()
    if subset.empty:
        subset = df.copy()
    tier_order = {
        "core_key_candidate": 0,
        "strong_key_potential": 1,
        "watchlist_key_potential": 2,
        "weak_signal_only": 3,
        "insufficient_evidence": 4,
        "not_key_core_candidate": 5,
    }
    subset["_tier_order"] = subset["key_core_tier"].map(lambda value: tier_order.get(_safe_text(value), 9))
    subset["_score"] = subset["key_core_score"].map(_safe_float)
    subset["_quality"] = subset.get("candidate_core_evidence_quality", 0).map(_safe_float)
    subset = subset.sort_values(
        by=["_tier_order", "_score", "_quality"],
        ascending=[True, False, False],
    )
    subset["_dedupe_key"] = subset.apply(
        lambda row: "::".join(
            _safe_text(row.get(column)).lower()
            for column in ["candidate_name", "final_research_object_name", "tech_chain_name"]
            if _safe_text(row.get(column))
        )
        or _safe_text(row.get("candidate_id")).lower(),
        axis=1,
    )
    subset = subset.drop_duplicates(subset=["_dedupe_key"], keep="first")
    subset = subset.head(top_k).copy()
    subset["key_core_rank"] = range(1, len(subset) + 1)
    for column in KEY_CORE_CANDIDATE_COLUMNS:
        if column not in subset.columns:
            subset[column] = ""
    return subset[KEY_CORE_CANDIDATE_COLUMNS].reset_index(drop=True)


__all__ = [
    "KEY_CORE_SCORE_COLUMNS",
    "KEY_CORE_CANDIDATE_COLUMNS",
    "score_key_core_candidates",
    "build_key_core_candidate_table",
]
