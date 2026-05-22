from __future__ import annotations

from typing import Tuple

import pandas as pd


FINAL_SHORTLIST_COLUMNS = [
    "shortlist_rank",
    "shortlist_tier",
    "shortlist_reason",
    "display_candidate_name",
    "final_research_object_name",
    "family_id",
    "family_priority",
    "final_research_bucket",
    "signal_type",
    "weak_signal_score",
    "quality_adjusted_rank_score",
    "candidate_evidence_quality",
    "candidate_core_evidence_quality",
    "low_quality_evidence_ratio",
    "quality_risk_flag",
    "tech_chain_node_id",
    "tech_chain_name",
    "mapping_relation",
    "mapping_confidence",
    "mapping_method",
    "bottleneck_level",
    "strategic_importance_level",
    "tech_chain_mapping_risk",
    "hotspot_score",
    "source_count",
    "cluster_evidence_count",
    "reverse_validation_status",
    "source_semantic_consistency",
    "object_semantic_alignment",
    "release_alignment_risk",
    "risk_flags",
]

DEDUP_MAP_COLUMNS = [
    "display_candidate_name",
    "kept_display_candidate_name",
    "dedup_key",
    "dedup_reason",
    "final_research_bucket",
    "weak_signal_score",
    "source_count",
    "cluster_evidence_count",
]


def _safe_text(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null", "nat"} else text


def _safe_float(value, default=0.0) -> float:
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return default


def _shortlist_tier(row: pd.Series) -> Tuple[str, str]:
    bucket = _safe_text(row.get("final_research_bucket"))
    signal_type = _safe_text(row.get("signal_type"))
    reverse_status = _safe_text(row.get("reverse_validation_status"))
    consistency = _safe_text(row.get("source_semantic_consistency"))

    if bucket == "core_result":
        return "primary", "核心研究结果，优先进入报告主线"
    if bucket == "candidate_result":
        return "candidate", "候选研究结果，可进入人工复核或后续跟踪"
    if bucket == "family_backbone":
        return "backbone", "对象族主干，适合作为报告背景或承接结构"
    if bucket == "observation_pool":
        return "watch", "观察池对象，保留为后续监测线索"
    if signal_type == "weak_signal" and reverse_status == "natural_topic_in_source":
        return "candidate", "弱信号且原始证据中存在自然小主题表达"
    if signal_type == "weak_signal" and consistency in {"high", "medium"}:
        return "watch", "弱信号语义一致性尚可，但仍需更多验证"
    return "internal", "未进入最终短名单主层"


def _risk_flags(row: pd.Series) -> str:
    flags = []
    if _safe_text(row.get("release_alignment_risk")) in {"high", "medium"}:
        flags.append(f"release_alignment:{_safe_text(row.get('release_alignment_risk'))}")
    if _safe_text(row.get("object_semantic_alignment")) in {"topic_close_object_mismatch", "evidence_insufficient"}:
        flags.append(f"object_alignment:{_safe_text(row.get('object_semantic_alignment'))}")
    if _safe_text(row.get("source_semantic_consistency")) == "low":
        flags.append("semantic_consistency:low")
    if bool(row.get("should_go_to_failure_case_section", False)):
        flags.append("failure_case_candidate")
    if _safe_text(row.get("quality_risk_flag")) in {"high_quality_risk", "quality_unknown"}:
        flags.append(f"evidence_quality:{_safe_text(row.get('quality_risk_flag'))}")
    if _safe_text(row.get("mapping_relation")) == "no_match":
        flags.append("tech_chain:no_match")
    elif _safe_text(row.get("tech_chain_mapping_risk")) in {"low_confidence_mapping", "broad_chain_mapping", "semantic_mapping_review"}:
        flags.append(f"tech_chain:{_safe_text(row.get('tech_chain_mapping_risk'))}")
    return "; ".join(flags)


def _dedup_key(row: pd.Series) -> str:
    family_id = _safe_text(row.get("family_id"))
    if family_id:
        return f"family:{family_id}"
    name = _safe_text(row.get("final_research_object_name")) or _safe_text(row.get("display_candidate_name"))
    return f"name:{name.lower()}"


def _empty_shortlist() -> pd.DataFrame:
    return pd.DataFrame(columns=FINAL_SHORTLIST_COLUMNS)


def _empty_dedup_map() -> pd.DataFrame:
    return pd.DataFrame(columns=DEDUP_MAP_COLUMNS)


def build_final_shortlist(scored_df: pd.DataFrame, top_k: int = 20) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if scored_df is None or scored_df.empty:
        return _empty_shortlist(), _empty_dedup_map()

    df = scored_df.copy()
    for column in ["final_research_bucket", "signal_type", "display_candidate_name"]:
        if column not in df.columns:
            df[column] = ""

    keep_buckets = {"core_result", "candidate_result", "family_backbone", "observation_pool"}
    keep_mask = df["final_research_bucket"].astype(str).isin(keep_buckets) | df["signal_type"].astype(str).isin(
        {"weak_signal", "hotspot"}
    )
    df = df[keep_mask].copy()
    if df.empty:
        return _empty_shortlist(), _empty_dedup_map()

    df["final_research_object_name"] = df.apply(
        lambda row: _safe_text(row.get("final_research_object_name"))
        or _safe_text(row.get("topic_summary_name"))
        or _safe_text(row.get("display_candidate_name")),
        axis=1,
    )
    df["_dedup_key"] = df.apply(_dedup_key, axis=1)
    df["_tier_tuple"] = df.apply(_shortlist_tier, axis=1)
    df["shortlist_tier"] = df["_tier_tuple"].map(lambda item: item[0])
    df["shortlist_reason"] = df["_tier_tuple"].map(lambda item: item[1])
    df["risk_flags"] = df.apply(_risk_flags, axis=1)

    bucket_priority = {
        "core_result": 0,
        "candidate_result": 1,
        "family_backbone": 2,
        "observation_pool": 3,
    }
    tier_priority = {"primary": 0, "candidate": 1, "backbone": 2, "watch": 3, "internal": 4}
    df["_bucket_order"] = df["final_research_bucket"].map(lambda value: bucket_priority.get(_safe_text(value), 9))
    df["_tier_order"] = df["shortlist_tier"].map(lambda value: tier_priority.get(_safe_text(value), 9))
    df["_weak_score"] = df.get("weak_signal_score", 0).map(_safe_float)
    df["_quality_adjusted_score"] = df.get("quality_adjusted_rank_score", df.get("weak_signal_score", 0)).map(_safe_float)
    df["_candidate_quality"] = df.get("candidate_evidence_quality", 0).map(_safe_float)
    df["_mapping_confidence"] = df.get("mapping_confidence", 0).map(_safe_float)
    df["_hotspot_score"] = df.get("hotspot_score", 0).map(_safe_float)
    df["_source_count"] = df.get("source_count", 0).map(_safe_float)
    df["_evidence_count"] = df.get("cluster_evidence_count", 0).map(_safe_float)

    df = df.sort_values(
        by=[
            "_tier_order",
            "_bucket_order",
            "_weak_score",
            "_quality_adjusted_score",
            "_candidate_quality",
            "_mapping_confidence",
            "_hotspot_score",
            "_source_count",
            "_evidence_count",
        ],
        ascending=[True, True, False, False, False, False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)

    kept = df.drop_duplicates(subset=["_dedup_key"], keep="first").head(top_k).copy()
    kept_names = dict(zip(kept["_dedup_key"], kept["display_candidate_name"]))
    kept["shortlist_rank"] = range(1, len(kept) + 1)

    dedup_rows = []
    for _, row in df.iterrows():
        key = row.get("_dedup_key", "")
        kept_name = kept_names.get(key, "")
        if not kept_name:
            continue
        dedup_rows.append(
            {
                "display_candidate_name": row.get("display_candidate_name", ""),
                "kept_display_candidate_name": kept_name,
                "dedup_key": key,
                "dedup_reason": "kept" if row.get("display_candidate_name", "") == kept_name else "same_family_or_name",
                "final_research_bucket": row.get("final_research_bucket", ""),
                "weak_signal_score": row.get("weak_signal_score", 0),
                "source_count": row.get("source_count", 0),
                "cluster_evidence_count": row.get("cluster_evidence_count", 0),
            }
        )

    for column in FINAL_SHORTLIST_COLUMNS:
        if column not in kept.columns:
            kept[column] = ""

    shortlist = kept[FINAL_SHORTLIST_COLUMNS].reset_index(drop=True)
    dedup_map = pd.DataFrame(dedup_rows)
    for column in DEDUP_MAP_COLUMNS:
        if column not in dedup_map.columns:
            dedup_map[column] = ""
    return shortlist, dedup_map[DEDUP_MAP_COLUMNS].reset_index(drop=True)


__all__ = ["build_final_shortlist"]
