"""Event quality scoring for weak-signal evidence.

This module assigns a lightweight, explainable quality score to each extracted
event and propagates the score to candidate-level evidence. Low-quality events
are never removed here; downstream stages can use the scores as ranking or
report-evidence weights.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any, Dict, List, Tuple

import pandas as pd


EVENT_QUALITY_COLUMNS = [
    "event_id",
    "source_id",
    "source_type",
    "title",
    "event_quality_score",
    "event_completeness_score",
    "tech_relevance_score",
    "traceability_score",
    "source_reliability_score",
    "non_marketing_score",
    "event_quality_tier",
    "event_quality_reason",
]

EVENT_QUALITY_SCORE_COLUMNS = [
    "event_quality_score",
    "event_completeness_score",
    "tech_relevance_score",
    "traceability_score",
    "source_reliability_score",
    "non_marketing_score",
    "event_quality_tier",
    "event_quality_reason",
]

CANDIDATE_QUALITY_COLUMNS = [
    "candidate_evidence_quality",
    "candidate_core_evidence_quality",
    "low_quality_evidence_ratio",
    "high_quality_evidence_count",
    "quality_risk_flag",
    "quality_adjusted_rank_score",
    "candidate_evidence_quality_reason",
]

UNKNOWN_VALUES = {"", "nan", "nat", "none", "null", "unknown", "未知", "无", "n/a"}

SOURCE_TYPE_NORMALIZATION = {
    "专利": "patent",
    "patent": "patent",
    "文献": "paper",
    "literature": "paper",
    "paper": "paper",
    "论文": "paper",
    "研报": "report",
    "行业研报": "report",
    "report": "report",
    "资讯": "news",
    "新闻": "news",
    "news": "news",
    "policy": "policy",
    "政策": "policy",
}

SOURCE_RELIABILITY_SCORES = {
    "patent": 8.5,
    "paper": 8.0,
    "policy": 8.5,
    "report": 7.0,
    "news": 5.5,
    "unknown": 4.0,
}

TECH_KEYWORDS = {
    "algorithm", "model", "system", "method", "framework", "architecture",
    "sensor", "control", "planning", "training", "simulation", "learning",
    "network", "robot", "robots", "robotics", "slam", "lidar", "camera",
    "imu", "multimodal", "perception", "navigation", "manipulation",
    "算法", "模型", "系统", "方法", "框架", "架构", "技术", "工艺",
    "材料", "装置", "设备", "传感", "控制", "规划", "训练", "仿真",
    "学习", "网络", "机器人", "感知", "导航", "操作", "定位", "建图",
    "多模态", "激光雷达", "摄像头", "强化学习", "世界模型", "具身智能",
}

MARKETING_PATTERNS = [
    "全球领先", "国际领先", "国内首个", "重磅发布", "颠覆", "革命性",
    "赋能", "引爆", "震撼", "史诗级", "划时代", "遥遥领先", "爆款",
    "全面升级", "新质生产力", "突破性进展", "world-leading", "leading",
    "revolutionary", "game-changing", "disruptive", "breakthrough",
]


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


def _normalize_source_type(value: Any) -> str:
    text = _safe_text(value)
    if not text:
        return "unknown"
    return SOURCE_TYPE_NORMALIZATION.get(text, SOURCE_TYPE_NORMALIZATION.get(text.lower(), text.lower()))


def _safe_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple) or isinstance(value, set):
        return list(value)
    if value is None:
        return []
    try:
        if pd.isna(value):
            return []
    except Exception:
        pass
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in UNKNOWN_VALUES:
            return []
        if (text.startswith("[") and text.endswith("]")) or (text.startswith("{") and text.endswith("}")):
            for parser in (json.loads, ast.literal_eval):
                try:
                    parsed = parser(text)
                    if isinstance(parsed, list):
                        return parsed
                    if isinstance(parsed, dict):
                        return [parsed]
                except Exception:
                    pass
        return [part.strip() for part in re.split(r"[;,，、/|]", text) if part.strip()]
    return [value]


def _safe_items(value: Any) -> List[Dict[str, Any]]:
    items = _safe_list(value)
    return [item for item in items if isinstance(item, dict)]


def _has_value(value: Any) -> bool:
    if isinstance(value, list):
        return any(_has_value(item) for item in value)
    return bool(_safe_text(value))


def _raw_index(raw_data: pd.DataFrame | None) -> Dict[str, Dict[str, Any]]:
    if raw_data is None or raw_data.empty or "id" not in raw_data.columns:
        return {}
    records = raw_data.drop_duplicates(subset="id", keep="last").to_dict(orient="records")
    return {str(record.get("id", "")).strip(): record for record in records if str(record.get("id", "")).strip()}


def _event_id(row: pd.Series, index: int) -> str:
    event_id = _safe_text(row.get("id"))
    return event_id or f"event_row_{index}"


def _combined_text(event: pd.Series, raw_record: Dict[str, Any]) -> str:
    parts = [
        event.get("title", ""),
        raw_record.get("title", ""),
        raw_record.get("text", ""),
        event.get("subject", ""),
        event.get("action", ""),
        event.get("scene", ""),
        " ".join(str(item) for item in _safe_list(event.get("technology", []))),
        " ".join(str(item) for item in _safe_list(event.get("observation_scopes", []))),
        " ".join(str(item) for item in _safe_list(event.get("weak_signal_reasons", []))),
    ]
    return " ".join(_safe_text(part) for part in parts if _safe_text(part))


def _score_completeness(event: pd.Series, raw_record: Dict[str, Any]) -> Tuple[float, str]:
    checks = [
        ("主体", event.get("subject")),
        ("行为", event.get("action")),
        ("技术", event.get("technology")),
        ("场景", event.get("scene")),
        ("时间", event.get("time") or raw_record.get("date")),
    ]
    present = [label for label, value in checks if _has_value(value)]
    missing = [label for label, _ in checks if label not in present]
    score = round(len(present) / len(checks) * 10.0, 2)
    reason = f"完整字段={','.join(present) or '无'}"
    if missing:
        reason += f"，缺失={','.join(missing)}"
    return score, reason


def _score_tech_relevance(event: pd.Series, raw_record: Dict[str, Any]) -> Tuple[float, str]:
    score = 0.0
    reasons = []
    technologies = [_safe_text(item) for item in _safe_list(event.get("technology", []))]
    technologies = [item for item in technologies if item]
    if technologies:
        score += 3.5
        reasons.append("技术字段明确")

    candidate_units = _safe_list(event.get("candidate_units", []))
    if candidate_units:
        score += 2.0
        reasons.append("存在候选技术单元")

    scopes = _safe_list(event.get("observation_scopes", []))
    if scopes:
        score += 1.2
        reasons.append("命中观察范围")

    text = _combined_text(event, raw_record).lower()
    keyword_hits = sorted({keyword for keyword in TECH_KEYWORDS if keyword.lower() in text})
    if keyword_hits:
        score += min(3.3, 0.7 + len(keyword_hits[:5]) * 0.5)
        reasons.append("技术关键词=" + "/".join(keyword_hits[:5]))

    if not reasons:
        reasons.append("未发现明确技术线索")
    return round(min(score, 10.0), 2), "；".join(reasons)


def _score_traceability(event: pd.Series, raw_record: Dict[str, Any], event_id: str) -> Tuple[float, str]:
    checks = [
        ("id", event_id),
        ("标题", event.get("title") or raw_record.get("title")),
        ("来源", event.get("source_type") or raw_record.get("source_type")),
        ("日期", event.get("time") or raw_record.get("date")),
        ("原文", raw_record.get("text")),
        ("链接", raw_record.get("url")),
    ]
    present = [label for label, value in checks if _has_value(value)]
    score = round(min(len(present) / 5.0 * 10.0, 10.0), 2)
    return score, "可追溯字段=" + ("/".join(present) or "无")


def _score_source_reliability(source_type: str, raw_record: Dict[str, Any]) -> Tuple[float, str]:
    normalized = _normalize_source_type(source_type or raw_record.get("source_type"))
    score = SOURCE_RELIABILITY_SCORES.get(normalized, 5.0)
    org = _safe_text(raw_record.get("org"))
    if org and org.lower() not in {"unknown", "未知"}:
        score = min(score + 0.5, 10.0)
    return round(score, 2), f"来源类型={normalized}" + (f"，机构={org}" if org else "")


def _score_non_marketing(event: pd.Series, raw_record: Dict[str, Any], source_type: str) -> Tuple[float, str]:
    text = _combined_text(event, raw_record)
    lowered = text.lower()
    hits = []
    for pattern in MARKETING_PATTERNS:
        if pattern.lower() in lowered:
            hits.append(pattern)

    normalized = _normalize_source_type(source_type or raw_record.get("source_type"))
    base = 9.0 if normalized in {"patent", "paper", "policy"} else 8.0
    penalty = min(len(hits) * 1.4, 5.5)
    score = max(2.0, base - penalty)
    reason = "宣传性表达较少" if not hits else "宣传性表达=" + "/".join(hits[:5])
    return round(score, 2), reason


def _quality_tier(score: float) -> str:
    if score >= 8.0:
        return "high"
    if score >= 6.0:
        return "medium"
    if score >= 4.0:
        return "low"
    return "very_low"


def _quality_row(event: pd.Series, raw_record: Dict[str, Any], index: int) -> Dict[str, Any]:
    event_id = _event_id(event, index)
    source_type = _normalize_source_type(event.get("source_type") or raw_record.get("source_type"))
    completeness, completeness_reason = _score_completeness(event, raw_record)
    tech_relevance, tech_reason = _score_tech_relevance(event, raw_record)
    traceability, trace_reason = _score_traceability(event, raw_record, event_id)
    reliability, reliability_reason = _score_source_reliability(source_type, raw_record)
    non_marketing, non_marketing_reason = _score_non_marketing(event, raw_record, source_type)
    score = round(
        completeness * 0.20
        + tech_relevance * 0.25
        + traceability * 0.20
        + reliability * 0.20
        + non_marketing * 0.15,
        2,
    )
    tier = _quality_tier(score)
    reason = (
        f"完整性{completeness:.1f}({completeness_reason})；"
        f"技术相关{tech_relevance:.1f}({tech_reason})；"
        f"追溯性{traceability:.1f}({trace_reason})；"
        f"来源可信{reliability:.1f}({reliability_reason})；"
        f"非营销{non_marketing:.1f}({non_marketing_reason})"
    )
    return {
        "event_id": event_id,
        "source_id": _safe_text(event.get("id")) or event_id,
        "source_type": source_type,
        "title": _safe_text(event.get("title") or raw_record.get("title")),
        "event_quality_score": score,
        "event_completeness_score": completeness,
        "tech_relevance_score": tech_relevance,
        "traceability_score": traceability,
        "source_reliability_score": reliability,
        "non_marketing_score": non_marketing,
        "event_quality_tier": tier,
        "event_quality_reason": reason,
    }


def build_event_quality_table(events_df: pd.DataFrame, raw_data: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build an event-level quality table.

    Every event row receives a score. Empty or malformed fields reduce the
    score, but never cause the event to be dropped.
    """
    if events_df is None or events_df.empty:
        return pd.DataFrame(columns=EVENT_QUALITY_COLUMNS)

    raw_by_id = _raw_index(raw_data)
    rows = []
    for index, event in events_df.reset_index(drop=True).iterrows():
        event_id = _event_id(event, index)
        raw_record = raw_by_id.get(_safe_text(event.get("id")), {})
        rows.append(_quality_row(event, raw_record, index))
    return pd.DataFrame(rows, columns=EVENT_QUALITY_COLUMNS)


def _quality_lookup(event_quality_df: pd.DataFrame | None) -> Dict[str, Dict[str, Any]]:
    if event_quality_df is None or event_quality_df.empty:
        return {}
    lookup = {}
    for record in event_quality_df.to_dict(orient="records"):
        for key in (record.get("event_id"), record.get("source_id")):
            text = _safe_text(key)
            if text:
                lookup[text] = record
    return lookup


def merge_event_quality_into_events(events_df: pd.DataFrame, event_quality_df: pd.DataFrame) -> pd.DataFrame:
    """Merge event quality columns back into the event table."""
    if events_df is None or events_df.empty:
        return events_df.copy() if isinstance(events_df, pd.DataFrame) else pd.DataFrame()
    if event_quality_df is None or event_quality_df.empty:
        merged = events_df.copy()
        for column in EVENT_QUALITY_SCORE_COLUMNS:
            if column not in merged.columns:
                merged[column] = 0.0 if column.endswith("_score") else ""
        return merged

    working = events_df.reset_index(drop=True).copy()
    working = working.drop(columns=[column for column in EVENT_QUALITY_SCORE_COLUMNS if column in working.columns])
    working["_event_quality_join_id"] = [
        _event_id(row, index) for index, row in working.iterrows()
    ]
    quality = event_quality_df.copy()
    quality["_event_quality_join_id"] = quality["event_id"].astype(str)
    columns = ["_event_quality_join_id"] + EVENT_QUALITY_SCORE_COLUMNS
    merged = working.merge(quality[columns], on="_event_quality_join_id", how="left")
    merged = merged.drop(columns=["_event_quality_join_id"])
    for column in EVENT_QUALITY_SCORE_COLUMNS:
        if column.endswith("_score"):
            merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0.0)
        else:
            merged[column] = merged[column].fillna("")
    return merged


def merge_event_quality_into_raw_data(raw_data: pd.DataFrame, event_quality_df: pd.DataFrame) -> pd.DataFrame:
    """Merge event quality into raw data so regenerated evidence can carry it."""
    if raw_data is None or raw_data.empty:
        return raw_data.copy() if isinstance(raw_data, pd.DataFrame) else pd.DataFrame()
    if "id" not in raw_data.columns or event_quality_df is None or event_quality_df.empty:
        return raw_data.copy()
    quality = event_quality_df.copy()
    quality["_raw_quality_join_id"] = quality["source_id"].astype(str)
    raw = raw_data.copy()
    raw = raw.drop(columns=[column for column in EVENT_QUALITY_SCORE_COLUMNS if column in raw.columns])
    raw["_raw_quality_join_id"] = raw["id"].astype(str)
    columns = ["_raw_quality_join_id"] + EVENT_QUALITY_SCORE_COLUMNS
    merged = raw.merge(quality[columns], on="_raw_quality_join_id", how="left")
    merged = merged.drop(columns=["_raw_quality_join_id"])
    return merged


def _event_quality_payload(event_id: Any, lookup: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    record = lookup.get(_safe_text(event_id), {})
    return {
        "event_quality_score": float(record.get("event_quality_score", 0.0) or 0.0),
        "event_quality_tier": _safe_text(record.get("event_quality_tier")),
        "event_quality_reason": _safe_text(record.get("event_quality_reason")),
    }


def _with_item_quality(item: Dict[str, Any], lookup: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    updated = dict(item)
    event_id = _safe_text(updated.get("id"))
    payload = _event_quality_payload(event_id, lookup)
    if payload["event_quality_score"] <= 0 and not event_id:
        # Some legacy signal evidence lacks an id; leave it intact and mark as unknown.
        updated.setdefault("event_quality_score", 0.0)
        updated.setdefault("event_quality_tier", "")
        updated.setdefault("event_quality_reason", "")
        return updated
    updated.update(payload)
    return updated


def _candidate_event_ids(row: pd.Series) -> List[str]:
    ids = []
    for value in _safe_list(row.get("mention_ids", [])):
        text = _safe_text(value)
        if text:
            ids.append(text)
    for item in _safe_items(row.get("evidence_items", [])):
        text = _safe_text(item.get("id"))
        if text:
            ids.append(text)
    seen = set()
    deduped = []
    for event_id in ids:
        if event_id not in seen:
            deduped.append(event_id)
            seen.add(event_id)
    return deduped


def _candidate_quality_metrics(
    row: pd.Series,
    lookup: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    ids = _candidate_event_ids(row)
    scores = [
        float(lookup[event_id].get("event_quality_score", 0.0) or 0.0)
        for event_id in ids
        if event_id in lookup
    ]
    if not scores:
        return {
            "candidate_evidence_quality": 0.0,
            "candidate_core_evidence_quality": 0.0,
            "low_quality_evidence_ratio": 0.0,
            "high_quality_evidence_count": 0,
            "quality_risk_flag": "quality_unknown",
            "quality_adjusted_rank_score": round(float(row.get("weak_signal_score", row.get("score", 0.0)) or 0.0), 2),
            "candidate_evidence_quality_reason": "未能按 mention_ids/evidence_items 回挂事件质量",
        }

    average = round(sum(scores) / len(scores), 2)
    sorted_scores = sorted(scores, reverse=True)
    core_scores = sorted_scores[: min(3, len(sorted_scores))]
    core_average = round(sum(core_scores) / len(core_scores), 2)
    low_ratio = round(sum(1 for score in scores if score < 4.0) / len(scores), 3)
    high_count = sum(1 for score in scores if score >= 8.0)
    if average < 4.0 or low_ratio >= 0.5:
        risk = "high_quality_risk"
    elif average < 6.0 or low_ratio > 0:
        risk = "medium_quality_risk"
    else:
        risk = "low_quality_risk"

    base_score = float(row.get("weak_signal_score", row.get("score", 0.0)) or 0.0)
    quality_factor = 0.7 + min(average, 10.0) / 10.0 * 0.3
    adjusted = round(base_score * quality_factor, 2)
    reason = (
        f"证据事件数={len(scores)}，平均质量={average:.2f}，"
        f"核心证据均值={core_average:.2f}，低质量占比={low_ratio:.2f}"
    )
    return {
        "candidate_evidence_quality": average,
        "candidate_core_evidence_quality": core_average,
        "low_quality_evidence_ratio": low_ratio,
        "high_quality_evidence_count": high_count,
        "quality_risk_flag": risk,
        "quality_adjusted_rank_score": adjusted,
        "candidate_evidence_quality_reason": reason,
    }


def merge_event_quality_into_candidates(
    candidate_df: pd.DataFrame,
    event_quality_df: pd.DataFrame,
) -> pd.DataFrame:
    """Attach event quality to candidate rows and their evidence items."""
    if candidate_df is None or candidate_df.empty:
        return candidate_df.copy() if isinstance(candidate_df, pd.DataFrame) else pd.DataFrame()

    merged = candidate_df.copy()
    lookup = _quality_lookup(event_quality_df)
    if not lookup:
        for column in CANDIDATE_QUALITY_COLUMNS:
            if column not in merged.columns:
                merged[column] = 0.0 if column.endswith("_quality") or column.endswith("_score") or column.endswith("_ratio") or column.endswith("_count") else ""
        return merged

    if "evidence_items" in merged.columns:
        merged["evidence_items"] = merged["evidence_items"].apply(
            lambda value: [_with_item_quality(item, lookup) for item in _safe_items(value)]
        )

    metrics = merged.apply(lambda row: _candidate_quality_metrics(row, lookup), axis=1)
    for column in CANDIDATE_QUALITY_COLUMNS:
        merged[column] = [item.get(column, "") for item in metrics]
    return merged
