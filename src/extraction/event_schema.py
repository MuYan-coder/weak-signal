"""Event schema contract and normalization helpers."""

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


WEAK_SIGNAL_EVENT_SCHEMA_VERSION = "weak_signal_event_v2"

EVENT_COMPAT_FIELDS = [
    "id", "subject", "action", "technology", "scene", "time",
    "date", "event_date", "source_type", "title",
]

WEAK_SIGNAL_EVENT_FIELDS = [
    "event_schema_version", "event_id", "event_type", "technical_object",
    "mechanism", "task", "data_modality", "method", "capability_change",
    "problem_solved", "maturity_stage", "novelty_signal", "adoption_signal",
    "cross_domain_signal", "weak_signal_reason", "uncertainty",
    "evidence_span", "confidence",
]

EVENT_AUXILIARY_FIELDS = [
    "mechanism_core_tokens", "task_constraint_tokens", "object_modifier_tokens",
    "data_modifier_tokens", "method_modifier_tokens", "candidate_units",
]

EVENT_DERIVED_FIELDS = [
    "observation_scopes", "scope_candidates", "scope_candidate_scopes",
    "scope_match_mode", "observation_scopes_detected", "scope_direct_matches",
    "scope_alias_matches", "scope_proxy_scopes", "scope_rejected_scopes",
    "scope_detection_reason", "candidate_unit_count",
    "scope_echo_candidate_count", "low_attention_hint", "niche_actor_hint",
    "non_dominant_hint", "cross_domain_hint", "traceable_hint",
    "weak_signal_event_score", "weak_signal_event_candidate",
    "weak_signal_reasons", "source_extraction_mode", "schema_migration_mode",
]

EVENT_LIST_FIELDS = {
    "technology", "data_modality", "method", "mechanism_core_tokens",
    "task_constraint_tokens", "object_modifier_tokens", "data_modifier_tokens",
    "method_modifier_tokens", "candidate_units", "observation_scopes",
    "scope_candidates", "scope_direct_matches", "scope_alias_matches",
    "scope_proxy_scopes", "scope_rejected_scopes", "weak_signal_reasons",
}

EVENT_LLM_JSON_FIELDS = [
    "doc_index", "event_schema_version", "event_id",
    "subject", "action", "technology", "scene", "time",
    "event_type", "technical_object", "mechanism", "task", "data_modality",
    "method", "capability_change", "problem_solved", "maturity_stage",
    "novelty_signal", "adoption_signal", "cross_domain_signal",
    "weak_signal_reason", "uncertainty", "evidence_span", "confidence",
]


def event_cache_columns() -> List[str]:
    return EVENT_COMPAT_FIELDS + WEAK_SIGNAL_EVENT_FIELDS + EVENT_AUXILIARY_FIELDS + EVENT_DERIVED_FIELDS


def safe_event_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, (list, tuple, set)):
        text = "、".join(str(item).strip() for item in value if str(item).strip())
        if text.lower() in {"nan", "nat", "none", "null"}:
            return default
        return text or default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if text.lower() in {"nan", "nat", "none", "null"}:
        return default
    return text or default


def _dedupe_preserve_order(values: Any) -> List[Any]:
    seen = set()
    deduped = []
    for value in values or []:
        if isinstance(value, dict):
            key = json.dumps(value, ensure_ascii=False, sort_keys=True)
            item = value
        else:
            key = str(value).strip()
            item = key
        if not key or key in seen:
            continue
        deduped.append(item)
        seen.add(key)
    return deduped


def coerce_event_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        items = value
    elif isinstance(value, (tuple, set)):
        items = list(value)
    else:
        items = None
    try:
        if items is None and pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass
    if items is None:
        text = str(value).strip()
        if not text or text.lower() in {"nan", "nat", "none", "null", "未知"}:
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    items = parsed
                else:
                    items = [parsed]
            except Exception:
                items = re.split(r"[,，;；、]\s*", text.strip("[]"))
        else:
            items = re.split(r"[,，;；、]\s*", text)

    if any(isinstance(item, dict) for item in items):
        return [item for item in items if item]
    return _dedupe_preserve_order(items)


def coerce_confidence(value: Any, default: float = 0.0) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        confidence = default
    return max(0.0, min(1.0, confidence))


def source_date_text(row: Any) -> str:
    for field in ["date", "event_date", "publication_date", "application_date", "grant_date", "time"]:
        value = str(row.get(field, "")).strip()
        if value and value.lower() not in {"nan", "nat", "none", "null", "未知"}:
            return value
    return ""


def _sanitize_event_id_part(value: Any) -> str:
    raw = safe_event_text(value, "doc")
    sanitized = re.sub(r"[^a-zA-Z0-9_-]+", "_", raw).strip("_")
    if sanitized:
        return sanitized[:80]
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
    return f"doc_{digest}"


def make_event_id(source_id: Any, event_index: Any) -> str:
    try:
        index = max(1, int(event_index))
    except (TypeError, ValueError):
        index = 1
    return f"{_sanitize_event_id_part(source_id)}_evt_{index:03d}"


def normalize_event_schema(
    event: Optional[Dict[str, Any]],
    source_row: Any = None,
    doc_event_index: int = 1,
    source_extraction_mode: str = "legacy_backfill",
) -> Dict[str, Any]:
    event = dict(event or {})
    had_schema_version = bool(safe_event_text(event.get("event_schema_version")))
    if source_row is not None:
        if not safe_event_text(event.get("id")):
            event["id"] = source_row.get("id", "")
        source_date = source_date_text(source_row)
        if source_date and not safe_event_text(event.get("date")):
            event["date"] = source_date
        if source_date and not safe_event_text(event.get("event_date")):
            event["event_date"] = source_date
        for field in ["source_type", "title"]:
            if not safe_event_text(event.get(field)):
                event[field] = source_row.get(field, "")

    event["event_schema_version"] = (
        safe_event_text(event.get("event_schema_version"))
        or WEAK_SIGNAL_EVENT_SCHEMA_VERSION
    )
    event["id"] = safe_event_text(event.get("id"))
    event["event_id"] = safe_event_text(event.get("event_id")) or make_event_id(
        event.get("id") or "doc",
        doc_event_index,
    )

    for field in ["subject", "action", "scene", "time"]:
        event[field] = safe_event_text(event.get(field), "未知")
    for field in ["date", "event_date", "source_type", "title"]:
        event[field] = safe_event_text(event.get(field))

    for field in [
        "event_type", "technical_object", "mechanism", "task",
        "capability_change", "problem_solved", "maturity_stage",
        "novelty_signal", "adoption_signal", "cross_domain_signal",
        "weak_signal_reason", "uncertainty", "evidence_span",
    ]:
        event[field] = safe_event_text(event.get(field))

    for field in EVENT_LIST_FIELDS:
        event[field] = coerce_event_list(event.get(field))

    if not event["technology"]:
        event["technology"] = ["未知"]
    if not event["weak_signal_reason"] and event.get("weak_signal_reasons"):
        event["weak_signal_reason"] = "；".join(event["weak_signal_reasons"])
    if event["weak_signal_reason"] and not event.get("weak_signal_reasons"):
        event["weak_signal_reasons"] = [event["weak_signal_reason"]]

    event["confidence"] = coerce_confidence(event.get("confidence"), default=0.0)
    event.setdefault("source_extraction_mode", source_extraction_mode)
    event.setdefault("schema_migration_mode", "native" if had_schema_version else "legacy_backfill")
    return event


def normalize_events_dataframe(events_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if events_df is None or events_df.empty:
        return pd.DataFrame(columns=event_cache_columns())
    normalized = []
    per_doc_counts = {}
    for _, row in events_df.iterrows():
        record = row.to_dict()
        source_id = safe_event_text(record.get("id")) or f"event_{len(normalized)}"
        per_doc_counts[source_id] = per_doc_counts.get(source_id, 0) + 1
        normalized.append(
            normalize_event_schema(
                record,
                doc_event_index=per_doc_counts[source_id],
                source_extraction_mode=record.get("source_extraction_mode", "legacy_backfill"),
            )
        )
    return pd.DataFrame(normalized).reindex(columns=event_cache_columns(), fill_value=None)


def backfill_events_dataframe(events_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Backfill legacy event records to the current weak-signal event schema."""
    return normalize_events_dataframe(events_df)


def read_events_file(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=event_cache_columns())
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, encoding="utf-8-sig")
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return pd.DataFrame(payload)
        if isinstance(payload, dict) and isinstance(payload.get("events"), list):
            return pd.DataFrame(payload.get("events", []))
        if isinstance(payload, dict):
            return pd.DataFrame([payload])
    return pd.DataFrame(columns=event_cache_columns())


def write_events_file(path: Path, events_df: pd.DataFrame) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    events_df = backfill_events_dataframe(events_df)
    if path.suffix.lower() == ".csv":
        events_df.to_csv(path, index=False, encoding="utf-8-sig")
    else:
        events_df.to_json(path, orient="records", force_ascii=False, indent=2)
    return path


def event_schema_summary(events_df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    events_df = backfill_events_dataframe(events_df)
    if events_df.empty:
        return {
            "event_count": 0,
            "schema_version_counts": {},
            "schema_migration_mode_counts": {},
            "evidence_span_coverage": 0.0,
            "confidence_coverage": 0.0,
            "technical_object_coverage": 0.0,
            "mechanism_coverage": 0.0,
            "weak_signal_reason_coverage": 0.0,
        }

    def coverage(column: str) -> float:
        if column not in events_df.columns:
            return 0.0
        present = events_df[column].apply(lambda value: bool(safe_event_text(value)))
        return round(float(present.mean()), 3)

    confidence_present = (
        pd.to_numeric(events_df.get("confidence", pd.Series([0.0] * len(events_df))), errors="coerce")
        .fillna(0.0)
        .gt(0)
    )
    return {
        "event_count": int(len(events_df)),
        "schema_version_counts": events_df.get("event_schema_version", pd.Series(dtype="object")).fillna("").astype(str).value_counts().to_dict(),
        "schema_migration_mode_counts": events_df.get("schema_migration_mode", pd.Series(dtype="object")).fillna("").astype(str).value_counts().to_dict(),
        "evidence_span_coverage": coverage("evidence_span"),
        "confidence_coverage": round(float(confidence_present.mean()), 3),
        "technical_object_coverage": coverage("technical_object"),
        "mechanism_coverage": coverage("mechanism"),
        "weak_signal_reason_coverage": coverage("weak_signal_reason"),
    }


def backfill_events_file(input_path: Path, output_path: Optional[Path] = None) -> pd.DataFrame:
    input_path = Path(input_path)
    output_path = Path(output_path) if output_path else input_path.with_name(f"{input_path.stem}_schema_backfilled.json")
    events_df = backfill_events_dataframe(read_events_file(input_path))
    write_events_file(output_path, events_df)
    summary_path = output_path.with_name(f"{output_path.stem}_summary.json")
    summary_path.write_text(
        json.dumps(event_schema_summary(events_df), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return events_df


__all__ = [
    "WEAK_SIGNAL_EVENT_SCHEMA_VERSION",
    "EVENT_COMPAT_FIELDS",
    "WEAK_SIGNAL_EVENT_FIELDS",
    "EVENT_AUXILIARY_FIELDS",
    "EVENT_DERIVED_FIELDS",
    "EVENT_LIST_FIELDS",
    "EVENT_LLM_JSON_FIELDS",
    "event_cache_columns",
    "safe_event_text",
    "coerce_event_list",
    "coerce_confidence",
    "source_date_text",
    "make_event_id",
    "normalize_event_schema",
    "normalize_events_dataframe",
    "backfill_events_dataframe",
    "read_events_file",
    "write_events_file",
    "event_schema_summary",
    "backfill_events_file",
]
