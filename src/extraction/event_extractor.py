import json
import os
import re
from pathlib import Path

import pandas as pd

from ..utils.api_stats import record_call
from ..utils.env_config import ensure_env_loaded
from ..utils.llm_client import chat_text, get_provider_and_client
from .event_schema import (
    EVENT_LLM_JSON_FIELDS as _EVENT_JSON_FIELDS,
    WEAK_SIGNAL_EVENT_SCHEMA_VERSION,
    coerce_confidence as _coerce_confidence,
    coerce_event_list as _coerce_event_list,
    event_cache_columns as _event_cache_columns,
    normalize_event_schema as _normalize_event_schema,
    normalize_events_dataframe as _normalize_events_dataframe,
    safe_event_text as _safe_event_text,
    source_date_text as _source_date_text,
)
from .tech_lexicon import (
    BROAD_TECH_TERMS,
    DOMINANT_TECH_TERMS,
    OBSERVATION_SCOPE_SET,
    aliases_for,
    canonicalize_term,
    diagnose_observation_scope_detection,
    detect_supported_observation_scopes,
    discover_candidate_terms,
    extract_data_modifier_tokens,
    extract_mechanism_core_tokens,
    extract_method_modifier_tokens,
    extract_object_modifier_tokens,
    extract_scene_tokens,
    extract_task_constraint_tokens,
    extract_technologies,
    has_robot_domain_anchor,
    has_non_scope_constraint,
    is_bare_mechanism_candidate,
    is_scope_echo_candidate,
    normalize_signal_phrase,
    normalize_proxy_token,
    normalize_technologies,
)


HEAD_ORGS = {
    "google", "openai", "microsoft", "meta", "amazon", "ibm", "nvidia",
    "baidu", "alibaba", "tencent", "huawei", "deepmind", "anthropic",
    "mit", "stanford", "tsinghua", "pku",
}

SCOPE_LABELS_ZH = {
    "humanoid robot": "人形机器人",
    "world model": "世界模型",
    "embodied intelligence": "具身智能",
}

MECHANISM_LABELS_ZH = {
    "training": "训练",
    "planning": "规划",
    "simulation": "仿真",
    "reasoning": "推理",
    "control": "控制",
    "grounding": "落地",
    "memory": "记忆",
}

ensure_env_loaded()


def _api_config_available():
    return bool(
        (os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_BASE_URL"))
        or (os.getenv("ANTHROPIC_AUTH_TOKEN") and os.getenv("ANTHROPIC_BASE_URL"))
        or os.getenv("DEEPSEEK_API_KEY")
    )

def _dedupe_preserve_order(values):
    seen = set()
    deduped = []
    for value in values or []:
        text = str(value).strip()
        if not text or text in seen:
            continue
        deduped.append(text)
        seen.add(text)
    return deduped


def _fix_event_dict(event):
    if not isinstance(event, dict):
        return {}
    if not _coerce_event_list(event.get("technology")):
        event["technology"] = ["未知"]
    for field in ["subject", "action", "scene", "time"]:
        if not _safe_event_text(event.get(field)):
            event[field] = "未知"
    event["event_schema_version"] = (
        _safe_event_text(event.get("event_schema_version"))
        or WEAK_SIGNAL_EVENT_SCHEMA_VERSION
    )
    for field in ["data_modality", "method"]:
        event[field] = _coerce_event_list(event.get(field))
    event["confidence"] = _coerce_confidence(event.get("confidence"), default=0.0)
    return event


def _event_extraction_max_events_per_doc():
    try:
        value = int(os.getenv("EVENT_EXTRACTION_MAX_EVENTS_PER_DOC", "6"))
    except (TypeError, ValueError):
        value = 6
    return max(1, value)


def _event_extraction_min_confidence():
    try:
        value = float(os.getenv("EVENT_EXTRACTION_MIN_CONFIDENCE", "0"))
    except (TypeError, ValueError):
        value = 0.0
    return max(0.0, min(1.0, value))


def _retry_empty_batch_enabled():
    value = str(os.getenv("EVENT_EXTRACTION_RETRY_EMPTY_BATCH", "1")).strip().lower()
    return value not in {"0", "false", "no", "off"}


def _event_extraction_batch_timeout():
    try:
        value = int(os.getenv("EVENT_EXTRACTION_BATCH_TIMEOUT", "60"))
    except (TypeError, ValueError):
        value = 60
    return max(30, value)


def _event_extraction_single_timeout():
    try:
        value = int(os.getenv("EVENT_EXTRACTION_SINGLE_TIMEOUT", "45"))
    except (TypeError, ValueError):
        value = 45
    return max(30, value)


def _event_extraction_batch_retries():
    try:
        value = int(os.getenv("EVENT_EXTRACTION_BATCH_RETRIES", "0"))
    except (TypeError, ValueError):
        value = 0
    return max(0, value)


def _timeout_fallback_to_local_enabled():
    value = str(os.getenv("EVENT_EXTRACTION_TIMEOUT_FALLBACK_TO_LOCAL", "1")).strip().lower()
    return value not in {"0", "false", "no", "off"}


def _event_selection_key(index_event):
    index, event = index_event
    return (
        _coerce_confidence(event.get("confidence"), default=0.0),
        1 if _safe_event_text(event.get("evidence_span")) else 0,
        1 if any(
            _safe_event_text(event.get(field))
            for field in ["technical_object", "mechanism", "task", "weak_signal_reason"]
        ) else 0,
        -index,
    )


def _filter_events_for_doc(events, max_events_per_doc=None, min_confidence=None):
    """Apply per-document event explosion controls without changing event order."""
    max_events_per_doc = max_events_per_doc or _event_extraction_max_events_per_doc()
    min_confidence = _event_extraction_min_confidence() if min_confidence is None else min_confidence
    indexed_events = [(index, event) for index, event in enumerate(events or []) if isinstance(event, dict)]
    eligible = [
        (index, event)
        for index, event in indexed_events
        if _coerce_confidence(event.get("confidence"), default=0.0) >= min_confidence
    ]
    selected_ranked = sorted(eligible, key=_event_selection_key, reverse=True)[:max_events_per_doc]
    selected_indexes = {index for index, _ in selected_ranked}
    selected = [event for index, event in eligible if index in selected_indexes]
    return selected, len(indexed_events) - len(eligible), max(0, len(eligible) - len(selected))


def _parse_json_from_response(raw: str):
    raw = raw.strip()
    if not raw:
        return None

    # 首先尝试直接解析
    try:
        return json.loads(raw)
    except Exception:
        pass

    # 移除可能的markdown代码块标记
    if raw.startswith("```"):
        lines = raw.split("\n")
        if len(lines) > 1:
            raw = "\n".join(lines[1:])
        if "```" in raw:
            raw = raw.rsplit("```", 1)[0]
        raw = raw.strip()

    try:
        return json.loads(raw)
    except Exception:
        pass

    first_obj_idx = raw.find("{")
    last_obj_idx = raw.rfind("}")
    first_array_idx = raw.find("[")
    if first_obj_idx != -1 and last_obj_idx > first_obj_idx and (
        first_array_idx == -1 or first_obj_idx < first_array_idx
    ):
        candidate = raw[first_obj_idx:last_obj_idx + 1]
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # 尝试提取数组
    first_idx = raw.find("[")
    last_idx = raw.rfind("]")
    if first_idx != -1 and last_idx != -1 and last_idx > first_idx:
        candidate = raw[first_idx:last_idx + 1]
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # 尝试解析数组格式
    json_pattern = r"\[\s*\{.*?\}\s*\]"
    matches = re.findall(json_pattern, raw, re.DOTALL)
    if matches:
        candidate = max(matches, key=len)
        try:
            return json.loads(candidate)
        except Exception:
            pass

    recovered_events = _parse_event_list_lenient(raw)
    if recovered_events:
        return recovered_events

    # 尝试提取对象
    first_idx = raw.find("{")
    last_idx = raw.rfind("}")
    if first_idx != -1 and last_idx != -1 and last_idx > first_idx:
        candidate = raw[first_idx:last_idx + 1]
        try:
            return json.loads(candidate)
        except Exception:
            pass

        recovered_event = _parse_event_object_lenient(candidate)
        if recovered_event:
            return recovered_event

    obj_pattern = r"\{[^{}]*\}"
    matches = re.findall(obj_pattern, raw, re.DOTALL)
    if matches:
        candidate = max(matches, key=len)
        try:
            return json.loads(candidate)
        except Exception:
            pass

    return None


def _split_object_snippets(raw: str):
    return re.findall(r"\{[^{}]*\}", raw or "", re.DOTALL)


def _read_lenient_string_field(obj_text: str, field: str) -> str:
    marker = f'"{field}"'
    key_index = obj_text.find(marker)
    if key_index == -1:
        return ""
    colon_index = obj_text.find(":", key_index + len(marker))
    if colon_index == -1:
        return ""
    quote_index = obj_text.find('"', colon_index + 1)
    if quote_index == -1:
        return ""

    end_candidates = []
    try:
        field_position = _EVENT_JSON_FIELDS.index(field)
    except ValueError:
        field_position = -1
    for next_field in _EVENT_JSON_FIELDS[field_position + 1:]:
        match = re.search(r',\s*"' + re.escape(next_field) + r'"\s*:', obj_text[quote_index + 1:], re.DOTALL)
        if match:
            end_candidates.append(quote_index + 1 + match.start())

    close_index = obj_text.rfind("}")
    if close_index > quote_index:
        end_candidates.append(close_index)
    if not end_candidates:
        return ""

    value = obj_text[quote_index + 1:min(end_candidates)].strip()
    if value.endswith(","):
        value = value[:-1].rstrip()
    if value.endswith('"'):
        value = value[:-1]
    return value.replace('\\"', '"').strip()


def _read_lenient_technology_field(obj_text: str):
    marker = '"technology"'
    key_index = obj_text.find(marker)
    if key_index == -1:
        return []
    colon_index = obj_text.find(":", key_index + len(marker))
    if colon_index == -1:
        return []

    array_start = obj_text.find("[", colon_index + 1)
    if array_start != -1:
        array_end = obj_text.find("]", array_start + 1)
        if array_end != -1:
            array_text = obj_text[array_start + 1:array_end]
            values = [item.strip() for item in re.findall(r'"([^"]+)"', array_text) if item.strip()]
            if values:
                return values
            return [
                item.strip().strip('"').strip("'")
                for item in array_text.split(",")
                if item.strip().strip('"').strip("'")
            ]

    value = _read_lenient_string_field(obj_text, "technology")
    return [value] if value else []


def _parse_event_object_lenient(obj_text: str):
    try:
        parsed = json.loads(obj_text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    event = {
        "subject": _read_lenient_string_field(obj_text, "subject"),
        "action": _read_lenient_string_field(obj_text, "action"),
        "technology": _read_lenient_technology_field(obj_text),
        "scene": _read_lenient_string_field(obj_text, "scene"),
        "time": _read_lenient_string_field(obj_text, "time"),
    }
    if any(event.get(key) for key in ["subject", "action", "technology", "scene", "time"]):
        return event
    return None


def _parse_event_list_lenient(raw: str):
    events = []
    for snippet in _split_object_snippets(raw):
        event = _parse_event_object_lenient(snippet)
        if event:
            events.append(event)
    return events


def _normalize_cache_path(cache_path):
    if not cache_path:
        return None
    path = Path(cache_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    return path


def load_event_cache(cache_path, expected_ids=None):
    cache_file = _normalize_cache_path(cache_path)
    if cache_file is None or not cache_file.exists():
        return None
    try:
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
    except Exception:
        return None
    metadata = {}
    if isinstance(payload, dict):
        metadata = payload.get("metadata", {}) if isinstance(payload.get("metadata"), dict) else {}
        events = payload.get("events", [])
    elif isinstance(payload, list):
        events = payload
    else:
        events = []
    if not isinstance(events, list) or not events:
        return None
    events_df = _normalize_events_dataframe(pd.DataFrame(events))
    if expected_ids:
        expected = [str(item) for item in expected_ids]
        metadata_source_ids = [str(item) for item in metadata.get("source_ids", [])]
        cached_ids = _dedupe_preserve_order(events_df.get("id", pd.Series(dtype="object")).astype(str).tolist())
        if metadata_source_ids and metadata_source_ids != expected:
            return None
        if not metadata_source_ids and cached_ids != expected:
            return None
    return events_df


def save_event_cache(cache_path, events_df, metadata=None):
    cache_file = _normalize_cache_path(cache_path)
    if cache_file is None or events_df is None or events_df.empty:
        return
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    events_df = _normalize_events_dataframe(events_df)
    metadata = dict(metadata or {})
    metadata.setdefault("event_schema_version", WEAK_SIGNAL_EVENT_SCHEMA_VERSION)
    metadata.setdefault(
        "source_ids",
        _dedupe_preserve_order(events_df.get("id", pd.Series(dtype="object")).astype(str).tolist()),
    )
    payload = {
        "metadata": metadata,
        "events": events_df.to_dict(orient="records"),
    }
    cache_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _supports_scope_context(snippet: str, scopes):
    haystack = normalize_signal_phrase(snippet)
    if not haystack:
        return False
    for scope in scopes:
        for alias in aliases_for(scope):
            if normalize_signal_phrase(alias) in haystack:
                return True
    return False


def _strip_html_noise(text: str) -> str:
    raw_text = str(text or "").strip()
    if not raw_text:
        return ""
    cleaned = re.sub(r"<[^>]+>", " ", raw_text)
    cleaned = cleaned.replace("&nbsp;", " ").replace("&amp;", "&")
    cleaned = re.sub(r"https?://\S+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _looks_formable_candidate(
    mechanism_tokens,
    task_tokens,
    object_tokens,
    data_tokens,
    scene_tokens,
    method_tokens,
    strict_mode=True,
):
    if not mechanism_tokens:
        return False
    
    # 宽松模式：只要有机制核心词就认为可以成形
    if not strict_mode:
        return True
    
    if is_bare_mechanism_candidate(
        mechanism_tokens=mechanism_tokens,
        task_tokens=task_tokens,
        object_tokens=object_tokens,
        data_tokens=data_tokens,
        scene_tokens=scene_tokens,
        method_tokens=method_tokens,
    ):
        return False
    return has_non_scope_constraint(
        task_tokens=task_tokens,
        object_tokens=object_tokens,
        data_tokens=data_tokens,
        scene_tokens=scene_tokens,
        mechanism_tokens=mechanism_tokens,
        method_tokens=method_tokens,
    )


def _is_broad_world_model_candidate(
    scope_names,
    mechanism_tokens,
    task_tokens,
    object_tokens,
    data_tokens,
    method_tokens,
):
    scope_names = _dedupe_preserve_order(scope_names)
    if "world model" not in scope_names:
        return False
    has_specific_constraint = bool(task_tokens or object_tokens or data_tokens)
    if has_specific_constraint:
        return False
    return bool(mechanism_tokens or method_tokens)


def _primary_mechanism_core(mechanism_tokens):
    values = _dedupe_preserve_order(mechanism_tokens)
    return values[0] if values else ""


def _candidate_phrase_type(task_tokens, object_tokens, data_tokens, method_tokens):
    if object_tokens or task_tokens:
        return "object_task_mechanism"
    if data_tokens:
        return "data_mechanism"
    if method_tokens:
        return "method_mechanism"
    return "mechanism_only"


def _zh_label(token: str) -> str:
    normalized = normalize_proxy_token(str(token or "").strip())
    if not normalized:
        return ""
    if normalized in SCOPE_LABELS_ZH:
        return SCOPE_LABELS_ZH[normalized]
    if normalized in MECHANISM_LABELS_ZH:
        return MECHANISM_LABELS_ZH[normalized]
    return normalized


def _relation_semantics(
    scope_names,
    mechanism_core_tokens,
    task_constraint_tokens,
    object_modifier_tokens,
    data_modifier_tokens,
    method_modifier_tokens,
):
    scope_names = _dedupe_preserve_order(scope_names)
    mechanism_core_tokens = _dedupe_preserve_order(mechanism_core_tokens)
    task_constraint_tokens = _dedupe_preserve_order(task_constraint_tokens)
    object_modifier_tokens = _dedupe_preserve_order(object_modifier_tokens)
    data_modifier_tokens = _dedupe_preserve_order(data_modifier_tokens)
    method_modifier_tokens = _dedupe_preserve_order(method_modifier_tokens)

    role_target = object_modifier_tokens[0] if object_modifier_tokens else (task_constraint_tokens[0] if task_constraint_tokens else "")
    role_task = task_constraint_tokens[0] if task_constraint_tokens else ""
    role_data = data_modifier_tokens[0] if data_modifier_tokens else ""
    role_method = method_modifier_tokens[0] if method_modifier_tokens else ""
    mechanism_core = mechanism_core_tokens[0] if mechanism_core_tokens else ""
    primary_scope = scope_names[0] if scope_names else ""

    target_zh = _zh_label(role_target)
    task_zh = _zh_label(role_task)
    data_zh = _zh_label(role_data)
    method_zh = _zh_label(role_method)
    mechanism_zh = _zh_label(mechanism_core)
    scope_zh = SCOPE_LABELS_ZH.get(primary_scope, _zh_label(primary_scope))

    relation_parts = []
    if data_zh:
        relation_parts.append(f"基于{data_zh}数据")
    if target_zh:
        relation_parts.append(f"用于{target_zh}")
    elif task_zh:
        relation_parts.append(f"面向{task_zh}任务")
    if method_zh:
        relation_parts.append(f"采用{method_zh}方法")
    if scope_zh and mechanism_zh:
        relation_parts.append(f"{scope_zh}{mechanism_zh}")
    elif mechanism_zh:
        relation_parts.append(mechanism_zh)
    elif scope_zh:
        relation_parts.append(scope_zh)
    relation_summary = "，".join(part for part in relation_parts if part)

    relation_signature = " | ".join(
        [
            f"target={role_target or 'none'}",
            f"task={role_task or 'none'}",
            f"data={role_data or 'none'}",
            f"method={role_method or 'none'}",
            f"scope={primary_scope or 'none'}",
            f"mechanism={mechanism_core or 'none'}",
        ]
    )
    return {
        "relation_target": role_target,
        "relation_task": role_task,
        "relation_data_modality": role_data,
        "relation_method": role_method,
        "relation_summary": relation_summary,
        "relation_signature": relation_signature,
    }


def _split_candidate_snippets(*texts: str):
    snippets = []
    for text in texts:
        raw_text = str(text or "").strip()
        if not raw_text:
            continue
        parts = re.split(r"[。！？!?；;:\n]+", raw_text)
        for part in parts:
            normalized = normalize_signal_phrase(part)
            if normalized and normalized not in snippets:
                snippets.append(normalized)
    return snippets


def _build_candidate_unit(
    raw_text: str,
    scope_names,
    action_text: str,
    scene_text: str,
    mechanism_tokens=None,
    task_tokens=None,
    scene_tokens=None,
    object_tokens=None,
    data_tokens=None,
    method_tokens=None,
    scope_context_supported=False,
    source_extraction_mode="local",
    scope_match_mode="explicit",
):
    raw_candidate_text = normalize_signal_phrase(raw_text)
    if not raw_candidate_text or not scope_names:
        return None

    mechanism_core_tokens = _dedupe_preserve_order(
        mechanism_tokens or extract_mechanism_core_tokens(action_text, raw_candidate_text)
    )
    task_constraint_tokens = _dedupe_preserve_order(
        task_tokens or extract_task_constraint_tokens(scene_text, raw_candidate_text)
    )
    scene_tokens = _dedupe_preserve_order(
        scene_tokens or extract_scene_tokens(scene_text, raw_candidate_text)
    )
    object_modifier_tokens = _dedupe_preserve_order(
        object_tokens or extract_object_modifier_tokens(scene_text, action_text, raw_candidate_text)
    )
    data_modifier_tokens = _dedupe_preserve_order(
        data_tokens or extract_data_modifier_tokens(scene_text, action_text, raw_candidate_text)
    )
    method_modifier_tokens = _dedupe_preserve_order(
        method_tokens or extract_method_modifier_tokens(scene_text, action_text, raw_candidate_text)
    )
    action_tokens = _dedupe_preserve_order(extract_mechanism_core_tokens(action_text, raw_candidate_text))
    is_scope_echo = is_scope_echo_candidate(
        raw_candidate_text,
        scope_names,
        mechanism_tokens=mechanism_core_tokens,
        task_tokens=task_constraint_tokens,
    )
    has_mechanism_core = bool(mechanism_core_tokens)
    has_task_constraint = bool(task_constraint_tokens)
    has_non_scope_info = has_non_scope_constraint(
        task_constraint_tokens,
        object_modifier_tokens,
        data_modifier_tokens,
        scene_tokens,
        mechanism_core_tokens,
        method_modifier_tokens,
    )
    mechanism_core = _primary_mechanism_core(mechanism_core_tokens)
    generic_core_only = bool(has_mechanism_core and not has_non_scope_info)
    raw_phrase_type = _candidate_phrase_type(
        task_constraint_tokens,
        object_modifier_tokens,
        data_modifier_tokens,
        method_modifier_tokens,
    )
    relation_meta = _relation_semantics(
        scope_names,
        mechanism_core_tokens,
        task_constraint_tokens,
        object_modifier_tokens,
        data_modifier_tokens,
        method_modifier_tokens,
    )

    return {
        "raw_phrase": raw_candidate_text,
        "raw_candidate_text": raw_candidate_text,
        "mechanism_core": mechanism_core,
        "secondary_mechanism_cores": mechanism_core_tokens[1:],
        "scope_names": _dedupe_preserve_order(scope_names),
        "mechanism_core_tokens": mechanism_core_tokens,
        "task_constraint_tokens": task_constraint_tokens,
        "object_modifier_tokens": object_modifier_tokens,
        "data_modifier_tokens": data_modifier_tokens,
        "method_modifier_tokens": method_modifier_tokens,
        "scene_tokens": scene_tokens,
        "action_tokens": action_tokens,
        "is_scope_echo": bool(is_scope_echo),
        "has_mechanism_core": has_mechanism_core,
        "has_task_constraint": has_task_constraint,
        "scope_context_supported": bool(scope_context_supported),
        "has_non_scope_constraint": bool(has_non_scope_info),
        "raw_phrase_type": raw_phrase_type,
        "source_extraction_mode": source_extraction_mode,
        "scope_match_mode": scope_match_mode,
        "generic_core_only": generic_core_only,
        "relation_target": relation_meta["relation_target"],
        "relation_task": relation_meta["relation_task"],
        "relation_data_modality": relation_meta["relation_data_modality"],
        "relation_method": relation_meta["relation_method"],
        "relation_summary": relation_meta["relation_summary"],
        "relation_signature": relation_meta["relation_signature"],
    }


def _extract_candidate_units(
    event,
    title_text="",
    fallback_text="",
    source_extraction_mode="local",
    source_type="",
    observation_scopes=None,
    scope_match_mode="explicit",
):
    title_text = _strip_html_noise(title_text)
    fallback_text = _strip_html_noise(fallback_text)
    combined_text = " ".join(
        part for part in [str(title_text or "").strip(), str(fallback_text or "").strip()] if part
    )
    observation_scopes = observation_scopes or detect_supported_observation_scopes(combined_text)
    if not observation_scopes:
        return []
    robot_scopes = {"humanoid robot", "embodied intelligence"}
    if robot_scopes.intersection(observation_scopes) and not has_robot_domain_anchor(combined_text):
        return []

    technologies = event.get("technology", [])
    if not isinstance(technologies, list):
        technologies = [technologies]
    non_scope_technologies = []
    for technology in technologies:
        canonical = canonicalize_term(technology)
        if canonical == "未知":
            continue
        if canonical in OBSERVATION_SCOPE_SET or canonical in BROAD_TECH_TERMS:
            continue
        non_scope_technologies.append(canonical)
    non_scope_technologies = _dedupe_preserve_order(non_scope_technologies)

    action_text = str(event.get("action", "")).strip()
    scene_text = str(event.get("scene", "")).strip()
    fallback_mechanisms = _dedupe_preserve_order(
        extract_mechanism_core_tokens(action_text, title_text, fallback_text)
    )
    fallback_tasks = _dedupe_preserve_order(
        extract_task_constraint_tokens(scene_text, title_text, fallback_text)
    )
    fallback_scenes = _dedupe_preserve_order(
        extract_scene_tokens(scene_text, title_text, fallback_text)
    )
    fallback_object_tokens = _dedupe_preserve_order(
        extract_object_modifier_tokens(scene_text, title_text, fallback_text, *non_scope_technologies)
    )
    fallback_data_tokens = _dedupe_preserve_order(
        extract_data_modifier_tokens(scene_text, title_text, fallback_text)
    )
    fallback_method_tokens = _dedupe_preserve_order(
        extract_method_modifier_tokens(action_text, scene_text, title_text, fallback_text)
    )

    units = []
    seen = set()
    for snippet in _split_candidate_snippets(title_text, fallback_text):
        snippet_mechanisms = _dedupe_preserve_order(extract_mechanism_core_tokens(action_text, snippet))
        snippet_tasks = _dedupe_preserve_order(extract_task_constraint_tokens(scene_text, snippet))
        snippet_scenes = _dedupe_preserve_order(extract_scene_tokens(scene_text, snippet))
        snippet_object_tokens = _dedupe_preserve_order(
            extract_object_modifier_tokens(scene_text, snippet, *non_scope_technologies)
        )
        snippet_data_tokens = _dedupe_preserve_order(
            extract_data_modifier_tokens(scene_text, snippet)
        )
        snippet_method_tokens = _dedupe_preserve_order(
            extract_method_modifier_tokens(action_text, scene_text, snippet)
        )
        has_scope_context = _supports_scope_context(snippet, observation_scopes)
        if source_type == "patent" and scope_match_mode == "proxy_patent":
            has_scope_context = True

        if not snippet_mechanisms:
            continue
        if not has_scope_context and not fallback_mechanisms:
            continue
        if not _looks_formable_candidate(
            snippet_mechanisms,
            snippet_tasks,
            snippet_object_tokens,
            snippet_data_tokens,
            snippet_scenes,
            snippet_method_tokens,
            strict_mode=(source_extraction_mode != "local"),
        ):
            continue
        if _is_broad_world_model_candidate(
            observation_scopes,
            snippet_mechanisms,
            snippet_tasks,
            snippet_object_tokens,
            snippet_data_tokens,
            snippet_method_tokens,
        ):
            continue

        raw_parts = []
        raw_parts.extend(snippet_object_tokens[:2])
        raw_parts.extend(snippet_data_tokens[:2])
        raw_parts.extend(non_scope_technologies[:1])
        raw_parts.extend(snippet_mechanisms[:2])
        raw_parts.extend(
            token for token in snippet_method_tokens if token not in raw_parts
        )
        raw_parts.extend(
            token
            for token in snippet_tasks
            if token not in raw_parts
        )
        raw_parts = _dedupe_preserve_order(raw_parts)
        raw_text = " ".join(raw_parts[:5]) if raw_parts else snippet
        unit = _build_candidate_unit(
            raw_text=raw_text,
            scope_names=observation_scopes,
            action_text=action_text,
            scene_text=scene_text,
            mechanism_tokens=snippet_mechanisms,
            task_tokens=snippet_tasks,
            scene_tokens=snippet_scenes,
            object_tokens=snippet_object_tokens,
            data_tokens=snippet_data_tokens,
            method_tokens=snippet_method_tokens,
            scope_context_supported=has_scope_context,
            source_extraction_mode=source_extraction_mode,
            scope_match_mode=scope_match_mode,
        )
        if not unit:
            continue
        signature = (
            unit["raw_candidate_text"],
            tuple(unit["scope_names"]),
            tuple(unit["mechanism_core_tokens"]),
            tuple(unit["task_constraint_tokens"]),
            tuple(unit["object_modifier_tokens"]),
            tuple(unit["data_modifier_tokens"]),
            tuple(unit["method_modifier_tokens"]),
        )
        if signature in seen:
            continue
        units.append(unit)
        seen.add(signature)
        if len(units) >= 6:
            break

    if not units:
        if _looks_formable_candidate(
            fallback_mechanisms,
            fallback_tasks,
            fallback_object_tokens,
            fallback_data_tokens,
            fallback_scenes,
            fallback_method_tokens,
        ):
            if _is_broad_world_model_candidate(
                observation_scopes,
                fallback_mechanisms,
                fallback_tasks,
                fallback_object_tokens,
                fallback_data_tokens,
                fallback_method_tokens,
            ):
                return units
            raw_parts = []
            raw_parts.extend(fallback_object_tokens[:2])
            raw_parts.extend(fallback_data_tokens[:2])
            raw_parts.extend(non_scope_technologies[:1])
            raw_parts.extend(fallback_mechanisms[:2])
            raw_parts.extend(
                token for token in fallback_method_tokens if token not in raw_parts
            )
            raw_parts.extend(
                token
                for token in fallback_tasks
                if token not in raw_parts
            )
            fallback_raw_text = " ".join(_dedupe_preserve_order(raw_parts))
            unit = _build_candidate_unit(
                raw_text=fallback_raw_text,
                scope_names=observation_scopes,
                action_text=action_text,
                scene_text=scene_text,
                mechanism_tokens=fallback_mechanisms,
                task_tokens=fallback_tasks,
                scene_tokens=fallback_scenes,
                object_tokens=fallback_object_tokens,
                data_tokens=fallback_data_tokens,
                method_tokens=fallback_method_tokens,
                scope_context_supported=True,
                source_extraction_mode=source_extraction_mode,
                scope_match_mode=scope_match_mode,
            )
            if unit:
                units.append(unit)

    return units


def _normalize_event_technologies(
    event,
    fallback_text="",
    title_text="",
    source_extraction_mode="local",
    source_type="",
):
    technologies = event.get("technology", [])
    if not isinstance(technologies, list):
        technologies = [technologies]

    title_text = _strip_html_noise(title_text)
    fallback_text = _strip_html_noise(fallback_text)
    combined_text = " ".join(str(part) for part in [title_text, fallback_text] if str(part).strip())
    normalized = normalize_technologies(technologies, fallback_text=combined_text)
    discovered_terms = discover_candidate_terms(combined_text, limit=4)

    if normalized and all(item in DOMINANT_TECH_TERMS for item in normalized if item != "未知"):
        for term in discovered_terms:
            if term not in normalized and term not in DOMINANT_TECH_TERMS:
                normalized.append(term)

    event["technology"] = normalized
    source_type = str(source_type or "").strip().lower()
    fallback_mechanisms = _dedupe_preserve_order(
        extract_mechanism_core_tokens(title_text, fallback_text, " ".join(normalized))
    )
    fallback_tasks = _dedupe_preserve_order(
        extract_task_constraint_tokens(title_text, fallback_text)
    )
    fallback_scenes = _dedupe_preserve_order(
        extract_scene_tokens(title_text, fallback_text)
    )
    fallback_object_tokens = _dedupe_preserve_order(
        extract_object_modifier_tokens(title_text, fallback_text, " ".join(normalized))
    )
    fallback_data_tokens = _dedupe_preserve_order(
        extract_data_modifier_tokens(title_text, fallback_text)
    )
    fallback_method_tokens = _dedupe_preserve_order(
        extract_method_modifier_tokens(title_text, fallback_text)
    )
    if not _coerce_event_list(event.get("mechanism_core_tokens")):
        event["mechanism_core_tokens"] = fallback_mechanisms
    if not _coerce_event_list(event.get("task_constraint_tokens")):
        event["task_constraint_tokens"] = fallback_tasks
    if not _coerce_event_list(event.get("object_modifier_tokens")):
        event["object_modifier_tokens"] = fallback_object_tokens
    if not _coerce_event_list(event.get("data_modifier_tokens")):
        event["data_modifier_tokens"] = fallback_data_tokens
    if not _coerce_event_list(event.get("method_modifier_tokens")):
        event["method_modifier_tokens"] = fallback_method_tokens
    if not _safe_event_text(event.get("technical_object")) and fallback_object_tokens:
        event["technical_object"] = " ".join(fallback_object_tokens[:2])
    if not _safe_event_text(event.get("mechanism")) and fallback_mechanisms:
        event["mechanism"] = " ".join(fallback_mechanisms[:2])
    if not _safe_event_text(event.get("task")) and fallback_tasks:
        event["task"] = " ".join(fallback_tasks[:2])
    if not _coerce_event_list(event.get("data_modality")) and fallback_data_tokens:
        event["data_modality"] = fallback_data_tokens[:3]
    if not _coerce_event_list(event.get("method")) and fallback_method_tokens:
        event["method"] = fallback_method_tokens[:3]
    scope_diag = diagnose_observation_scope_detection(
        combined_text,
        source_type=source_type,
        mechanism_tokens=fallback_mechanisms,
        task_tokens=fallback_tasks,
        object_tokens=fallback_object_tokens,
        data_tokens=fallback_data_tokens,
        scene_tokens=fallback_scenes,
        method_tokens=fallback_method_tokens,
    )
    observation_scopes = scope_diag["supported_scopes"]
    scope_match_mode = scope_diag["scope_match_mode"] or ""

    candidate_units = _extract_candidate_units(
        event,
        title_text=title_text,
        fallback_text=fallback_text,
        source_extraction_mode=source_extraction_mode,
        source_type=source_type,
        observation_scopes=observation_scopes,
        scope_match_mode=scope_match_mode or "explicit",
    )
    scope_candidates = [unit["raw_candidate_text"] for unit in candidate_units]
    scope_candidate_scopes = {
        unit["raw_candidate_text"]: unit["scope_names"]
        for unit in candidate_units
        if unit.get("raw_candidate_text")
    }

    event["observation_scopes"] = observation_scopes
    event["observation_scopes_detected"] = bool(observation_scopes)
    event["scope_match_mode"] = scope_match_mode or "explicit"
    event["scope_direct_matches"] = scope_diag["direct_matches"]
    event["scope_alias_matches"] = scope_diag["alias_matches"]
    event["scope_proxy_scopes"] = scope_diag["proxy_scopes"]
    event["scope_rejected_scopes"] = scope_diag["rejected_scopes"]
    event["scope_detection_reason"] = scope_diag["reason"]
    event["scope_candidates"] = scope_candidates
    event["scope_candidate_scopes"] = scope_candidate_scopes
    event["candidate_units"] = candidate_units
    event["candidate_unit_count"] = len(candidate_units)
    event["scope_echo_candidate_count"] = sum(
        1 for unit in candidate_units if bool(unit.get("is_scope_echo", False))
    )
    return event


def _weak_signal_event_prompt(text):
    return f"""
你是用于技术预见的弱信号事件抽取器。请从文本中抽取 0 到 N 条有明确原文证据的技术事件。
只输出纯 JSON 数组，不要添加解释、Markdown、编号或多余文字。没有有效技术事件时输出 []。

文本：
{text}

抽取要求：
- 保留兼容字段：subject、action、technology、scene、time。
- 补充弱信号字段：event_type、technical_object、mechanism、task、data_modality、method、capability_change、problem_solved、maturity_stage、novelty_signal、adoption_signal、cross_domain_signal、weak_signal_reason、uncertainty、evidence_span、confidence。
- evidence_span 必须是原文中连续出现的证据片段，不能编造；找不到证据片段的事件不要输出。
- subject/action/technology/scene/time 只描述原文明确证据；capability_change、weak_signal_reason、uncertainty 可以是基于证据的审慎推断。
- 一篇文本可拆成多条事件，优先拆分不同技术对象、机制、任务或应用场景。
- confidence 为 0 到 1 的数字。

输出格式示例：
[
  {{
    "event_schema_version": "{WEAK_SIGNAL_EVENT_SCHEMA_VERSION}",
    "event_id": "",
    "subject": "string",
    "action": "string",
    "technology": ["tech1", "tech2"],
    "scene": "string",
    "time": "string",
    "event_type": "research",
    "technical_object": "string",
    "mechanism": "string",
    "task": "string",
    "data_modality": ["video", "sensor"],
    "method": ["retrieval-based"],
    "capability_change": "string",
    "problem_solved": "string",
    "maturity_stage": "lab",
    "novelty_signal": "string",
    "adoption_signal": "string",
    "cross_domain_signal": "string",
    "weak_signal_reason": "string",
    "uncertainty": "string",
    "evidence_span": "原文连续片段",
    "confidence": 0.82
  }}
]
"""


def _weak_signal_batch_prompt(items, batch_size):
    return f"""
你是用于技术预见的弱信号事件抽取器。请从下面多个文本中抽取 0 到 N 条有明确原文证据的技术事件。
只输出纯 JSON 数组，不要添加解释、Markdown、编号或多余文字。

文本列表：
{items}

抽取要求：
- 每条事件必须包含 doc_index，取值为对应文本编号 1 到 {batch_size}。
- 每篇文本可以输出 0 到 N 条事件；不要为了凑数量强行输出事件。
- 保留兼容字段：subject、action、technology、scene、time。
- 补充弱信号字段：event_type、technical_object、mechanism、task、data_modality、method、capability_change、problem_solved、maturity_stage、novelty_signal、adoption_signal、cross_domain_signal、weak_signal_reason、uncertainty、evidence_span、confidence。
- evidence_span 必须是对应原文中连续出现的证据片段，不能编造；找不到证据片段的事件不要输出。
- subject/action/technology/scene/time 只描述原文明确证据；capability_change、weak_signal_reason、uncertainty 可以是基于证据的审慎推断。
- confidence 为 0 到 1 的数字。

输出格式示例：
[
  {{
    "doc_index": 1,
    "event_schema_version": "{WEAK_SIGNAL_EVENT_SCHEMA_VERSION}",
    "event_id": "",
    "subject": "string",
    "action": "string",
    "technology": ["tech1", "tech2"],
    "scene": "string",
    "time": "string",
    "event_type": "research",
    "technical_object": "string",
    "mechanism": "string",
    "task": "string",
    "data_modality": ["video", "sensor"],
    "method": ["retrieval-based"],
    "capability_change": "string",
    "problem_solved": "string",
    "maturity_stage": "lab",
    "novelty_signal": "string",
    "adoption_signal": "string",
    "cross_domain_signal": "string",
    "weak_signal_reason": "string",
    "uncertainty": "string",
    "evidence_span": "原文连续片段",
    "confidence": 0.82
  }}
]
"""


def _event_records_from_parsed(parsed, inherited_doc_index=None):
    records = []
    if isinstance(parsed, dict):
        doc_index = (
            parsed.get("doc_index")
            or parsed.get("document_index")
            or parsed.get("source_index")
            or parsed.get("text_index")
            or inherited_doc_index
        )
        events = parsed.get("events")
        if isinstance(events, list):
            for event in events:
                records.extend(_event_records_from_parsed(event, inherited_doc_index=doc_index))
            return records
        record = dict(parsed)
        if doc_index is not None and not record.get("doc_index"):
            record["doc_index"] = doc_index
        records.append(record)
    elif isinstance(parsed, list):
        for item in parsed:
            records.extend(_event_records_from_parsed(item, inherited_doc_index=inherited_doc_index))
    return records


def _doc_index_hint(event):
    for key in ["doc_index", "document_index", "source_index", "text_index", "input_index", "编号"]:
        value = event.get(key)
        if value is None:
            continue
        match = re.search(r"\d+", str(value))
        if match:
            try:
                return int(match.group(0))
            except ValueError:
                continue
    return None


def _batch_local_index(event, batch_len):
    hint = _doc_index_hint(event)
    if hint is None:
        return None
    if 1 <= hint <= batch_len:
        return hint - 1
    if 0 <= hint < batch_len:
        return hint
    return None


def _strip_extraction_mapping_fields(event):
    event = dict(event or {})
    for field in [
        "doc_index", "document_index", "source_index", "text_index",
        "input_index", "编号", "events", "doc_id",
    ]:
        event.pop(field, None)
    return event


def extract_events_with_api(text):
    prompt = _weak_signal_event_prompt(text)

    try:
        provider, client = get_provider_and_client()
        if client is None or provider is None:
            event = extract_event_simulate(text)
            event["_source_extraction_mode"] = "local"
            return [event]

        extraction_model = os.getenv("EXTRACTION_MODEL", os.getenv("OPENAI_MODEL", "Qwen/Qwen2.5-7B-Instruct"))
        result, usage_info, _ = chat_text(
            prompt,
            model=extraction_model,
            temperature=0.1,
            max_tokens=4000,
            timeout=_event_extraction_single_timeout(),
        )

        if usage_info["prompt_tokens"] or usage_info["completion_tokens"]:
            record_call(
                call_type="事件抽取-单条",
                prompt_tokens=usage_info["prompt_tokens"],
                completion_tokens=usage_info["completion_tokens"],
                success=True,
            )

        parsed = _parse_json_from_response(result)
        if parsed is None:
            print(f"[事件抽取] 无法解析JSON响应: {result[:200]}")
            event = extract_event_simulate(text)
            event["_source_extraction_mode"] = "local"
            return [event]
        records = _event_records_from_parsed(parsed)
        normalized_records = []
        for record in records:
            if not isinstance(record, dict) or not record:
                continue
            event = _strip_extraction_mapping_fields(record)
            _fix_event_dict(event)
            _normalize_event_technologies(event, text, text, source_extraction_mode="api")
            normalized_records.append(event)
        return normalized_records
    except Exception as e:
        print(f"API调用失败: {e}")
        event = extract_event_simulate(text)
        event["_source_extraction_mode"] = "local"
        return [event]


def extract_event_with_api(text):
    events = extract_events_with_api(text)
    return _normalize_event_schema(events[0], source_extraction_mode="api") if events else _normalize_event_schema({})


def batch_extract_event_with_api(texts, batch_size=5):
    results = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        items = "\n\n".join([f"编号{idx + 1}: {t}" for idx, t in enumerate(batch)])
        prompt = _weak_signal_batch_prompt(items, len(batch))
        try:
            provider, client = get_provider_and_client()
            if client is None or provider is None:
                for idx, text in enumerate(batch):
                    event = extract_event_simulate(text)
                    event["_source_text_index"] = start + idx
                    event["_source_extraction_mode"] = "local"
                    results.append(event)
                continue

            extraction_model = os.getenv("EXTRACTION_MODEL", os.getenv("OPENAI_MODEL", "Qwen/Qwen2.5-7B-Instruct"))
            print(f"[事件抽取] 调用API，批次: {start//batch_size + 1}，模型: {extraction_model}")
            last_error = None
            max_attempts = _event_extraction_batch_retries() + 1
            for attempt in range(1, max_attempts + 1):
                try:
                    raw, usage_info, _ = chat_text(
                        prompt,
                        model=extraction_model,
                        temperature=0.1,
                        max_tokens=6000,
                        timeout=_event_extraction_batch_timeout(),
                    )
                    break
                except Exception as attempt_error:
                    last_error = attempt_error
                    if attempt < max_attempts:
                        print(
                            f"[事件抽取] 批次 {start//batch_size + 1} 第 {attempt} 次请求失败: "
                            f"{attempt_error}，准备重试"
                        )
                    else:
                        raise last_error
            print(f"[事件抽取] API响应接收完成，长度: {len(raw)}")

            if usage_info["prompt_tokens"] or usage_info["completion_tokens"]:
                record_call(
                    call_type="事件抽取-批量",
                    prompt_tokens=usage_info["prompt_tokens"],
                    completion_tokens=usage_info["completion_tokens"],
                    success=True,
                )

            extracted = _parse_json_from_response(raw)
            if extracted is None:
                print("[warning] 批量抽取返回内容无法解析，退回逐条抽取")
                print("[debug] 原始响应：", raw[:800])
                for idx, text in enumerate(batch):
                    for event in extract_events_with_api(text):
                        event["_source_text_index"] = start + idx
                        results.append(event)
                continue
            if isinstance(extracted, list) and not extracted:
                print(
                    f"[事件抽取] 批次 {start//batch_size + 1} 返回空数组 []，解析事件数: 0"
                )
                if _retry_empty_batch_enabled() and any(_safe_event_text(text) for text in batch):
                    print("[事件抽取] 空批次启用逐条重试，避免批量模式漏抽")
                    retry_count = 0
                    for idx, text in enumerate(batch):
                        for event in extract_events_with_api(text):
                            event["_source_text_index"] = start + idx
                            results.append(event)
                            retry_count += 1
                    print(f"[事件抽取] 空批次逐条重试完成，补回事件数: {retry_count}")
                continue

            extracted_events = _event_records_from_parsed(extracted)
            if not extracted_events:
                print(
                    f"[事件抽取] 批次 {start//batch_size + 1} 解析后事件数: 0，原始响应预览: {raw[:120]}"
                )
                continue
            print(
                f"[事件抽取] 批次 {start//batch_size + 1} 解析事件数: {len(extracted_events)}"
            )

            all_missing_doc_index = all(_batch_local_index(event, len(batch)) is None for event in extracted_events)
            if all_missing_doc_index and len(extracted_events) == len(batch):
                for idx, event in enumerate(extracted_events):
                    event = _strip_extraction_mapping_fields(event)
                    event["_source_text_index"] = start + idx
                    results.append(event)
                continue
            if all_missing_doc_index:
                print("[warning] 批量抽取返回多事件但缺少 doc_index，退回逐条抽取")
                for idx, text in enumerate(batch):
                    for event in extract_events_with_api(text):
                        event["_source_text_index"] = start + idx
                        results.append(event)
                continue

            unresolved_count = 0
            for event in extracted_events:
                local_index = _batch_local_index(event, len(batch))
                if local_index is None:
                    unresolved_count += 1
                    continue
                event = _strip_extraction_mapping_fields(event)
                event["_source_text_index"] = start + local_index
                results.append(event)
            if unresolved_count:
                print(f"[warning] 批量抽取忽略 {unresolved_count} 条无法映射到原文编号的事件")
        except Exception as e:
            use_local_timeout_fallback = "timed out" in str(e).lower() and _timeout_fallback_to_local_enabled()
            fallback_label = "本地规则兜底" if use_local_timeout_fallback else "逐条抽取"
            print(f"[事件抽取] 批量API调用失败: {e}，退回{fallback_label}")
            record_call(
                call_type="事件抽取-批量",
                prompt_tokens=0,
                completion_tokens=0,
                success=False,
            )
            fallback_count = 0
            for idx, text in enumerate(batch):
                fallback_events = [extract_event_simulate(text)] if use_local_timeout_fallback else extract_events_with_api(text)
                for event in fallback_events:
                    event["_source_text_index"] = start + idx
                    if use_local_timeout_fallback:
                        event["_source_extraction_mode"] = "local"
                    results.append(event)
                    fallback_count += 1
            print(f"[事件抽取] 批次 {start//batch_size + 1} {fallback_label}完成，补回事件数: {fallback_count}")
    return results


def _compact_local_phrase(*values, limit=120):
    parts = []
    for value in values:
        if isinstance(value, (list, tuple, set)):
            parts.extend(str(item).strip() for item in value if str(item).strip())
        else:
            text = str(value or "").strip()
            if text:
                parts.append(text)
    return " ".join(_dedupe_preserve_order(parts))[:limit].strip()


def _local_event_type(text_lower, source_hint=""):
    source_hint = str(source_hint or "").strip().lower()
    if source_hint in {"paper", "patent", "news", "report"}:
        if source_hint == "paper":
            return "research"
        if source_hint == "patent":
            return "patent_application"
        if source_hint == "news":
            return "market_signal"
        if source_hint == "report":
            return "market_signal"
    if any(word in text_lower for word in ["patent", "专利", "申请", "发明"]):
        return "patent_application"
    if any(word in text_lower for word in ["prototype", "原型", "样机", "demo", "试点"]):
        return "prototype"
    if any(word in text_lower for word in ["deploy", "deployment", "落地", "部署", "量产", "commercial"]):
        return "deployment"
    if any(word in text_lower for word in ["investment", "funding", "融资", "投资"]):
        return "investment"
    if any(word in text_lower for word in ["policy", "regulation", "政策", "标准", "指南"]):
        return "policy"
    return "research"


def _local_maturity_stage(text_lower, event_type):
    if any(word in text_lower for word in ["scaled", "量产", "规模化", "commercial", "commercialized"]):
        return "scaled"
    if any(word in text_lower for word in ["deployment", "deployed", "部署", "落地", "early adoption"]):
        return "early_adoption"
    if any(word in text_lower for word in ["pilot", "试点", "示范"]):
        return "pilot"
    if any(word in text_lower for word in ["prototype", "原型", "样机", "demo"]):
        return "prototype"
    if event_type == "patent_application":
        return "idea"
    return "lab"


def _local_weak_signal_reason(
    mechanism_tokens,
    object_tokens,
    data_tokens,
    method_tokens,
    technologies,
    source_hint="",
):
    reasons = []
    if mechanism_tokens and (object_tokens or data_tokens or method_tokens):
        reasons.append("机制与对象/数据/方法约束同时出现，具备候选成形线索")
    if len(_dedupe_preserve_order(technologies)) >= 2:
        reasons.append("涉及多个技术对象，存在组合式变化迹象")
    if data_tokens and method_tokens:
        reasons.append("数据模态与方法特征共同变化，可能对应能力边界变化")
    if str(source_hint or "").strip().lower() in {"paper", "patent"}:
        reasons.append("来源偏早期研究或专利，具备低关注弱信号特征")
    return "；".join(reasons)


def extract_event_simulate(text):
    """模拟事件抽取 - 支持中英文"""
    text_lower = text.lower()
    technologies = extract_technologies(text)
    subject = "未知"
    action = "未知"
    scene = "未知"
    time = "未知"

    # 主体识别（中英文）
    if any(kw in text for kw in ["研究", "实验室", "高校", "大学", "研究院"]):
        subject = "研究机构"
    elif any(kw in text for kw in ["公司", "企业", "集团", "厂商"]):
        subject = "企业"
    elif any(kw in text for kw in ["团队", "项目组", "课题组"]):
        subject = "研究团队"
    elif "research" in text_lower or "study" in text_lower:
        subject = "研究机构"
        action = "training"
    elif "company" in text_lower or "startup" in text_lower:
        subject = "科技公司"
        action = "training"
    elif "university" in text_lower or "lab" in text_lower:
        subject = "实验室"
        action = "planning"

    # 行为识别（中英文）- 使用机制核心词
    if any(kw in text for kw in ["训练", "学习", "优化", "微调"]):
        action = "training"
    elif any(kw in text for kw in ["规划", "计划", "路径规划"]):
        action = "planning"
    elif any(kw in text for kw in ["控制", "操控", "调节", "驱动"]):
        action = "control"
    elif any(kw in text for kw in ["仿真", "模拟", "建模"]):
        action = "simulation"
    elif any(kw in text for kw in ["推理", "推断", "判断"]):
        action = "reasoning"
    elif any(kw in text for kw in ["记忆", "存储", "检索"]):
        action = "memory"
    elif any(kw in text for kw in ["校准", "标定", "校正"]):
        action = "calibration"
    elif any(kw in text for kw in ["压缩", "量化", "蒸馏"]):
        action = "compression"
    elif any(kw in text for kw in ["对齐", "偏好优化"]):
        action = "alignment"
    elif any(kw in text for kw in ["策略", "政策", "方案"]):
        action = "policy"
    # 英文关键词
    elif "train" in text_lower or "training" in text_lower or "optimize" in text_lower:
        action = "training"
    elif "plan" in text_lower or "planning" in text_lower:
        action = "planning"
    elif "control" in text_lower:
        action = "control"
    elif "simulat" in text_lower:
        action = "simulation"
    elif "reason" in text_lower:
        action = "reasoning"
    elif "memory" in text_lower:
        action = "memory"

    # 场景识别（中英文）
    if any(kw in text for kw in ["机器人", "机械臂", "自动化", "智能设备"]):
        scene = "机器人"
    elif any(kw in text for kw in ["医疗", "医院", "诊断", "治疗", "健康", "药物"]):
        scene = "医疗健康"
    elif any(kw in text for kw in ["金融", "银行", "支付", "投资", "证券", "保险"]):
        scene = "金融科技"
    elif any(kw in text for kw in ["汽车", "车辆", "驾驶", "交通", "运输"]):
        scene = "智能交通"
    elif any(kw in text for kw in ["能源", "电力", "电池", "太阳能", "风能"]):
        scene = "能源技术"
    elif any(kw in text for kw in ["通信", "网络", "5G", "6G", "无线"]):
        scene = "通信技术"
    elif any(kw in text for kw in ["芯片", "半导体", "集成电路", "处理器"]):
        scene = "半导体"
    elif any(kw in text for kw in ["人工智能", "AI", "机器学习", "深度学习", "神经网络", "算法", "模型"]):
        scene = "人工智能"
    elif "robot" in text_lower or "automation" in text_lower:
        scene = "机器人应用"
    elif "health" in text_lower or "medical" in text_lower:
        scene = "医疗领域"
    elif "finance" in text_lower or "bank" in text_lower:
        scene = "金融科技"

    # 时间识别
    year_match = re.search(r"\b(20\d{2})\b", text)
    if year_match:
        time = year_match.group(1)

    mechanism_tokens = _dedupe_preserve_order(extract_mechanism_core_tokens(text, action))
    task_tokens = _dedupe_preserve_order(extract_task_constraint_tokens(text, scene))
    object_tokens = _dedupe_preserve_order(extract_object_modifier_tokens(text, scene, " ".join(technologies)))
    data_tokens = _dedupe_preserve_order(extract_data_modifier_tokens(text))
    method_tokens = _dedupe_preserve_order(extract_method_modifier_tokens(text, action))
    event_type = _local_event_type(text_lower)
    maturity_stage = _local_maturity_stage(text_lower, event_type)
    primary_technology = next(
        (tech for tech in technologies if str(tech).strip() and str(tech).strip() != "未知"),
        "",
    )
    technical_object = _compact_local_phrase(
        " ".join(object_tokens[:2]),
        primary_technology,
        limit=80,
    )
    mechanism = _compact_local_phrase(mechanism_tokens[:2], action if action != "未知" else "", limit=80)
    task = _compact_local_phrase(task_tokens[:2], scene if scene != "未知" else "", limit=80)
    evidence_span = _strip_html_noise(text)[:240]
    weak_signal_reason = _local_weak_signal_reason(
        mechanism_tokens,
        object_tokens,
        data_tokens,
        method_tokens,
        technologies,
    )

    return {
        "event_schema_version": WEAK_SIGNAL_EVENT_SCHEMA_VERSION,
        "subject": subject,
        "action": action,
        "technology": technologies,
        "scene": scene,
        "time": time,
        "event_type": event_type,
        "technical_object": technical_object,
        "mechanism": mechanism,
        "task": task,
        "data_modality": data_tokens,
        "method": method_tokens,
        "capability_change": "、".join(_dedupe_preserve_order([task, mechanism])) if task and mechanism else "",
        "problem_solved": task,
        "maturity_stage": maturity_stage,
        "novelty_signal": "、".join(_dedupe_preserve_order(object_tokens + data_tokens + method_tokens)[:4]),
        "adoption_signal": "",
        "cross_domain_signal": "、".join(_dedupe_preserve_order(data_tokens + method_tokens)[:4]),
        "weak_signal_reason": weak_signal_reason,
        "uncertainty": "本地规则回退抽取，需人工复核",
        "evidence_span": evidence_span,
        "confidence": 0.55 if evidence_span else 0.0,
        "mechanism_core_tokens": mechanism_tokens,
        "task_constraint_tokens": task_tokens,
        "object_modifier_tokens": object_tokens,
        "data_modifier_tokens": data_tokens,
        "method_modifier_tokens": method_tokens,
    }


def extract_events(text, use_api=True):
    if use_api and _api_config_available():
        return extract_events_with_api(text)
    return [extract_event_simulate(text)]


def extract_event(text, use_api=True):
    events = extract_events(text, use_api=use_api)
    mode = "api" if use_api and _api_config_available() else "local"
    return _normalize_event_schema(events[0], source_extraction_mode=mode) if events else _normalize_event_schema({})


def assess_weak_signal_event(event, source_row):
    source_type = str(source_row.get("source_type", "unknown")).strip().lower()
    org = str(source_row.get("org", "Unknown")).strip()
    title = str(source_row.get("title", "")).strip()
    technologies = event.get("technology", [])
    if not isinstance(technologies, list):
        technologies = [technologies]
    technologies = [
        str(item).strip()
        for item in technologies
        if str(item).strip() and str(item).strip() != "未知"
    ]

    reasons = []
    traceable = bool(str(source_row.get("url", "")).strip() or str(source_row.get("date", "")).strip())

    low_attention_hint = source_type in {"paper", "patent"}
    if low_attention_hint:
        reasons.append("来源不属于高曝光新闻流，具备低关注度特征")

    org_lower = org.lower()
    niche_actor_hint = bool(org and org != "Unknown" and all(head not in org_lower for head in HEAD_ORGS))
    if niche_actor_hint:
        reasons.append("事件主体不属于典型头部机构，具备小众主体特征")

    specific_techs = [tech for tech in technologies if tech.lower() not in DOMINANT_TECH_TERMS]
    non_dominant_hint = bool(specific_techs)
    if non_dominant_hint:
        reasons.append("技术表述不完全停留在主流泛化术语，具备非主导特征")

    cross_domain_hint = len(technologies) >= 2 or event.get("scene", "未知") != "未知"
    if cross_domain_hint:
        reasons.append("事件同时涉及多个技术/场景信息，存在跨界融合线索")

    traceable_hint = traceable and bool(title)
    if traceable_hint:
        reasons.append("事件具备可追溯载体与时间/链接信息")

    event_score = sum(
        int(flag)
        for flag in [
            low_attention_hint,
            niche_actor_hint,
            non_dominant_hint,
            cross_domain_hint,
            traceable_hint,
        ]
    )

    event["technology"] = technologies if technologies else ["未知"]
    event["low_attention_hint"] = low_attention_hint
    event["niche_actor_hint"] = niche_actor_hint
    event["non_dominant_hint"] = non_dominant_hint
    event["cross_domain_hint"] = cross_domain_hint
    event["traceable_hint"] = traceable_hint
    event["weak_signal_event_score"] = event_score
    event["weak_signal_event_candidate"] = bool(traceable_hint and event_score >= 2)
    event["weak_signal_reasons"] = reasons
    return event


def _prepare_event_for_source_row(event, row, doc_event_index=1, source_extraction_mode="local"):
    event = dict(event or {})
    event["id"] = row.get("id", event.get("id", ""))
    _normalize_event_technologies(
        event,
        row.get("text", ""),
        row.get("title", ""),
        source_extraction_mode=source_extraction_mode,
        source_type=row.get("source_type", ""),
    )
    assess_weak_signal_event(event, row)
    source_date = _source_date_text(row)
    if source_date:
        event["date"] = source_date
        event["event_date"] = source_date
    event["source_type"] = row.get("source_type", "")
    event["title"] = row.get("title", "")
    return _normalize_event_schema(
        event,
        source_row=row,
        doc_event_index=doc_event_index,
        source_extraction_mode=source_extraction_mode,
    )


def process_events(df, use_api=True, batch_size=5, cache_path=None, refresh_cache=False):
    event_columns = _event_cache_columns()
    if df is None or df.empty:
        return pd.DataFrame(columns=event_columns)

    # 确保数据框有必需的列
    if "text" not in df.columns:
        print(f"[ERROR] 数据框缺少 'text' 列，现有列: {list(df.columns)}")
        # 尝试从其他列创建text列
        text_cols = ['content', 'abstract', '标题', '摘要', '内容', 
                    'title', '专利名称', '发明名称', 'name', '专利标题', '名称',
                    'title_cn', 'abstract_first']
        for col in text_cols:
            if col in df.columns:
                df = df.copy()
                df['text'] = df[col].astype(str)
                print(f"[INFO] 使用列 '{col}' 作为 text 列")
                break
        else:
            # 如果没有找到文本列，使用第一列
            df = df.copy()
            df['text'] = df.iloc[:, 0].astype(str)
            print(f"[INFO] 使用第一列 '{df.columns[0]}' 作为 text 列")

    if "id" not in df.columns:
        df = df.copy()
        df['id'] = [f"row_{i}" for i in range(len(df))]
        print("[INFO] 自动生成 id 列")

    source_df = df.reset_index(drop=True)
    expected_ids = source_df["id"].astype(str).tolist() if "id" in source_df.columns else None
    if not refresh_cache:
        cached_df = load_event_cache(cache_path, expected_ids=expected_ids)
        if cached_df is not None and not cached_df.empty:
            return cached_df.reindex(columns=event_columns, fill_value=None)

    if use_api and _api_config_available():
        texts = source_df["text"].tolist()
        print(f"[事件抽取] 使用API，文本数量: {len(texts)}，批次大小: {batch_size}")
        extracted_events = batch_extract_event_with_api(texts, batch_size=batch_size)
        print(f"[事件抽取] API调用完成，事件数量: {len(extracted_events)}")
        normalized_events = []
        per_doc_counts = {}
        max_events_per_doc = _event_extraction_max_events_per_doc()
        min_confidence = _event_extraction_min_confidence()
        events_by_source_index = {}
        for event in extracted_events:
            try:
                source_index = int(event.get("_source_text_index"))
            except (KeyError, TypeError, ValueError):
                print("[warning] 忽略一条缺少来源索引的抽取事件")
                continue
            if source_index < 0 or source_index >= len(source_df):
                print("[warning] 忽略一条来源索引越界的抽取事件")
                continue
            events_by_source_index.setdefault(source_index, []).append(dict(event))

        filtered_total = 0
        trimmed_total = 0
        for source_index, source_events in events_by_source_index.items():
            selected_events, filtered_count, trimmed_count = _filter_events_for_doc(
                source_events,
                max_events_per_doc=max_events_per_doc,
                min_confidence=min_confidence,
            )
            filtered_total += filtered_count
            trimmed_total += trimmed_count
            if filtered_count or trimmed_count:
                print(
                    f"[事件抽取] 文档{source_index} 控制多事件: "
                    f"保留{len(selected_events)}条，低置信过滤{filtered_count}条，上限截断{trimmed_count}条"
                )
            for event in selected_events:
                event.pop("_source_text_index", None)
                source_mode = event.pop("_source_extraction_mode", "api")
                row = source_df.iloc[source_index]
                source_id = str(row.get("id", ""))
                per_doc_counts[source_id] = per_doc_counts.get(source_id, 0) + 1
                normalized_events.append(
                    _prepare_event_for_source_row(
                        event,
                        row,
                        doc_event_index=per_doc_counts[source_id],
                        source_extraction_mode=source_mode,
                    )
                )
        if filtered_total or trimmed_total:
            print(f"[事件抽取] 多事件控制汇总: 低置信过滤{filtered_total}条，上限截断{trimmed_total}条")
        events_df = pd.DataFrame(normalized_events).reindex(columns=event_columns, fill_value=None)
        save_event_cache(
            cache_path,
            events_df,
            metadata={
                "mode": "api",
                "batch_size": batch_size,
                "rows": len(events_df),
                "source_ids": expected_ids or [],
                "event_schema_version": WEAK_SIGNAL_EVENT_SCHEMA_VERSION,
                "max_events_per_doc": max_events_per_doc,
                "min_confidence": min_confidence,
            },
        )
        return events_df

    events = []
    per_doc_counts = {}
    max_events_per_doc = _event_extraction_max_events_per_doc()
    min_confidence = _event_extraction_min_confidence()
    filtered_total = 0
    trimmed_total = 0
    for _, row in source_df.iterrows():
        source_id = str(row.get("id", ""))
        doc_events, filtered_count, trimmed_count = _filter_events_for_doc(
            extract_events(row["text"], use_api=False),
            max_events_per_doc=max_events_per_doc,
            min_confidence=min_confidence,
        )
        filtered_total += filtered_count
        trimmed_total += trimmed_count
        for event in doc_events:
            per_doc_counts[source_id] = per_doc_counts.get(source_id, 0) + 1
            events.append(
                _prepare_event_for_source_row(
                    event,
                    row,
                    doc_event_index=per_doc_counts[source_id],
                    source_extraction_mode="local",
                )
            )
    if filtered_total or trimmed_total:
        print(f"[事件抽取] 多事件控制汇总: 低置信过滤{filtered_total}条，上限截断{trimmed_total}条")
    events_df = pd.DataFrame(events).reindex(columns=event_columns, fill_value=None)
    save_event_cache(
        cache_path,
        events_df,
        metadata={
            "mode": "api" if use_api and _api_config_available() else "local",
            "batch_size": batch_size,
            "rows": len(events_df),
            "source_ids": expected_ids or [],
            "event_schema_version": WEAK_SIGNAL_EVENT_SCHEMA_VERSION,
            "max_events_per_doc": max_events_per_doc,
            "min_confidence": min_confidence,
        },
    )
    return events_df


def summarize_events(events_df):
    """生成展示版可直接使用的事件层摘要。"""
    if events_df is None or events_df.empty:
        return {
            "event_count": 0,
            "technology_hit_rate": 0.0,
            "scene_hit_rate": 0.0,
            "weak_signal_candidate_ratio": 0.0,
            "low_attention_ratio": 0.0,
            "niche_actor_ratio": 0.0,
            "non_dominant_ratio": 0.0,
            "traceable_ratio": 0.0,
            "known_subject_ratio": 0.0,
            "known_action_ratio": 0.0,
        }

    def _known_ratio(series):
        cleaned = series.astype(str).str.strip()
        return round((cleaned != "未知").mean(), 2)

    technology_hit_rate = round(
        events_df["technology"].apply(
            lambda items: bool(items and any(str(item).strip() and str(item).strip() != "未知" for item in items))
            if isinstance(items, list) else str(items).strip() not in {"", "未知"}
        ).mean(),
        2,
    ) if "technology" in events_df.columns else 0.0

    scene_hit_rate = _known_ratio(events_df["scene"]) if "scene" in events_df.columns else 0.0
    known_subject_ratio = _known_ratio(events_df["subject"]) if "subject" in events_df.columns else 0.0
    known_action_ratio = _known_ratio(events_df["action"]) if "action" in events_df.columns else 0.0
    weak_signal_candidate_ratio = round(events_df.get("weak_signal_event_candidate", pd.Series(dtype=bool)).fillna(False).mean(), 2)
    low_attention_ratio = round(events_df.get("low_attention_hint", pd.Series(dtype=bool)).fillna(False).mean(), 2)
    niche_actor_ratio = round(events_df.get("niche_actor_hint", pd.Series(dtype=bool)).fillna(False).mean(), 2)
    non_dominant_ratio = round(events_df.get("non_dominant_hint", pd.Series(dtype=bool)).fillna(False).mean(), 2)
    traceable_ratio = round(events_df.get("traceable_hint", pd.Series(dtype=bool)).fillna(False).mean(), 2)

    return {
        "event_count": int(len(events_df)),
        "technology_hit_rate": technology_hit_rate,
        "scene_hit_rate": scene_hit_rate,
        "weak_signal_candidate_ratio": weak_signal_candidate_ratio,
        "low_attention_ratio": low_attention_ratio,
        "niche_actor_ratio": niche_actor_ratio,
        "non_dominant_ratio": non_dominant_ratio,
        "traceable_ratio": traceable_ratio,
        "known_subject_ratio": known_subject_ratio,
        "known_action_ratio": known_action_ratio,
    }
