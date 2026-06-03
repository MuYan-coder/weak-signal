import json
import os
import re
import hashlib
import time
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from ..utils.api_stats import record_call
from ..utils.env_config import ensure_env_loaded
from ..utils.llm_client import chat_text, get_provider_and_client
from .event_schema import (
    EVENT_LLM_JSON_FIELDS as _EVENT_JSON_FIELDS,
    WEAK_SIGNAL_EVENT_SCHEMA_VERSION,
    clean_event_list as _clean_event_list,
    clean_event_text as _clean_event_text,
    coerce_confidence as _coerce_confidence,
    event_cache_columns as _event_cache_columns,
    normalize_event_schema as _normalize_event_schema,
    normalize_events_dataframe as _normalize_events_dataframe,
    normalize_subject as _normalize_subject,
    safe_event_text as _safe_event_text,
    source_date_text as _source_date_text,
)
from .tech_lexicon import (
    BROAD_TECH_TERMS,
    DOMINANT_TECH_TERMS,
    OBSERVATION_SCOPE_SET,
    aliases_for,
    build_domain_lexicon,
    canonicalize_term,
    discover_candidate_terms,
    has_non_scope_constraint,
    is_bare_mechanism_candidate,
    normalize_signal_phrase,
    normalize_proxy_token,
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
    event["technology"] = _clean_event_list(event.get("technology")) or ["未知"]
    event["subject"] = _normalize_subject(event.get("subject"))
    for field in ["subject", "action", "scene", "time"]:
        if not _clean_event_text(event.get(field)):
            event[field] = "未知"
    event["event_schema_version"] = (
        _clean_event_text(event.get("event_schema_version"))
        or WEAK_SIGNAL_EVENT_SCHEMA_VERSION
    )
    for field in ["data_modality", "method"]:
        event[field] = _clean_event_list(event.get(field))
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
        value = int(os.getenv("EVENT_EXTRACTION_BATCH_RETRIES", "3"))
    except (TypeError, ValueError):
        value = 3
    return max(0, value)


def _event_extraction_concurrency():
    try:
        value = int(os.getenv("EVENT_EXTRACTION_CONCURRENCY", "2"))
    except (TypeError, ValueError):
        value = 2
    return max(1, value)


def _event_extraction_batch_size():
    try:
        value = int(os.getenv("EVENT_EXTRACTION_BATCH_SIZE", "5"))
    except (TypeError, ValueError):
        value = 5
    return max(1, value)




def _timeout_fallback_to_local_enabled():
    value = str(os.getenv("EVENT_EXTRACTION_TIMEOUT_FALLBACK_TO_LOCAL", "1")).strip().lower()
    return value not in {"0", "false", "no", "off"}


def _event_selection_key(index_event):
    index, event = index_event
    return (
        _coerce_confidence(event.get("confidence"), default=0.0),
        1 if _clean_event_text(event.get("evidence_span")) else 0,
        1 if any(
            _clean_event_text(event.get(field))
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

def _get_doc_level_cache_dir(cache_path):
    cache_file = _normalize_cache_path(cache_path)
    if cache_file is None:
        return None
    return cache_file.parent / "doc_level" / cache_file.stem


def _load_doc_cache(doc_hash, doc_level_dir):
    if not doc_level_dir or not doc_level_dir.exists():
        return None
    cache_file = doc_level_dir / f"{doc_hash}.json"
    if not cache_file.exists():
        return None
    try:
        events = json.loads(cache_file.read_text(encoding="utf-8"))
        if isinstance(events, list):
            return events
    except Exception:
        pass
    return None


def _save_doc_cache(doc_hash, events, doc_level_dir):
    if not doc_level_dir:
        return
    try:
        doc_level_dir.mkdir(parents=True, exist_ok=True)
        cache_file = doc_level_dir / f"{doc_hash}.json"
        cache_file.write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"[warning] 写入文档级缓存失败: {e}")


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


def _supports_scope_context(snippet: str, scopes, domain_lexicon=None):
    haystack = normalize_signal_phrase(snippet)
    if not haystack:
        return False
    for scope in scopes:
        scope_aliases = domain_lexicon.aliases_for(scope) if domain_lexicon is not None else aliases_for(scope)
        for alias in scope_aliases:
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


def _analysis_scope_from_row(row, domain_lexicon=None) -> str:
    if row is None:
        return ""
    for field in ["analysis_tech_field_name", "tech_field_name", "analysis_domain", "selected_domain"]:
        try:
            value = row.get(field, "")
        except AttributeError:
            value = ""
        text = _clean_event_text(value)
        if not text or text == "未知":
            continue
        canonical = domain_lexicon.canonicalize_term(text) if domain_lexicon is not None else canonicalize_term(text)
        return canonical if canonical and canonical != "未知" else text
    return ""


def _freeform_candidate_token(value, *, allow_generic_action=False, max_len=80) -> str:
    text = _clean_event_text(value)
    if not text or text == "未知":
        return ""
    text = _strip_html_noise(text)
    text = re.sub(r"[\[\]{}<>]", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" ，,。.;；:：")
    if not text or len(text) > max_len:
        return ""
    generic_actions = {
        "提出", "研发", "研制", "设计", "发布", "开源", "测试", "验证",
        "propose", "proposed", "develop", "developed", "design", "designed",
        "release", "released", "test", "tested", "validate", "validated",
    }
    if not allow_generic_action and text.lower() in generic_actions:
        return ""
    return text


def _looks_formable_candidate(
    mechanism_tokens,
    task_tokens,
    object_tokens,
    data_tokens,
    scene_tokens,
    method_tokens,
    strict_mode=True,
    domain_lexicon=None,
):
    if not mechanism_tokens:
        if domain_lexicon is not None and not domain_lexicon.use_legacy_robot_rules:
            return domain_lexicon.valid_candidate_pattern_matches(
                task_tokens=task_tokens,
                object_tokens=object_tokens,
                data_tokens=data_tokens,
                scene_tokens=scene_tokens,
                mechanism_tokens=mechanism_tokens,
                method_tokens=method_tokens,
                evidence_present=True,
            )
        return False

    # 宽松模式：只要有机制核心词就认为可以成形
    if not strict_mode:
        return True

    if domain_lexicon is not None and not domain_lexicon.use_legacy_robot_rules:
        return domain_lexicon.has_non_scope_constraint(
            task_tokens=task_tokens,
            object_tokens=object_tokens,
            data_tokens=data_tokens,
            scene_tokens=scene_tokens,
            mechanism_tokens=mechanism_tokens,
            method_tokens=method_tokens,
        )

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
    domain_lexicon=None,
):
    domain_lexicon = domain_lexicon or build_domain_lexicon(None)
    raw_candidate_text = normalize_signal_phrase(raw_text)
    if not raw_candidate_text or not scope_names:
        return None

    mechanism_core_tokens = _dedupe_preserve_order(
        mechanism_tokens or domain_lexicon.extract_mechanism_core_tokens(action_text, raw_candidate_text)
    )
    task_constraint_tokens = _dedupe_preserve_order(
        task_tokens or domain_lexicon.extract_task_constraint_tokens(scene_text, raw_candidate_text)
    )
    scene_tokens = _dedupe_preserve_order(
        scene_tokens or domain_lexicon.extract_scene_tokens(scene_text, raw_candidate_text)
    )
    object_modifier_tokens = _dedupe_preserve_order(
        object_tokens or domain_lexicon.extract_object_modifier_tokens(scene_text, action_text, raw_candidate_text)
    )
    data_modifier_tokens = _dedupe_preserve_order(
        data_tokens or domain_lexicon.extract_data_modifier_tokens(scene_text, action_text, raw_candidate_text)
    )
    method_modifier_tokens = _dedupe_preserve_order(
        method_tokens or domain_lexicon.extract_method_modifier_tokens(scene_text, action_text, raw_candidate_text)
    )
    action_tokens = _dedupe_preserve_order(domain_lexicon.extract_mechanism_core_tokens(action_text, raw_candidate_text))
    is_scope_echo = domain_lexicon.is_scope_echo_candidate(
        raw_candidate_text,
        scope_names,
        mechanism_tokens=mechanism_core_tokens,
        task_tokens=task_constraint_tokens,
    )
    has_mechanism_core = bool(mechanism_core_tokens)
    has_task_constraint = bool(task_constraint_tokens)
    has_non_scope_info = domain_lexicon.has_non_scope_constraint(
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
    domain_lexicon=None,
):
    domain_lexicon = domain_lexicon or build_domain_lexicon(None)
    title_text = _strip_html_noise(title_text)
    fallback_text = _strip_html_noise(fallback_text)
    combined_text = " ".join(
        part for part in [str(title_text or "").strip(), str(fallback_text or "").strip()] if part
    )
    observation_scopes = observation_scopes or domain_lexicon.detect_supported_observation_scopes(combined_text)
    if not observation_scopes:
        return []
    robot_scopes = {"humanoid robot", "embodied intelligence"}
    if (
        domain_lexicon.use_legacy_robot_rules
        and robot_scopes.intersection(observation_scopes)
        and not domain_lexicon.has_robot_domain_anchor(combined_text)
    ):
        return []

    technologies = event.get("technology", [])
    if not isinstance(technologies, list):
        technologies = [technologies]
    non_scope_technologies = []
    for technology in technologies:
        canonical = domain_lexicon.canonicalize_term(technology)
        if canonical == "未知" or canonical.lower() == "unknown":
            continue
        if domain_lexicon.use_legacy_robot_rules and (canonical in OBSERVATION_SCOPE_SET or canonical in BROAD_TECH_TERMS):
            continue
        if not domain_lexicon.use_legacy_robot_rules and (
            domain_lexicon.is_observation_scope(canonical) or domain_lexicon.is_generic_or_shell(canonical)
        ):
            continue
        non_scope_technologies.append(canonical)
    non_scope_technologies = _dedupe_preserve_order(non_scope_technologies)

    action_text = str(event.get("action", "")).strip()
    scene_text = str(event.get("scene", "")).strip()
    fallback_mechanisms = _dedupe_preserve_order(
        domain_lexicon.extract_mechanism_core_tokens(action_text, title_text, fallback_text)
    )
    fallback_tasks = _dedupe_preserve_order(
        domain_lexicon.extract_task_constraint_tokens(scene_text, title_text, fallback_text)
    )
    fallback_scenes = _dedupe_preserve_order(
        domain_lexicon.extract_scene_tokens(scene_text, title_text, fallback_text)
    )
    fallback_object_tokens = _dedupe_preserve_order(
        domain_lexicon.extract_object_modifier_tokens(scene_text, title_text, fallback_text, *non_scope_technologies)
    )
    fallback_data_tokens = _dedupe_preserve_order(
        domain_lexicon.extract_data_modifier_tokens(scene_text, title_text, fallback_text)
    )
    fallback_method_tokens = _dedupe_preserve_order(
        domain_lexicon.extract_method_modifier_tokens(action_text, scene_text, title_text, fallback_text)
    )

    units = []
    seen = set()
    for snippet in _split_candidate_snippets(title_text, fallback_text):
        snippet_mechanisms = _dedupe_preserve_order(domain_lexicon.extract_mechanism_core_tokens(action_text, snippet))
        snippet_tasks = _dedupe_preserve_order(domain_lexicon.extract_task_constraint_tokens(scene_text, snippet))
        snippet_scenes = _dedupe_preserve_order(domain_lexicon.extract_scene_tokens(scene_text, snippet))
        snippet_object_tokens = _dedupe_preserve_order(
            domain_lexicon.extract_object_modifier_tokens(scene_text, snippet, *non_scope_technologies)
        )
        snippet_data_tokens = _dedupe_preserve_order(
            domain_lexicon.extract_data_modifier_tokens(scene_text, snippet)
        )
        snippet_method_tokens = _dedupe_preserve_order(
            domain_lexicon.extract_method_modifier_tokens(action_text, scene_text, snippet)
        )
        has_scope_context = _supports_scope_context(snippet, observation_scopes, domain_lexicon=domain_lexicon)
        if str(scope_match_mode or "").startswith("analysis_field"):
            has_scope_context = True
        if source_type == "patent" and scope_match_mode == "proxy_patent":
            has_scope_context = True

        if not snippet_mechanisms and not domain_lexicon.valid_candidate_pattern_matches(
            task_tokens=snippet_tasks,
            object_tokens=snippet_object_tokens,
            data_tokens=snippet_data_tokens,
            scene_tokens=snippet_scenes,
            mechanism_tokens=snippet_mechanisms,
            method_tokens=snippet_method_tokens,
            evidence_present=True,
        ):
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
            domain_lexicon=domain_lexicon,
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
            domain_lexicon=domain_lexicon,
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
            domain_lexicon=domain_lexicon,
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
                domain_lexicon=domain_lexicon,
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
    analysis_scope="",
    domain_lexicon=None,
):
    domain_lexicon = domain_lexicon or build_domain_lexicon(None)
    technologies = _clean_event_list(event.get("technology"))

    title_text = _strip_html_noise(title_text)
    fallback_text = _strip_html_noise(fallback_text)
    combined_text = " ".join(str(part) for part in [title_text, fallback_text] if str(part).strip())
    normalized = domain_lexicon.normalize_technologies(technologies, fallback_text=combined_text)
    discovered_terms = discover_candidate_terms(combined_text, limit=4) if domain_lexicon.use_legacy_robot_rules else []

    if domain_lexicon.use_legacy_robot_rules and normalized and all(item in DOMINANT_TECH_TERMS for item in normalized if item != "未知"):
        for term in discovered_terms:
            if term not in normalized and term not in DOMINANT_TECH_TERMS:
                normalized.append(term)

    event["technology"] = normalized
    source_type = str(source_type or "").strip().lower()
    fallback_mechanisms = _dedupe_preserve_order(
        domain_lexicon.extract_mechanism_core_tokens(title_text, fallback_text, " ".join(normalized))
    )
    fallback_tasks = _dedupe_preserve_order(
        domain_lexicon.extract_task_constraint_tokens(title_text, fallback_text)
    )
    fallback_scenes = _dedupe_preserve_order(
        domain_lexicon.extract_scene_tokens(title_text, fallback_text)
    )
    fallback_object_tokens = _dedupe_preserve_order(
        domain_lexicon.extract_object_modifier_tokens(title_text, fallback_text, " ".join(normalized))
    )
    if not fallback_object_tokens:
        object_fallback = _freeform_candidate_token(event.get("technical_object"), max_len=90)
        if not object_fallback:
            object_fallback = next(
                (
                    _freeform_candidate_token(item, max_len=80)
                    for item in normalized
                    if str(item or "").strip() and str(item).strip() != "未知"
                ),
                "",
            )
        if object_fallback:
            fallback_object_tokens = [object_fallback]
    fallback_data_tokens = _dedupe_preserve_order(
        domain_lexicon.extract_data_modifier_tokens(title_text, fallback_text)
    )
    fallback_method_tokens = _dedupe_preserve_order(
        domain_lexicon.extract_method_modifier_tokens(title_text, fallback_text)
    )
    if not fallback_mechanisms:
        mechanism_fallback = (
            _freeform_candidate_token(event.get("mechanism"), max_len=80)
            or _freeform_candidate_token(event.get("capability_change"), max_len=80)
            or _freeform_candidate_token(event.get("action"), allow_generic_action=True, max_len=24)
        )
        if mechanism_fallback:
            fallback_mechanisms = [mechanism_fallback]
    if not _clean_event_list(event.get("mechanism_core_tokens")):
        event["mechanism_core_tokens"] = fallback_mechanisms
    if not _clean_event_list(event.get("task_constraint_tokens")):
        event["task_constraint_tokens"] = fallback_tasks
    if not _clean_event_list(event.get("object_modifier_tokens")):
        event["object_modifier_tokens"] = fallback_object_tokens
    if not _clean_event_list(event.get("data_modifier_tokens")):
        event["data_modifier_tokens"] = fallback_data_tokens
    if not _clean_event_list(event.get("method_modifier_tokens")):
        event["method_modifier_tokens"] = fallback_method_tokens
    if not _clean_event_text(event.get("technical_object")) and fallback_object_tokens:
        event["technical_object"] = " ".join(fallback_object_tokens[:2])
    if not _clean_event_text(event.get("mechanism")) and fallback_mechanisms:
        event["mechanism"] = " ".join(fallback_mechanisms[:2])
    if not _clean_event_text(event.get("task")) and fallback_tasks:
        event["task"] = " ".join(fallback_tasks[:2])
    if not _clean_event_list(event.get("data_modality")) and fallback_data_tokens:
        event["data_modality"] = fallback_data_tokens[:3]
    if not _clean_event_list(event.get("method")) and fallback_method_tokens:
        event["method"] = fallback_method_tokens[:3]
    scope_diag = domain_lexicon.diagnose_observation_scope_detection(
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
    analysis_scope = _freeform_candidate_token(analysis_scope, max_len=80)
    if analysis_scope:
        observation_scopes = [analysis_scope]
        scope_match_mode = "analysis_field" if not scope_match_mode else f"analysis_field+{scope_match_mode}"

    candidate_units = _extract_candidate_units(
        event,
        title_text=title_text,
        fallback_text=fallback_text,
        source_extraction_mode=source_extraction_mode,
        source_type=source_type,
        observation_scopes=observation_scopes,
        scope_match_mode=scope_match_mode or "explicit",
        domain_lexicon=domain_lexicon,
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
  * subject (主语)：实施该事件的具体主体机构、团队、人员或文献/专利发明人/申请人，例如：“麻省理工团队”、“特斯拉”、“上海海事大学”。必须是原文或来源元数据中可核验的真实主体；如果只有“我们/本文/本发明/该方法/该系统/研究人员/日本科学家”等代词、泛称、国别+角色代称或无法确认主体，请填“未知”。**禁止将具体的技术概念、产品名词、系统/方法/模型/算法/装置名称填入该字段。**
  * action (动作)：描述该技术事件的研发/学术动作的**核心动词**，通常为：“研发”、“研制”、“提出”、“设计”、“发布”、“开源”、“测试”、“验证”等。**绝对禁止将“全身反应规划控制”、“运动规划”、“控制系统”、“高速跑酷导航”等技术类目、任务短语或技术名词填入 action 字段。**
  * technology (技术)：事件涉及的核心技术，需为列表。
  * scene (场景)：应用场景。
  * time (时间)：事件发生的明确时间。
- 补充弱信号字段：event_type、technical_object、mechanism、task、data_modality、method、capability_change、problem_solved、maturity_stage、novelty_signal、adoption_signal、cross_domain_signal、weak_signal_reason、uncertainty、evidence_span、confidence。
- technical_object 必须是原文中的具体技术对象（材料、器件、系统、方法或结构）；mechanism 是作用机制/关键原理；task 是技术任务或性能目标；data_modality 和 method 必须是列表。没有证据的补充字段填空字符串或空数组。
- evidence_span 必须是原文中连续出现的证据片段，不能编造；找不到证据片段的事件不要输出。
- subject/action/technology/scene/time 只描述原文明确证据；subject 无明确证据时必须为“未知”；capability_change、weak_signal_reason、uncertainty 可以是基于证据的审慎推断。
- 所有字段都禁止输出 string、text、value、placeholder、example、sample、null、N/A、待填、示例、占位符等占位值。
- 一篇文本可拆成多条事件，优先拆分不同技术对象、机制、任务或应用场景。
- confidence 为 0 到 1 的数字。

输出格式示例：
[
  {{
    "event_schema_version": "{WEAK_SIGNAL_EVENT_SCHEMA_VERSION}",
    "event_id": "",
    "subject": "上海交通大学团队",
    "action": "提出",
    "technology": ["钙钛矿薄膜", "界面钝化材料"],
    "scene": "柔性电子器件",
    "time": "2026年",
    "event_type": "research",
    "technical_object": "钙钛矿薄膜界面钝化材料",
    "mechanism": "界面缺陷钝化",
    "task": "提升器件长期稳定性",
    "data_modality": ["电化学测试数据", "结构表征数据"],
    "method": ["掺杂改性", "界面工程"],
    "capability_change": "提高材料导电性与稳定性",
    "problem_solved": "降低界面缺陷导致的性能衰减",
    "maturity_stage": "lab",
    "novelty_signal": "新型掺杂体系出现早期实验验证",
    "adoption_signal": "论文或专利披露初步样品制备",
    "cross_domain_signal": "材料改性方法迁移到电子器件",
    "weak_signal_reason": "具体材料对象、机制和性能任务同时出现",
    "uncertainty": "仍需更多实验重复与规模化验证",
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
  * subject (主语)：实施该事件的具体主体机构、团队、人员或文献/专利发明人/申请人，例如：“麻省理工团队”、“特斯拉”、“上海海事大学”。必须是原文或来源元数据中可核验的真实主体；如果只有“我们/本文/本发明/该方法/该系统/研究人员/日本科学家”等代词、泛称、国别+角色代称或无法确认主体，请填“未知”。**禁止将具体的技术概念、产品名词、系统/方法/模型/算法/装置名称填入该字段。**
  * action (动作)：描述该技术事件的研发/学术动作的**核心动词**，通常为：“研发”、“研制”、“提出”、“设计”、“发布”、“开源”、“测试”、“验证”等。**绝对禁止将“全身反应规划控制”、“运动规划”、“控制系统”、“高速跑酷导航”等技术类目、任务短语或技术名词填入 action 字段。**
  * technology (技术)：事件涉及的核心技术，需为列表。
  * scene (场景)：应用场景。
  * time (时间)：事件发生的明确时间。
- 补充弱信号字段：event_type、technical_object、mechanism、task、data_modality、method、capability_change、problem_solved、maturity_stage、novelty_signal、adoption_signal、cross_domain_signal、weak_signal_reason、uncertainty、evidence_span、confidence。
- technical_object 必须是原文中的具体技术对象（材料、器件、系统、方法或结构）；mechanism 是作用机制/关键原理；task 是技术任务或性能目标；data_modality 和 method 必须是列表。没有证据的补充字段填空字符串或空数组。
- evidence_span 必须是对应原文中连续出现的证据片段，不能编造；找不到证据片段的事件不要输出。
- subject/action/technology/scene/time 只描述原文明确证据；subject 无明确证据时必须为“未知”；capability_change、weak_signal_reason、uncertainty 可以是基于证据的审慎推断。
- 所有字段都禁止输出 string、text、value、placeholder、example、sample、null、N/A、待填、示例、占位符等占位值。
- confidence 为 0 到 1 的数字。

输出格式示例：
[
  {{
    "doc_index": 1,
    "event_schema_version": "{WEAK_SIGNAL_EVENT_SCHEMA_VERSION}",
    "event_id": "",
    "subject": "上海交通大学团队",
    "action": "提出",
    "technology": ["钙钛矿薄膜", "界面钝化材料"],
    "scene": "柔性电子器件",
    "time": "2026年",
    "event_type": "research",
    "technical_object": "钙钛矿薄膜界面钝化材料",
    "mechanism": "界面缺陷钝化",
    "task": "提升器件长期稳定性",
    "data_modality": ["电化学测试数据", "结构表征数据"],
    "method": ["掺杂改性", "界面工程"],
    "capability_change": "提高材料导电性与稳定性",
    "problem_solved": "降低界面缺陷导致的性能衰减",
    "maturity_stage": "lab",
    "novelty_signal": "新型掺杂体系出现早期实验验证",
    "adoption_signal": "论文或专利披露初步样品制备",
    "cross_domain_signal": "材料改性方法迁移到电子器件",
    "weak_signal_reason": "具体材料对象、机制和性能任务同时出现",
    "uncertainty": "仍需更多实验重复与规模化验证",
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

        result, usage_info, _ = chat_text(
            prompt,
            system="你是用于技术预见的弱信号事件抽取器。请严格按照要求抽取技术事件并输出合法的 JSON 数组，必须保证 JSON 键名拼写完全正确，不要输出 Markdown 标记或任何解释性文字。",
            model=extraction_model,
            temperature=0.3,
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


def _extract_batch_worker(batch, start, batch_idx, total_batches, extraction_model):
    items = "\n\n".join([f"编号{idx + 1}: {t}" for idx, t in enumerate(batch)])
    prompt = _weak_signal_batch_prompt(items, len(batch))

    provider, client = get_provider_and_client()
    if client is None or provider is None:
        batch_results = []
        for idx, text in enumerate(batch):
            event = extract_event_simulate(text)
            event["_source_text_index"] = start + idx
            event["_source_extraction_mode"] = "local"
            batch_results.append(event)
        return batch_results

    print(f"[事件抽取] 调用API，批次: {batch_idx}/{total_batches}，模型: {extraction_model}")
    last_error = None
    max_attempts = _event_extraction_batch_retries() + 1

    raw = None
    usage_info = {"prompt_tokens": 0, "completion_tokens": 0}
    for attempt in range(1, max_attempts + 1):
        try:
            raw, usage_info, _ = chat_text(
                prompt,
                system="你是用于技术预见的弱信号事件抽取器。请严格按照要求抽取技术事件并输出合法的 JSON 数组，必须保证 JSON 键名拼写完全正确，不要输出 Markdown 标记或任何解释性文字。",
                model=extraction_model,
                temperature=0.3,
                max_tokens=6000,
                timeout=_event_extraction_batch_timeout(),
            )
            break
        except Exception as attempt_error:
            last_error = attempt_error
            if attempt < max_attempts:
                sleep_time = (2.0 ** attempt) + random.uniform(0.5, 1.5)
                print(
                    f"[事件抽取] 批次 {batch_idx}/{total_batches} 第 {attempt} 次请求失败: "
                    f"{attempt_error}，将在 {sleep_time:.2f} 秒后重试"
                )
                time.sleep(sleep_time)
            else:
                raise last_error

    print(f"[事件抽取] API响应接收完成，批次 {batch_idx}/{total_batches}，长度: {len(raw) if raw else 0}")

    if usage_info.get("prompt_tokens") or usage_info.get("completion_tokens"):
        record_call(
            call_type="事件抽取-批量",
            prompt_tokens=usage_info.get("prompt_tokens", 0),
            completion_tokens=usage_info.get("completion_tokens", 0),
            success=True,
        )

    if not raw:
        raise ValueError("API returned empty raw content")

    extracted = _parse_json_from_response(raw)
    batch_results = []

    if extracted is None:
        print(f"[warning] 批量抽取第 {batch_idx} 批次返回内容无法解析，退回逐条抽取")
        print(f"[debug] 原始响应预览：{raw[:800] if raw else 'None'}")
        for idx, text in enumerate(batch):
            for event in extract_events_with_api(text):
                event["_source_text_index"] = start + idx
                batch_results.append(event)
        return batch_results

    if isinstance(extracted, list) and not extracted:
        print(f"[事件抽取] 批次 {batch_idx}/{total_batches} 返回空数组 []")
        if _retry_empty_batch_enabled() and any(_safe_event_text(text) for text in batch):
            print(f"[事件抽取] 批次 {batch_idx} 空批次启用逐条重试，避免批量模式漏抽")
            for idx, text in enumerate(batch):
                for event in extract_events_with_api(text):
                    event["_source_text_index"] = start + idx
                    batch_results.append(event)
        return batch_results

    extracted_events = _event_records_from_parsed(extracted)
    if not extracted_events:
        print(f"[事件抽取] 批次 {batch_idx}/{total_batches} 解析后事件数: 0")
        return batch_results

    all_missing_doc_index = all(_batch_local_index(event, len(batch)) is None for event in extracted_events)
    if all_missing_doc_index and len(extracted_events) == len(batch):
        for idx, event in enumerate(extracted_events):
            event = _strip_extraction_mapping_fields(event)
            event["_source_text_index"] = start + idx
            batch_results.append(event)
        return batch_results

    if all_missing_doc_index:
        print(f"[warning] 批量抽取第 {batch_idx} 批次返回多事件但缺少 doc_index，退回逐条抽取")
        for idx, text in enumerate(batch):
            for event in extract_events_with_api(text):
                event["_source_text_index"] = start + idx
                batch_results.append(event)
        return batch_results

    for event in extracted_events:
        local_index = _batch_local_index(event, len(batch))
        if local_index is None:
            continue
        event = _strip_extraction_mapping_fields(event)
        event["_source_text_index"] = start + local_index
        batch_results.append(event)

    return batch_results


def _extract_batch_worker_wrapper(batch, start, batch_idx, total_batches, extraction_model):
    try:
        return _extract_batch_worker(batch, start, batch_idx, total_batches, extraction_model)
    except Exception as e:
        use_local_timeout_fallback = "timed out" in str(e).lower() and _timeout_fallback_to_local_enabled()
        fallback_label = "本地规则兜底" if use_local_timeout_fallback else "逐条抽取"
        print(f"[事件抽取] 批次 {batch_idx}/{total_batches} 批量API调用发生异常: {e}，退回 {fallback_label}")

        record_call(
            call_type="事件抽取-批量",
            prompt_tokens=0,
            completion_tokens=0,
            success=False,
        )

        batch_results = []
        for idx, text in enumerate(batch):
            fallback_events = []
            single_err = None
            if use_local_timeout_fallback:
                fallback_events = [extract_event_simulate(text)]
            else:
                try:
                    fallback_events = extract_events_with_api(text)
                except Exception as single_err_exc:
                    single_err = single_err_exc
                    print(f"[事件抽取] 逐条降级请求也失败: {single_err}，强制使用本地规则兜底")
                    fallback_events = [extract_event_simulate(text)]

            for event in fallback_events:
                event["_source_text_index"] = start + idx
                if use_local_timeout_fallback or single_err:
                    event["_source_extraction_mode"] = "local"
                batch_results.append(event)
        return batch_results


def batch_extract_event_with_api(texts, batch_size=5):
    results = []
    concurrency = _event_extraction_concurrency()
    extraction_model = os.getenv("EXTRACTION_MODEL", os.getenv("OPENAI_MODEL", "Qwen/Qwen2.5-7B-Instruct"))

    batches = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        batches.append((batch, start))

    total_batches = len(batches)
    print(f"[事件抽取] 开始并行抽取，文本总数: {len(texts)}，分批数: {total_batches}，并发数: {concurrency}")

    if concurrency <= 1 or total_batches <= 1:
        for idx, (batch, start) in enumerate(batches):
            batch_idx = idx + 1
            batch_res = _extract_batch_worker_wrapper(batch, start, batch_idx, total_batches, extraction_model)
            results.extend(batch_res)
    else:
        futures = {}
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            for idx, (batch, start) in enumerate(batches):
                batch_idx = idx + 1
                future = executor.submit(
                    _extract_batch_worker_wrapper,
                    batch, start, batch_idx, total_batches, extraction_model
                )
                futures[future] = batch_idx

            for future in as_completed(futures):
                batch_idx = futures[future]
                try:
                    batch_res = future.result()
                    results.extend(batch_res)
                except Exception as future_err:
                    print(f"[事件抽取] 线程任务 {batch_idx} 执行发生严重未捕获错误: {future_err}")

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


def extract_event_simulate(text, domain_lexicon=None):
    """模拟事件抽取 - 支持中英文"""
    domain_lexicon = domain_lexicon or build_domain_lexicon(None)
    text_lower = text.lower()
    technologies = domain_lexicon.extract_technologies(text)
    subject = "未知"
    action = "未知"
    scene = "未知"
    time = "未知"

    # 主体识别（中英文）。这里只做弱提示，最终会按 schema 严格清洗。
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
    subject = _normalize_subject(subject)

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

    mechanism_tokens = _dedupe_preserve_order(domain_lexicon.extract_mechanism_core_tokens(text, action))
    task_tokens = _dedupe_preserve_order(domain_lexicon.extract_task_constraint_tokens(text, scene))
    object_tokens = _dedupe_preserve_order(domain_lexicon.extract_object_modifier_tokens(text, scene, " ".join(technologies)))
    data_tokens = _dedupe_preserve_order(domain_lexicon.extract_data_modifier_tokens(text))
    method_tokens = _dedupe_preserve_order(domain_lexicon.extract_method_modifier_tokens(text, action))
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


def extract_events(text, use_api=True, domain_lexicon=None):
    domain_lexicon = domain_lexicon or build_domain_lexicon(None)
    if use_api and _api_config_available():
        return extract_events_with_api(text)
    return [extract_event_simulate(text, domain_lexicon=domain_lexicon)]


def extract_event(text, use_api=True, domain_lexicon=None):
    domain_lexicon = domain_lexicon or build_domain_lexicon(None)
    events = extract_events(text, use_api=use_api, domain_lexicon=domain_lexicon)
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


def _prepare_event_for_source_row(event, row, doc_event_index=1, source_extraction_mode="local", domain_lexicon=None):
    domain_lexicon = domain_lexicon or build_domain_lexicon(None)
    event = dict(event or {})
    event["id"] = row.get("id", event.get("id", ""))
    analysis_scope = _analysis_scope_from_row(row, domain_lexicon=domain_lexicon)
    _normalize_event_technologies(
        event,
        row.get("text", ""),
        row.get("title", ""),
        source_extraction_mode=source_extraction_mode,
        source_type=row.get("source_type", ""),
        analysis_scope=analysis_scope,
        domain_lexicon=domain_lexicon,
    )
    assess_weak_signal_event(event, row)
    source_date = _source_date_text(row)
    if source_date:
        event["date"] = source_date
        event["event_date"] = source_date
    event["source_type"] = row.get("source_type", "")
    event["title"] = row.get("title", "")
    if analysis_scope:
        event["analysis_tech_field_name"] = analysis_scope
    return _normalize_event_schema(
        event,
        source_row=row,
        doc_event_index=doc_event_index,
        source_extraction_mode=source_extraction_mode,
    )


def _renormalize_cached_events_with_source(cached_df, source_df, domain_lexicon=None):
    domain_lexicon = domain_lexicon or build_domain_lexicon(None)
    if cached_df is None or cached_df.empty or source_df is None or source_df.empty:
        return cached_df
    source_rows = {
        str(row.get("id", "")).strip(): row
        for _, row in source_df.iterrows()
        if str(row.get("id", "")).strip()
    }
    normalized = []
    per_doc_counts = {}
    for _, event_row in cached_df.iterrows():
        event = event_row.to_dict()
        source_id = str(event.get("id", "")).strip()
        per_doc_counts[source_id] = per_doc_counts.get(source_id, 0) + 1
        source_row = source_rows.get(source_id)
        if source_row is not None:
            normalized.append(
                _prepare_event_for_source_row(
                    event,
                    source_row,
                    doc_event_index=per_doc_counts[source_id],
                    source_extraction_mode=event.get("source_extraction_mode", "cache"),
                    domain_lexicon=domain_lexicon,
                )
            )
        else:
            normalized.append(
                _normalize_event_schema(
                    event,
                    doc_event_index=per_doc_counts[source_id],
                    source_extraction_mode=event.get("source_extraction_mode", "cache"),
                )
            )
    return pd.DataFrame(normalized)


def process_events(
    df,
    use_api=True,
    batch_size=None,
    cache_path=None,
    refresh_cache=False,
    domain_context=None,
    domain_pack=None,
):
    domain_lexicon = build_domain_lexicon(domain_context or domain_pack)
    event_columns = _event_cache_columns()
    if df is None or df.empty:
        return pd.DataFrame(columns=event_columns)

    if batch_size is None:
        batch_size = _event_extraction_batch_size()


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
            cached_df = _renormalize_cached_events_with_source(cached_df, source_df, domain_lexicon=domain_lexicon)
            return cached_df.reindex(columns=event_columns, fill_value=None)

    if use_api and _api_config_available():
        texts = source_df["text"].tolist()

        # 1. 查找文档级缓存
        doc_level_dir = _get_doc_level_cache_dir(cache_path) if (cache_path and not refresh_cache) else None

        cached_events_by_idx = {}
        miss_indices = []
        miss_texts = []

        for idx, text in enumerate(texts):
            doc_hash = hashlib.sha1(str(text).encode("utf-8", errors="ignore")).hexdigest()
            cached_events = _load_doc_cache(doc_hash, doc_level_dir)
            if cached_events is not None:
                cached_events_by_idx[idx] = cached_events
            else:
                miss_indices.append(idx)
                miss_texts.append(text)

        extracted_events = []

        # 2. 如果有未命中的，调用 API 进行批量提取
        if miss_texts:
            print(f"[事件抽取] 使用API，总文献数: {len(texts)}，缓存命中: {len(cached_events_by_idx)}，未命中(需调用API): {len(miss_texts)}，批次大小: {batch_size}")
            api_extracted = batch_extract_event_with_api(miss_texts, batch_size=batch_size)

            # 将提取出的事件归类到对应的 miss_indices 中
            extracted_by_miss_idx = {}
            for event in api_extracted:
                try:
                    local_index = int(event.get("_source_text_index"))
                except (KeyError, TypeError, ValueError):
                    continue
                if 0 <= local_index < len(miss_texts):
                    orig_idx = miss_indices[local_index]
                    extracted_by_miss_idx.setdefault(orig_idx, []).append(event)

            # 保存结果到文档级缓存中，并写入 extracted_events
            for miss_idx, orig_idx in enumerate(miss_indices):
                events_for_doc = extracted_by_miss_idx.get(orig_idx, [])

                # 剥离 _source_text_index 并深度复制以保存到独立缓存文件
                cleaned_events = []
                for ev in events_for_doc:
                    ev_copy = dict(ev)
                    ev_copy.pop("_source_text_index", None)
                    cleaned_events.append(ev_copy)

                if doc_level_dir:
                    doc_hash = hashlib.sha1(str(miss_texts[miss_idx]).encode("utf-8", errors="ignore")).hexdigest()
                    _save_doc_cache(doc_hash, cleaned_events, doc_level_dir)

                # 将原始事件（包含临时 _source_text_index）放入当前总提取列表中用于归档
                for ev in events_for_doc:
                    # 必须重设 _source_text_index 指向原始 DataFrame 中的行索引
                    ev_copy = dict(ev)
                    ev_copy["_source_text_index"] = orig_idx
                    extracted_events.append(ev_copy)
        else:
            print(f"[事件抽取] 使用API，全部命中缓存({len(texts)}/{len(texts)})，跳过API请求")

        # 3. 将缓存中命中的事件也加入总结果，并设置正确的 _source_text_index
        for orig_idx, events_for_doc in cached_events_by_idx.items():
            for ev in events_for_doc:
                ev_copy = dict(ev)
                ev_copy["_source_text_index"] = orig_idx
                extracted_events.append(ev_copy)

        print(f"[事件抽取] API及缓存阶段处理完成，待归档事件数: {len(extracted_events)}")
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
                        domain_lexicon=domain_lexicon,
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
            extract_events(row["text"], use_api=False, domain_lexicon=domain_lexicon),
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
                    domain_lexicon=domain_lexicon,
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
