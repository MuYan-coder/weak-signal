import json
import os
import re
from pathlib import Path

import pandas as pd

from ..utils.api_stats import record_call
from ..utils.env_config import ensure_env_loaded
from ..utils.llm_client import chat_text, get_provider_and_client
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
    if event.get("technology") is None:
        event["technology"] = ["未知"]
    if event.get("subject") is None:
        event["subject"] = "未知"
    if event.get("action") is None:
        event["action"] = "未知"
    if event.get("scene") is None:
        event["scene"] = "未知"
    if event.get("time") is None:
        event["time"] = "未知"
    return event


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

    # 尝试解析数组格式
    json_pattern = r"\[\s*\{.*?\}\s*\]"
    matches = re.findall(json_pattern, raw, re.DOTALL)
    if matches:
        candidate = max(matches, key=len)
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # 尝试解析对象格式
    obj_pattern = r"\{[^{}]*\}"
    matches = re.findall(obj_pattern, raw, re.DOTALL)
    if matches:
        candidate = max(matches, key=len)
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

    return None


_EVENT_JSON_FIELDS = ["subject", "action", "technology", "scene", "time"]


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


def _event_cache_columns():
    return [
        "subject", "action", "technology", "scene", "time", "date", "event_date", "id", "source_type", "title",
        "observation_scopes", "scope_candidates", "scope_candidate_scopes",
        "scope_match_mode",
        "observation_scopes_detected", "scope_direct_matches", "scope_alias_matches",
        "scope_proxy_scopes", "scope_rejected_scopes", "scope_detection_reason",
        "candidate_units", "candidate_unit_count", "scope_echo_candidate_count",
        "low_attention_hint", "niche_actor_hint", "non_dominant_hint",
        "cross_domain_hint", "traceable_hint", "weak_signal_event_score",
        "weak_signal_event_candidate", "weak_signal_reasons",
    ]


def _source_date_text(row):
    for field in ["date", "event_date", "publication_date", "application_date", "grant_date", "time"]:
        value = str(row.get(field, "")).strip()
        if value and value.lower() not in {"nan", "nat", "none", "null", "未知"}:
            return value
    return ""


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
    events = payload.get("events", [])
    if not isinstance(events, list) or not events:
        return None
    events_df = pd.DataFrame(events)
    if expected_ids:
        cached_ids = events_df.get("id", pd.Series(dtype="object")).astype(str).tolist()
        if list(map(str, expected_ids)) != cached_ids:
            return None
    return events_df


def save_event_cache(cache_path, events_df, metadata=None):
    cache_file = _normalize_cache_path(cache_path)
    if cache_file is None or events_df is None or events_df.empty:
        return
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": metadata or {},
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


def extract_event_with_api(text):
    prompt = f"""
请从以下文本中抽取结构化事件信息，返回JSON格式。

文本：{text}

要求：
- 主体（subject）：谁在行动
- 行为（action）：做什么
- 技术（technology）：涉及哪些技术（列表）
- 场景（scene）：应用场景
- 时间（time）：时间信息

如果某项信息不存在，用"未知"填充。

输出格式：
{{
  "subject": "string",
  "action": "string",
  "technology": ["tech1", "tech2"],
  "scene": "string",
  "time": "string"
}}
"""

    try:
        provider, client = get_provider_and_client()
        if client is None or provider is None:
            return extract_event_simulate(text)
        
        # 使用配置的事件抽取模型
        extraction_model = os.getenv("EXTRACTION_MODEL", os.getenv("OPENAI_MODEL", "Qwen/Qwen2.5-7B-Instruct"))
        result, usage_info, _ = chat_text(
            prompt,
            model=extraction_model,
            temperature=0.1,
            max_tokens=300,
            timeout=60,
        )
        result = result.strip()
        
        # 移除可能的markdown代码块标记
        if result.startswith("```"):
            lines = result.split("\n")
            if len(lines) > 1:
                result = "\n".join(lines[1:])
            if "```" in result:
                result = result.rsplit("```", 1)[0]
            result = result.strip()
        
        if usage_info["prompt_tokens"] or usage_info["completion_tokens"]:
            record_call(
                call_type="事件抽取-单条",
                prompt_tokens=usage_info["prompt_tokens"],
                completion_tokens=usage_info["completion_tokens"],
                success=True,
            )
        
        # 尝试解析JSON
        try:
            event = json.loads(result)
        except json.JSONDecodeError:
            # 尝试从文本中提取JSON
            json_match = re.search(r'\{[^{}]*\}', result, re.DOTALL)
            if json_match:
                event = json.loads(json_match.group(0))
            else:
                print(f"[事件抽取] 无法解析JSON响应: {result[:200]}")
                return extract_event_simulate(text)
        
        _fix_event_dict(event)
        _normalize_event_technologies(event, text, text, source_extraction_mode="api")
        return event
    except Exception as e:
        print(f"API调用失败: {e}")
        return extract_event_simulate(text)


def batch_extract_event_with_api(texts, batch_size=10):
    results = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        items = "\n\n".join([f"编号{idx + 1}: {t}" for idx, t in enumerate(batch)])
        prompt = f"""
请按照编号顺序，从下面多个文本中抽取结构化事件信息，返回一个纯 JSON 数组。
不要添加任何说明性文字、编号、列表或解释。只输出 JSON 数组。

文本列表：
{items}

输出格式示例：
[
  {{
    "subject": "string",
    "action": "string",
    "technology": ["tech1", "tech2"],
    "scene": "string",
    "time": "string"
  }}
]

重要：返回的数组必须包含正好 {len(batch)} 个对象，对应每个文本。
"""
        try:
            provider, client = get_provider_and_client()
            if client is None or provider is None:
                for text in batch:
                    results.append(extract_event_simulate(text))
                continue
            
            # 使用配置的事件抽取模型
            extraction_model = os.getenv("EXTRACTION_MODEL", os.getenv("OPENAI_MODEL", "Qwen/Qwen2.5-7B-Instruct"))
            print(f"[事件抽取] 调用API，批次: {start//batch_size + 1}，模型: {extraction_model}")
            raw, usage_info, _ = chat_text(
                prompt,
                model=extraction_model,
                temperature=0.1,
                max_tokens=3000,
                timeout=120,
            )
            print(f"[事件抽取] API响应接收完成，长度: {len(raw)}")
            raw = raw.strip()
            
            # 移除可能的markdown代码块标记
            if raw.startswith("```"):
                lines = raw.split("\n")
                if len(lines) > 1:
                    raw = "\n".join(lines[1:])
                if "```" in raw:
                    raw = raw.rsplit("```", 1)[0]
                raw = raw.strip()
            
            if usage_info["prompt_tokens"] or usage_info["completion_tokens"]:
                record_call(
                    call_type="事件抽取-批量",
                    prompt_tokens=usage_info["prompt_tokens"],
                    completion_tokens=usage_info["completion_tokens"],
                    success=True,
                )
            
            # 尝试解析JSON
            extracted = _parse_json_from_response(raw)
            if extracted is None:
                # 尝试直接解析
                try:
                    extracted = json.loads(raw)
                except json.JSONDecodeError:
                    pass
            
            if isinstance(extracted, list) and extracted:
                recovered_count = min(len(extracted), len(batch))
                if len(extracted) != len(batch):
                    print(
                        f"[warning] 批量抽取仅恢复 {recovered_count}/{len(batch)} 条，缺失部分退回逐条抽取"
                    )
                for event, source_text in zip(extracted[:recovered_count], batch[:recovered_count]):
                    _fix_event_dict(event)
                    _normalize_event_technologies(event, source_text, source_text, source_extraction_mode="api")
                results.extend(extracted[:recovered_count])
                for text in batch[recovered_count:]:
                    results.append(extract_event_with_api(text))
            else:
                print("[warning] 批量抽取返回内容无法解析为符合长度的列表，退回逐条抽取")
                print("[debug] 原始响应：", raw[:800])
                for text in batch:
                    results.append(extract_event_with_api(text))
        except Exception as e:
            print(f"批量API调用失败: {e}")
            record_call(
                call_type="事件抽取-批量",
                prompt_tokens=0,
                completion_tokens=0,
                success=False,
            )
            for text in batch:
                results.append(extract_event_with_api(text))
    return results


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

    return {
        "subject": subject,
        "action": action,
        "technology": technologies,
        "scene": scene,
        "time": time,
    }


def extract_event(text, use_api=True):
    if use_api and _api_config_available():
        return extract_event_with_api(text)
    return extract_event_simulate(text)


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


def process_events(df, use_api=True, batch_size=10, cache_path=None, refresh_cache=False):
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

    expected_ids = df["id"].astype(str).tolist() if "id" in df.columns else None
    if not refresh_cache:
        cached_df = load_event_cache(cache_path, expected_ids=expected_ids)
        if cached_df is not None and not cached_df.empty:
            return cached_df.reindex(columns=event_columns, fill_value=None)

    if use_api and _api_config_available():
        texts = df["text"].tolist()
        print(f"[事件抽取] 使用API，文本数量: {len(texts)}，批次大小: {batch_size}")
        events = batch_extract_event_with_api(texts, batch_size=batch_size)
        print(f"[事件抽取] API调用完成，事件数量: {len(events)}")
        if len(events) != len(df):
            print("[warning] 批量抽取结果数量与输入数量不一致，退回逐条抽取")
        else:
            normalized_events = []
            for event, (_, row) in zip(events, df.iterrows()):
                event["id"] = row["id"]
                _normalize_event_technologies(
                    event,
                    row.get("text", ""),
                    row.get("title", ""),
                    source_extraction_mode="api",
                    source_type=row.get("source_type", ""),
                )
                assess_weak_signal_event(event, row)
                source_date = _source_date_text(row)
                if source_date:
                    event["date"] = source_date
                    event["event_date"] = source_date
                event["source_type"] = row.get("source_type", "")
                event["title"] = row.get("title", "")
                normalized_events.append(event)
            events_df = pd.DataFrame(normalized_events).reindex(columns=event_columns, fill_value=None)
            save_event_cache(
                cache_path,
                events_df,
                metadata={"mode": "api", "batch_size": batch_size, "rows": len(events_df)},
            )
            return events_df

    events = []
    for _, row in df.iterrows():
        event = extract_event(row["text"], use_api)
        event["id"] = row["id"]
        _normalize_event_technologies(
            event,
            row.get("text", ""),
            row.get("title", ""),
            source_extraction_mode="local",
            source_type=row.get("source_type", ""),
        )
        assess_weak_signal_event(event, row)
        source_date = _source_date_text(row)
        if source_date:
            event["date"] = source_date
            event["event_date"] = source_date
        event["source_type"] = row.get("source_type", "")
        event["title"] = row.get("title", "")
        events.append(event)
    events_df = pd.DataFrame(events).reindex(columns=event_columns, fill_value=None)
    save_event_cache(
        cache_path,
        events_df,
        metadata={
            "mode": "api" if use_api and _api_config_available() else "local",
            "batch_size": batch_size,
            "rows": len(events_df),
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
