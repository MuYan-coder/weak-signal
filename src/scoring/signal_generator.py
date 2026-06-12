from collections import defaultdict
import hashlib
import re

import pandas as pd

from ..utils.semantic_utils import semantic_similarity
from ..validation.object_family_canonicalizer import (
    CanonicalizationResult,
    get_canonicalizer,
)
from .candidate_eligibility import apply_candidate_eligibility


_ANALYSIS_SCOPE_FIELDS = [
    "analysis_tech_field_name", "tech_field_name", "analysis_domain", "selected_domain",
]

_OBJECT_FAMILY_SCOPE_HINTS = {
    "robot", "robotics", "robotic", "机器人", "机械臂", "具身", "embodied",
    "world model", "世界模型", "ai", "artificial intelligence", "人工智能",
    "智能体", "agent", "agents", "large model", "大模型", "llm",
}

_MATERIAL_SCOPE_HINTS = {
    "材料", "电子材料", "新型电子材料", "material", "materials",
    "electronic material", "semiconductor", "battery", "电池", "半导体",
}

_MATERIAL_RELEVANCE_TERMS = {
    "材料", "电子材料", "半导体", "电池", "薄膜", "钙钛矿", "氧化物",
    "颗粒", "纳米", "界面", "钝化", "掺杂", "导电", "电化学",
    "晶体", "晶格", "正极", "负极", "电解质", "电容", "介电", "柔性电子",
    "material", "materials", "electronic", "semiconductor", "battery",
    "thin film", "film", "perovskite", "oxide", "nanoparticle",
    "interface", "passivation", "doping", "conductive", "electrochemical",
    "crystal", "lattice", "cathode", "anode", "electrolyte", "dielectric",
}

_OFF_DOMAIN_AI_ROBOT_TOKENS = {
    "机器人", "机械臂", "具身", "世界模型", "导航", "跑酷", "抓取",
    "操控", "仿真", "训练技术", "规划技术", "控制技术",
    "robot", "robotic", "robotics", "humanoid", "embodied", "agent",
    "world model", "navigation", "parkour", "grasping", "manipulation",
    "training", "planning", "control", "policy", "simulation",
}

_EMPTY_TEXT_VALUES = {"", "nan", "nat", "none", "null", "未知"}


def _safe_raw_text(value):
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    return "" if text.lower() in _EMPTY_TEXT_VALUES else text


def _dedupe_preserve(values):
    deduped = []
    seen = set()
    for value in values:
        text = _safe_raw_text(value)
        if not text:
            continue
        marker = text.lower()
        if marker in seen:
            continue
        seen.add(marker)
        deduped.append(text)
    return deduped


def _normalize_scope_key(value):
    return "".join(ch.lower() for ch in _safe_raw_text(value) if ch.isalnum() or "\u4e00" <= ch <= "\u9fff")


def _coerce_text_list(value):
    if isinstance(value, list):
        raw_values = value
    elif isinstance(value, (tuple, set)):
        raw_values = list(value)
    elif isinstance(value, str) and value.strip().startswith("[") and value.strip().endswith("]"):
        raw_values = [part.strip(" '\"\t\r\n") for part in value.strip("[]").split(",")]
    elif isinstance(value, str) and any(separator in value for separator in [",", "，", "、", ";", "；"]):
        raw_values = re.split(r"[,，、;；]\s*", value)
    else:
        raw_values = [value]
    values = []
    for item in raw_values:
        text = _safe_raw_text(item)
        if text and text not in values:
            values.append(text)
    return values


def _stable_document_id(record, index):
    source_type = _safe_raw_text((record or {}).get("source_type"))
    title = _safe_raw_text((record or {}).get("title"))
    text = _safe_raw_text((record or {}).get("text")) or _safe_raw_text((record or {}).get("full_text"))
    seed = "|".join([source_type, title, text[:240], str(index)])
    digest = hashlib.sha256(seed.encode("utf-8", errors="ignore")).hexdigest()[:16]
    return f"source_{digest}"


def _normalize_document_identity(data_df):
    stats = {
        "id_filled_from_source_id_count": 0,
        "id_filled_from_doc_id_count": 0,
        "id_generated_count": 0,
        "source_id_filled_from_id_count": 0,
    }
    if data_df is None or data_df.empty:
        return data_df.copy() if isinstance(data_df, pd.DataFrame) else pd.DataFrame(), stats

    normalized = data_df.copy()
    if "id" not in normalized.columns:
        normalized["id"] = ""
    if "source_id" not in normalized.columns:
        normalized["source_id"] = ""

    for index, row in normalized.reset_index(drop=True).iterrows():
        row_index = normalized.index[index]
        current_id = _safe_raw_text(row.get("id"))
        source_id = _safe_raw_text(row.get("source_id"))
        doc_id = _safe_raw_text(row.get("doc_id"))
        if not current_id and source_id:
            normalized.at[row_index, "id"] = source_id
            current_id = source_id
            stats["id_filled_from_source_id_count"] += 1
        elif not current_id and doc_id:
            normalized.at[row_index, "id"] = doc_id
            current_id = doc_id
            stats["id_filled_from_doc_id_count"] += 1
        elif not current_id:
            generated_id = _stable_document_id(row.to_dict(), index)
            normalized.at[row_index, "id"] = generated_id
            current_id = generated_id
            stats["id_generated_count"] += 1

        if not source_id and current_id:
            normalized.at[row_index, "source_id"] = current_id
            stats["source_id_filled_from_id_count"] += 1

    normalized["id"] = normalized["id"].fillna("").astype(str)
    normalized["source_id"] = normalized["source_id"].fillna("").astype(str)
    return normalized, stats


def _analysis_scope_from_record(record):
    for field in _ANALYSIS_SCOPE_FIELDS:
        text = _safe_raw_text((record or {}).get(field, ""))
        if text:
            return text
    return ""


def _analysis_scope_from_data(data_df):
    if data_df is None or data_df.empty:
        return ""
    counts = defaultdict(int)
    for _, row in data_df.iterrows():
        scope = _analysis_scope_from_record(row.to_dict())
        if scope:
            counts[scope] += 1
    if not counts:
        return ""
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _pack_list_terms(pack, section_name, field_names):
    if pack is None:
        return []
    section = getattr(pack, section_name, {}) or {}
    if not isinstance(section, dict):
        return []
    terms = []
    for field_name in field_names:
        terms.extend(_coerce_text_list(section.get(field_name, [])))
    return terms


def _domain_pack_scope_terms(domain_context, analysis_scope=""):
    pack = _extract_domain_pack(domain_context)
    terms = [analysis_scope]
    if pack is None:
        return _dedupe_preserve(terms)
    terms.extend(_pack_list_terms(pack, "domain_identity", ["field_name"]))
    terms.extend(_pack_list_terms(pack, "observation_scopes", ["main_scope", "sub_scopes", "scope_aliases"]))
    source = getattr(pack, "source", {}) or {}
    if isinstance(source, dict):
        user_input = source.get("based_on_user_input", {})
        if isinstance(user_input, dict):
            terms.extend(_coerce_text_list(user_input.get("field_name", "")))
    return _dedupe_preserve(terms)


def _concept_scope_values(concept):
    values = []
    for field in ["analysis_tech_field_name", "primary_scope", "scope_name"]:
        text = _safe_raw_text((concept or {}).get(field, ""))
        if text:
            values.append(text)
    values.extend(_coerce_text_list((concept or {}).get("scope_names", [])))
    return [value for value in values if value]


def _scope_matches_analysis(concept, analysis_scope, domain_context=None):
    if not analysis_scope:
        return True
    target_key = _normalize_scope_key(analysis_scope)
    if not target_key:
        return True
    scope_values = _concept_scope_values(concept)
    if not scope_values:
        return True
    for value in scope_values:
        key = _normalize_scope_key(value)
        if key and (key == target_key or key in target_key or target_key in key):
            return True
    alias_keys = {
        _normalize_scope_key(term)
        for term in _domain_pack_scope_terms(domain_context, analysis_scope)
        if _normalize_scope_key(term)
    }
    if target_key in alias_keys:
        for value in scope_values:
            key = _normalize_scope_key(value)
            if key in alias_keys:
                return True
    return False


def _attach_analysis_scope(concept, analysis_scope):
    if not analysis_scope:
        return concept
    concept = dict(concept or {})
    concept["analysis_tech_field_name"] = analysis_scope
    if not _safe_raw_text(concept.get("primary_scope")):
        concept["primary_scope"] = analysis_scope
    if not _safe_raw_text(concept.get("scope_name")):
        concept["scope_name"] = analysis_scope
    if not _coerce_text_list(concept.get("scope_names", [])):
        concept["scope_names"] = [analysis_scope]
    return concept


def _extract_domain_pack(domain_context):
    if domain_context is None:
        return None
    if hasattr(domain_context, "domain_pack"):
        return getattr(domain_context, "domain_pack")
    if hasattr(domain_context, "canonicalization"):
        return domain_context
    return None


def _domain_allows_object_family(concept, domain_context=None, canonicalizer=None):
    if canonicalizer is not None and getattr(canonicalizer, "domain_pack", None) is not None:
        return bool(getattr(canonicalizer, "object_family_enabled", False))
    pack = _extract_domain_pack(domain_context)
    if pack is not None:
        canonicalization = getattr(pack, "canonicalization", {}) or {}
        return bool(canonicalization.get("object_family_enabled", False))
    scope_text = " ".join(_concept_scope_values(concept)).lower()
    if not scope_text:
        return True
    return any(hint in scope_text for hint in _OBJECT_FAMILY_SCOPE_HINTS)


def _contains_any_term(text, terms):
    lowered = _safe_raw_text(text).lower()
    if not lowered:
        return False
    return any(str(term).lower() in lowered for term in terms if str(term).strip())


def _is_material_analysis_scope(analysis_scope):
    scope_text = _safe_raw_text(analysis_scope).lower()
    return any(str(term).lower() in scope_text for term in _MATERIAL_SCOPE_HINTS)


def _candidate_surface_text(concept):
    parts = []
    for field in [
        "display_candidate_name", "canonical_candidate_name_en", "raw_phrase",
        "raw_candidate_text", "normalized_candidate_text", "mechanism_core",
        "constraint_signature", "topic_summary_name",
    ]:
        text = _safe_raw_text((concept or {}).get(field, ""))
        if text:
            parts.append(text)
    for field in [
        "mechanism_core_tokens", "task_constraint_tokens", "object_modifier_tokens",
        "data_modifier_tokens", "method_modifier_tokens",
    ]:
        parts.extend(_coerce_text_list((concept or {}).get(field, [])))
    return " ".join(parts)


def _analysis_relevance_terms(record, analysis_scope):
    terms = []
    for field in ["analysis_keywords", "analysis_synonyms", "keywords", "keyword"]:
        terms.extend(_coerce_text_list((record or {}).get(field, [])))
    if _is_material_analysis_scope(analysis_scope):
        terms.extend(_MATERIAL_RELEVANCE_TERMS)
    cleaned = []
    for term in terms:
        text = _safe_raw_text(term)
        if text and text not in cleaned and _normalize_scope_key(text) != _normalize_scope_key(analysis_scope):
            cleaned.append(text)
    return cleaned


def _domain_pack_relevance_terms(domain_context):
    terms = []
    pack = _extract_domain_pack(domain_context)
    terms.extend(_domain_pack_scope_terms(domain_context))
    if pack is not None:
        terms.extend(_pack_list_terms(pack, "search_strategy", ["core_keywords", "synonyms", "english_terms"]))
        terms.extend(
            _pack_list_terms(
                pack,
                "candidate_formation",
                [
                    "technical_object_types",
                    "mechanism_types",
                    "task_or_performance_types",
                    "data_or_method_types",
                    "scene_or_application_types",
                ],
            )
        )
    try:
        from src.extraction.tech_lexicon import build_domain_lexicon
        lexicon = build_domain_lexicon(domain_context)
    except Exception:
        lexicon = None
    if lexicon is not None:
        terms.extend(_coerce_text_list(getattr(lexicon, "domain_anchor_terms", [])))
        terms.extend(_coerce_text_list(getattr(lexicon, "domain_specific_terms", [])))
    return _dedupe_preserve(terms)


def _domain_pack_exclusion_terms(domain_context):
    pack = _extract_domain_pack(domain_context)
    if pack is None:
        return []
    terms = []
    terms.extend(_pack_list_terms(pack, "domain_identity", ["out_of_scope_domains"]))
    terms.extend(_pack_list_terms(pack, "search_strategy", ["exclude_terms"]))
    terms.extend(_pack_list_terms(pack, "observation_scopes", ["off_domain_anchor_terms"]))
    terms.extend(_pack_list_terms(pack, "evidence_rules", ["evidence_rejection_patterns"]))
    source = getattr(pack, "source", {}) or {}
    if isinstance(source, dict):
        user_input = source.get("based_on_user_input", {})
        if isinstance(user_input, dict):
            terms.extend(_coerce_text_list(user_input.get("exclude_terms", [])))
    return _dedupe_preserve(terms)


def _matching_terms(text, terms):
    lowered = _safe_raw_text(text).lower()
    if not lowered:
        return []
    matches = []
    normalized_text = _normalize_scope_key(lowered)
    for term in terms:
        term_text = _safe_raw_text(term)
        if not term_text:
            continue
        term_lower = term_text.lower()
        term_key = _normalize_scope_key(term_text)
        if term_lower in lowered or (term_key and term_key in normalized_text):
            matches.append(term_text)
    return _dedupe_preserve(matches)


def _concept_relevance_decision(concept, record, analysis_scope, domain_context=None, canonicalizer=None):
    if not analysis_scope or _domain_allows_object_family(concept, domain_context, canonicalizer):
        return True, "not_applicable"
    candidate_text = _candidate_surface_text(concept)
    evidence_text = " ".join(
        _safe_raw_text((record or {}).get(field, ""))
        for field in ["title", "text", "abstract", "summary", "keywords", "keyword"]
    )
    relevance_terms = _analysis_relevance_terms(record, analysis_scope)
    relevance_terms.extend(_domain_pack_relevance_terms(domain_context))
    relevance_terms = _dedupe_preserve(relevance_terms)
    exclusion_terms = _domain_pack_exclusion_terms(domain_context)
    positive_term_keys = {_normalize_scope_key(term) for term in relevance_terms if _normalize_scope_key(term)}
    exclusion_hits = _matching_terms(" ".join([candidate_text, evidence_text]), exclusion_terms)
    blocking_exclusion_hits = [
        term for term in exclusion_hits if _normalize_scope_key(term) not in positive_term_keys
    ]
    if blocking_exclusion_hits:
        return False, "explicit_off_domain"

    has_candidate_domain_anchor = bool(_matching_terms(candidate_text, relevance_terms))
    has_evidence_domain_anchor = bool(_matching_terms(evidence_text, relevance_terms))
    has_off_domain_candidate_anchor = _contains_any_term(candidate_text, _OFF_DOMAIN_AI_ROBOT_TOKENS)

    if has_off_domain_candidate_anchor and not (has_candidate_domain_anchor or has_evidence_domain_anchor):
        return False, "generic_off_domain_without_domain_anchor"
    if _is_material_analysis_scope(analysis_scope) and not (has_candidate_domain_anchor or has_evidence_domain_anchor):
        return False, "material_scope_without_domain_anchor"
    return True, "passed"


def _concept_relevant_to_analysis(concept, record, analysis_scope, domain_context=None, canonicalizer=None):
    return _concept_relevance_decision(
        concept,
        record,
        analysis_scope,
        domain_context=domain_context,
        canonicalizer=canonicalizer,
    )[0]


def _neutral_canonicalization(term):
    text = _safe_raw_text(term)
    return CanonicalizationResult(
        surface_term=text,
        canonical_term=text,
        parent_term="",
        term_type="unknown",
        family_id="",
        normalization_type="none",
        cross_source_pattern="single_source",
        patent_role="optional",
        priority="low",
    )


def _canonical_group_token(value):
    token = str(value or "").strip().lower()
    alias_map = {
        "robotics": "robot",
        "robots": "robot",
        "agents": "agent",
        "videos": "video",
        "visuals": "visual",
        "sensors": "sensor",
        "trajectories": "trajectory",
    }
    return alias_map.get(token, token)


def _first_token(values, preferred=None):
    normalized = [_canonical_group_token(value) for value in _normalize_aliases(values)]
    normalized = [value for value in normalized if value]
    if not normalized:
        return ""
    preferred = preferred or []
    for token in preferred:
        if token in normalized:
            return token
    return normalized[0]


def _object_family_candidate_key(concept, canonicalizer, object_family_allowed=True):
    """
    对象族感知的聚合键生成。

    优先级：
    1. 若短语可命中 object family registry → 按 family_id 聚合
    2. 特殊处理：simulation (of_006) 按 mechanism_core 细分
    3. 若无法命中 → 退回通用 _coarse_family_key

    Returns:
        (cluster_key, family_info)
    """
    if not object_family_allowed:
        return _coarse_family_key(concept, canonicalizer, use_canonicalizer=False), None

    family = canonicalizer.match_candidate_family(concept) if canonicalizer is not None else None

    if family:
        family_id = family.get('family_id', '')
        canonical_term = family.get('canonical_term', '')
        return f"family:{family_id}:{canonical_term}", family

    # 退回通用键
    return _coarse_family_key(concept, canonicalizer), None


def _coarse_family_key(concept, canonicalizer=None, use_canonicalizer=True):
    """
    候选聚合阶段的粗粒度家族键。

    优化策略 (v4 - 对象族感知聚合):
    1. 优先使用对象族感知聚合（由 _object_family_candidate_key 调用）
    2. 当 mechanism_core 存在时：scope + mechanism + anchors
    3. 当 mechanism_core 缺失时：使用 fallback 策略
       - 优先使用归一化后的 raw_phrase_core
       - 使用 task + object + data anchors 组合
       - 避免生成 scope + nan 这类空键
    4. 目标：提升跨源聚合率，减少无效空键
    """
    if use_canonicalizer and canonicalizer is None:
        canonicalizer = get_canonicalizer()

    primary_scope = str(concept.get("primary_scope", "")).strip()
    mechanism = str(concept.get("mechanism_core", "")).strip()

    # 获取 task anchor（优先选择具体的任务类型）
    task_anchor = _first_token(
        concept.get("task_constraint_tokens", []),
        preferred=["driving", "navigation", "manipulation", "grasping", "robot", "agent", "training", "planning", "control"],
    )

    # 获取 object anchor
    object_anchor = _first_token(
        concept.get("object_modifier_tokens", []),
        preferred=["robot", "agent", "environment", "perception", "memory", "policy", "model"],
    )

    # 获取 data anchor
    data_anchor = _first_token(
        concept.get("data_modifier_tokens", []),
        preferred=["video", "sensor", "trajectory", "visual", "3d", "multimodal", "simulation"],
    )

    # 获取 scene anchor (新增)
    scene_anchor = _first_token(
        concept.get("scene_tokens", []),
        preferred=["indoor", "outdoor", "driving", "navigation", "manipulation"],
    )

    # 获取 raw_phrase 的核心词作为 fallback，并使用 canonicalizer 归一化
    raw_phrase = str(concept.get("raw_phrase", "")).strip()
    raw_phrase_core = ""
    canonical_term = ""
    if raw_phrase:
        # 首先尝试从整个短语中提取归一化术语
        canonical_term = ""
        norm_type = "none"
        if use_canonicalizer and canonicalizer is not None:
            canonical_term, norm_type, _ = canonicalizer.extract_canonical_from_phrase(raw_phrase)
        if norm_type != 'none':
            raw_phrase_core = canonical_term.lower()
        else:
            # 如果没有找到，提取 raw_phrase 中的核心词
            raw_tokens = raw_phrase.lower().split()
            # 过滤掉常见停用词和 scope 词
            stop_words = {"the", "a", "an", "for", "with", "based", "using", "via", "through"}
            scope_words = {"embodied", "intelligence", "world", "model"}
            filtered = [t for t in raw_tokens if t not in stop_words and t not in scope_words and len(t) >= 3]
            if filtered:
                raw_phrase_core = filtered[0]

    # 主要区分维度：task 或 object（优先 task）
    primary_anchor = task_anchor or object_anchor
    secondary_anchor = data_anchor or scene_anchor

    # 策略1: mechanism_core 存在时，使用原有逻辑
    if mechanism and mechanism != "nan":
        parts = [primary_scope, mechanism, primary_anchor, secondary_anchor]
        return " || ".join(part for part in parts if part)

    # 策略2: mechanism_core 缺失时，使用 fallback
    # 构建 fallback 键：scope + anchors + raw_phrase_core
    fallback_parts = [primary_scope]

    # 添加 anchors
    if primary_anchor:
        fallback_parts.append(primary_anchor)
    if secondary_anchor and secondary_anchor != primary_anchor:
        fallback_parts.append(secondary_anchor)

    # 如果有归一化后的 raw_phrase_core，添加作为补充
    if raw_phrase_core and raw_phrase_core not in fallback_parts:
        fallback_parts.append(raw_phrase_core)

    # 如果 fallback_parts 只有 scope，尝试从 display_candidate_name 提取并归一化
    if len(fallback_parts) == 1:
        display_name = str(concept.get("display_candidate_name", "")).strip()
        if display_name:
            # 使用 canonicalizer 归一化 display_name
            canonical_display = ""
            norm_type = "none"
            if use_canonicalizer and canonicalizer is not None:
                canonical_display, norm_type, _ = canonicalizer.canonicalize(display_name)
            if norm_type != 'none':
                fallback_parts.append(canonical_display.lower())
            else:
                # 提取 display_name 中的关键词
                name_tokens = display_name.lower().split()
                filtered_name = [t for t in name_tokens if t not in {"the", "a", "an", "for", "with"} and len(t) >= 2]
                if filtered_name:
                    fallback_parts.append(filtered_name[0])

    return " || ".join(fallback_parts) if len(fallback_parts) > 1 else f"{primary_scope} || ungrouped"


def _normalize_scope_names(raw_value):
    if isinstance(raw_value, list):
        return [str(item).strip() for item in raw_value if str(item).strip()]
    if isinstance(raw_value, str) and raw_value.strip():
        return [item.strip() for item in raw_value.split(",") if item.strip()]
    return []


def _normalize_aliases(raw_value):
    if isinstance(raw_value, list):
        return [str(item).strip() for item in raw_value if str(item).strip()]
    if isinstance(raw_value, str) and raw_value.strip():
        return [raw_value.strip()]
    return []


SOURCE_TYPE_NORMALIZATION = {
    "专利": "patent",
    "文献": "paper",
    "研报": "report",
    "资讯": "news",
    "patent": "patent",
    "paper": "paper",
    "literature": "paper",
    "report": "report",
    "news": "news",
}


def _safe_text(value):
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "nat", "none"} else text


_APPLICATION_SURFACE_SUFFIXES = (
    "应用",
    "场景",
    "方向",
    "落地",
    "部署",
    "示范",
    "试点",
    " application",
    " applications",
    " scenario",
    " scenarios",
    " use case",
    " use cases",
    " deployment",
    " deployments",
)

_TECHNICAL_NAME_ANCHORS = (
    "技术",
    "方法",
    "模型",
    "算法",
    "工艺",
    "材料",
    "器件",
    "芯片",
    "电路",
    "装置",
    "模块",
    "控制",
    "规划",
    "训练",
    "仿真",
    "推理",
    "制造",
    "制备",
    "钝化",
    "掺杂",
    "沉积",
    "刻蚀",
    "封装",
    "互连",
    "集成",
    "导航",
    "操控",
    "抓取",
    "technology",
    "method",
    "model",
    "algorithm",
    "process",
    "material",
    "device",
    "circuit",
    "integrated circuit",
    "control",
    "planning",
    "training",
    "simulation",
    "reasoning",
    "manufacturing",
    "fabrication",
    "passivation",
    "doping",
    "deposition",
    "etching",
    "packaging",
    "interconnect",
    "integration",
    "navigation",
    "manipulation",
)


def _contains_technical_anchor(text):
    raw = _safe_text(text)
    if not raw:
        return False
    lowered = raw.lower()
    return any(term in (lowered if term.isascii() else raw) for term in _TECHNICAL_NAME_ANCHORS)


def _is_application_surface_name(text):
    raw = _safe_text(text)
    if not raw:
        return False
    lowered = raw.lower()
    return any(raw.endswith(term) if not term.isascii() else lowered.endswith(term) for term in _APPLICATION_SURFACE_SUFFIXES)


def _technical_name_profile(text):
    raw = _safe_text(text)
    if not raw:
        return {"quality": "missing", "issue": "missing_name", "is_technical": False}
    has_technical_anchor = _contains_technical_anchor(raw)
    application_surface = _is_application_surface_name(raw)
    if application_surface and not has_technical_anchor:
        return {
            "quality": "application_surface",
            "issue": "application_only_surface",
            "is_technical": False,
        }
    if has_technical_anchor:
        return {"quality": "technical", "issue": "", "is_technical": True}
    return {
        "quality": "unclear_surface",
        "issue": "missing_technical_anchor",
        "is_technical": False,
    }


def _first_nonempty_token(values):
    for value in _normalize_aliases(values):
        text = _safe_text(value)
        if text:
            return text
    return ""


def _can_add_technical_suffix(row):
    row = row or {}
    status = _safe_text(row.get("candidate_eligibility", ""))
    technical_item_stage = _safe_text(row.get("technical_item_stage", ""))
    source_type_values = [row.get("source_type", "")]
    source_types = row.get("source_types", [])
    if isinstance(source_types, (list, tuple, set)):
        source_type_values.extend(source_types)
    else:
        source_type_values.append(source_types)
    source_type_set = {_safe_text(item).lower() for item in source_type_values if _safe_text(item)}
    return (
        status == "eligible"
        and technical_item_stage in {"technical_item", "technical_item_draft"}
        and not _safe_bool(row.get("scope_shell_heavy", False))
        and _safe_bool(row.get("survives_without_scope", False))
        and not (source_type_set & {"policy", "market", "finance"})
    )


def _compose_slot_technical_name(row):
    mechanism = _safe_text((row or {}).get("mechanism_core", ""))
    if not mechanism:
        mechanism = _first_nonempty_token((row or {}).get("mechanism_core_tokens", []))
    if not mechanism:
        return ""
    subject = _first_nonempty_token((row or {}).get("object_modifier_tokens", []))
    if not subject:
        subject = _first_nonempty_token((row or {}).get("task_constraint_tokens", []))
    if not subject:
        subject = _safe_text((row or {}).get("scope_name", ""))
    if not subject:
        return ""
    if re.search(r"[A-Za-z]", subject + mechanism):
        parts = [subject, mechanism]
        name = " ".join(part for part in parts if part).strip()
        if name and "technology" not in name.lower() and not _contains_technical_anchor(name):
            if not _can_add_technical_suffix(row):
                return ""
            name = f"{name} technology"
        return name
    name = f"{subject}{mechanism}".strip()
    if name and not _contains_technical_anchor(name):
        if not _can_add_technical_suffix(row):
            return ""
        name = f"{name}技术"
    return name


def _resolve_final_technical_name(row):
    candidates = [
        ("rule_display_candidate_name", _safe_text((row or {}).get("rule_display_candidate_name", ""))),
        ("display_candidate_name", _safe_text((row or {}).get("display_candidate_name", ""))),
        ("llm_refined_topic_name", _safe_text((row or {}).get("llm_refined_topic_name", ""))),
        ("topic_summary_name", _safe_text((row or {}).get("topic_summary_name", ""))),
        ("canonical_candidate_name_en", _safe_text((row or {}).get("canonical_candidate_name_en", ""))),
        ("normalized_candidate_text", _safe_text((row or {}).get("normalized_candidate_text", ""))),
        ("raw_phrase", _safe_text((row or {}).get("raw_phrase", ""))),
        ("raw_candidate_text", _safe_text((row or {}).get("raw_candidate_text", ""))),
    ]
    first_surface = ""
    first_profile = None
    for source, name in candidates:
        if not name:
            continue
        profile = _technical_name_profile(name)
        if first_surface == "":
            first_surface = name
            first_profile = profile
        if profile["is_technical"]:
            return {
                "name": name,
                "source": source,
                "quality": profile["quality"],
                "issue": profile["issue"],
            }

    slot_name = _compose_slot_technical_name(row)
    slot_profile = _technical_name_profile(slot_name)
    if slot_profile["is_technical"]:
        return {
            "name": slot_name,
            "source": "slot_technical_name",
            "quality": slot_profile["quality"],
            "issue": slot_profile["issue"],
        }

    profile = first_profile or _technical_name_profile(first_surface)
    return {
        "name": first_surface,
        "source": "application_surface" if profile["issue"] == "application_only_surface" else "fallback_surface",
        "quality": profile["quality"],
        "issue": profile["issue"],
    }


def _safe_float(value, default=0.0):
    try:
        if value is None or pd.isna(value):
            return default
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value, default=0):
    try:
        if value is None or pd.isna(value):
            return default
    except Exception:
        pass
    try:
        return int(float(value))
    except Exception:
        return default


def _safe_bool(value):
    if isinstance(value, bool):
        return value
    text = _safe_text(value).lower()
    return text in {"1", "true", "yes", "y", "是", "通过"}


def _normalize_source_type(value):
    text = _safe_text(value)
    return SOURCE_TYPE_NORMALIZATION.get(text, SOURCE_TYPE_NORMALIZATION.get(text.lower(), text.lower() or "unknown"))


def _build_evidence_item(record, concept):
    title = _safe_text(record.get("title", ""))
    text = _safe_text(record.get("text", ""))
    # 优化：延长 snippet 从 120 到 300 字符，增加语义信息量
    snippet = text[:300].replace("\n", " ")
    if len(text) > 300:
        snippet += "..."
    item = {
        "id": _safe_text(record.get("id", "")),
        "source_type": _normalize_source_type(record.get("source_type", "unknown")),
        "org": _safe_text(record.get("org", "Unknown")) or "Unknown",
        "date": _safe_text(record.get("date", "")),
        "url": _safe_text(record.get("url", "")),
        "title": title,
        "snippet": snippet,
        "text": text[:600],  # 延长 text 从 400 到 600
        "raw_candidate_text": _safe_text(concept.get("raw_candidate_text", "")),
        "display_candidate_name": _safe_text(concept.get("display_candidate_name", "")),
    }
    for key in ["event_quality_score", "event_quality_tier", "event_quality_reason"]:
        if key in record:
            item[key] = record.get(key, "")
    return item


# 技术术语权重增强列表
_TECH_TERMS = {
    "robot", "robotic", "robotics", "agent", "agents",
    "training", "planning", "control", "navigation", "manipulation",
    "video", "visual", "trajectory", "trajectories", "sensor", "sensors",
    "multimodal", "simulation", "simulator", "embodied", "intelligence",
    "world model", "world models", "reinforcement", "learning",
    "perception", "action", "policy", "policies", "environment",
    "3d", "three-dimensional", "autonomous", "autonomy",
    "grasping", "grasp", "motion", "dynamics", "kinematics",
    "benchmark", "benchmarks", "dataset", "datasets", "evaluation",
    "material", "materials", "electronic", "semiconductor", "perovskite",
    "thin", "film", "oxide", "cathode", "anode", "electrolyte", "doping",
    "interface", "passivation", "conductive", "stability", "nanoparticle",
    "材料", "电子材料", "半导体", "钙钛矿", "薄膜", "氧化物", "正极",
    "负极", "电解质", "掺杂", "界面", "钝化", "导电", "稳定性", "纳米",
}


def _tech_term_weighted_similarity(text_a, text_b):
    """
    带技术术语权重的相似度计算。
    在 fast_mode (Jaccard) 基础上，对技术术语给予额外权重。
    """
    tokens_a = set(token for token in text_a.lower().split() if token)
    tokens_b = set(token for token in text_b.lower().split() if token)

    if not tokens_a or not tokens_b:
        return 0.0

    # 基础 Jaccard 相似度
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    base_score = len(intersection) / len(union) if union else 0.0

    # 技术术语加成
    tech_intersection = intersection & _TECH_TERMS
    tech_bonus = len(tech_intersection) * 0.05  # 每个匹配的技术术语加 0.05

    # 组合分数，上限 1.0
    final_score = min(base_score + tech_bonus, 1.0)
    return round(final_score, 3)


def _focused_evidence_text(item, focus_terms):
    base_text = " ".join(
        [
            str(item.get("title", "")).strip(),
            str(item.get("snippet", "")).strip(),
            str(item.get("text", "")).strip(),
        ]
    ).strip()
    lowered = base_text.lower()
    snippets = []
    for term in focus_terms or []:
        alias_lower = str(term).strip().lower()
        if not alias_lower:
            continue
        start = lowered.find(alias_lower)
        while start != -1:
            left = max(0, start - 120)  # 扩大上下文窗口
            right = min(len(base_text), start + len(alias_lower) + 120)
            snippets.append(base_text[left:right].strip())
            start = lowered.find(alias_lower, start + len(alias_lower))
    return " ".join(snippets[:4]) if snippets else base_text


def _cross_source_semantic_validation(evidence_items, focus_terms, threshold=0.15):
    """
    跨源语义验证 (优化版)。

    优化策略：
    1. 降低阈值从 0.28 到 0.15
    2. 使用技术术语加权相似度
    3. 分层阈值：基础阈值 0.15，强验证阈值 0.25
    """
    if not evidence_items or len(evidence_items) < 2:
        return False, 0.0, 0
    best_score = 0.0
    validated_pairs = 0
    for idx, left in enumerate(evidence_items):
        for right in evidence_items[idx + 1:]:
            if left.get("source_type") == right.get("source_type"):
                continue
            # 使用技术术语加权相似度
            score = _tech_term_weighted_similarity(
                _focused_evidence_text(left, focus_terms),
                _focused_evidence_text(right, focus_terms),
            )
            if score >= threshold:
                validated_pairs += 1
            if score > best_score:
                best_score = score
    return best_score >= threshold, round(best_score, 3), validated_pairs


def _family_scoped_semantic_validation(evidence_items, family_info, focus_terms):
    """
    对象族范围的语义验证。

    只在以下对象上做：
    - 已命中 object family registry 的 candidate
    - 或 source_count >= 2 的高价值 candidate

    Args:
        evidence_items: 证据项列表
        family_info: 对象族信息（来自 match_family）
        focus_terms: 关注术语列表

    Returns:
        dict: {
            "family_semantic_consistency": "high" | "medium" | "low",
            "family_semantic_validation_score": 0.0-1.0,
            "family_semantic_validation_method": "term_weighted" | "embedding",
            "family_semantic_validation_note": "解释"
        }
    """
    if not family_info:
        return None

    if not evidence_items or len(evidence_items) < 2:
        return {
            "family_semantic_consistency": "low",
            "family_semantic_validation_score": 0.0,
            "family_semantic_validation_method": "term_weighted",
            "family_semantic_validation_note": "证据项不足，无法验证"
        }

    # 获取对象族的别名作为额外的 focus_terms
    family_aliases = family_info.get('alias_terms', [])
    zh_terms = family_info.get('zh_en_pairs', {}).get('zh', [])
    en_terms = family_info.get('zh_en_pairs', {}).get('en', [])
    canonical = family_info.get('canonical_term', '')

    # 合并所有关注术语
    all_focus_terms = list(set(focus_terms + family_aliases + zh_terms + en_terms + [canonical]))

    # 使用更低的阈值进行对象族范围验证
    threshold = 0.10
    best_score = 0.0
    validated_pairs = 0
    total_pairs = 0

    for idx, left in enumerate(evidence_items):
        for right in evidence_items[idx + 1:]:
            if left.get("source_type") == right.get("source_type"):
                continue
            total_pairs += 1
            # 使用技术术语加权相似度
            score = _tech_term_weighted_similarity(
                _focused_evidence_text(left, all_focus_terms),
                _focused_evidence_text(right, all_focus_terms),
            )
            if score >= threshold:
                validated_pairs += 1
            if score > best_score:
                best_score = score

    # 判断一致性级别
    if best_score >= 0.25:
        consistency = "high"
        note = f"对象族内语义一致性高，最佳分数={best_score:.3f}，验证对数={validated_pairs}/{total_pairs}"
    elif best_score >= 0.15:
        consistency = "medium"
        note = f"对象族内语义一致性中等，最佳分数={best_score:.3f}，验证对数={validated_pairs}/{total_pairs}"
    else:
        consistency = "low"
        note = f"对象族内语义一致性低，最佳分数={best_score:.3f}，验证对数={validated_pairs}/{total_pairs}"

    return {
        "family_semantic_consistency": consistency,
        "family_semantic_validation_score": round(best_score, 3),
        "family_semantic_validation_method": "term_weighted",
        "family_semantic_validation_note": note
    }


def _best_pair_score(left_items, right_items, focus_terms, threshold=0.15):
    best_score = 0.0
    validated_pairs = 0
    for left in left_items:
        for right in right_items:
            score = _tech_term_weighted_similarity(
                _focused_evidence_text(left, focus_terms),
                _focused_evidence_text(right, focus_terms),
            )
            if score >= threshold:
                validated_pairs += 1
            if score > best_score:
                best_score = score
    return round(best_score, 3), validated_pairs


def _event_document_validation(evidence_items, focus_terms, threshold=0.15):
    news_items = [item for item in evidence_items if item.get("source_type") == "news"]
    paper_items = [item for item in evidence_items if item.get("source_type") in {"paper", "report"}]
    patent_items = [item for item in evidence_items if item.get("source_type") == "patent"]
    event_paper_score, paper_pairs = _best_pair_score(news_items, paper_items, focus_terms, threshold=threshold)
    event_patent_score, patent_pairs = _best_pair_score(news_items, patent_items, focus_terms, threshold=threshold)
    best_score = max(event_paper_score, event_patent_score)
    return {
        "event_doc_semantic_validated": best_score >= threshold,
        "event_doc_validation_score": round(best_score, 3),
        "event_doc_semantic_pairs": paper_pairs + patent_pairs,
        "event_paper_semantic_score": event_paper_score,
        "event_patent_semantic_score": event_patent_score,
    }


def _time_validation_metrics(date_values):
    parsed = []
    for raw in date_values:
        dt = pd.to_datetime(str(raw).strip(), errors="coerce", utc=True)
        if not pd.isna(dt):
            parsed.append(dt)
    if len(parsed) < 2:
        return 0, len(parsed), 0.0, False
    parsed = sorted(parsed)
    split_index = max(1, len(parsed) // 2)
    early_count = split_index
    recent_count = len(parsed) - split_index
    validation_ratio = round(recent_count / max(early_count, 1), 2)
    time_validated = recent_count >= early_count and recent_count >= 2
    return early_count, recent_count, validation_ratio, time_validated


def _empty_candidate_columns():
    return [
        "tech_name", "technology", "display_candidate_name", "normalized_candidate_text",
        "canonical_candidate_name_en", "constraint_signature", "internal_candidate_label",
        "relation_target", "relation_task", "relation_data_modinality", "relation_method",
        "relation_summary", "relation_signature", "topic_summary_name", "topic_naturalness_reason",
        "compression_mode",
        "mechanism_core", "news_count", "paper_count", "report_count", "patent_count",
        "total_mentions", "org_count", "orgs", "source_types", "source_count", "mention_ids",
        "mention_dates", "evidence_titles", "evidence_items", "weak_signal_event_count",
        "candidate_evidence_quality", "candidate_core_evidence_quality",
        "low_quality_evidence_ratio", "high_quality_evidence_count",
        "quality_risk_flag", "quality_adjusted_rank_score",
        "candidate_evidence_quality_reason",
        "candidate_id",
        "first_seen_date", "last_seen_date", "temporal_validation_tier",
        "temporal_validation_status", "temporal_validation_passed",
        "temporal_momentum_score", "growth_rate", "source_growth_rate",
        "org_growth_rate", "high_quality_growth_rate", "date_coverage_ratio",
        "monitoring_priority", "monitoring_action", "next_observation_window_start",
        "next_observation_window_end", "temporal_validation_reason",
        "weak_signal_event_ratio", "low_attention_ratio", "niche_actor_ratio",
        "non_dominant_ratio", "cross_domain_ratio", "traceable_ratio", "multi_source_validated",
        "multi_source_validation_score", "semantic_validated", "semantic_validation_score",
        "semantic_validation_pairs", "event_doc_semantic_validated", "event_doc_validation_score",
        "event_doc_semantic_pairs", "event_paper_semantic_score", "event_patent_semantic_score",
        "literature_validated", "patent_validated", "time_validation_early_count",
        "time_validation_recent_count", "time_validation_ratio", "time_validated",
        "weak_signal_focus_candidate", "analysis_tech_field_name", "is_observation_scope", "scope_name", "scope_names",
        "is_scope_internal_candidate", "candidate_cluster_id", "display_candidate_aliases",
        "alias_count", "true_alias_count", "cluster_evidence_count", "cluster_item_count",
        "is_scope_echo", "has_mechanism_core", "has_task_constraint", "has_non_scope_constraint",
        "generic_core_only", "template_variant_count", "candidate_stage", "raw_phrase_type",
        "mechanism_core_tokens", "task_constraint_tokens", "object_modifier_tokens",
        "data_modifier_tokens", "method_modifier_tokens", "constraint_diversity",
        "merged_from_long_phrase_count", "generic_core_collision_count", "strong_ready_reason",
        "topic_granularity", "display_tier", "non_scope_constraint_count",
        "survives_without_scope", "scope_shell_heavy", "scope_shell_reason",
        "weak_signal_score", "hotspot_score", "score", "explanation",
        "upstream_signal_type",
        # 新增：内部标签与最终表述分离
        "raw_phrase_cluster", "final_research_object_name",
        "final_name_source", "technical_name_quality", "technical_name_issue",
        # 新增：对象族归一化信息
        "canonical_term", "normalization_type", "parent_term", "cross_source_pattern", "patent_role",
        # 新增：对象族感知聚合信息
        "family_id", "family_matched", "term_type", "family_priority",
        # 新增：跨源模式解释
        "cross_source_pattern_strength", "cross_source_pattern_reason",
        # 新增：专利侧角色解释
        "patent_role_reason", "patent_support_mode",
        "family_semantic_consistency", "family_semantic_validation_score",
        "family_semantic_validation_method", "family_semantic_validation_note",
        "candidate_eligibility", "eligibility_reason_codes", "eligible_for_scoring",
        "eligible_for_signal_generation", "eligible_for_weak_signal",
    ]


def generate_candidate_outputs(candidate_forms_df, data_df, domain_context=None):
    empty_columns = _empty_candidate_columns()
    diagnostics = {
        "input_candidate_count": int(len(candidate_forms_df)) if isinstance(candidate_forms_df, pd.DataFrame) else 0,
        "input_document_count": int(len(data_df)) if isinstance(data_df, pd.DataFrame) else 0,
        "active_candidate_count": 0,
        "upstream_weak_signal_count": 0,
        "scope_mismatch_filtered_count": 0,
        "domain_relevance_filtered_count": 0,
        "explicit_off_domain_filtered_count": 0,
        "output_candidate_count": 0,
        "final_weak_signal_count": 0,
        "near_strong_count": 0,
        "application_surface_rewritten_count": 0,
        "application_only_rejected_count": 0,
        "scope_mismatch_examples": [],
        "domain_relevance_filtered_examples": [],
    }
    if candidate_forms_df is None or candidate_forms_df.empty or data_df is None or data_df.empty:
        empty_df = pd.DataFrame(columns=empty_columns)
        return {"candidates_df": empty_df, "near_strong_candidates_df": empty_df.copy(), "diagnostics": diagnostics}

    data_df, identity_stats = _normalize_document_identity(data_df)
    diagnostics.update(identity_stats)

    canonicalizer = get_canonicalizer(domain_context=domain_context)

    active_forms = candidate_forms_df[
        candidate_forms_df["candidate_stage"].isin(["scope_overview", "formed_candidate", "formed_candidate_strong"])
    ].copy()

    if (
        "eligible_for_signal_generation" not in active_forms.columns
        or "candidate_eligibility" not in active_forms.columns
    ):
        active_forms = apply_candidate_eligibility(
            active_forms,
            phase="signal_generation_ready",
            domain_context=domain_context,
        )
    scope_overview_mask = active_forms["candidate_stage"] == "scope_overview"
    signal_eligible_mask = active_forms["eligible_for_signal_generation"].apply(_safe_bool)
    active_forms = active_forms[scope_overview_mask | signal_eligible_mask].copy()

    diagnostics["active_candidate_count"] = int(len(active_forms))
    if "signal_type" in active_forms.columns:
        diagnostics["upstream_weak_signal_count"] = int((active_forms["signal_type"].astype(str) == "weak_signal").sum())
    if active_forms.empty or "id" not in active_forms.columns or "id" not in data_df.columns:
        empty_df = pd.DataFrame(columns=empty_columns)
        return {"candidates_df": empty_df, "near_strong_candidates_df": empty_df.copy(), "diagnostics": diagnostics}

    metadata_by_id = data_df.drop_duplicates(subset="id", keep="last").set_index("id").to_dict("index")
    default_analysis_scope = _analysis_scope_from_data(data_df)
    group_maps = defaultdict(list)
    mechanism_collision = defaultdict(set)
    family_maps = {}  # 记录每个 cluster_key 对应的 family_info

    for _, concept_row in active_forms.iterrows():
        concept = concept_row.to_dict()
        source_record = metadata_by_id.get(concept.get("id"), {})
        record_scope = _analysis_scope_from_record(source_record)
        analysis_scope = record_scope or default_analysis_scope
        if analysis_scope and not _scope_matches_analysis(concept, analysis_scope, domain_context=domain_context):
            diagnostics["scope_mismatch_filtered_count"] += 1
            if len(diagnostics["scope_mismatch_examples"]) < 10:
                diagnostics["scope_mismatch_examples"].append(
                    {
                        "id": str(concept.get("id", "")).strip(),
                        "display_candidate_name": str(concept.get("display_candidate_name", "")).strip(),
                        "analysis_scope": analysis_scope,
                        "candidate_scope": str(concept.get("scope_name", concept.get("primary_scope", ""))).strip(),
                    }
                )
            continue
        concept = _attach_analysis_scope(concept, analysis_scope)
        relevance_passed, relevance_reason = _concept_relevance_decision(
            concept,
            source_record,
            analysis_scope,
            domain_context=domain_context,
            canonicalizer=canonicalizer,
        )
        if analysis_scope and not relevance_passed:
            diagnostics["domain_relevance_filtered_count"] += 1
            if relevance_reason == "explicit_off_domain":
                diagnostics["explicit_off_domain_filtered_count"] += 1
            if len(diagnostics["domain_relevance_filtered_examples"]) < 10:
                diagnostics["domain_relevance_filtered_examples"].append(
                    {
                        "id": str(concept.get("id", "")).strip(),
                        "display_candidate_name": str(concept.get("display_candidate_name", "")).strip(),
                        "analysis_scope": analysis_scope,
                        "reason": relevance_reason,
                    }
                )
            continue

        primary_scope = str(concept.get("primary_scope", "")).strip()
        mechanism_core = str(concept.get("mechanism_core", "")).strip()
        constraint_signature = str(concept.get("constraint_signature", "")).strip()
        canonical_name = str(concept.get("canonical_candidate_name_en", "")).strip() or str(concept.get("normalized_candidate_text", "")).strip()
        object_family_allowed = _domain_allows_object_family(
            concept,
            domain_context=domain_context,
            canonicalizer=canonicalizer,
        )

        # 优先使用 candidate_forms 中已有的 candidate_cluster_id，确保整个流程中键的一致性
        existing_cluster_id = str(concept.get("candidate_cluster_id", "")).strip()
        if existing_cluster_id and existing_cluster_id not in {"", "nan", "None"}:
            cluster_key = existing_cluster_id
            # 仍然尝试获取 family_info
            family_info = None
            family = canonicalizer.match_family(canonical_name) if object_family_allowed and canonical_name else None
            if family:
                family_info = family
                family_maps[cluster_key] = family_info
        else:
            # 使用对象族感知聚合生成新的 cluster_key
            cluster_key, family_info = _object_family_candidate_key(
                concept,
                canonicalizer,
                object_family_allowed=object_family_allowed,
            )

            # 记录 family_info
            if family_info:
                family_maps[cluster_key] = family_info

        # 如果没有生成有效的 cluster_key，使用 fallback
        if not cluster_key:
            cluster_key = " || ".join([primary_scope, mechanism_core, constraint_signature, canonical_name]) or str(concept.get("raw_candidate_text", "")).strip()

        group_maps[cluster_key].append(concept)
        if mechanism_core and constraint_signature:
            mechanism_collision[mechanism_core].add(constraint_signature)

    candidates = []
    for cluster_key, items in group_maps.items():
        first = items[0]
        counts = defaultdict(int)
        orgs = set()
        mention_ids = set()
        mention_dates = []
        evidence_titles = []
        evidence_items = []
        mention_records_by_id = {}
        preserved_cluster_evidence_count = 0
        preserved_total_mentions = 0
        preserved_source_count = 0
        preserved_source_types = set()
        for item in items:
            preserved_cluster_evidence_count = max(
                preserved_cluster_evidence_count,
                _safe_int(item.get("cluster_evidence_count"), 0),
            )
            preserved_total_mentions = max(
                preserved_total_mentions,
                _safe_int(item.get("total_mentions"), 0),
            )
            preserved_source_count = max(
                preserved_source_count,
                _safe_int(item.get("source_count"), 0),
            )
            for source_type in _coerce_text_list(item.get("source_types", [])):
                normalized_source_type = _normalize_source_type(source_type)
                preserved_source_types.add(normalized_source_type)
            item_mention_ids = _coerce_text_list(item.get("mention_ids", []))
            if not item_mention_ids and item.get("id") is not None:
                item_mention_ids = [item.get("id")]
            for mention_id in item_mention_ids:
                mention_ids.add(mention_id)
                record = metadata_by_id.get(mention_id) or metadata_by_id.get(item.get("id"))
                if record:
                    mention_records_by_id.setdefault(mention_id, record)
            for date_text in _coerce_text_list(item.get("mention_dates", [])):
                if date_text:
                    mention_dates.append(date_text)
            for title in _coerce_text_list(item.get("evidence_titles", [])):
                if title and title not in evidence_titles:
                    evidence_titles.append(title)
            for evidence_item in item.get("evidence_items", []) or []:
                if isinstance(evidence_item, dict):
                    title = _safe_text(evidence_item.get("title", ""))
                    if title and title not in evidence_titles:
                        evidence_titles.append(title)
                    evidence_items.append(evidence_item)

        for mention_id in sorted(mention_ids):
            record = mention_records_by_id.get(mention_id)
            if not record:
                continue
            source_type = _normalize_source_type(record.get("source_type", "unknown"))
            counts[source_type] += 1
            org = _safe_text(record.get("org", "Unknown")) or "Unknown"
            orgs.add(org)
            date_text = _safe_text(record.get("date"))
            if date_text:
                mention_dates.append(date_text)
            title = _safe_text(record.get("title", ""))
            if title and title not in evidence_titles:
                evidence_titles.append(title)
                evidence_items.append(_build_evidence_item(record, first))

        for source_type in preserved_source_types:
            counts[source_type] = max(counts[source_type], 1)
        source_types = sorted({k for k, v in counts.items() if v > 0} | preserved_source_types)
        cluster_evidence_count = max(
            len(mention_ids),
            preserved_cluster_evidence_count,
            len(items),
        )
        observed_total_mentions = len(mention_ids) or sum(counts.values())
        total_mentions = max(
            preserved_total_mentions,
            cluster_evidence_count,
            observed_total_mentions,
        )
        source_count = max(len(source_types), preserved_source_count)
        mention_records = list(mention_records_by_id.values())
        raw_variant_aliases = sorted(
            {
                str(item.get("raw_phrase", "")).strip()
                for item in items
                if str(item.get("raw_phrase", "")).strip()
            }
        )
        alias_terms = sorted(
            {
                alias
                for item in items
                for alias in _normalize_aliases(item.get("display_candidate_aliases", []))
                if alias
            }
        )
        if not alias_terms and first.get("canonical_candidate_name_en"):
            alias_terms = [first["canonical_candidate_name_en"]]
        alias_terms = sorted({*alias_terms, *raw_variant_aliases[:4]})
        early_mentions, recent_mentions, validation_ratio, time_validated = _time_validation_metrics(mention_dates)
        multi_source_validated = source_count >= 2
        multi_source_validation_score = 2.0 if source_count >= 3 else 1.5 if source_count == 2 else 0.0
        semantic_validated, semantic_score, semantic_pair_count = _cross_source_semantic_validation(evidence_items, alias_terms)
        event_doc_validation = _event_document_validation(evidence_items, alias_terms)
        candidate_stage = str(first.get("candidate_stage", "")).strip()
        evidence_ready_formed_candidate = bool(
            candidate_stage == "formed_candidate"
            and cluster_evidence_count >= 2
            and bool(first.get("is_scope_internal_candidate", False))
            and not bool(first.get("is_scope_echo", False))
            and bool(first.get("has_mechanism_core", False))
            and bool(first.get("has_non_scope_constraint", False))
            and str(first.get("topic_granularity", "")).strip() == "fine_grained_topic"
            and str(first.get("display_tier", "")).strip() == "weak_signal"
            and (
                multi_source_validated
                or semantic_validated
                or event_doc_validation["event_doc_semantic_validated"]
            )
        )
        weak_signal_focus_candidate = (
            candidate_stage in {"formed_candidate_strong", "formed_candidate"}
            and bool(first.get("is_scope_internal_candidate", False))
            and not bool(first.get("is_scope_echo", False))
            and bool(first.get("has_mechanism_core", False))
            and bool(first.get("has_non_scope_constraint", False))
            and (candidate_stage == "formed_candidate_strong" or evidence_ready_formed_candidate)
            and (multi_source_validated or semantic_validated or event_doc_validation["event_doc_semantic_validated"])
        )
        constraint_signatures = {
            str(item.get("constraint_signature", "")).strip()
            for item in items
            if str(item.get("constraint_signature", "")).strip()
        }

        # 对象族归一化（使用分层归一）
        display_name = str(first.get("display_candidate_name", "")).strip()
        canonical_name_en = str(first.get("canonical_candidate_name_en", "")).strip()
        raw_phrase = str(first.get("raw_phrase", "")).strip()
        object_family_allowed = _domain_allows_object_family(
            first,
            domain_context=domain_context,
            canonicalizer=canonicalizer,
        )

        # 使用分层归一化（始终执行，用于获取 canonical_term 等信息）
        norm_input = canonical_name_en or raw_phrase or display_name
        norm_result = (
            canonicalizer.canonicalize_with_layers(norm_input)
            if object_family_allowed
            else _neutral_canonicalization(norm_input)
        )

        # 如果没有匹配，尝试从 raw_variant_aliases 中提取
        if object_family_allowed and norm_result.normalization_type == 'none' and raw_variant_aliases:
            for alias in raw_variant_aliases:
                alias_result = canonicalizer.canonicalize_with_layers(alias)
                if alias_result.normalization_type != 'none':
                    norm_result = alias_result
                    break

        # 优先从 family_maps 获取拆分后的 family_info（覆盖 norm_result）
        family_info = family_maps.get(cluster_key) if object_family_allowed else None
        if family_info:
            family_matched = True
            family_id = family_info.get('family_id', '')
            term_type = family_info.get('term_type', 'unknown')
            family_priority = family_info.get('priority', 'low')
        else:
            # 使用 norm_result 的信息
            family_matched = norm_result.family_id != ""
            family_id = norm_result.family_id
            term_type = norm_result.term_type
            family_priority = norm_result.priority

        # 获取跨源模式信息
        cross_source_pattern_name, pattern_info = canonicalizer.get_cross_source_pattern(source_types)
        cross_source_pattern_strength = pattern_info.get('strength', 'weak') if pattern_info else 'weak'
        cross_source_pattern_reason = pattern_info.get('reason', '') if pattern_info else ''

        # 获取专利角色信息
        if family_info:
            patent_role = family_info.get('patent_role', 'optional')
        else:
            patent_role = norm_result.patent_role
        patent_role_reason = ""
        patent_support_mode = ""
        if patent_role == 'engineering_signal':
            patent_role_reason = "专利侧提供工程信号补充"
            patent_support_mode = "技术实现细节、工程参数、应用场景"
        elif patent_role == 'application_trace':
            patent_role_reason = "专利侧提供产业化证据"
            patent_support_mode = "产品化、商业化、落地场景"
        elif patent_role == 'validation':
            patent_role_reason = "专利侧提供技术成熟度证据"
            patent_support_mode = "技术路线稳定性、市场认可度"
        else:
            patent_role_reason = "专利侧不强制要求"
            patent_support_mode = "任何补充信息"

        # 获取对象族信息（用于 family-scoped semantic validation）
        # family_info 已在上面设置

        # 执行 family-scoped semantic validation
        family_semantic_result = _family_scoped_semantic_validation(
            evidence_items, family_info, alias_terms
        )
        if family_semantic_result is None:
            family_semantic_result = {
                "family_semantic_consistency": "",
                "family_semantic_validation_score": 0.0,
                "family_semantic_validation_method": "",
                "family_semantic_validation_note": ""
            }

        candidate_quality_values = [
            _safe_float(item.get("candidate_evidence_quality"), default=-1.0)
            for item in items
        ]
        candidate_quality_values = [value for value in candidate_quality_values if value >= 0]
        candidate_evidence_quality = (
            round(sum(candidate_quality_values) / len(candidate_quality_values), 2)
            if candidate_quality_values
            else 0.0
        )
        candidate_core_quality_values = [
            _safe_float(item.get("candidate_core_evidence_quality"), default=-1.0)
            for item in items
        ]
        candidate_core_quality_values = [value for value in candidate_core_quality_values if value >= 0]
        candidate_core_evidence_quality = (
            round(sum(candidate_core_quality_values) / len(candidate_core_quality_values), 2)
            if candidate_core_quality_values
            else candidate_evidence_quality
        )
        low_quality_evidence_ratio = _safe_float(first.get("low_quality_evidence_ratio"), 0.0)
        high_quality_evidence_count = int(_safe_float(first.get("high_quality_evidence_count"), 0.0))
        quality_risk_flag = str(first.get("quality_risk_flag", "")).strip()
        quality_adjusted_rank_score = _safe_float(
            first.get("quality_adjusted_rank_score"),
            _safe_float(first.get("weak_signal_score", first.get("score", 0.0))),
        )
        candidate_evidence_quality_reason = str(first.get("candidate_evidence_quality_reason", "")).strip()
        name_resolution = _resolve_final_technical_name(first)
        final_technical_name = _safe_text(name_resolution.get("name", ""))
        display_surface_name = _safe_text(first.get("display_candidate_name", ""))
        if (
            final_technical_name
            and display_surface_name
            and final_technical_name != display_surface_name
            and _is_application_surface_name(display_surface_name)
            and name_resolution.get("quality") == "technical"
        ):
            diagnostics["application_surface_rewritten_count"] += 1
        candidates.append(
            {
                "tech_name": final_technical_name or str(first.get("canonical_candidate_name_en", "")).strip() or str(first.get("raw_candidate_text", "")).strip(),
                "technology": final_technical_name or str(first.get("canonical_candidate_name_en", "")).strip(),
                "display_candidate_name": final_technical_name or str(first.get("display_candidate_name", "")).strip() or str(first.get("canonical_candidate_name_en", "")).strip(),
                "normalized_candidate_text": str(first.get("normalized_candidate_text", "")).strip(),
                "canonical_candidate_name_en": str(first.get("canonical_candidate_name_en", "")).strip(),
                "constraint_signature": str(first.get("constraint_signature", "")).strip(),
                "internal_candidate_label": str(first.get("internal_candidate_label", "")).strip(),
                "relation_target": str(first.get("relation_target", "")).strip(),
                "relation_task": str(first.get("relation_task", "")).strip(),
                "relation_data_modality": str(first.get("relation_data_modality", "")).strip(),
                "relation_method": str(first.get("relation_method", "")).strip(),
                "relation_summary": str(first.get("relation_summary", "")).strip(),
                "relation_signature": str(first.get("relation_signature", "")).strip(),
                "topic_summary_name": str(first.get("topic_summary_name", "")).strip(),
                "topic_naturalness_reason": str(first.get("topic_naturalness_reason", "")).strip(),
                "compression_mode": str(first.get("compression_mode", "")).strip(),
                "mechanism_core": str(first.get("mechanism_core", "")).strip(),
                "news_count": counts.get("news", 0),
                "paper_count": counts.get("paper", 0) + counts.get("report", 0),
                "report_count": counts.get("paper", 0) + counts.get("report", 0),
                "patent_count": counts.get("patent", 0),
                "total_mentions": total_mentions,
                "org_count": len(orgs),
                "orgs": sorted(orgs),
                "source_types": source_types,
                "source_count": source_count,
                "mention_ids": sorted(mention_ids),
                "mention_dates": mention_dates,
                "evidence_titles": evidence_titles[:5],
                "evidence_items": evidence_items[:5],
                "weak_signal_event_count": cluster_evidence_count,
                "candidate_evidence_quality": candidate_evidence_quality,
                "candidate_core_evidence_quality": candidate_core_evidence_quality,
                "low_quality_evidence_ratio": low_quality_evidence_ratio,
                "high_quality_evidence_count": high_quality_evidence_count,
                "quality_risk_flag": quality_risk_flag,
                "quality_adjusted_rank_score": quality_adjusted_rank_score,
                "candidate_evidence_quality_reason": candidate_evidence_quality_reason,
                "candidate_id": str(first.get("candidate_id", "")).strip(),
                "first_seen_date": str(first.get("first_seen_date", "")).strip(),
                "last_seen_date": str(first.get("last_seen_date", "")).strip(),
                "temporal_validation_tier": str(first.get("temporal_validation_tier", "")).strip(),
                "temporal_validation_status": str(first.get("temporal_validation_status", "")).strip(),
                "temporal_validation_passed": _safe_bool(first.get("temporal_validation_passed", False)),
                "temporal_momentum_score": _safe_float(first.get("temporal_momentum_score"), 0.0),
                "growth_rate": _safe_float(first.get("growth_rate"), 0.0),
                "source_growth_rate": _safe_float(first.get("source_growth_rate"), 0.0),
                "org_growth_rate": _safe_float(first.get("org_growth_rate"), 0.0),
                "high_quality_growth_rate": _safe_float(first.get("high_quality_growth_rate"), 0.0),
                "date_coverage_ratio": _safe_float(first.get("date_coverage_ratio"), 0.0),
                "monitoring_priority": str(first.get("monitoring_priority", "")).strip(),
                "monitoring_action": str(first.get("monitoring_action", "")).strip(),
                "next_observation_window_start": str(first.get("next_observation_window_start", "")).strip(),
                "next_observation_window_end": str(first.get("next_observation_window_end", "")).strip(),
                "temporal_validation_reason": str(first.get("temporal_validation_reason", "")).strip(),
                "weak_signal_event_ratio": round(cluster_evidence_count / max(total_mentions, 1), 2),
                "low_attention_ratio": round(sum(1 for record in mention_records if record.get("source_type") in {"paper", "patent"}) / max(total_mentions, 1), 2),
                "niche_actor_ratio": round(sum(1 for org in orgs if org not in {"Unknown", "arXiv"}) / max(len(orgs), 1), 2),
                "non_dominant_ratio": round(sum(1 for item in items if item.get("has_mechanism_core")) / max(len(items), 1), 2),
                "cross_domain_ratio": round(sum(1 for item in items if item.get("has_non_scope_constraint")) / max(len(items), 1), 2),
                "traceable_ratio": round(sum(1 for record in mention_records if bool(record.get("url") or record.get("date"))) / max(total_mentions, 1), 2),
                "multi_source_validated": multi_source_validated,
                "multi_source_validation_score": multi_source_validation_score,
                "semantic_validated": semantic_validated,
                "semantic_validation_score": semantic_score,
                "semantic_validation_pairs": semantic_pair_count,
                "event_doc_semantic_validated": event_doc_validation["event_doc_semantic_validated"],
                "event_doc_validation_score": event_doc_validation["event_doc_validation_score"],
                "event_doc_semantic_pairs": event_doc_validation["event_doc_semantic_pairs"],
                "event_paper_semantic_score": event_doc_validation["event_paper_semantic_score"],
                "event_patent_semantic_score": event_doc_validation["event_patent_semantic_score"],
                "literature_validated": event_doc_validation["event_paper_semantic_score"] > 0,
                "patent_validated": event_doc_validation["event_patent_semantic_score"] > 0,
                "time_validation_early_count": early_mentions,
                "time_validation_recent_count": recent_mentions,
                "time_validation_ratio": validation_ratio,
                "time_validated": time_validated,
                "weak_signal_focus_candidate": weak_signal_focus_candidate,
                "analysis_tech_field_name": str(first.get("analysis_tech_field_name", "")).strip() or str(first.get("primary_scope", "")).strip(),
                "is_observation_scope": bool(first.get("is_observation_scope", False)),
                "scope_name": str(first.get("scope_name", "")).strip(),
                "scope_names": _normalize_scope_names(first.get("scope_names", [])),
                "primary_scope": str(first.get("primary_scope", "")).strip(),
                "is_scope_internal_candidate": bool(first.get("is_scope_internal_candidate", False)),
                "candidate_cluster_id": cluster_key,
                "display_candidate_aliases": alias_terms,
                "alias_count": len(alias_terms),
                "true_alias_count": len(alias_terms),
                "cluster_evidence_count": cluster_evidence_count,
                "cluster_item_count": len(items),
                "is_scope_echo": bool(first.get("is_scope_echo", False)),
                "has_mechanism_core": bool(first.get("has_mechanism_core", False)),
                "has_task_constraint": bool(first.get("has_task_constraint", False)),
                "has_non_scope_constraint": bool(first.get("has_non_scope_constraint", False)),
                "generic_core_only": bool(first.get("generic_core_only", False)),
                "template_variant_count": len({str(item.get('raw_phrase', '')).strip() for item in items if str(item.get('raw_phrase', '')).strip()}) - len(alias_terms),
                "candidate_stage": candidate_stage,
                "raw_phrase_type": str(first.get("raw_phrase_type", "")).strip(),
                "mechanism_core_tokens": _normalize_aliases(first.get("mechanism_core_tokens", [])),
                "task_constraint_tokens": _normalize_aliases(first.get("task_constraint_tokens", [])),
                "object_modifier_tokens": _normalize_aliases(first.get("object_modifier_tokens", [])),
                "data_modifier_tokens": _normalize_aliases(first.get("data_modifier_tokens", [])),
                "method_modifier_tokens": _normalize_aliases(first.get("method_modifier_tokens", [])),
                "constraint_diversity": len(constraint_signatures),
                "merged_from_long_phrase_count": sum(1 for item in items if len(str(item.get("raw_phrase", "")).split()) >= 4),
                "generic_core_collision_count": max(len(mechanism_collision.get(str(first.get("mechanism_core", "")).strip(), set())) - 1, 0),
                "strong_ready_reason": str(first.get("strong_ready_reason", "")).strip(),
                "topic_granularity": str(first.get("topic_granularity", "")).strip(),
                "display_tier": str(first.get("display_tier", "")).strip(),
                "non_scope_constraint_count": int(first.get("non_scope_constraint_count", 0) or 0),
                "survives_without_scope": bool(first.get("survives_without_scope", False)),
                "scope_shell_heavy": bool(first.get("scope_shell_heavy", False)),
                "scope_shell_reason": str(first.get("scope_shell_reason", "")).strip(),
                "weak_signal_score": float(first.get("weak_signal_score", 0.0) or 0.0),
                "hotspot_score": float(first.get("hotspot_score", 0.0) or 0.0),
                "score": float(first.get("score", first.get("weak_signal_score", first.get("hotspot_score", 0.0))) or 0.0),
                "explanation": str(first.get("explanation", "")).strip(),
                "upstream_signal_type": str(first.get("signal_type", "")).strip(),
                "candidate_eligibility": str(first.get("candidate_eligibility", "")).strip(),
                "eligibility_reason_codes": first.get("eligibility_reason_codes", []),
                "eligible_for_scoring": _safe_bool(first.get("eligible_for_scoring", False)),
                "eligible_for_signal_generation": _safe_bool(first.get("eligible_for_signal_generation", False)),
                "eligible_for_weak_signal": _safe_bool(first.get("eligible_for_weak_signal", False)),
                # 新增：内部标签与最终表述分离
                "raw_phrase_cluster": "；".join(raw_variant_aliases[:5]) if raw_variant_aliases else "",
                "final_research_object_name": final_technical_name or str(first.get("topic_summary_name", "")).strip() or str(first.get("display_candidate_name", "")).strip(),
                "final_name_source": str(name_resolution.get("source", "")).strip(),
                "technical_name_quality": str(name_resolution.get("quality", "")).strip(),
                "technical_name_issue": str(name_resolution.get("issue", "")).strip(),
                # 新增：对象族归一化信息
                "canonical_term": norm_result.canonical_term,
                "normalization_type": norm_result.normalization_type,
                "parent_term": norm_result.parent_term,
                "cross_source_pattern": cross_source_pattern_name,
                "patent_role": patent_role,
                # 新增：对象族感知聚合信息
                "family_id": family_id,
                "family_matched": family_matched,
                "term_type": term_type,
                "family_priority": family_priority,
                # 新增：跨源模式解释
                "cross_source_pattern_strength": cross_source_pattern_strength,
                "cross_source_pattern_reason": cross_source_pattern_reason,
                # 新增：专利侧角色解释
                "patent_role_reason": patent_role_reason,
                "patent_support_mode": patent_support_mode,
                # 新增：对象族范围语义验证
                "family_semantic_consistency": family_semantic_result.get("family_semantic_consistency", ""),
                "family_semantic_validation_score": family_semantic_result.get("family_semantic_validation_score", 0.0),
                "family_semantic_validation_method": family_semantic_result.get("family_semantic_validation_method", ""),
                "family_semantic_validation_note": family_semantic_result.get("family_semantic_validation_note", ""),
            }
        )

    candidates_df = pd.DataFrame(candidates, columns=_empty_candidate_columns())
    if candidates_df.empty:
        return {"candidates_df": candidates_df, "near_strong_candidates_df": candidates_df.copy(), "diagnostics": diagnostics}
    
    # 设置信号类型
    def _set_signal_type(row):
        if str(row.get("technical_name_issue", "")).strip() == "application_only_surface":
            return "other"
        if row.get("candidate_stage") != "scope_overview" and not _safe_bool(row.get("eligible_for_weak_signal", False)):
            return "other"
        upstream_signal_type = str(row.get("upstream_signal_type", "")).strip()
        if upstream_signal_type in {"weak_signal", "hotspot", "near_strong", "scope_overview", "other"}:
            return upstream_signal_type
        if row.get("candidate_stage") == "scope_overview":
            return "scope_overview"
        elif row.get("candidate_stage") in ["formed_candidate", "formed_candidate_strong"]:
            evidence_ready = bool(
                _safe_int(row.get("cluster_evidence_count"), 0) >= 2
                and _safe_int(row.get("source_count"), 0) >= 2
                and str(row.get("topic_granularity", "")).strip() == "fine_grained_topic"
                and str(row.get("display_tier", "")).strip() == "weak_signal"
                and row.get("is_scope_internal_candidate")
                and row.get("has_mechanism_core")
                and row.get("has_non_scope_constraint")
                and not row.get("is_scope_echo")
            )
            if evidence_ready:
                return "weak_signal"
            if row.get("is_scope_internal_candidate") and row.get("has_mechanism_core") and row.get("has_non_scope_constraint"):
                return "near_strong"
            else:
                return "other"
        else:
            return "other"
    
    candidates_df["signal_type"] = candidates_df.apply(_set_signal_type, axis=1)
    diagnostics["application_only_rejected_count"] = int(
        (candidates_df["technical_name_issue"].astype(str) == "application_only_surface").sum()
    )

    candidates_df = candidates_df.sort_values(
        by=["total_mentions", "source_count", "org_count", "tech_name"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)

    if "candidate_eligibility" in candidates_df.columns:
        trends_mask = candidates_df["candidate_eligibility"] == "candidate_monitoring"
        trends_df = candidates_df[trends_mask].copy().reset_index(drop=True)
        candidates_df = candidates_df[~trends_mask].copy().reset_index(drop=True)
    else:
        trends_df = pd.DataFrame(columns=candidates_df.columns)

    near_strong_candidates_df = candidates_df[
        (candidates_df["signal_type"] == "near_strong")
    ].copy()
    diagnostics["output_candidate_count"] = int(len(candidates_df))
    diagnostics["final_weak_signal_count"] = int((candidates_df["signal_type"] == "weak_signal").sum())
    diagnostics["near_strong_count"] = int(len(near_strong_candidates_df))
    diagnostics["trends_count"] = int(len(trends_df))
    return {
        "candidates_df": candidates_df,
        "near_strong_candidates_df": near_strong_candidates_df.reset_index(drop=True),
        "trends_df": trends_df,
        "diagnostics": diagnostics,
    }


def generate_candidates(candidate_forms_df, data_df, domain_context=None):
    return generate_candidate_outputs(candidate_forms_df, data_df, domain_context=domain_context)["candidates_df"]
