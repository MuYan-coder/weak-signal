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


_NULL_TEXT_VALUES = {"", "nan", "nat", "none", "null"}


def safe_event_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, (list, tuple, set)):
        text = "、".join(str(item).strip() for item in value if str(item).strip())
        if text.lower() in _NULL_TEXT_VALUES:
            return default
        return text or default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if text.lower() in _NULL_TEXT_VALUES:
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


_UNKNOWN_TEXT_VALUES = _NULL_TEXT_VALUES | {"unknown", "未知", "无", "n/a", "na"}
_PLACEHOLDER_TEXT_VALUES = _UNKNOWN_TEXT_VALUES | {
    "string", "str", "text", "value", "field", "placeholder", "example",
    "sample", "todo", "tbd", "n.a.", "not applicable",
    "示例", "样例", "占位", "占位符", "待填", "待补充", "待定",
    "字段", "文本", "内容", "空", "不详", "无信息",
}
_PLACEHOLDER_TEXT_RE = re.compile(
    r"^(?:string|str|text|value|field|placeholder|example|sample|todo|tbd|"
    r"示例|样例|占位|占位符|待填|待补充|待定|字段|文本|内容)(?:[\s_\-:：]*\d+)?$",
    re.IGNORECASE,
)


def _compact_placeholder_key(text: str) -> str:
    return re.sub(r"[\s\W_]+", "", str(text or "").lower())


_PLACEHOLDER_COMPACT_KEYS = {_compact_placeholder_key(value) for value in _PLACEHOLDER_TEXT_VALUES}


def _strip_value_markup(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"^(?:字段|field|value|text)\s*[:：]\s*", "", text, flags=re.IGNORECASE).strip()
    return text.strip(" \t\r\n\"'“”‘’[]{}()（）<>《》")


def is_placeholder_event_value(value: Any) -> bool:
    text = safe_event_text(value)
    if not text:
        return True
    text = _strip_value_markup(text)
    if not text:
        return True
    lowered = text.lower()
    compact = _compact_placeholder_key(text)
    if lowered in _PLACEHOLDER_TEXT_VALUES or compact in _PLACEHOLDER_COMPACT_KEYS:
        return True
    if _PLACEHOLDER_TEXT_RE.fullmatch(text):
        return True
    if re.fullmatch(r"(?:string|str|text|value|field)\d*", compact):
        return True
    return False


def clean_event_text(value: Any, default: str = "") -> str:
    text = safe_event_text(value, default)
    if not text:
        return default
    text = _strip_value_markup(text)
    if is_placeholder_event_value(text):
        return default
    return text or default


def _clean_event_mapping(value: Dict[str, Any]) -> Dict[str, Any]:
    cleaned: Dict[str, Any] = {}
    for key, item in (value or {}).items():
        if isinstance(item, dict):
            nested = _clean_event_mapping(item)
            if nested:
                cleaned[key] = nested
        elif isinstance(item, (list, tuple, set)):
            nested_list = clean_event_list(item)
            if nested_list:
                cleaned[key] = nested_list
        elif isinstance(item, bool):
            cleaned[key] = item
        elif isinstance(item, (int, float)):
            try:
                if pd.isna(item):
                    continue
            except (TypeError, ValueError):
                pass
            cleaned[key] = item
        else:
            text = clean_event_text(item)
            if text:
                cleaned[key] = text
    return cleaned


def clean_event_list(value: Any) -> List[Any]:
    cleaned = []
    for item in coerce_event_list(value):
        if isinstance(item, dict):
            nested = _clean_event_mapping(item)
            if nested:
                cleaned.append(nested)
            continue
        text = clean_event_text(item)
        if text:
            cleaned.append(text)
    return _dedupe_preserve_order(cleaned)

_SUBJECT_METADATA_FIELDS = [
    "org", "organization", "organizations", "affiliation", "affiliations",
    "institution", "institutions", "company", "assignee", "assignees",
    "applicant", "applicants", "applicant_cn", "applicant_en",
    "assignee_cn", "assignee_en", "owner", "patentee", "applicant_name",
    "申请人", "申请单位", "专利申请人", "申请机构", "专利权人", "权利人",
    "机构", "单位", "所属机构", "来源机构", "inventor",
    "inventors", "inventor_cn", "inventor_en", "author", "authors",
    "author_name", "发明人", "作者",
]

_SUBJECT_CONTEXT_FIELDS = [
    "title", "text", "abstract", "summary", "content", "snippet",
    "main_content", "description",
]

_GENERIC_SUBJECT_VALUES = {
    "我们", "我", "我方", "本文", "本研究", "该研究", "本论文", "论文",
    "作者", "研究", "研究人员", "研究者", "研究机构", "研究团队", "团队",
    "项目组", "课题组", "实验室", "企业", "公司", "科技公司", "厂商",
    "申请人", "发明人", "本发明", "该发明", "本申请", "本公开", "该公开",
    "本实用新型", "该方法", "本方法", "该系统", "本系统", "该模型",
    "本模型", "该装置", "本装置", "该技术", "该方案", "所述方法",
    "所述系统", "所述装置", "所述模型",
    "we", "us", "our", "ours", "i", "me", "they", "them", "this paper",
    "the paper", "this study", "the study", "this work", "the work",
    "authors", "the authors", "researchers", "the researchers",
    "inventors", "applicants", "company", "team", "lab", "laboratory",
    "method", "system", "model", "framework", "technology",
}

_SUBJECT_ACTOR_MARKERS_ZH = (
    "大学", "学院", "研究院", "研究所", "实验室", "公司", "集团", "企业",
    "厂商", "团队", "课题组", "项目组", "中心", "医院", "组织",
    "协会", "联盟", "委员会", "部门", "研究室", "工作室", "申请人",
    "发明人", "作者", "株式会社", "有限公司", "股份有限公司",
)

_SUBJECT_ACTOR_MARKERS_EN = (
    "university", "institute", "laboratory", " lab", " labs", "company",
    "corporation", " corp", "inc", "ltd", "llc", "gmbh", "team", "group",
    "researchers", "authors", "inventor", "inventors",
    "applicant", "applicants", "assignee", "assignees",
)

_COMMON_ORG_NAMES = {
    "google", "openai", "microsoft", "meta", "amazon", "ibm", "nvidia",
    "baidu", "alibaba", "tencent", "huawei", "deepmind", "anthropic",
    "tesla", "mit", "stanford", "tsinghua", "pku", "清华", "清华大学",
    "北大", "北京大学", "中科院", "特斯拉", "谷歌", "微软", "英伟达",
    "华为", "腾讯", "阿里巴巴", "百度",
}

_TECHNICAL_SUBJECT_KEYWORDS_ZH = (
    "世界模型", "大模型", "模型", "算法", "方法", "系统", "框架", "技术",
    "方案", "策略", "机制", "结构", "装置", "设备", "模块", "组件",
    "单元", "控制器", "传感器", "执行器", "机器人", "机械臂", "机械手",
    "规划", "控制", "训练", "学习", "仿真", "推理", "记忆", "生成",
    "数据", "网络", "平台", "工具", "应用", "场景", "路径", "任务",
    "多模态", "触觉", "视觉", "点云", "强化学习", "扩散", "变换器",
)

_SUBJECT_GEO_TERMS_ZH = (
    "中国", "美国", "日本", "韩国", "德国", "法国", "英国", "欧洲",
    "印度", "加拿大", "澳大利亚", "新加坡", "以色列", "瑞士", "瑞典",
    "荷兰", "俄罗斯", "国内", "国外", "海外", "国际", "亚洲", "北美",
)

_SUBJECT_GEO_TERMS_EN = (
    "chinese", "china", "american", "us", "u.s.", "united states",
    "japanese", "japan", "korean", "german", "french", "british", "uk",
    "european", "indian", "canadian", "australian", "singaporean",
    "israeli", "swiss", "swedish", "dutch", "russian", "overseas",
    "international",
)

_GENERIC_ACTOR_ROLE_TERMS_ZH = (
    "科学家", "研究人员", "研究者", "研究团队", "团队", "工程师", "专家",
    "学者", "作者", "发明人", "申请人", "开发者", "人员",
)

_GENERIC_ACTOR_ROLE_TERMS_EN = (
    "scientist", "scientists", "researcher", "researchers", "team",
    "teams", "engineer", "engineers", "expert", "experts", "scholar",
    "scholars", "author", "authors", "inventor", "inventors",
    "applicant", "applicants", "developer", "developers",
)

_CONCRETE_ACTOR_SUFFIX_ZH = (
    "大学", "学院", "研究院", "研究所", "实验室", "公司", "集团", "企业",
    "团队", "课题组", "项目组", "中心", "医院", "联盟", "委员会",
    "株式会社", "有限公司", "股份有限公司",
)

_CONCRETE_ACTOR_SUFFIX_EN = (
    "University", "Institute", "Laboratory", "Lab", "Labs", "Corporation",
    "Corp", "Inc", "Ltd", "LLC", "Group", "College", "School",
)

_TECHNICAL_SUBJECT_KEYWORDS_EN = (
    "model", "algorithm", "method", "system", "framework", "technology",
    "technique", "approach", "architecture", "device", "apparatus",
    "module", "component", "controller", "sensor", "actuator", "robot",
    "robotic", "planning", "control", "training", "learning",
    "simulation", "reasoning", "memory", "generation", "data", "network",
    "platform", "tool", "application", "scenario", "policy",
    "transformer", "diffusion", "benchmark", "dataset",
)

_STANDARD_ACTION_VERBS = [
    ("开源", ["开源", "开放源代码", "open-source", "open sourced", "open-sourced"]),
    ("发布", ["发布", "推出", "公布", "release", "released", "launch", "launched", "announce", "announced"]),
    ("提出", ["提出", "提议", "propose", "proposed", "introduce", "introduced", "present", "presented"]),
    ("研发", ["研发", "开发", "研制", "develop", "developed", "create", "created", "build", "built"]),
    ("设计", ["设计", "design", "designed"]),
    ("实现", ["实现", "构建", "implement", "implemented", "realize", "realized"]),
    ("展示", ["展示", "演示", "demonstrate", "demonstrated", "show", "showed"]),
    ("验证", ["验证", "validate", "validated", "verify", "verified"]),
    ("测试", ["测试", "test", "tested", "evaluate", "evaluated"]),
    ("申请", ["申请", "申报", "file", "filed"]),
    ("公开", ["公开", "披露", "publish", "published", "disclose", "disclosed"]),
    ("部署", ["部署", "落地", "应用", "deploy", "deployed", "adopt", "adopted"]),
    ("训练", ["训练", "微调", "优化", "train", "trained", "training", "fine-tune", "fine-tuned", "optimize", "optimized"]),
]

_ACTION_NOUN_HINTS_ZH = (
    "系统", "方法", "算法", "模型", "框架", "技术", "方案", "平台", "装置",
    "设备", "结构", "模块", "组件", "任务", "场景", "能力", "策略",
    "导航", "跑酷", "规划", "控制", "感知", "识别", "检测", "分割",
    "预测", "定位", "建图", "避障", "抓取", "操控", "搬运", "装配",
    "生成", "推理", "记忆", "数据", "视觉", "触觉",
)

_ACTION_NOUN_HINTS_EN = (
    "system", "method", "algorithm", "model", "framework", "technology",
    "platform", "device", "apparatus", "architecture", "module", "task",
    "scenario", "capability", "policy", "navigation", "parkour",
    "planning", "control", "perception", "recognition", "detection",
    "segmentation", "prediction", "localization", "mapping", "grasping",
    "manipulation", "assembly", "generation", "reasoning", "memory",
    "vision", "tactile", "dataset", "benchmark",
)


def _row_get(row: Any, field: str) -> Any:
    if row is None:
        return ""
    if hasattr(row, "get"):
        try:
            return row.get(field, "")
        except Exception:
            return ""
    return getattr(row, field, "")


def _clean_subject_text(value: Any) -> str:
    text = safe_event_text(value, "未知")
    text = re.sub(r"[\u200b\xa0]+", " ", text).strip()
    text = re.sub(r"^(?:subject|主语|主体|事件主体)\s*[:：]\s*", "", text, flags=re.IGNORECASE).strip()
    text = text.strip(" \t\r\n\"'“”‘’[]{}()（）<>《》")
    text = re.sub(r"\s+", " ", text).strip()
    if is_placeholder_event_value(text):
        return "未知"
    return text or "未知"


def _compact_subject_key(text: str) -> str:
    return re.sub(r"[\s\W_]+", "", str(text or "").lower())


_GENERIC_SUBJECT_KEYS = {_compact_subject_key(value) for value in _GENERIC_SUBJECT_VALUES}


def _source_context_text(source_row: Any) -> str:
    parts = []
    for field in _SUBJECT_CONTEXT_FIELDS:
        value = safe_event_text(_row_get(source_row, field))
        if value:
            parts.append(value)
    return " ".join(parts)


def _source_subject_candidates(source_row: Any) -> List[str]:
    candidates = []
    source_type = safe_event_text(_row_get(source_row, "source_type")).lower()
    org_fallback_blocked = source_type in {"news", "rss", "media", "article", "新闻", "媒体"}
    org_fallback_fields = {"org", "organization", "organizations", "company", "机构", "单位", "来源机构"}
    for field in _SUBJECT_METADATA_FIELDS:
        if org_fallback_blocked and field in org_fallback_fields:
            continue
        value = _clean_subject_text(_row_get(source_row, field))
        if value != "未知":
            candidates.append(value)
    return _dedupe_preserve_order(candidates)


def _trim_context_subject_candidate(text: str) -> str:
    value = _clean_subject_text(text)
    value = re.split(r"[。！？!?；;\n\r]", value)[-1].strip()
    value = re.sub(r"^(?:由|来自|来自于|联合|据|根据|with|from|at|by)\s*", "", value, flags=re.IGNORECASE).strip()
    value = re.sub(r"^(?:the\s+)?(?:researchers|scientists|team|engineers|authors|inventors)\s+(?:at|from|with|of)\s+", "", value, flags=re.IGNORECASE).strip()
    return _clean_subject_text(value)


def _context_subject_candidates(source_row: Any) -> List[str]:
    context = _source_context_text(source_row)
    if not context:
        return []

    candidates = []
    zh_suffix = "|".join(re.escape(item) for item in sorted(_CONCRETE_ACTOR_SUFFIX_ZH, key=len, reverse=True))
    zh_actor = rf"([\u4e00-\u9fffA-Za-z0-9·（）()&.\- ]{{2,50}}?(?:{zh_suffix}))"
    zh_actions = r"(?:提出|发布|推出|研发|研制|开发|设计|实现|展示|演示|开源|验证|测试|申请|公开|披露|发明|构建|部署|落地)"
    zh_patterns = [
        rf"(?:由|来自|来自于|联合|据|根据)\s*{zh_actor}(?:的)?(?:研究团队|团队|研究人员|科学家|工程师|作者|发明人)?\s*{zh_actions}",
        rf"{zh_actor}(?:的)?(?:研究团队|团队|研究人员|科学家|工程师|作者|发明人)?\s*{zh_actions}",
        rf"(?:研究人员|科学家|工程师|团队|作者|发明人)(?:来自|就职于|隶属于)\s*{zh_actor}",
    ]
    for pattern in zh_patterns:
        for match in re.finditer(pattern, context):
            candidates.append(_trim_context_subject_candidate(match.group(1)))

    en_suffix = "|".join(re.escape(item) for item in _CONCRETE_ACTOR_SUFFIX_EN)
    en_actor = rf"([A-Z][A-Za-z0-9&.'\- ]{{2,80}}?(?:{en_suffix}))"
    en_actions = r"(?:proposed|introduced|presented|developed|released|launched|announced|designed|built|created|implemented|demonstrated|validated|tested|published|disclosed|deployed)"
    en_patterns = [
        rf"(?:researchers|scientists|team|engineers|authors|inventors)\s+(?:at|from|with|of)\s+{en_actor}",
        rf"{en_actor}\s+(?:researchers|scientists|team|engineers|authors|inventors)?\s*{en_actions}",
    ]
    for pattern in en_patterns:
        for match in re.finditer(pattern, context):
            candidates.append(_trim_context_subject_candidate(match.group(1)))

    lowered = context.lower()
    for org in sorted(_COMMON_ORG_NAMES, key=len, reverse=True):
        org_text = str(org)
        if org_text.lower() in lowered or org_text in context:
            candidates.append(org_text)

    return _dedupe_preserve_order(candidate for candidate in candidates if candidate and candidate != "未知")


def _has_subject_actor_marker(text: str) -> bool:
    if not text:
        return False
    for marker in _SUBJECT_ACTOR_MARKERS_ZH:
        if marker in text and len(text.replace(marker, "").strip()) >= 2:
            return True
    lower = text.lower()
    return any(
        re.search(r"\b" + re.escape(marker.strip()) + r"\b", lower)
        for marker in _SUBJECT_ACTOR_MARKERS_EN
    )


def _contains_technical_subject_keyword(text: str) -> bool:
    if not text:
        return False
    lower = text.lower()
    if any(keyword in text for keyword in _TECHNICAL_SUBJECT_KEYWORDS_ZH):
        return True
    return any(
        re.search(r"\b" + re.escape(keyword) + r"\b", lower)
        for keyword in _TECHNICAL_SUBJECT_KEYWORDS_EN
    )


def _is_generic_subject(text: str) -> bool:
    key = _compact_subject_key(text)
    if key in _PLACEHOLDER_COMPACT_KEYS or key in _GENERIC_SUBJECT_KEYS:
        return True
    lower = str(text or "").strip().lower()
    if lower in _PLACEHOLDER_TEXT_VALUES:
        return True
    if re.match(r"^(我们|我方|本人|本团队|本公司|本发明|本申请|本公开|本文|本研究|该|此|所述|上述)", text):
        return True
    if re.match(r"^(we|our|this|the)\s+", lower):
        return True
    return False


def _is_demographic_proxy_subject(text: str) -> bool:
    cleaned = _clean_subject_text(text)
    if not cleaned:
        return False
    zh_geo = "|".join(re.escape(item) for item in _SUBJECT_GEO_TERMS_ZH)
    zh_role = "|".join(re.escape(item) for item in _GENERIC_ACTOR_ROLE_TERMS_ZH)
    if re.fullmatch(rf"(?:来自)?(?:{zh_geo})(?:的)?(?:{zh_role})(?:团队)?", cleaned):
        return True
    lower = cleaned.lower().strip()
    en_geo = "|".join(re.escape(item) for item in _SUBJECT_GEO_TERMS_EN)
    en_role = "|".join(re.escape(item) for item in _GENERIC_ACTOR_ROLE_TERMS_EN)
    return bool(re.fullmatch(rf"(?:{en_geo})(?:\s+|-)+(?:{en_role})", lower))


def _looks_like_person_name(text: str) -> bool:
    compact = re.sub(r"[\s·,，;；、.-]+", "", text)
    if re.fullmatch(r"[\u4e00-\u9fff]{2,4}", compact):
        return True
    return bool(re.fullmatch(r"[A-Z][a-z]+(?:\s+[A-Z]\.){0,3}(?:\s+[A-Z][a-z]+){1,3}", text.strip()))


def _subject_parts(text: str) -> List[str]:
    return [part.strip() for part in re.split(r"[;；、|]+", text) if part.strip()]


def _is_valid_subject_single(text: str, source_context: str = "", from_metadata: bool = False) -> bool:
    text = _clean_subject_text(text)
    if text == "未知" or _is_generic_subject(text) or _is_demographic_proxy_subject(text) or len(text) > 80:
        return False

    lower = text.lower().strip(" .")
    if lower in _COMMON_ORG_NAMES or text in _COMMON_ORG_NAMES:
        return True

    has_actor_marker = _has_subject_actor_marker(text)
    if _contains_technical_subject_keyword(text) and not has_actor_marker:
        return False
    if has_actor_marker:
        return True

    if from_metadata and _looks_like_person_name(text):
        return True

    cjk_chars = re.findall(r"[\u4e00-\u9fff]", text)
    if cjk_chars and 2 <= len(cjk_chars) <= 20:
        if not source_context or text in source_context or from_metadata:
            return True

    if from_metadata and re.fullmatch(r"[A-Za-z][A-Za-z0-9&.' -]{2,60}", text):
        return True

    return False


def _is_valid_subject(text: str, source_context: str = "", from_metadata: bool = False) -> bool:
    parts = _subject_parts(text)
    if len(parts) > 1:
        return all(_is_valid_subject_single(part, source_context, from_metadata) for part in parts)
    return _is_valid_subject_single(text, source_context, from_metadata)


def normalize_subject(value: Any, source_row: Any = None) -> str:
    """Normalize subject to a concrete actor, otherwise return 未知."""
    subject = _clean_subject_text(value)
    source_context = _source_context_text(source_row)
    if _is_valid_subject(subject, source_context=source_context, from_metadata=False):
        return subject

    for candidate in _source_subject_candidates(source_row):
        if _is_valid_subject(candidate, source_context=source_context, from_metadata=True):
            return candidate

    for candidate in _context_subject_candidates(source_row):
        if _is_valid_subject(candidate, source_context=source_context, from_metadata=False):
            return candidate

    return "未知"


def _clean_action_text(value: Any) -> str:
    text = safe_event_text(value, "未知")
    text = re.sub(r"^(?:action|动作|行为)\s*[:：]\s*", "", text, flags=re.IGNORECASE).strip()
    text = text.strip(" \t\r\n\"'“”‘’[]{}()（）<>《》")
    text = re.sub(r"\s+", " ", text).strip()
    if is_placeholder_event_value(text):
        return "未知"
    return text or "未知"


def _source_action_text(source_row: Any) -> str:
    return " ".join(
        safe_event_text(_row_get(source_row, field))
        for field in ["title", "text", "abstract", "summary", "content", "snippet", "main_content"]
        if safe_event_text(_row_get(source_row, field))
    )


def _find_standard_action(text: str) -> str:
    if not text:
        return ""
    lowered = text.lower()
    for canonical, aliases in _STANDARD_ACTION_VERBS:
        for alias in aliases:
            alias_text = str(alias)
            if re.search(r"[\u4e00-\u9fff]", alias_text):
                if alias_text in text:
                    return canonical
            elif re.search(r"\b" + re.escape(alias_text.lower()) + r"\b", lowered):
                return canonical
    return ""


def _looks_like_action_noun_phrase(action_text: str) -> bool:
    action_text = _clean_action_text(action_text)
    if action_text == "未知":
        return False
    if _find_standard_action(action_text):
        return False

    zh_chars = re.findall(r"[\u4e00-\u9fff]", action_text)
    if zh_chars:
        if len(zh_chars) >= 4:
            return True
        return any(keyword in action_text for keyword in _ACTION_NOUN_HINTS_ZH)

    lowered = action_text.lower()
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9\-]*", lowered)
    if len(tokens) >= 2 and any(token in _ACTION_NOUN_HINTS_EN for token in tokens):
        return True
    if len(tokens) > 3:
        return True
    return any(re.search(r"\b" + re.escape(keyword) + r"\b", lowered) for keyword in _ACTION_NOUN_HINTS_EN)


def normalize_action(value: Any, source_row: Any = None) -> str:
    """Normalize action to a core event verb; task/object phrases fall back to evidence verbs."""
    action = _clean_action_text(value)
    if action == "未知":
        return _find_standard_action(_source_action_text(source_row)) or "未知"

    context_action = _find_standard_action(_source_action_text(source_row))
    if action in {"研发", "propose"} and context_action and context_action != action:
        return context_action

    explicit = _find_standard_action(action)
    if explicit and not _looks_like_action_noun_phrase(action):
        return explicit
    if explicit:
        return explicit

    if _looks_like_action_noun_phrase(action):
        if context_action:
            return context_action
        return "propose" if re.search(r"[a-zA-Z]", action) else "研发"

    return action


def source_date_text(row: Any) -> str:
    for field in ["date", "event_date", "publication_date", "application_date", "grant_date", "time"]:
        value = clean_event_text(row.get(field, ""))
        if value:
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
    had_schema_version = bool(clean_event_text(event.get("event_schema_version")))
    if source_row is not None:
        if not clean_event_text(event.get("id")):
            event["id"] = source_row.get("id", "")
        source_date = source_date_text(source_row)
        if source_date and not clean_event_text(event.get("date")):
            event["date"] = source_date
        if source_date and not clean_event_text(event.get("event_date")):
            event["event_date"] = source_date
        for field in ["source_type", "title"]:
            if not clean_event_text(event.get(field)):
                event[field] = source_row.get(field, "")

    event["event_schema_version"] = (
        clean_event_text(event.get("event_schema_version"))
        or WEAK_SIGNAL_EVENT_SCHEMA_VERSION
    )
    event["id"] = clean_event_text(event.get("id"))
    event["event_id"] = clean_event_text(event.get("event_id")) or make_event_id(
        event.get("id") or "doc",
        doc_event_index,
    )

    event["subject"] = normalize_subject(event.get("subject"), source_row=source_row)
    for field in ["scene", "time"]:
        event[field] = clean_event_text(event.get(field), "未知")

    event["action"] = normalize_action(event.get("action"), source_row=source_row)
    for field in ["date", "event_date", "source_type", "title"]:
        event[field] = clean_event_text(event.get(field))

    for field in [
        "event_type", "technical_object", "mechanism", "task",
        "capability_change", "problem_solved", "maturity_stage",
        "novelty_signal", "adoption_signal", "cross_domain_signal",
        "weak_signal_reason", "uncertainty", "evidence_span",
    ]:
        event[field] = clean_event_text(event.get(field))

    for field in EVENT_LIST_FIELDS:
        event[field] = clean_event_list(event.get(field))

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
        present = events_df[column].apply(lambda value: bool(clean_event_text(value)))
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
    "clean_event_text",
    "clean_event_list",
    "is_placeholder_event_value",
    "coerce_confidence",
    "normalize_subject",
    "normalize_action",
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
