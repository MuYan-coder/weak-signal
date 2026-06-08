from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Callable, Dict, List

import yaml

from ..utils.env_config import get_preferred_chat_model
from ..utils.llm_client import chat_text
from .models import DOMAIN_PACK_SCHEMA_VERSION, DomainPack


DOMAIN_PACK_GENERATOR_PROMPT_VERSION = "domain_pack_generator_v3"
DEFAULT_MEMORY_DIR = Path("memory") / "domain_packs"
WEAK_SIGNAL_MARKER_FIELDS = [
    "early_stage_markers",
    "low_attention_markers",
    "niche_actor_markers",
    "cross_domain_markers",
    "engineering_trace_markers",
    "commercialization_noise_markers",
    "policy_or_market_noise_markers",
]
WEAK_SIGNAL_MARKER_DEFAULTS = {
    "early_stage_markers": ["初步研究", "实验室验证", "原型验证", "小规模试验"],
    "low_attention_markers": ["较少关注", "低引用", "少量报道", "小众研究"],
    "niche_actor_markers": ["小团队", "初创企业", "高校团队", "非主流机构"],
    "cross_domain_markers": ["跨领域", "交叉应用", "异质集成", "跨学科方法"],
    "engineering_trace_markers": ["工艺优化", "参数调整", "原型测试", "工程验证"],
    "commercialization_noise_markers": ["市场推广", "融资", "产品发布", "商业合作"],
    "policy_or_market_noise_markers": ["政策支持", "市场趋势", "行业报告", "产业规划"],
}
DEFAULT_SHELL_TERMS = [
    "技术", "方法", "系统", "应用", "数据", "性能", "效果", "场景",
    "technology", "method", "system", "application", "data", "performance", "effect", "scene"
]
GENERATOR_SYSTEM_PROMPT = (
    "你是弱信号识别系统的领域运行包生成器。\n"
    "【严格格式约束】：\n"
    "1. 必须且只能输出合法的纯 JSON 或 YAML 格式。\n"
    "2. 绝对禁止在输出中包含任何形式的代码注释（如 // 或 /* */）。\n"
    "3. 绝对禁止输出任何解释性对话、前言或后语。"
)
ALLOWED_SLOT_NAMES = [
    "technical_object",
    "mechanism",
    "task",
    "performance",
    "data_modality",
    "method",
    "scene",
    "evidence_span",
]


class DomainPackGenerationError(RuntimeError):
    """Raised when a generated Domain Pack cannot be safely constructed."""


@dataclass
class DomainPackGenerationRequest:
    field_id: str
    field_name: str
    keywords: List[str] = field(default_factory=list)
    synonyms: List[str] = field(default_factory=list)
    exclude_terms: List[str] = field(default_factory=list)
    source_types: List[str] = field(default_factory=list)
    sample_documents: List[Dict[str, Any]] = field(default_factory=list)

    def normalized(self) -> Dict[str, Any]:
        return {
            "field_id": _clean_text(self.field_id),
            "field_name": _clean_text(self.field_name),
            "keywords": _dedupe_texts(self.keywords),
            "synonyms": _dedupe_texts(self.synonyms),
            "exclude_terms": _dedupe_texts(self.exclude_terms),
            "source_types": _dedupe_texts(self.source_types),
            "sample_documents": [_sample_document_preview(item) for item in self.sample_documents],
        }


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _dedupe_texts(values: List[Any]) -> List[str]:
    seen = set()
    cleaned = []
    for value in values or []:
        text = _clean_text(value)
        if not text:
            continue
        marker = text.lower()
        if marker in seen:
            continue
        seen.add(marker)
        cleaned.append(text)
    return cleaned


def _is_shell_like_term(value: Any) -> bool:
    text = _clean_text(value).lower()
    return not text or text in {item.lower() for item in DEFAULT_SHELL_TERMS}


def _flatten_text_values(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        values: List[str] = []
        for item in value:
            values.extend(_flatten_text_values(item))
        return values
    if isinstance(value, dict):
        values: List[str] = []
        for key in ("keywords", "terms", "markers", "aliases", "values"):
            if key in value:
                values.extend(_flatten_text_values(value.get(key)))
        if values:
            return values
        for key in ("term", "name", "value", "marker", "label", "description"):
            if value.get(key):
                return _flatten_text_values(value.get(key))
        for item in value.values():
            values.extend(_flatten_text_values(item))
        if values:
            return values
        return []
    text = _clean_text(value)
    return [text] if text else []


def _coerce_text_list(value: Any) -> List[str]:
    return _dedupe_texts(_flatten_text_values(value))


def is_current_generated_domain_pack(
    pack: DomainPack,
    *,
    current_prompt_version: str = DOMAIN_PACK_GENERATOR_PROMPT_VERSION,
) -> bool:
    source = getattr(pack, "source", {}) if pack is not None else {}
    if not isinstance(source, dict):
        return True
    mode = _clean_text(source.get("mode", "")).lower()
    if mode != "llm_generated":
        return True
    return _clean_text(source.get("prompt_version", "")) == _clean_text(current_prompt_version)


def _sample_document_preview(item: Dict[str, Any]) -> Dict[str, str]:
    item = item or {}
    return {
        "source_type": _clean_text(item.get("source_type", ""))[:40],
        "title": _clean_text(item.get("title", ""))[:160],
        "text": _clean_text(item.get("text", item.get("abstract", "")))[:240],
    }


def _json_block(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _request_payload(request: DomainPackGenerationRequest) -> Dict[str, Any]:
    return request.normalized()


def build_domain_modeling_prompt(request: DomainPackGenerationRequest) -> str:
    payload = _request_payload(request)
    return f"""
阶段一：领域建模。

请根据用户选择的技术领域、关键词、排除词、数据源类型和样本文档，输出合法 JSON 对象。

必须输出字段：
- domain_boundary
- adjacent_domains
- out_of_scope_domains: 必须包含明显不属于该领域的常见干扰领域，例如：人力资源、金融分析、市场营销、社会新闻、电子商务、娱乐传媒、房地产、健康养生等。
- sub_directions
- noise_terms
- search_strategy.core_keywords
- search_strategy.synonyms
- search_strategy.english_terms
- search_strategy.exclude_terms: 必须包含与该领域无关的常见干扰词。除了用户提供的排除词外，还应包含：招聘、培训、广告、股票分析、行情、娱乐、美食、旅游、健康养生、房地产、直播带货、二手交易、电商、退款、维权、社会新闻等。

输入：
{_json_block(payload)}
""".strip()


def build_candidate_formation_prompt(
    request: DomainPackGenerationRequest,
    domain_model: Dict[str, Any],
) -> str:
    payload = {
        "request": _request_payload(request),
        "domain_model": domain_model,
        "allowed_slots": ALLOWED_SLOT_NAMES,
        "reference_candidate_pattern_summary": {
            "valid_pattern": "technical_object + mechanism + task/performance + evidence_span",
            "invalid_pattern": "scope or shell term only",
            "generic_shell_examples": ["技术", "方法", "系统", "应用", "性能", "效果", "数据", "场景", "technology", "method", "system", "application", "data", "performance", "effect", "scene"],
        },
    }
    return f"""
阶段二：候选成形范式生成。

请输出合法 JSON 对象，所有 pattern 必须有稳定、全局唯一且具有语义描述性的 pattern_id（例如 `object_mechanism_task`、`shell_term_only`）。
绝对禁止使用简单的数字（如 1, 2）作为 pattern_id。valid_candidate_patterns 和 invalid_candidate_patterns 中的 pattern_id 严禁重复。

【通用与精炼约束，极其重要】：
1. 领域绝对隔离：必须完全根据当前输入的领域生成对应词汇，绝不允许生搬硬套其他领域的名词（如在非AI领域不要生成“训练”，非机器领域不要生成“规划”，非材料领域不要生成“掺杂”、“晶胞”等）。
2. 强制精炼：所有输出的类别词汇（types）必须是简短、精炼的技术专有名词或短语（通常在 10 个字符以内）。绝对禁止输出长句或包含逻辑关系的描述句式（例如，严禁出现“通过优化XXX提高XXX”）。
3. 剔除无意义词：坚决过滤掉毫无核心技术属性的日常通用动词，如“展示”、“实现”、“使用”、“包含”、“具有”等。

必须输出字段：
- technical_object_types: 必须包含具体的非泛化技术对象短语，不能仅有通用词。
  必须保留用户输入的中文关键词，并补充对应英文/缩写；例如中文领域输入为“碳化硅/氮化镓/二维材料”时，不得只输出 SiC/GaN/2D materials。
- mechanism_types: 必须同时包含两类 mechanism：
  (a) 理论/底层机制：当前领域特有的基础物理、数学、化学或逻辑原理；
  (b) 工程/操作机制：当前领域特有的核心工艺、加工手段、算法步骤或系统操作交互动作。
  重要：必须是强技术属性的动作或原理，不得包含“展示”、“实现”等废话。
- task_or_performance_types: 必须包含具体的任务或性能指标。
- data_or_method_types: 必须包含具体的数据或方法类别。
- scene_or_application_types: 必须包含具体的场景或应用。
- generic_terms: 必须包含通用技术描述词，且必须包含常见通用词（如 "技术", "方法", "系统", "应用", "数据", "性能", "效果", "场景" 及其英文单词）。
- shell_terms: 必须包含壳词/泛化词，且必须包含常见通用词（如 "技术", "方法", "系统", "应用", "数据", "性能", "效果", "场景" 及其英文单词）。
- valid_candidate_patterns: 每一个 valid_candidate_pattern 都必须要求至少一个具体的非空壳 slot。
- invalid_candidate_patterns: 必须设计针对仅匹配通用壳词、无具体技术主语的过滤模式，reject_terms 必须包含常见通用词。

required_slots、optional_slots、reject_if_slots_only 只能使用这些槽位：
{", ".join(ALLOWED_SLOT_NAMES)}

输入：
{_json_block(payload)}
""".strip()


def build_weak_signal_markers_prompt(
    request: DomainPackGenerationRequest,
    domain_model: Dict[str, Any],
    candidate_formation: Dict[str, Any],
) -> str:
    payload = {
        "request": _request_payload(request),
        "domain_model": domain_model,
        "candidate_formation": candidate_formation,
        "current_scoring_dimensions": [
            "early_stage",
            "low_attention",
            "niche_actor",
            "cross_domain",
            "engineering_trace",
            "commercialization_noise",
            "policy_or_market_noise",
        ],
    }
    return f"""
阶段 3a：弱信号维度标记词生成。

请严格只输出一个合法 JSON 对象，绝对不要包含任何注释（如 //）、Markdown 代码围栏、解释性文字、前言或后语。
每个字段的值必须是一维字符串数组；禁止输出对象数组、分组对象、嵌套数组或说明句。

必须输出字段：
- early_stage_markers
- low_attention_markers
- niche_actor_markers
- cross_domain_markers
- engineering_trace_markers
- commercialization_noise_markers
- policy_or_market_noise_markers

必须遵循此 JSON 骨架：
{{
  "early_stage_markers": ["实验室验证", "原型验证"],
  "low_attention_markers": ["低引用", "少量报道"],
  "niche_actor_markers": ["小团队", "初创企业"],
  "cross_domain_markers": ["跨领域", "交叉应用"],
  "engineering_trace_markers": ["工艺优化", "参数调整"],
  "commercialization_noise_markers": ["市场推广", "融资"],
  "policy_or_market_noise_markers": ["政策支持", "市场趋势"]
}}

输入：
{_json_block(payload)}
""".strip()


def build_scoring_adjustments_prompt(
    request: DomainPackGenerationRequest,
    domain_model: Dict[str, Any],
    candidate_formation: Dict[str, Any],
) -> str:
    payload = {
        "request": _request_payload(request),
        "domain_model": domain_model,
        "candidate_formation": candidate_formation,
    }
    return f"""
阶段 3b：弱信号计分调整规则生成。

请输出合法 JSON 对象，包含 `scoring_adjustments` 字段，该字段为一个规则数组。
每条规则必须有 rule_id，score_delta 和 max_delta 必须是有界数值。绝对不要包含任何注释（如 //）。

必须输出字段：
- scoring_adjustments

输入：
{_json_block(payload)}
""".strip()


def build_integration_prompt(
    request: DomainPackGenerationRequest,
    domain_model: Dict[str, Any],
    candidate_formation: Dict[str, Any],
    weak_signal_rules: Dict[str, Any],
) -> str:
    payload = {
        "request": _request_payload(request),
        "domain_model": domain_model,
        "candidate_formation": candidate_formation,
        "weak_signal_rules": weak_signal_rules,
        "schema_contract": {
            "schema_version": DOMAIN_PACK_SCHEMA_VERSION,
            "required_top_level_fields": [
                "schema_version",
                "pack_id",
                "pack_name",
                "pack_version",
                "source",
                "domain_identity",
                "search_strategy",
                "observation_scopes",
                "candidate_formation",
                "weak_signal_rules",
                "canonicalization",
                "evidence_rules",
                "reporting",
            ],
            "allowed_slots": ALLOWED_SLOT_NAMES,
            "source_mode": "llm_generated",
        },
    }
    return f"""
阶段四：整合与压缩。

请将前三阶段结果压缩为标准 Domain Pack。只输出合法 JSON 或 YAML，不要输出解释。
schema_version 必须为 domain_pack_v1。
required_slots、optional_slots、reject_if_slots_only 只能使用：
{", ".join(ALLOWED_SLOT_NAMES)}
observation_scopes 不得为空：
- main_scope 必须使用用户技术领域名称；
- sub_scopes 必须包含阶段一 sub_directions 和用户核心关键词；
- scope_aliases 必须包含用户 synonyms/english_terms；
- scope_echo_terms 必须包含用户 field_name 和 keywords；
- off_domain_anchor_terms 必须包含 exclude_terms、noise_terms、out_of_scope_domains。
candidate_formation.technical_object_types 必须保留中文关键词和英文同义词，禁止只保留英文缩写。

输入：
{_json_block(payload)}
""".strip()


def build_domain_pack_cache_key(
    request: DomainPackGenerationRequest,
    *,
    model: str,
    prompt_version: str = DOMAIN_PACK_GENERATOR_PROMPT_VERSION,
) -> str:
    payload = {
        "field_id": _clean_text(request.field_id),
        "field_name": _clean_text(request.field_name),
        "keywords": _dedupe_texts(request.keywords),
        "synonyms": _dedupe_texts(request.synonyms),
        "exclude_terms": _dedupe_texts(request.exclude_terms),
        "source_types": _dedupe_texts(request.source_types),
        "prompt_version": prompt_version,
        "model": _clean_text(model),
    }
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]
    slug = _slugify(payload["field_id"] or payload["field_name"] or "domain")
    model_slug = _slugify(payload["model"] or "default-model")
    prompt_slug = _slugify(prompt_version)
    return f"{slug}__{prompt_slug}__{model_slug}__{digest}"


def _slugify(value: str) -> str:
    text = _clean_text(value).lower()
    text = re.sub(r"[^a-z0-9_\-\u4e00-\u9fff]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "domain"


def _parse_structured_response(raw: str, stage_name: str) -> Dict[str, Any]:
    text = str(raw or "").strip()
    if not text:
        raise DomainPackGenerationError(f"{stage_name} returned empty response")

    candidates = [text]

    # 优先提取 Markdown JSON/YAML 代码块内容
    for code_block_match in re.finditer(r"```(?:json|yaml|yml)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE):
        candidates.append(code_block_match.group(1))

    # 提取最外层的大括号内的内容
    start_idx = text.find("{")
    end_idx = text.rfind("}")
    if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
        candidates.append(text[start_idx : end_idx + 1])

    for candidate in candidates:
        # 清理常见的模型自作聪明加的单行注释
        clean_candidate = re.sub(r"(?m)^\s*//.*$", "", candidate)

        try:
            parsed = json.loads(clean_candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        try:
            parsed = yaml.safe_load(clean_candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    raise DomainPackGenerationError(f"{stage_name} response is not valid JSON/YAML mapping")


def _build_structured_repair_prompt(stage_name: str, original_prompt: str, raw: str, error: Exception) -> str:
    weak_signal_schema = ""
    if stage_name == "weak_signal_markers":
        weak_signal_schema = f"""
必须只返回如下字段，且每个字段必须是一维字符串数组：
{_json_block({field: WEAK_SIGNAL_MARKER_DEFAULTS[field][:2] for field in WEAK_SIGNAL_MARKER_FIELDS})}
""".strip()
    return f"""
上一次 `{stage_name}` 阶段输出无法解析：{error}

请根据原始任务重新输出。只返回一个合法 JSON 对象，不要包含 Markdown、解释、注释、前言或后语。
{weak_signal_schema}

原始任务：
{original_prompt}

上一次无法解析的输出：
{str(raw or "")[:4000]}
""".strip()


def _merge_missing_or_empty_fields(primary: Dict[str, Any], fallback: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(primary, dict):
        return copy.deepcopy(fallback) if isinstance(fallback, dict) else {}
    if not isinstance(fallback, dict):
        return copy.deepcopy(primary)

    merged = copy.deepcopy(primary)
    for key, fallback_value in fallback.items():
        current_value = merged.get(key)
        if current_value in (None, [], {}) and fallback_value not in (None, [], {}):
            merged[key] = copy.deepcopy(fallback_value)
    return merged


def _normalize_weak_signal_markers(payload: Dict[str, Any]) -> Dict[str, Any]:
    source = payload if isinstance(payload, dict) else {}
    for wrapper_key in ("weak_signal_markers", "weak_signal_rules", "markers"):
        wrapped = source.get(wrapper_key)
        if isinstance(wrapped, dict):
            source = {**wrapped, **{key: source.get(key) for key in WEAK_SIGNAL_MARKER_FIELDS if key in source}}
            break

    normalized: Dict[str, Any] = {}
    for field_name in WEAK_SIGNAL_MARKER_FIELDS:
        values = _coerce_text_list(source.get(field_name))
        if not values:
            values = list(WEAK_SIGNAL_MARKER_DEFAULTS[field_name])
        normalized[field_name] = values
    return normalized


def _fallback_specific_candidate_terms(
    request_payload: Dict[str, Any] | None,
    domain_model: Dict[str, Any] | None,
) -> List[str]:
    request_payload = request_payload if isinstance(request_payload, dict) else {}
    domain_model = domain_model if isinstance(domain_model, dict) else {}
    search_strategy = domain_model.get("search_strategy", {}) if isinstance(domain_model.get("search_strategy"), dict) else {}

    sources = [
        request_payload.get("keywords", []),
        search_strategy.get("core_keywords", []),
        domain_model.get("sub_directions", []),
        request_payload.get("synonyms", []),
        search_strategy.get("synonyms", []),
        search_strategy.get("english_terms", []),
        request_payload.get("field_name", ""),
    ]
    candidates: List[str] = []
    for source in sources:
        for term in _coerce_text_list(source):
            if _is_shell_like_term(term):
                continue
            candidates.append(term)
    return _dedupe_texts(candidates)[:12]


def _ensure_observation_scopes_safety(
    scopes: Dict[str, Any],
    *,
    request_payload: Dict[str, Any] | None = None,
    domain_model: Dict[str, Any] | None = None,
) -> None:
    if not isinstance(scopes, dict):
        return
    request_payload = request_payload if isinstance(request_payload, dict) else {}
    domain_model = domain_model if isinstance(domain_model, dict) else {}
    search_strategy = domain_model.get("search_strategy", {}) if isinstance(domain_model.get("search_strategy"), dict) else {}

    field_name = _clean_text(request_payload.get("field_name", ""))
    field_id = _clean_text(request_payload.get("field_id", ""))
    main_scope = _clean_text(scopes.get("main_scope", "")) or field_name or field_id or _clean_text(domain_model.get("domain_boundary", ""))
    scopes["main_scope"] = main_scope

    scopes["sub_scopes"] = _dedupe_texts(
        _coerce_text_list(scopes.get("sub_scopes"))
        + _coerce_text_list(domain_model.get("sub_directions"))
        + _coerce_text_list(request_payload.get("keywords"))
        + _coerce_text_list(search_strategy.get("core_keywords"))
    )[:24]
    scopes["scope_aliases"] = _dedupe_texts(
        _coerce_text_list(scopes.get("scope_aliases"))
        + _coerce_text_list(request_payload.get("synonyms"))
        + _coerce_text_list(search_strategy.get("synonyms"))
        + _coerce_text_list(search_strategy.get("english_terms"))
    )[:32]
    scopes["scope_echo_terms"] = _dedupe_texts(
        _coerce_text_list(scopes.get("scope_echo_terms"))
        + _coerce_text_list([field_name, field_id])
        + _coerce_text_list(request_payload.get("keywords"))
        + _coerce_text_list(search_strategy.get("core_keywords"))
    )[:32]
    scopes["off_domain_anchor_terms"] = _dedupe_texts(
        _coerce_text_list(scopes.get("off_domain_anchor_terms"))
        + _coerce_text_list(request_payload.get("exclude_terms"))
        + _coerce_text_list(domain_model.get("noise_terms"))
        + _coerce_text_list(domain_model.get("out_of_scope_domains"))
        + _coerce_text_list(search_strategy.get("exclude_terms"))
    )[:48]


class DomainPackGenerator:
    def __init__(
        self,
        *,
        chat_fn: Callable | None = None,
        memory_dir: Path | str = DEFAULT_MEMORY_DIR,
        model: str | None = None,
        prompt_version: str = DOMAIN_PACK_GENERATOR_PROMPT_VERSION,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        timeout: float | None = None,
        max_parse_retries: int = 1,
    ):
        self.chat_fn = chat_fn or chat_text
        self.memory_dir = Path(memory_dir)
        self.model = model or get_preferred_chat_model()
        self.prompt_version = prompt_version
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout or float(os.getenv("DOMAIN_PACK_GENERATOR_TIMEOUT", "300.0"))
        self.max_parse_retries = max(0, int(max_parse_retries or 0))

    def generate(
        self,
        request: DomainPackGenerationRequest,
        *,
        refresh_cache: bool = False,
        progress_fn: Callable[[str, int], None] | None = None,
    ) -> DomainPack:
        cache_key = build_domain_pack_cache_key(
            request,
            model=self.model,
            prompt_version=self.prompt_version,
        )
        cache_path = self._cache_path(cache_key)
        if not refresh_cache:
            cached = self._load_cached_pack(cache_path)
            if cached is not None:
                return cached

        stage_cache_path = self.memory_dir / "cache" / "generator_stages" / f"{cache_key}__stage_payloads.json"
        if refresh_cache:
            try:
                if stage_cache_path.exists():
                    stage_cache_path.unlink()
            except Exception:
                pass
            stage_payloads = {}
        else:
            stage_payloads = self._load_stage_cache(stage_cache_path)

        # 阶段 1
        if progress_fn:
            progress_fn("domain_modeling", 1)
        if "domain_modeling" in stage_payloads:
            domain_model = stage_payloads["domain_modeling"]
        else:
            domain_model = self._call_stage("domain_modeling", build_domain_modeling_prompt(request))
            stage_payloads["domain_modeling"] = domain_model
            self._save_stage_cache(stage_cache_path, stage_payloads)

        # 阶段 2
        if progress_fn:
            progress_fn("candidate_formation", 2)
        if "candidate_formation" in stage_payloads:
            candidate_formation = stage_payloads["candidate_formation"]
        else:
            candidate_formation = self._call_stage(
                "candidate_formation",
                build_candidate_formation_prompt(request, domain_model),
            )
            stage_payloads["candidate_formation"] = candidate_formation
            self._save_stage_cache(stage_cache_path, stage_payloads)

        # 阶段 3a
        if progress_fn:
            progress_fn("weak_signal_markers", 3)
        if "weak_signal_markers" in stage_payloads:
            weak_signal_markers = _normalize_weak_signal_markers(stage_payloads["weak_signal_markers"])
        else:
            weak_signal_markers = self._call_stage(
                "weak_signal_markers",
                build_weak_signal_markers_prompt(request, domain_model, candidate_formation),
            )
            stage_payloads["weak_signal_markers"] = weak_signal_markers
            self._save_stage_cache(stage_cache_path, stage_payloads)

        # 阶段 3b
        if progress_fn:
            progress_fn("scoring_adjustments", 3)
        if "scoring_adjustments" in stage_payloads:
            scoring_adjustments = stage_payloads["scoring_adjustments"]
        else:
            scoring_adjustments = self._call_stage(
                "scoring_adjustments",
                build_scoring_adjustments_prompt(request, domain_model, candidate_formation),
            )
            stage_payloads["scoring_adjustments"] = scoring_adjustments
            self._save_stage_cache(stage_cache_path, stage_payloads)

        weak_signal_rules = {**weak_signal_markers}
        weak_signal_rules["scoring_adjustments"] = scoring_adjustments.get("scoring_adjustments", [])

        # 阶段 4
        if progress_fn:
            progress_fn("integration", 4)
        final_payload = self._call_stage(
            "integration",
            build_integration_prompt(request, domain_model, candidate_formation, weak_signal_rules),
        )

        pack = DomainPack.from_dict(
            self._finalize_pack_payload(
                final_payload,
                request,
                candidate_formation=candidate_formation,
                weak_signal_rules=weak_signal_rules,
                domain_model=domain_model,
            )
        )
        if pack.pack_id == "neutral":
            raise DomainPackGenerationError("Generated pack resolved to neutral; refusing to save")

        # 验证大模型生成的结构（不包含 dry_run 检查）
        from .domain_pack_validator import validate_domain_pack
        report = validate_domain_pack(pack, require_dry_run=False)
        if not report.is_valid:
            try:
                if stage_cache_path.exists():
                    stage_cache_path.unlink()
            except Exception:
                pass
            raise DomainPackGenerationError(f"Generated pack failed validation: {report.errors}")

        pack_path = self._save_pack(pack)
        self._save_cache(cache_path, cache_key, pack_path, pack, stage_payloads)

        # 成功后清理中间缓存
        try:
            if stage_cache_path.exists():
                stage_cache_path.unlink()
        except Exception:
            pass

        return pack

    def _save_stage_cache(self, path: Path, payloads: Dict[str, Dict[str, Any]]) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payloads, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"[Warning] Failed to save stage cache: {e}")

    def _load_stage_cache(self, path: Path) -> Dict[str, Dict[str, Any]]:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8")) or {}
        except Exception:
            return {}

    def _call_stage(self, stage_name: str, prompt: str) -> Dict[str, Any]:
        current_prompt = prompt
        attempts = self.max_parse_retries + 1
        for attempt in range(1, attempts + 1):
            raw, _usage, _response = self.chat_fn(
                current_prompt,
                system=GENERATOR_SYSTEM_PROMPT,
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                timeout=self.timeout,
            )
            try:
                parsed = _parse_structured_response(raw, stage_name)
                return self._normalize_stage_payload(stage_name, parsed)
            except DomainPackGenerationError as exc:
                self._save_stage_parse_failure(stage_name, current_prompt, raw, exc, attempt)
                if attempt >= attempts:
                    raise
                current_prompt = _build_structured_repair_prompt(stage_name, prompt, raw, exc)
        raise DomainPackGenerationError(f"{stage_name} response is not valid JSON/YAML mapping")

    def _normalize_stage_payload(self, stage_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if stage_name == "weak_signal_markers":
            return _normalize_weak_signal_markers(payload)
        return payload

    def _save_stage_parse_failure(
        self,
        stage_name: str,
        prompt: str,
        raw: str,
        error: Exception,
        attempt: int,
    ) -> None:
        try:
            debug_dir = self.memory_dir / "cache" / "stage_failures"
            debug_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            safe_stage = _slugify(stage_name)
            path = debug_dir / f"{timestamp}_{safe_stage}_attempt{attempt}.txt"
            path.write_text(
                "\n".join(
                    [
                        f"stage={stage_name}",
                        f"attempt={attempt}",
                        f"error={error}",
                        "",
                        "PROMPT:",
                        str(prompt or "")[:8000],
                        "",
                        "RAW_RESPONSE:",
                        str(raw or "")[:12000],
                    ]
                ),
                encoding="utf-8",
            )
        except Exception as exc:
            print(f"[Warning] Failed to save stage parse failure: {exc}")

    def _finalize_pack_payload(
        self,
        payload: Dict[str, Any],
        request: DomainPackGenerationRequest,
        *,
        candidate_formation: Dict[str, Any],
        weak_signal_rules: Dict[str, Any],
        domain_model: Dict[str, Any],
    ) -> Dict[str, Any]:
        request_payload = request.normalized()
        final = dict(payload or {})
        field_id = request_payload["field_id"] or _slugify(request_payload["field_name"])
        field_name = request_payload["field_name"] or field_id
        final["schema_version"] = DOMAIN_PACK_SCHEMA_VERSION
        final["pack_id"] = _clean_text(final.get("pack_id") or field_id)
        final["pack_name"] = _clean_text(final.get("pack_name") or f"{field_name} Domain Pack")
        final["pack_version"] = _clean_text(final.get("pack_version") or final.get("domain_pack_version") or "generated.v1")
        final["generated_at"] = datetime.now(timezone.utc).isoformat()
        final["source"] = {
            **(final.get("source", {}) if isinstance(final.get("source"), dict) else {}),
            "mode": "llm_generated",
            "model": self.model,
            "prompt_version": self.prompt_version,
            "based_on_user_input": {
                "field_id": field_id,
                "field_name": field_name,
                "keywords": request_payload["keywords"],
                "synonyms": request_payload["synonyms"],
                "exclude_terms": request_payload["exclude_terms"],
            },
        }
        final["domain_identity"] = {
            **(final.get("domain_identity", {}) if isinstance(final.get("domain_identity"), dict) else {}),
            "field_id": field_id,
            "field_name": field_name,
            "domain_boundary": _clean_text(
                (final.get("domain_identity", {}) if isinstance(final.get("domain_identity"), dict) else {}).get("domain_boundary")
                or domain_model.get("domain_boundary", "")
            ),
            "adjacent_domains": (final.get("domain_identity", {}) if isinstance(final.get("domain_identity"), dict) else {}).get(
                "adjacent_domains",
                domain_model.get("adjacent_domains", []),
            ),
            "out_of_scope_domains": (final.get("domain_identity", {}) if isinstance(final.get("domain_identity"), dict) else {}).get(
                "out_of_scope_domains",
                domain_model.get("out_of_scope_domains", []),
            ),
        }
        if not isinstance(final.get("observation_scopes"), dict):
            final["observation_scopes"] = {}
        _ensure_observation_scopes_safety(
            final["observation_scopes"],
            request_payload=request_payload,
            domain_model=domain_model,
        )
        if not isinstance(final.get("search_strategy"), dict):
            final["search_strategy"] = domain_model.get("search_strategy", {})
        final["candidate_formation"] = _merge_missing_or_empty_fields(
            final.get("candidate_formation", {}),
            candidate_formation,
        )
        self._ensure_candidate_formation_safety(
            final["candidate_formation"],
            request_payload=request_payload,
            domain_model=domain_model,
        )
        if isinstance(final.get("weak_signal_rules"), dict):
            final_scoring_adjustments = final["weak_signal_rules"].get(
                "scoring_adjustments",
                weak_signal_rules.get("scoring_adjustments", []),
            )
            final["weak_signal_rules"] = _normalize_weak_signal_markers(final["weak_signal_rules"])
            final["weak_signal_rules"]["scoring_adjustments"] = (
                final_scoring_adjustments if isinstance(final_scoring_adjustments, list) else []
            )
        else:
            generated_rules = _normalize_weak_signal_markers(weak_signal_rules)
            generated_rules["scoring_adjustments"] = weak_signal_rules.get("scoring_adjustments", [])
            final["weak_signal_rules"] = generated_rules
        if not isinstance(final.get("canonicalization"), dict):
            final["canonicalization"] = {
                "object_families": [],
                "alias_groups": [],
                "parent_child_terms": [],
                "do_not_merge_rules": [],
                "object_family_enabled": False,
            }
        if not isinstance(final.get("evidence_rules"), dict):
            final["evidence_rules"] = {
                "strong_evidence_patterns": [],
                "weak_evidence_patterns": [],
                "source_reliability_hints": request_payload["source_types"],
                "traceability_fields": ["title", "text", "evidence_span", "source_type"],
                "evidence_rejection_patterns": request_payload["exclude_terms"],
            }
        if not isinstance(final.get("reporting"), dict):
            final["reporting"] = {
                "display_labels": {"domain": field_name},
                "explanation_templates": [],
                "review_hints": [],
            }
        return final

    def _ensure_candidate_formation_safety(
        self,
        cf: Dict[str, Any],
        *,
        request_payload: Dict[str, Any] | None = None,
        domain_model: Dict[str, Any] | None = None,
    ) -> None:
        if not isinstance(cf, dict):
            return
        for field_name in [
            "technical_object_types",
            "mechanism_types",
            "task_or_performance_types",
            "data_or_method_types",
            "scene_or_application_types",
        ]:
            values = _coerce_text_list(cf.get(field_name))
            if field_name in {"technical_object_types", "mechanism_types"}:
                values = [item for item in values if not _is_shell_like_term(item)]
            cf[field_name] = values

        if not cf["technical_object_types"] and not cf["mechanism_types"]:
            cf["technical_object_types"] = _fallback_specific_candidate_terms(request_payload, domain_model)
        else:
            cf["technical_object_types"] = _dedupe_texts(
                cf["technical_object_types"]
                + _fallback_specific_candidate_terms(request_payload, domain_model)
            )[:32]

        for field_name in ["generic_terms", "shell_terms"]:
            cleaned_terms = _coerce_text_list(cf.get(field_name))
            seen = {item.lower() for item in cleaned_terms}
            if len(cleaned_terms) < 2:
                for default_word in DEFAULT_SHELL_TERMS:
                    if default_word.lower() not in seen:
                        seen.add(default_word.lower())
                        cleaned_terms.append(default_word)
            cf[field_name] = cleaned_terms

        invalid_patterns = cf.get("invalid_candidate_patterns")
        if not isinstance(invalid_patterns, list):
            invalid_patterns = []
        for pattern in invalid_patterns:
            if not isinstance(pattern, dict):
                continue
            reject_terms = pattern.get("reject_terms")
            if not reject_terms:
                continue
            if pattern.get("max_specific_slot_count") in (None, ""):
                pattern["max_specific_slot_count"] = 0
        has_shell_rejection = any(
            isinstance(p, dict) and p.get("pattern_id") == "shell_only_rejection"
            for p in invalid_patterns
        )
        if not invalid_patterns or not has_shell_rejection:
            default_pattern = {
                "pattern_id": "shell_only_rejection",
                "reject_terms": ["技术", "方法", "系统", "应用", "数据", "性能", "效果", "场景"],
                "max_specific_slot_count": 0
            }
            if not isinstance(invalid_patterns, list):
                invalid_patterns = []
            invalid_patterns.append(default_pattern)
        cf["invalid_candidate_patterns"] = invalid_patterns

    def _pack_path(self, pack: DomainPack) -> Path:
        return self.memory_dir / f"{pack.domain_pack_hash}.yaml"

    def _cache_path(self, cache_key: str) -> Path:
        return self.memory_dir / "cache" / f"{cache_key}.json"

    def _save_pack(self, pack: DomainPack) -> Path:
        return save_domain_pack_to_memory(pack, memory_dir=self.memory_dir)

    def _save_cache(
        self,
        cache_path: Path,
        cache_key: str,
        pack_path: Path,
        pack: DomainPack,
        stage_payloads: Dict[str, Dict[str, Any]],
    ) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "cache_key": cache_key,
            "prompt_version": self.prompt_version,
            "model": self.model,
            "pack_path": str(pack_path),
            "domain_pack_hash": pack.domain_pack_hash,
            "pack_id": pack.pack_id,
            "stage_payloads": stage_payloads,
        }
        cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_cached_pack(self, cache_path: Path) -> DomainPack | None:
        if not cache_path.exists():
            return None
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            if _clean_text(payload.get("prompt_version", "")) != _clean_text(self.prompt_version):
                return None
            pack_path = Path(str(payload.get("pack_path", "")))
            if not pack_path.exists():
                return None
            loaded = yaml.safe_load(pack_path.read_text(encoding="utf-8")) or {}
            if not isinstance(loaded, dict):
                return None
            stage_payloads = payload.get("stage_payloads", {})
            if isinstance(stage_payloads, dict):
                cached_candidate_formation = stage_payloads.get("candidate_formation", {})
                if isinstance(cached_candidate_formation, dict):
                    loaded["candidate_formation"] = _merge_missing_or_empty_fields(
                        loaded.get("candidate_formation", {}),
                        cached_candidate_formation,
                    )
            if "candidate_formation" not in loaded:
                loaded["candidate_formation"] = {}
            self._ensure_candidate_formation_safety(loaded["candidate_formation"])
            if isinstance(loaded.get("weak_signal_rules"), dict):
                scoring_adjustments = loaded["weak_signal_rules"].get("scoring_adjustments", [])
                loaded["weak_signal_rules"] = _normalize_weak_signal_markers(loaded["weak_signal_rules"])
                loaded["weak_signal_rules"]["scoring_adjustments"] = (
                    scoring_adjustments if isinstance(scoring_adjustments, list) else []
                )
            pack = DomainPack.from_dict(loaded)
            if not is_current_generated_domain_pack(pack, current_prompt_version=self.prompt_version):
                return None
            return pack
        except Exception:
            return None


def _coerce_pack(pack_or_payload: Any) -> DomainPack:
    if isinstance(pack_or_payload, dict):
        return DomainPack.from_dict(pack_or_payload)
    if hasattr(pack_or_payload, "to_dict"):
        return DomainPack.from_dict(pack_or_payload.to_dict())
    if isinstance(pack_or_payload, DomainPack):
        return pack_or_payload
    return DomainPack.from_dict(pack_or_payload)


def save_domain_pack_to_memory(
    pack_or_payload: DomainPack | Dict[str, Any],
    *,
    memory_dir: Path | str = DEFAULT_MEMORY_DIR,
) -> Path:
    pack = _coerce_pack(pack_or_payload)
    target_dir = Path(memory_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{pack.domain_pack_hash}.yaml"
    path.write_text(
        yaml.safe_dump(pack.to_dict(), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return path


def save_domain_pack_snapshot(
    pack_or_payload: DomainPack | Dict[str, Any],
    *,
    result_dir: Path | str,
    filename: str = "domain_pack.yaml",
) -> Path:
    pack = _coerce_pack(pack_or_payload)
    target_dir = Path(result_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / filename
    path.write_text(
        yaml.safe_dump(pack.to_dict(), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return path
