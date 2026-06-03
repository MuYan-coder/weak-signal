from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable, Dict, List

import yaml

from ..utils.env_config import get_preferred_chat_model
from ..utils.llm_client import chat_text
from .models import DOMAIN_PACK_SCHEMA_VERSION, DomainPack


DOMAIN_PACK_GENERATOR_PROMPT_VERSION = "domain_pack_generator_v1"
DEFAULT_MEMORY_DIR = Path("memory") / "domain_packs"
GENERATOR_SYSTEM_PROMPT = (
    "你是弱信号识别系统的领域运行包生成器。"
    "只输出结构化 JSON 或 YAML，不输出解释性寒暄。"
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
- out_of_scope_domains
- sub_directions
- noise_terms
- search_strategy.core_keywords
- search_strategy.synonyms
- search_strategy.english_terms
- search_strategy.exclude_terms

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
        "humanoid_preset_pattern_summary": {
            "valid_pattern": "technical_object + mechanism + task/performance + evidence_span",
            "invalid_pattern": "scope or shell term only",
            "generic_shell_examples": ["技术", "方法", "系统", "应用", "technology", "method", "system"],
        },
    }
    return f"""
阶段二：候选成形范式生成。

请输出合法 JSON 对象，所有 pattern 必须有稳定 pattern_id。

必须输出字段：
- technical_object_types
- mechanism_types
- task_or_performance_types
- data_or_method_types
- scene_or_application_types
- generic_terms
- shell_terms
- valid_candidate_patterns
- invalid_candidate_patterns

required_slots、optional_slots、reject_if_slots_only 只能使用这些槽位：
{", ".join(ALLOWED_SLOT_NAMES)}

输入：
{_json_block(payload)}
""".strip()


def build_weak_signal_rules_prompt(
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
阶段三：弱信号识别范式生成。

请输出合法 JSON 对象，scoring_adjustments 中每条规则必须有 rule_id，score_delta 和 max_delta 必须是有界数值。

必须输出字段：
- early_stage_markers
- low_attention_markers
- niche_actor_markers
- cross_domain_markers
- engineering_trace_markers
- commercialization_noise_markers
- policy_or_market_noise_markers
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
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if json_match:
        candidates.append(json_match.group(0))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        try:
            parsed = yaml.safe_load(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    raise DomainPackGenerationError(f"{stage_name} response is not valid JSON/YAML mapping")


class DomainPackGenerator:
    def __init__(
        self,
        *,
        chat_fn: Callable | None = None,
        memory_dir: Path | str = DEFAULT_MEMORY_DIR,
        model: str | None = None,
        prompt_version: str = DOMAIN_PACK_GENERATOR_PROMPT_VERSION,
        max_tokens: int = 4096,
        temperature: float = 0.1,
    ):
        self.chat_fn = chat_fn or chat_text
        self.memory_dir = Path(memory_dir)
        self.model = model or get_preferred_chat_model()
        self.prompt_version = prompt_version
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate(self, request: DomainPackGenerationRequest, *, refresh_cache: bool = False) -> DomainPack:
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

        stage_payloads: Dict[str, Dict[str, Any]] = {}
        domain_model = self._call_stage("domain_modeling", build_domain_modeling_prompt(request))
        stage_payloads["domain_modeling"] = domain_model
        candidate_formation = self._call_stage(
            "candidate_formation",
            build_candidate_formation_prompt(request, domain_model),
        )
        stage_payloads["candidate_formation"] = candidate_formation
        weak_signal_rules = self._call_stage(
            "weak_signal_rules",
            build_weak_signal_rules_prompt(request, domain_model, candidate_formation),
        )
        stage_payloads["weak_signal_rules"] = weak_signal_rules
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

        pack_path = self._save_pack(pack)
        self._save_cache(cache_path, cache_key, pack_path, pack, stage_payloads)
        return pack

    def _call_stage(self, stage_name: str, prompt: str) -> Dict[str, Any]:
        raw, _usage, _response = self.chat_fn(
            prompt,
            system=GENERATOR_SYSTEM_PROMPT,
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return _parse_structured_response(raw, stage_name)

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
        if not isinstance(final.get("search_strategy"), dict):
            final["search_strategy"] = domain_model.get("search_strategy", {})
        if not isinstance(final.get("candidate_formation"), dict):
            final["candidate_formation"] = candidate_formation
        if not isinstance(final.get("weak_signal_rules"), dict):
            final["weak_signal_rules"] = weak_signal_rules
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
            pack_path = Path(str(payload.get("pack_path", "")))
            if not pack_path.exists():
                return None
            loaded = yaml.safe_load(pack_path.read_text(encoding="utf-8")) or {}
            if not isinstance(loaded, dict):
                return None
            return DomainPack.from_dict(loaded)
        except Exception:
            return None


def _coerce_pack(pack_or_payload: DomainPack | Dict[str, Any]) -> DomainPack:
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
