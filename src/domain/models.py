from __future__ import annotations

from dataclasses import dataclass, field
import copy
import hashlib
import json
import re
from typing import Any, Dict


DOMAIN_PACK_SCHEMA_VERSION = "domain_pack_v1"

SECTION_DEFAULTS: Dict[str, Any] = {
    "source": {
        "mode": "preset",
        "model": "",
        "prompt_version": "",
        "based_on_user_input": {
            "field_id": "",
            "field_name": "",
            "keywords": [],
            "synonyms": [],
            "exclude_terms": [],
        },
    },
    "domain_identity": {
        "field_id": "",
        "field_name": "",
        "parent_industries": [],
        "domain_boundary": "",
        "adjacent_domains": [],
        "out_of_scope_domains": [],
    },
    "search_strategy": {
        "core_keywords": [],
        "synonyms": [],
        "english_terms": [],
        "exclude_terms": [],
        "source_type_weights": {
            "paper": 1.0,
            "patent": 1.0,
            "news": 1.0,
            "report": 1.0,
            "policy": 1.0,
        },
        "count_policy": {
            "mode": "recommend_only",
            "default_total_sample_size": 0,
            "min_per_source": {},
            "max_per_source": {},
        },
        "query_expansion_rules": [],
    },
    "observation_scopes": {
        "main_scope": "",
        "sub_scopes": [],
        "scope_aliases": [],
        "scope_echo_terms": [],
        "off_domain_anchor_terms": [],
    },
    "candidate_formation": {
        "technical_object_types": [],
        "mechanism_types": [],
        "task_or_performance_types": [],
        "data_or_method_types": [],
        "scene_or_application_types": [],
        "generic_terms": [],
        "shell_terms": [],
        "valid_candidate_patterns": [],
        "invalid_candidate_patterns": [],
        "minimum_specificity_rule": {
            "min_non_shell_slots": 2,
            "require_evidence_span": True,
            "allow_scope_only_candidate": False,
        },
    },
    "weak_signal_rules": {
        "early_stage_markers": [],
        "low_attention_markers": [],
        "niche_actor_markers": [],
        "cross_domain_markers": [],
        "engineering_trace_markers": [],
        "commercialization_noise_markers": [],
        "policy_or_market_noise_markers": [],
        "scoring_adjustments": [],
    },
    "canonicalization": {
        "object_families": [],
        "alias_groups": [],
        "parent_child_terms": [],
        "do_not_merge_rules": [],
        "object_family_enabled": False,
    },
    "evidence_rules": {
        "strong_evidence_patterns": [],
        "weak_evidence_patterns": [],
        "source_reliability_hints": [],
        "traceability_fields": [],
        "evidence_rejection_patterns": [],
    },
    "reporting": {
        "display_labels": {},
        "explanation_templates": [],
        "review_hints": [],
    },
}

HASH_SECTION_KEYS = [
    "schema_version",
    "domain_identity",
    "search_strategy",
    "observation_scopes",
    "candidate_formation",
    "weak_signal_rules",
    "canonicalization",
    "evidence_rules",
    "reporting",
]


def _deep_merge(default: Any, value: Any) -> Any:
    if isinstance(default, dict):
        merged = copy.deepcopy(default)
        if isinstance(value, dict):
            for key, item in value.items():
                merged[key] = _deep_merge(merged[key], item) if key in merged else copy.deepcopy(item)
        return merged
    if value is None:
        return copy.deepcopy(default)
    return copy.deepcopy(value)


def _normalize_scalar(value: Any) -> Any:
    if isinstance(value, str):
        text = re.sub(r"\s+", " ", value.strip())
        return text.lower()
    return value


def _canonicalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _canonicalize(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, list):
        items = [_canonicalize(item) for item in value]
        serialized = []
        seen = set()
        for item in items:
            marker = json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            if marker in seen:
                continue
            seen.add(marker)
            serialized.append((marker, item))
        return [item for _, item in sorted(serialized, key=lambda pair: pair[0])]
    return _normalize_scalar(value)


def canonical_domain_pack_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    source = payload.get("source", {}) if isinstance(payload.get("source", {}), dict) else {}
    return {
        **{key: payload.get(key) for key in HASH_SECTION_KEYS},
        "source": {
            "based_on_user_input": source.get("based_on_user_input", {}),
            "prompt_version": source.get("prompt_version", ""),
            "model": source.get("model", ""),
        },
    }


def compute_domain_pack_hash(payload: Dict[str, Any]) -> str:
    canonical_payload = _canonicalize(canonical_domain_pack_payload(payload))
    encoded = json.dumps(canonical_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


@dataclass
class DomainPack:
    schema_version: str
    pack_id: str
    pack_name: str
    domain_pack_version: str
    raw: Dict[str, Any] = field(default_factory=dict)
    domain_pack_hash: str = ""
    source: Dict[str, Any] = field(default_factory=dict)
    domain_identity: Dict[str, Any] = field(default_factory=dict)
    search_strategy: Dict[str, Any] = field(default_factory=dict)
    observation_scopes: Dict[str, Any] = field(default_factory=dict)
    candidate_formation: Dict[str, Any] = field(default_factory=dict)
    weak_signal_rules: Dict[str, Any] = field(default_factory=dict)
    canonicalization: Dict[str, Any] = field(default_factory=dict)
    evidence_rules: Dict[str, Any] = field(default_factory=dict)
    reporting: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "DomainPack":
        if not isinstance(payload, dict):
            raise TypeError("Domain Pack payload must be a dictionary")
        normalized = copy.deepcopy(payload)
        normalized["schema_version"] = str(
            normalized.get("schema_version") or DOMAIN_PACK_SCHEMA_VERSION
        ).strip()
        for section, default in SECTION_DEFAULTS.items():
            normalized[section] = _deep_merge(default, normalized.get(section, {}))

        domain_identity = normalized["domain_identity"]
        source = normalized["source"]
        based_on_user_input = source.get("based_on_user_input", {})
        pack_id = str(
            normalized.get("pack_id")
            or domain_identity.get("field_id")
            or based_on_user_input.get("field_id")
            or "neutral"
        ).strip()
        pack_name = str(
            normalized.get("pack_name")
            or domain_identity.get("field_name")
            or based_on_user_input.get("field_name")
            or pack_id
        ).strip()
        pack_version = str(
            normalized.get("domain_pack_version")
            or normalized.get("pack_version")
            or "v1"
        ).strip()

        normalized["pack_id"] = pack_id
        normalized["pack_name"] = pack_name
        normalized["domain_pack_version"] = pack_version

        return cls(
            schema_version=normalized["schema_version"],
            pack_id=pack_id,
            pack_name=pack_name,
            domain_pack_version=pack_version,
            raw=normalized,
            domain_pack_hash=compute_domain_pack_hash(normalized),
            source=normalized["source"],
            domain_identity=normalized["domain_identity"],
            search_strategy=normalized["search_strategy"],
            observation_scopes=normalized["observation_scopes"],
            candidate_formation=normalized["candidate_formation"],
            weak_signal_rules=normalized["weak_signal_rules"],
            canonicalization=normalized["canonicalization"],
            evidence_rules=normalized["evidence_rules"],
            reporting=normalized["reporting"],
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = copy.deepcopy(self.raw)
        payload["domain_pack_hash"] = self.domain_pack_hash
        payload["domain_pack_version"] = self.domain_pack_version
        return payload


@dataclass
class DomainContext:
    domain_pack: DomainPack
    domain_pack_id: str
    domain_pack_version: str
    domain_pack_hash: str
    source_query: Dict[str, Any] = field(default_factory=dict)
    runtime_mode: str = "neutral"

    @classmethod
    def from_pack(cls, pack: DomainPack, runtime_mode: str = "") -> "DomainContext":
        source = pack.source if isinstance(pack.source, dict) else {}
        source_query = source.get("based_on_user_input", {})
        if not isinstance(source_query, dict):
            source_query = {}
        mode = runtime_mode or _runtime_mode_from_source(source.get("mode", ""), pack.pack_id)
        return cls(
            domain_pack=pack,
            domain_pack_id=pack.pack_id,
            domain_pack_version=pack.domain_pack_version,
            domain_pack_hash=pack.domain_pack_hash,
            source_query=copy.deepcopy(source_query),
            runtime_mode=mode,
        )


def _runtime_mode_from_source(source_mode: str, pack_id: str = "") -> str:
    mode = str(source_mode or "").strip().lower()
    if pack_id == "neutral" or mode == "neutral":
        return "neutral"
    if mode in {"llm_generated", "generated"}:
        return "generated"
    if mode == "user_edited":
        return "user_edited"
    if mode == "preset":
        return "preset"
    return "neutral"
