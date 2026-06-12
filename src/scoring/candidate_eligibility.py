"""Candidate eligibility contract for scoring.

The gate is domain-aware: when a Domain Pack is available, technical-anchor
checks use the pack's candidate-formation terms instead of legacy defaults.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Set

import pandas as pd

from src.extraction.tech_lexicon import build_domain_lexicon, normalize_signal_phrase


CANDIDATE_ELIGIBILITY_SCHEMA_VERSION = "candidate_eligibility_v2"
ELIGIBLE_STATES = {"eligible", "candidate_monitoring", "review_only", "rejected"}


POLICY_MARKET_NOISE_TERMS = {
    "政策",
    "补贴",
    "扶持",
    "规划",
    "战略",
    "产业链",
    "产业",
    "发展",
    "价格",
    "成本",
    "供需",
    "融资",
    "估值",
    "招商",
    "投资",
    "市场",
    "金融",
    "服务方案",
    "商业模式",
    "量产",
    "发售",
    "销售",
    "收入",
    "港口",
    "航线",
    "扩建工程",
    "泊位",
    "堆场",
    "能力建设",
    "网络外交",
    "policy",
    "subsidy",
    "strategy",
    "roadmap",
    "market",
    "finance",
    "investment",
    "valuation",
    "sales",
    "commercial",
    "port",
    "price",
    "cost",
    "demand",
    "supply",
}

TECHNICAL_ANCHOR_TERMS: Set[str] = {
    "工艺",
    "算法",
    "模型",
    "芯片",
    "材料",
    "传感器",
    "控制",
    "系统",
    "装置",
    "器件",
    "平台",
    "引擎",
    "协议",
    "架构",
    "电池",
    "电解质",
    "碳化硅",
    "氮化镓",
    "3d打印",
    "3D打印",
    "打印",
    "冷焊",
    "增材制造",
    "在轨装配",
    "微重力",
    "制造",
    "制备",
    "成形",
    "焊接",
    "model",
    "algorithm",
    "sensor",
    "chip",
    "material",
    "process",
    "device",
    "architecture",
    "protocol",
    "battery",
    "additive manufacturing",
    "space manufacturing",
    "cvd",
}

REJECT_NAME_ISSUES = {
    "bare_mechanism",
    "generic_method_only_without_domain_anchor",
}

REVIEW_NAME_ISSUES = {
    "scope_dependent",
    "scope_shell_heavy",
}


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def _safe_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, (tuple, set)):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    try:
        if pd.isna(value):
            return []
    except Exception:
        pass
    return [value]


def _safe_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "0", "false", "no", "n", "nan", "none", "null"}:
            return False
        if text in {"1", "true", "yes", "y"}:
            return True
        return default
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    return bool(value)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or pd.isna(value):
            return default
    except Exception:
        pass
    try:
        return int(float(value))
    except Exception:
        return default


def _issue_tokens(value: Any) -> List[str]:
    text = _safe_text(value)
    if not text:
        return []
    parts = re.split(r"[;,，；、\s|/]+", text)
    return [part for part in (item.strip() for item in parts) if part]


def _contains_any(text: str, terms: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(term and term.lower() in lowered for term in terms)


def _dedupe_terms(values: Iterable[Any]) -> Set[str]:
    terms: Set[str] = set()
    for value in values or []:
        text = _safe_text(value)
        if text:
            terms.add(text)
    return terms


def _domain_anchor_terms(domain_context: Any = None) -> Set[str]:
    if domain_context is None:
        return set()
    try:
        from src.extraction.domain_candidate_policy import build_domain_candidate_policy

        lexicon = build_domain_lexicon(domain_context)
        policy = build_domain_candidate_policy(lexicon)
        return _dedupe_terms(policy.technical_anchor_terms())
    except Exception:
        return set()


def _candidate_text_parts(row: Dict[str, Any]) -> List[str]:
    parts: List[str] = []
    scalar_fields = [
        "display_candidate_name",
        "topic_summary_name",
        "tech_name",
        "raw_phrase",
        "raw_candidate_text",
        "normalized_candidate_text",
        "canonical_candidate_name_en",
        "stable_object_label",
        "mechanism_core",
        "technical_subject_anchor",
        "relation_target",
        "relation_task",
        "constraint_signature",
    ]
    list_fields = [
        "object_modifier_tokens",
        "task_constraint_tokens",
        "data_modifier_tokens",
        "method_modifier_tokens",
        "technology",
        "technologies",
        "technical_object_types",
        "mechanism_core_tokens",
    ]
    for field_name in scalar_fields:
        text = _safe_text(row.get(field_name))
        if text:
            parts.append(text)
    for field_name in list_fields:
        for item in _safe_list(row.get(field_name)):
            text = _safe_text(item)
            if text:
                parts.append(text)
    for evidence in _safe_list(row.get("evidence_items")):
        if not isinstance(evidence, dict):
            continue
        for field_name in ["raw_candidate_text", "snippet", "title", "evidence_span"]:
            text = _safe_text(evidence.get(field_name))
            if text:
                parts.append(text)
    return parts


def _candidate_haystack(row: Dict[str, Any]) -> str:
    return normalize_signal_phrase(" ".join(_candidate_text_parts(row)))


def _term_in_haystack(term: str, haystack: str) -> bool:
    term_norm = normalize_signal_phrase(term)
    return bool(term_norm and term_norm in haystack)


def _has_technical_anchor(row: Dict[str, Any], domain_context: Any = None) -> bool:
    text = _candidate_haystack(row)
    if not text:
        return False
    domain_anchors = _domain_anchor_terms(domain_context)
    anchors = domain_anchors if domain_anchors else set(TECHNICAL_ANCHOR_TERMS)
    return any(_term_in_haystack(term, text) for term in anchors)


def _technical_envelope(row: Dict[str, Any], domain_context: Any = None) -> bool:
    if _safe_bool(row.get("is_observation_scope")):
        return False
    if _safe_bool(row.get("is_scope_echo")) or _safe_bool(row.get("generic_core_only")):
        return False
    if not _safe_text(row.get("display_candidate_name") or row.get("tech_name")):
        return False

    stage = _safe_text(row.get("candidate_stage"))
    if stage and stage not in {"formed_candidate", "formed_candidate_strong"}:
        return False

    granularity = _safe_text(row.get("topic_granularity"))
    if granularity == "generic_or_failed":
        return False

    if not _has_technical_anchor(row, domain_context=domain_context):
        return False

    has_mechanism = _safe_bool(row.get("has_mechanism_core")) or bool(_safe_text(row.get("mechanism_core")))
    has_constraint = _safe_bool(row.get("has_non_scope_constraint")) or _safe_bool(
        row.get("survives_without_scope"),
        default=False,
    )
    if not has_mechanism or not has_constraint:
        return False

    if _safe_bool(row.get("scope_shell_heavy")) and not _safe_bool(row.get("survives_without_scope")):
        return False

    return True


@dataclass
class CandidateEligibilityProfile:
    candidate_eligibility: str
    eligibility_reason_codes: List[str] = field(default_factory=list)
    weak_signal_raw_score: float = 0.0
    score_suppression_reason: str = ""
    eligible_for_scoring: bool = False
    eligible_for_signal_generation: bool = False
    eligible_for_weak_signal: bool = False
    score_applicability: str = "not_applicable"
    candidate_eligibility_schema_version: str = CANDIDATE_ELIGIBILITY_SCHEMA_VERSION
    candidate_eligibility_phase: str = "candidate_form_ready"
    has_technical_anchor: bool = False
    technical_envelope: bool = False
    domain_technical_anchor_count: int = 0

    @property
    def phase(self) -> str:
        return self.candidate_eligibility_phase

    @property
    def eligibility_reason(self) -> str:
        return "；".join(self.eligibility_reason_codes)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_eligibility": self.candidate_eligibility,
            "eligibility_reason_codes": list(self.eligibility_reason_codes),
            "eligibility_reason": self.eligibility_reason,
            "weak_signal_raw_score": self.weak_signal_raw_score,
            "score_suppression_reason": self.score_suppression_reason,
            "eligible_for_scoring": self.eligible_for_scoring,
            "eligible_for_signal_generation": self.eligible_for_signal_generation,
            "eligible_for_weak_signal": self.eligible_for_weak_signal,
            "score_applicability": self.score_applicability,
            "candidate_eligibility_schema_version": self.candidate_eligibility_schema_version,
            "candidate_eligibility_phase": self.candidate_eligibility_phase,
            "candidate_has_technical_anchor": bool(self.has_technical_anchor),
            "candidate_technical_envelope": bool(self.technical_envelope),
            "domain_technical_anchor_count": int(self.domain_technical_anchor_count),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CandidateEligibilityProfile":
        status = _safe_text(data.get("candidate_eligibility")) or "rejected"
        if status not in ELIGIBLE_STATES:
            status = "rejected"
        return cls.create(
            status,
            eligibility_reason_codes=list(_safe_list(data.get("eligibility_reason_codes", []))),
            phase=_safe_text(data.get("candidate_eligibility_phase")) or "candidate_form_ready",
            has_technical_anchor=_safe_bool(data.get("candidate_has_technical_anchor")),
            technical_envelope=_safe_bool(data.get("candidate_technical_envelope")),
            domain_technical_anchor_count=_safe_int(data.get("domain_technical_anchor_count"), 0),
        )

    @classmethod
    def create(
        cls,
        candidate_eligibility: str,
        eligibility_reason_codes: List[str] | None = None,
        *,
        phase: str = "candidate_form_ready",
        has_technical_anchor: bool = False,
        technical_envelope: bool = False,
        domain_technical_anchor_count: int = 0,
    ) -> "CandidateEligibilityProfile":
        status = candidate_eligibility if candidate_eligibility in ELIGIBLE_STATES else "rejected"
        reasons = list(dict.fromkeys(eligibility_reason_codes or []))
        is_eligible = status == "eligible"
        return cls(
            candidate_eligibility=status,
            eligibility_reason_codes=reasons,
            eligible_for_scoring=is_eligible,
            eligible_for_signal_generation=is_eligible,
            eligible_for_weak_signal=is_eligible,
            score_applicability="applicable" if is_eligible else "not_applicable",
            score_suppression_reason="" if is_eligible else f"candidate_eligibility={status}",
            candidate_eligibility_phase=phase,
            has_technical_anchor=has_technical_anchor,
            technical_envelope=technical_envelope,
            domain_technical_anchor_count=domain_technical_anchor_count,
        )


def _profile(
    status: str,
    reasons: List[str],
    *,
    phase: str,
    has_technical_anchor: bool,
    technical_envelope: bool,
    domain_technical_anchor_count: int,
) -> CandidateEligibilityProfile:
    return CandidateEligibilityProfile.create(
        status,
        reasons,
        phase=phase,
        has_technical_anchor=has_technical_anchor,
        technical_envelope=technical_envelope,
        domain_technical_anchor_count=domain_technical_anchor_count,
    )


def evaluate_candidate_eligibility(
    row: Dict[str, Any],
    *,
    phase: str = "candidate_form_ready",
    domain_context: Any = None,
) -> CandidateEligibilityProfile:
    row = dict(row or {})
    reasons: List[str] = []
    review_reasons: List[str] = []

    domain_anchor_count = len(_domain_anchor_terms(domain_context))
    has_technical_anchor = _has_technical_anchor(row, domain_context=domain_context)
    technical_envelope = _technical_envelope(row, domain_context=domain_context)

    stage = _safe_text(row.get("candidate_stage"))
    topic_granularity = _safe_text(row.get("topic_granularity"))
    display_tier = _safe_text(row.get("display_tier"))
    domain_pack_status = _safe_text(row.get("domain_pack_candidate_status"))
    source_type = _safe_text(row.get("source_type")).lower()
    source_types = row.get("source_types", [])
    if not isinstance(source_types, (list, tuple, set)):
        source_types = [source_types]
    source_type_set = {source_type} | {_safe_text(item).lower() for item in source_types}
    source_type_set.discard("")

    name_issues = set(_issue_tokens(row.get("display_candidate_name_issue")))
    has_mechanism_core = _safe_bool(row.get("has_mechanism_core"))
    has_non_scope_constraint = _safe_bool(row.get("has_non_scope_constraint"))
    scope_shell_heavy = _safe_bool(row.get("scope_shell_heavy"))
    survives_without_scope = _safe_bool(row.get("survives_without_scope"), default=True)
    is_scope_internal_candidate = _safe_bool(row.get("is_scope_internal_candidate"))
    text_surface = " ".join(
        _safe_text(row.get(key))
        for key in ["display_candidate_name", "raw_phrase", "raw_candidate_text", "mechanism_core"]
    )

    if stage in {"filtered_scope_echo", "unformed_generic_core"}:
        reasons.append(stage)
    if topic_granularity == "generic_or_failed":
        reasons.append("generic_or_failed")
    if domain_pack_status == "rejected":
        reasons.append("domain_pack_rejected")
    if _safe_bool(row.get("is_scope_echo")):
        reasons.append("is_scope_echo")
    if _safe_bool(row.get("generic_core_only")):
        reasons.append("generic_core_only")
    for issue in sorted(REJECT_NAME_ISSUES & name_issues):
        reasons.append(issue)

    if stage and stage != "scope_overview" and (not has_mechanism_core or not has_non_scope_constraint):
        reasons.append("bare_without_mechanism_or_constraint")

    enforce_anchor = bool(
        domain_anchor_count
        or stage in {"formed_candidate", "formed_candidate_strong"}
        or text_surface.strip()
        or row.get("technical_object_types")
    )
    if enforce_anchor and not has_technical_anchor:
        if domain_anchor_count:
            reasons.append("missing_domain_technical_anchor")
        else:
            reasons.append("missing_technical_anchor")
    if enforce_anchor and stage in {"formed_candidate", "formed_candidate_strong"} and not technical_envelope:
        reasons.append("technical_envelope_missing")

    has_noise_surface = _contains_any(text_surface, POLICY_MARKET_NOISE_TERMS)
    if has_noise_surface and (
        not has_technical_anchor
        or scope_shell_heavy
        or not survives_without_scope
        or not has_mechanism_core
        or not has_non_scope_constraint
    ):
        reasons.append("noise:policy_market_or_commercial_noise")
    if source_type_set & {"policy", "market", "finance"} and scope_shell_heavy:
        reasons.append("policy_market_source_scope_shell")

    if (
        stage in {"formed_candidate", "formed_candidate_strong"}
        and domain_pack_status == ""
        and not technical_envelope
        and not (scope_shell_heavy or not survives_without_scope)
    ):
        reasons.append("domain_pack_status_empty_without_technical_envelope")

    if reasons:
        return _profile(
            "rejected",
            list(dict.fromkeys(reasons)),
            phase=phase,
            has_technical_anchor=has_technical_anchor,
            technical_envelope=technical_envelope,
            domain_technical_anchor_count=domain_anchor_count,
        )

    if scope_shell_heavy:
        review_reasons.append("scope_shell_heavy")
    if not survives_without_scope:
        review_reasons.append("scope_dependent")
    if scope_shell_heavy and not survives_without_scope:
        review_reasons.append("scope_shell_dependent")
    if topic_granularity == "scope_internal_candidate":
        review_reasons.append("scope_internal_candidate")
    for issue in sorted(REVIEW_NAME_ISSUES & name_issues):
        review_reasons.append(issue)
    if _safe_text(row.get("topic_tier")) in {"upper_topic", "compressed_label", "unclear"}:
        review_reasons.append(f"llm_judged_{_safe_text(row.get('topic_tier'))}")

    if review_reasons:
        return _profile(
            "review_only",
            list(dict.fromkeys(review_reasons)),
            phase=phase,
            has_technical_anchor=has_technical_anchor,
            technical_envelope=technical_envelope,
            domain_technical_anchor_count=domain_anchor_count,
        )

    source_count = _safe_int(row.get("source_count"), 0)
    cluster_evidence_count = _safe_int(row.get("cluster_evidence_count"), 0)
    try:
        evidence_quality_score = float(row.get("candidate_evidence_quality", 0.0) or 0.0)
    except Exception:
        evidence_quality_score = 0.0
    sources = row.get("sources", [])
    if not isinstance(sources, (list, tuple, set)):
        sources = []
    has_authoritative_single_source = any(
        _safe_text(source.get("type", source.get("source_type", ""))).lower()
        in {"paper", "patent", "report", "journal"}
        for source in sources
        if isinstance(source, dict)
    )
    has_authoritative_source_type = bool(source_type_set & {"paper", "patent", "report", "journal"})
    evidence_ready = (
        (source_count >= 2 and cluster_evidence_count >= 2)
        or (
            source_count == 1
            and (has_authoritative_single_source or has_authoritative_source_type)
            and (evidence_quality_score >= 0.7 or cluster_evidence_count >= 2 or stage == "formed_candidate_strong")
        )
    )

    is_formed = stage in {"formed_candidate", "formed_candidate_strong"}
    is_fine_grained = topic_granularity == "fine_grained_topic"
    is_weak_tier = display_tier == "weak_signal"
    if (
        is_formed
        and is_fine_grained
        and is_weak_tier
        and (is_scope_internal_candidate or phase in {"scoring_ready", "signal_generation_ready"})
        and technical_envelope
        and evidence_ready
    ):
        return _profile(
            "eligible",
            [],
            phase=phase,
            has_technical_anchor=has_technical_anchor,
            technical_envelope=technical_envelope,
            domain_technical_anchor_count=domain_anchor_count,
        )

    if is_formed and is_fine_grained and is_weak_tier and technical_envelope:
        return _profile(
            "candidate_monitoring",
            ["insufficient_evidence_for_eligible"],
            phase=phase,
            has_technical_anchor=has_technical_anchor,
            technical_envelope=technical_envelope,
            domain_technical_anchor_count=domain_anchor_count,
        )

    return _profile(
        "review_only",
        ["does_not_meet_eligible_contract"],
        phase=phase,
        has_technical_anchor=has_technical_anchor,
        technical_envelope=technical_envelope,
        domain_technical_anchor_count=domain_anchor_count,
    )


def apply_candidate_eligibility(
    df: pd.DataFrame,
    *,
    phase: str = "candidate_form_ready",
    domain_context: Any = None,
) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    result = df.copy()

    if result.empty:
        empty_profile = CandidateEligibilityProfile.create("", phase=phase).to_dict()
        for column, default in empty_profile.items():
            result[column] = default
            df[column] = default
        return result

    profiles = [
        evaluate_candidate_eligibility(row.to_dict(), phase=phase, domain_context=domain_context).to_dict()
        for _, row in result.iterrows()
    ]
    profile_df = pd.DataFrame(profiles, index=result.index)
    for column in profile_df.columns:
        result[column] = profile_df[column]
        df[column] = profile_df[column]
    return result
