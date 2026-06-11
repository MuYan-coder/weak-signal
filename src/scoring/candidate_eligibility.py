"""Candidate eligibility contract for scoring.

The gate is intentionally domain-aware: non-humanoid domains should not inherit
humanoid robot anchors when deciding whether a candidate can enter scoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Set

import pandas as pd

from src.extraction.tech_lexicon import build_domain_lexicon, normalize_signal_phrase


TECHNICAL_ANCHOR_TERMS: Set[str] = {
    "3d printing",
    "additive manufacturing",
    "process",
    "材料",
    "工艺",
    "制备",
    "制造",
    "打印",
    "增材制造",
    "装配",
}


@dataclass
class CandidateEligibilityProfile:
    candidate_eligibility: str
    score_applicability: str
    eligibility_reason_codes: List[str] = field(default_factory=list)
    phase: str = "candidate_form_ready"
    has_technical_anchor: bool = False
    technical_envelope: bool = False
    domain_technical_anchor_count: int = 0

    @property
    def eligibility_reason(self) -> str:
        return "；".join(self.eligibility_reason_codes)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_eligibility": self.candidate_eligibility,
            "score_applicability": self.score_applicability,
            "eligibility_reason_codes": list(self.eligibility_reason_codes),
            "eligibility_reason": self.eligibility_reason,
            "candidate_eligibility_phase": self.phase,
            "candidate_has_technical_anchor": bool(self.has_technical_anchor),
            "candidate_technical_envelope": bool(self.technical_envelope),
            "domain_technical_anchor_count": int(self.domain_technical_anchor_count),
        }


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def _safe_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple) or isinstance(value, set):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    return [value]


def _dedupe_terms(values: Iterable[Any]) -> Set[str]:
    terms: Set[str] = set()
    for value in values:
        text = _safe_text(value)
        if text:
            terms.add(text)
    return terms


def _domain_anchor_terms(domain_context: Any = None) -> Set[str]:
    if domain_context is None:
        return set()
    try:
        from src.extraction.domain_candidate_policy import build_domain_candidate_policy
    except Exception:
        return set()
    try:
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


def _haystack(row: Dict[str, Any]) -> str:
    return normalize_signal_phrase(" ".join(_candidate_text_parts(row)))


def _term_in_haystack(term: str, haystack: str) -> bool:
    term_norm = normalize_signal_phrase(term)
    return bool(term_norm and term_norm in haystack)


def _has_technical_anchor(row: Dict[str, Any], domain_context: Any = None) -> bool:
    text = _haystack(row)
    if not text:
        return False
    domain_anchors = _domain_anchor_terms(domain_context)
    anchors = domain_anchors if domain_anchors else set(TECHNICAL_ANCHOR_TERMS)
    return any(_term_in_haystack(term, text) for term in anchors)


def _technical_envelope(row: Dict[str, Any], domain_context: Any = None) -> bool:
    if bool(row.get("is_observation_scope", False)):
        return False
    if bool(row.get("is_scope_echo", False)) or bool(row.get("generic_core_only", False)):
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

    has_mechanism = bool(row.get("has_mechanism_core", False)) or bool(_safe_text(row.get("mechanism_core")))
    has_constraint = bool(row.get("has_non_scope_constraint", False)) or bool(row.get("survives_without_scope", False))
    if not has_mechanism or not has_constraint:
        return False

    if bool(row.get("scope_shell_heavy", False)) and not bool(row.get("survives_without_scope", False)):
        return False

    return True


def _reason_codes(row: Dict[str, Any], *, has_anchor: bool, envelope: bool, domain_anchor_count: int) -> List[str]:
    codes: List[str] = []
    if not has_anchor:
        if domain_anchor_count:
            codes.append("missing_domain_technical_anchor")
        else:
            codes.append("missing_technical_anchor")
    if not envelope:
        codes.append("technical_envelope_missing")
    if bool(row.get("is_observation_scope", False)):
        codes.append("observation_scope_not_candidate")
    if bool(row.get("is_scope_echo", False)):
        codes.append("scope_echo")
    if bool(row.get("generic_core_only", False)):
        codes.append("generic_core_only")
    if bool(row.get("scope_shell_heavy", False)) and not bool(row.get("survives_without_scope", False)):
        codes.append("scope_shell_heavy")
    if _safe_text(row.get("topic_granularity")) == "generic_or_failed":
        codes.append("generic_or_failed")
    if not codes and envelope:
        codes.append("technical_envelope_ready")
    return list(dict.fromkeys(codes))


def evaluate_candidate_eligibility(
    row: Dict[str, Any],
    *,
    phase: str = "candidate_form_ready",
    domain_context: Any = None,
) -> CandidateEligibilityProfile:
    row = dict(row or {})
    domain_anchor_count = len(_domain_anchor_terms(domain_context))
    has_anchor = _has_technical_anchor(row, domain_context=domain_context)
    envelope = _technical_envelope(row, domain_context=domain_context)
    reason_codes = _reason_codes(
        row,
        has_anchor=has_anchor,
        envelope=envelope,
        domain_anchor_count=domain_anchor_count,
    )
    if not has_anchor or not envelope:
        return CandidateEligibilityProfile(
            candidate_eligibility="not_eligible",
            score_applicability="not_applicable",
            eligibility_reason_codes=reason_codes,
            phase=phase,
            has_technical_anchor=has_anchor,
            technical_envelope=envelope,
            domain_technical_anchor_count=domain_anchor_count,
        )

    source_count = int(row.get("source_count", 0) or 0)
    evidence_count = int(row.get("cluster_evidence_count", 0) or 0)
    stage = _safe_text(row.get("candidate_stage"))
    eligibility = (
        "eligible"
        if source_count >= 2 or evidence_count >= 2 or stage == "formed_candidate_strong"
        else "candidate_monitoring"
    )
    if eligibility == "candidate_monitoring":
        reason_codes = [*reason_codes, "needs_more_evidence"]

    return CandidateEligibilityProfile(
        candidate_eligibility=eligibility,
        score_applicability="applicable",
        eligibility_reason_codes=list(dict.fromkeys(reason_codes)),
        phase=phase,
        has_technical_anchor=has_anchor,
        technical_envelope=envelope,
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
        for column, default in CandidateEligibilityProfile(
            candidate_eligibility="",
            score_applicability="",
        ).to_dict().items():
            result[column] = default
        return result

    profiles = [
        evaluate_candidate_eligibility(row.to_dict(), phase=phase, domain_context=domain_context).to_dict()
        for _, row in result.iterrows()
    ]
    profile_df = pd.DataFrame(profiles, index=result.index)
    for column in profile_df.columns:
        result[column] = profile_df[column]
    return result
