from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Sequence

from .domain_pack_generator import DEFAULT_MEMORY_DIR
from .models import DomainPack


DRY_RUN_REPORT_SCHEMA_VERSION = "domain_pack_dry_run_v1"


@dataclass
class DomainPackDryRunReport:
    domain_pack_id: str
    domain_pack_hash: str
    sample_doc_count: int
    candidate_count: int
    specific_candidate_count: int
    shell_candidate_ratio: float
    off_domain_leakage_terms: List[str] = field(default_factory=list)
    off_domain_leakage_count: int = 0
    top_invalid_candidate_reasons: Dict[str, int] = field(default_factory=dict)
    recommended_pack_changes: List[str] = field(default_factory=list)
    manual_override: bool = False
    status: str = "completed"
    schema_version: str = DRY_RUN_REPORT_SCHEMA_VERSION
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["shell_candidate_ratio"] = round(float(self.shell_candidate_ratio), 6)
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "DomainPackDryRunReport":
        return cls(
            domain_pack_id=str(payload.get("domain_pack_id", "")),
            domain_pack_hash=str(payload.get("domain_pack_hash", "")),
            sample_doc_count=int(payload.get("sample_doc_count", 0) or 0),
            candidate_count=int(payload.get("candidate_count", 0) or 0),
            specific_candidate_count=int(payload.get("specific_candidate_count", 0) or 0),
            shell_candidate_ratio=float(payload.get("shell_candidate_ratio", 0.0) or 0.0),
            off_domain_leakage_terms=_dedupe_texts(payload.get("off_domain_leakage_terms", [])),
            off_domain_leakage_count=int(payload.get("off_domain_leakage_count", 0) or 0),
            top_invalid_candidate_reasons={
                str(key): int(value or 0)
                for key, value in (payload.get("top_invalid_candidate_reasons", {}) or {}).items()
            },
            recommended_pack_changes=_dedupe_texts(payload.get("recommended_pack_changes", [])),
            manual_override=bool(payload.get("manual_override", False)),
            status=str(payload.get("status", "completed") or "completed"),
            schema_version=str(payload.get("schema_version", DRY_RUN_REPORT_SCHEMA_VERSION)),
            metadata=payload.get("metadata", {}) if isinstance(payload.get("metadata", {}), dict) else {},
        )


def run_domain_pack_dry_run(
    pack: DomainPack,
    *,
    sample_documents: Sequence[Dict[str, Any]] | None = None,
    candidate_records: Sequence[Dict[str, Any]] | None = None,
    memory_dir: Path | str = DEFAULT_MEMORY_DIR,
    manual_override: bool = False,
) -> DomainPackDryRunReport:
    sample_docs = list(sample_documents or [])
    candidates = _normalize_candidate_records(candidate_records)
    if candidate_records is None:
        candidates = _derive_candidate_records(pack, sample_docs)

    term_sets = _build_term_sets(pack)
    specificity_rule = pack.candidate_formation.get("minimum_specificity_rule", {})
    min_non_shell_slots = _positive_int(specificity_rule.get("min_non_shell_slots"), default=2)
    require_evidence_span = bool(specificity_rule.get("require_evidence_span", True))

    invalid_reasons: Counter[str] = Counter()
    specific_candidate_count = 0
    shell_candidate_count = 0
    off_domain_hits: List[str] = []
    off_domain_leakage_count = 0

    for candidate in candidates:
        candidate_text = _candidate_text(candidate)
        evidence_span = _candidate_evidence(candidate)
        combined_text = " ".join(part for part in [candidate_text, evidence_span] if part)
        specific_matches = _matched_terms(combined_text, term_sets["specific"])
        shell_matches = _matched_terms(candidate_text, term_sets["shell"])
        leakage_matches = _matched_terms(combined_text, term_sets["off_domain"])

        reasons: List[str] = []
        if require_evidence_span and not evidence_span:
            reasons.append("missing_evidence_span")
        if leakage_matches:
            reasons.append("off_domain_leakage")
            off_domain_hits.extend(leakage_matches)
            off_domain_leakage_count += 1
        if _is_shell_candidate(candidate_text, specific_matches, shell_matches):
            reasons.append("shell_only")
            shell_candidate_count += 1
        elif len(specific_matches) < min_non_shell_slots:
            reasons.append("insufficient_specificity")

        if not reasons and len(specific_matches) >= min_non_shell_slots:
            specific_candidate_count += 1
        invalid_reasons.update(reasons)

    candidate_count = len(candidates)
    shell_candidate_ratio = shell_candidate_count / candidate_count if candidate_count else 0.0
    report = DomainPackDryRunReport(
        domain_pack_id=pack.pack_id,
        domain_pack_hash=pack.domain_pack_hash,
        sample_doc_count=len(sample_docs),
        candidate_count=candidate_count,
        specific_candidate_count=specific_candidate_count,
        shell_candidate_ratio=shell_candidate_ratio,
        off_domain_leakage_terms=_dedupe_texts(off_domain_hits),
        off_domain_leakage_count=off_domain_leakage_count,
        top_invalid_candidate_reasons=dict(invalid_reasons.most_common(8)),
        recommended_pack_changes=_recommend_pack_changes(
            candidate_count=candidate_count,
            specific_candidate_count=specific_candidate_count,
            shell_candidate_ratio=shell_candidate_ratio,
            off_domain_terms=off_domain_hits,
        ),
        manual_override=manual_override,
        status="manual_override" if manual_override else "completed",
        metadata={
            "min_non_shell_slots": min_non_shell_slots,
            "require_evidence_span": require_evidence_span,
        },
    )
    save_domain_pack_dry_run_report(report, memory_dir=memory_dir)
    # Layer 5: 输出 off-domain 泄漏诊断汇总，便于运行日志快速定位问题
    if off_domain_leakage_count > 0:
        leakage_terms_preview = ", ".join(list({t for t in off_domain_hits})[:8])
        print(
            f"[dry_run] 警告: off-domain 泄漏候选数={off_domain_leakage_count}/{candidate_count}  "
            f"(比例={off_domain_leakage_count / candidate_count:.1%})"
            f"，示例词: {leakage_terms_preview}"
        )
    print(
        f"[dry_run] 完成: pack_id={pack.pack_id} | "
        f"candidates={candidate_count} | specific={specific_candidate_count} | "
        f"shell_ratio={shell_candidate_ratio:.1%} | "
        f"off_domain={off_domain_leakage_count}"
    )
    return report


def save_domain_pack_dry_run_report(
    report: DomainPackDryRunReport,
    *,
    memory_dir: Path | str = DEFAULT_MEMORY_DIR,
) -> Path:
    target_dir = Path(memory_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{report.domain_pack_hash}_dry_run.json"
    path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def load_domain_pack_dry_run_report(path: Path | str) -> DomainPackDryRunReport:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("dry-run report payload must be a JSON object")
    return DomainPackDryRunReport.from_dict(payload)


def _build_term_sets(pack: DomainPack) -> Dict[str, List[str]]:
    candidate_formation = pack.candidate_formation if isinstance(pack.candidate_formation, dict) else {}
    search_strategy = pack.search_strategy if isinstance(pack.search_strategy, dict) else {}
    observation_scopes = pack.observation_scopes if isinstance(pack.observation_scopes, dict) else {}
    domain_identity = pack.domain_identity if isinstance(pack.domain_identity, dict) else {}
    evidence_rules = pack.evidence_rules if isinstance(pack.evidence_rules, dict) else {}

    shell_terms = _dedupe_texts(
        _list_values(candidate_formation.get("generic_terms"))
        + _list_values(candidate_formation.get("shell_terms"))
    )
    specific_terms = _dedupe_texts(
        _list_values(candidate_formation.get("technical_object_types"))
        + _list_values(candidate_formation.get("mechanism_types"))
        + _list_values(candidate_formation.get("task_or_performance_types"))
        + _list_values(candidate_formation.get("data_or_method_types"))
        + _list_values(candidate_formation.get("scene_or_application_types"))
        + _list_values(search_strategy.get("core_keywords"))
        + _list_values(search_strategy.get("synonyms"))
        + _list_values(search_strategy.get("english_terms"))
    )
    specific_terms = [term for term in specific_terms if term.lower() not in {item.lower() for item in shell_terms}]

    off_domain_terms = _dedupe_texts(
        _list_values(observation_scopes.get("off_domain_anchor_terms"))
        + _list_values(search_strategy.get("exclude_terms"))
        + _list_values(domain_identity.get("out_of_scope_domains"))
        + _list_values(evidence_rules.get("evidence_rejection_patterns"))
    )
    return {
        "specific": specific_terms,
        "shell": shell_terms,
        "off_domain": off_domain_terms,
    }


def _derive_candidate_records(
    pack: DomainPack,
    sample_documents: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    term_sets = _build_term_sets(pack)
    domain_terms = term_sets["specific"] + term_sets["shell"]
    records: List[Dict[str, Any]] = []
    seen = set()
    for document in sample_documents:
        title = _clean_text(document.get("title", ""))
        body = _clean_text(document.get("text", document.get("abstract", "")))
        combined = " ".join(part for part in [title, body] if part)
        matched = _matched_terms(combined, domain_terms)
        if not matched:
            continue
        candidate_text = title or " ".join(matched[:4])
        marker = candidate_text.lower()
        if marker in seen:
            continue
        seen.add(marker)
        records.append(
            {
                "candidate_text": candidate_text[:120],
                "evidence_span": combined[:240],
                "source_doc_id": document.get("id", ""),
            }
        )
    return records


def _normalize_candidate_records(
    candidate_records: Sequence[Dict[str, Any]] | None,
) -> List[Dict[str, Any]]:
    records = []
    for record in candidate_records or []:
        if isinstance(record, dict):
            records.append(dict(record))
    return records


def _candidate_text(record: Dict[str, Any]) -> str:
    for key in ["candidate_text", "display_candidate_name", "candidate_name", "name", "title", "text"]:
        text = _clean_text(record.get(key, ""))
        if text:
            return text
    return ""


def _candidate_evidence(record: Dict[str, Any]) -> str:
    for key in ["evidence_span", "evidence", "snippet", "context", "reason"]:
        text = _clean_text(record.get(key, ""))
        if text:
            return text
    return ""


def _matched_terms(text: str, terms: Iterable[str]) -> List[str]:
    haystack = _clean_text(text).lower()
    if not haystack:
        return []
    matches = []
    for term in terms:
        needle = _clean_text(term).lower()
        if needle and needle in haystack:
            matches.append(term)
    return _dedupe_texts(matches)


def _is_shell_candidate(
    candidate_text: str,
    specific_matches: Sequence[str],
    shell_matches: Sequence[str],
) -> bool:
    if specific_matches or not shell_matches:
        return False
    residue = _compact(candidate_text.lower())
    for term in sorted(shell_matches, key=len, reverse=True):
        residue = residue.replace(_compact(term.lower()), "")
    return len(residue) <= 2


def _recommend_pack_changes(
    *,
    candidate_count: int,
    specific_candidate_count: int,
    shell_candidate_ratio: float,
    off_domain_terms: Sequence[str],
) -> List[str]:
    recommendations = []
    if candidate_count == 0:
        recommendations.append("补充小样本候选或放宽候选成形入口以便 dry-run 产生可评估候选。")
    if specific_candidate_count == 0 and candidate_count > 0:
        recommendations.append("补充 technical_object_types、mechanism_types 或降低候选成形的槽位缺失。")
    if shell_candidate_ratio > 0:
        recommendations.append("将高频壳词加入 invalid_candidate_patterns，或提高 min_non_shell_slots。")
    if off_domain_terms:
        unique_terms = list({t for t in off_domain_terms})
        off_domain_count = len(off_domain_terms)
        preview = ", ".join(unique_terms[:6])
        recommendations.append(
            f"考虑扩展 exclude_terms、off_domain_anchor_terms 和 evidence_rejection_patterns；"
            f"当前调播到 {off_domain_count} 个候选的 off-domain 泄漏词示例：{preview}。"
        )
    return recommendations


def _list_values(value: Any) -> List[str]:
    if isinstance(value, list):
        return [_clean_text(item) for item in value if _clean_text(item)]
    if isinstance(value, str) and value.strip():
        return [_clean_text(value)]
    return []


def _dedupe_texts(values: Iterable[Any]) -> List[str]:
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


def _positive_int(value: Any, *, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _compact(value: str) -> str:
    return re.sub(r"[\s_\-，。,.、:：;；()（）\\[\\]{}]+", "", value or "")
