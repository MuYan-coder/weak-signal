from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Sequence

from .domain_pack_dry_run import DomainPackDryRunReport
from .models import DOMAIN_PACK_SCHEMA_VERSION, SECTION_DEFAULTS, DomainPack


BROAD_KEYWORDS = {
    "技术",
    "系统",
    "方法",
    "应用",
    "方案",
    "研究",
    "产品",
    "服务",
    "technology",
    "system",
    "method",
    "application",
    "solution",
    "research",
    "product",
    "service",
}


@dataclass
class DomainPackValidationReport:
    domain_pack_id: str
    domain_pack_hash: str
    is_valid: bool
    gate_status: str
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    checks: Dict[str, Any] = field(default_factory=dict)
    dry_run_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def validate_domain_pack(
    pack: DomainPack,
    *,
    dry_run_report: DomainPackDryRunReport | Dict[str, Any] | None = None,
    require_dry_run: bool = True,
    shell_candidate_ratio_threshold: float = 0.6,
) -> DomainPackValidationReport:
    errors: List[str] = []
    warnings: List[str] = []
    checks: Dict[str, Any] = {}

    _check_schema(pack, errors, checks)
    _check_required_sections(pack, errors, checks)
    _check_identity(pack, errors, checks)
    _check_search_strategy(pack, errors, warnings, checks)
    _check_candidate_formation(pack, errors, warnings, checks)
    _check_rule_conflicts(pack, errors, warnings, checks)
    _check_canonicalization(pack, warnings, checks)
    _check_scoring_rules(pack, errors, warnings, checks)
    _check_object_overlap(pack, errors, warnings, checks)

    report = _coerce_dry_run_report(dry_run_report)
    dry_run_metrics = _check_dry_run_gate(
        pack,
        report,
        errors,
        warnings,
        require_dry_run=require_dry_run,
        shell_candidate_ratio_threshold=shell_candidate_ratio_threshold,
    )
    if dry_run_metrics:
        checks["dry_run"] = "passed" if not any("dry-run" in item for item in errors) else "failed"

    is_valid = not errors
    return DomainPackValidationReport(
        domain_pack_id=pack.pack_id,
        domain_pack_hash=pack.domain_pack_hash,
        is_valid=is_valid,
        gate_status="passed" if is_valid else "blocked",
        errors=errors,
        warnings=warnings,
        checks=checks,
        dry_run_metrics=dry_run_metrics,
    )


def _check_schema(pack: DomainPack, errors: List[str], checks: Dict[str, Any]) -> None:
    if pack.schema_version != DOMAIN_PACK_SCHEMA_VERSION:
        errors.append(
            f"schema_version must be {DOMAIN_PACK_SCHEMA_VERSION}, got {pack.schema_version or '<empty>'}"
        )
        checks["schema_version"] = "failed"
        return
    checks["schema_version"] = "passed"


def _check_required_sections(pack: DomainPack, errors: List[str], checks: Dict[str, Any]) -> None:
    missing = []
    for section in SECTION_DEFAULTS:
        if not isinstance(getattr(pack, section, None), dict):
            missing.append(section)
    if missing:
        errors.append(f"required Domain Pack sections must be mappings: {', '.join(missing)}")
        checks["required_sections"] = "failed"
        return
    checks["required_sections"] = "passed"


def _check_identity(pack: DomainPack, errors: List[str], checks: Dict[str, Any]) -> None:
    field_id = _clean_text(pack.domain_identity.get("field_id") or pack.pack_id)
    field_name = _clean_text(pack.domain_identity.get("field_name") or pack.pack_name)
    if not pack.pack_id or pack.pack_id == "neutral":
        errors.append("pack_id must identify a non-neutral generated domain before entering main flow")
    if not field_id:
        errors.append("domain_identity.field_id is required")
    if not field_name:
        errors.append("domain_identity.field_name is required")
    checks["identity"] = "passed" if field_id and field_name and pack.pack_id != "neutral" else "failed"


def _check_search_strategy(
    pack: DomainPack,
    errors: List[str],
    warnings: List[str],
    checks: Dict[str, Any],
) -> None:
    core_keywords = _list_values(pack.search_strategy.get("core_keywords"))
    exclude_terms = _list_values(pack.search_strategy.get("exclude_terms"))
    synonyms = _list_values(pack.search_strategy.get("synonyms"))
    english_terms = _list_values(pack.search_strategy.get("english_terms"))

    if not core_keywords:
        errors.append("search_strategy.core_keywords must contain at least one domain-specific keyword")
    if not _is_list(pack.search_strategy.get("core_keywords")):
        errors.append("search_strategy.core_keywords must be a list")
    for field_name, values in {
        "synonyms": synonyms,
        "english_terms": english_terms,
        "exclude_terms": exclude_terms,
    }.items():
        if not _is_list(pack.search_strategy.get(field_name)):
            errors.append(f"search_strategy.{field_name} must be a list")
        elif not values and field_name != "english_terms":
            warnings.append(f"search_strategy.{field_name} is empty")

    if core_keywords and all(_is_broad_keyword(item) for item in core_keywords):
        errors.append("search_strategy.core_keywords are too broad; include concrete domain objects or mechanisms")
    checks["search_strategy"] = "passed" if not any("search_strategy" in item for item in errors) else "failed"


def _check_candidate_formation(
    pack: DomainPack,
    errors: List[str],
    warnings: List[str],
    checks: Dict[str, Any],
) -> None:
    candidate_formation = pack.candidate_formation
    list_fields = [
        "technical_object_types",
        "mechanism_types",
        "task_or_performance_types",
        "data_or_method_types",
        "scene_or_application_types",
        "generic_terms",
        "shell_terms",
        "valid_candidate_patterns",
        "invalid_candidate_patterns",
    ]
    for field_name in list_fields:
        if not _is_list(candidate_formation.get(field_name)):
            errors.append(f"candidate_formation.{field_name} must be a list")

    generic_terms = _list_values(candidate_formation.get("generic_terms"))
    shell_terms = _list_values(candidate_formation.get("shell_terms"))
    specific_terms = _list_values(candidate_formation.get("technical_object_types")) + _list_values(
        candidate_formation.get("mechanism_types")
    )
    if len(generic_terms) < 2:
        errors.append("candidate_formation.generic_terms must include at least two shell-like examples")
    if len(shell_terms) < 2:
        errors.append("candidate_formation.shell_terms must include at least two shell-like examples")
    if not specific_terms:
        errors.append("candidate_formation must include technical object or mechanism terms")

    invalid_patterns = _dict_items(candidate_formation.get("invalid_candidate_patterns"))
    if not invalid_patterns:
        errors.append("candidate_formation.invalid_candidate_patterns must contain at least one invalid pattern")

    task_or_perf = _list_values(candidate_formation.get("task_or_performance_types"))
    task_set = {t.lower() for t in task_or_perf}
    if task_set and task_set.issubset({"性能", "应用", "效果", "performance", "application", "effect"}):
        errors.append("candidate_formation.task_or_performance_types contains only generic/broad shell terms")

    data_or_method = _list_values(candidate_formation.get("data_or_method_types"))
    method_set = {t.lower() for t in data_or_method}
    if method_set and method_set.issubset({"数据", "方法", "技术", "data", "method", "technology"}):
        errors.append("candidate_formation.data_or_method_types contains only generic/broad shell terms")

    scene_or_app = _list_values(candidate_formation.get("scene_or_application_types"))
    scene_set = {t.lower() for t in scene_or_app}
    if scene_set and scene_set.issubset({"场景", "应用", "系统", "scene", "application", "system"}):
        errors.append("candidate_formation.scene_or_application_types contains only generic/broad shell terms")

    specificity_rule = candidate_formation.get("minimum_specificity_rule", {})
    if not isinstance(specificity_rule, dict):
        errors.append("candidate_formation.minimum_specificity_rule must be a mapping")
    checks["candidate_formation"] = "passed" if not any("candidate_formation" in item for item in errors) else "failed"


def _check_rule_conflicts(
    pack: DomainPack,
    errors: List[str],
    warnings: List[str],
    checks: Dict[str, Any],
) -> None:
    candidate_formation = pack.candidate_formation
    valid_patterns = _dict_items(candidate_formation.get("valid_candidate_patterns"))
    invalid_patterns = _dict_items(candidate_formation.get("invalid_candidate_patterns"))
    valid_ids = {_clean_text(item.get("pattern_id")) for item in valid_patterns if _clean_text(item.get("pattern_id"))}
    invalid_ids = {_clean_text(item.get("pattern_id")) for item in invalid_patterns if _clean_text(item.get("pattern_id"))}
    overlap_ids = sorted(valid_ids & invalid_ids)
    if overlap_ids:
        errors.append(f"valid and invalid candidate pattern_id conflict: {', '.join(overlap_ids)}")

    specific_terms = set(
        term.lower()
        for term in _list_values(candidate_formation.get("technical_object_types"))
        + _list_values(candidate_formation.get("mechanism_types"))
    )
    rejected_terms = set()
    for pattern in invalid_patterns:
        rejected_terms.update(term.lower() for term in _list_values(pattern.get("reject_terms")))
    risky_overlap = sorted(specific_terms & rejected_terms)
    if risky_overlap:
        warnings.append(f"invalid_candidate_patterns reject specific terms: {', '.join(risky_overlap)}")
    checks["rule_conflicts"] = "passed" if not overlap_ids else "failed"


def _check_canonicalization(
    pack: DomainPack,
    warnings: List[str],
    checks: Dict[str, Any],
) -> None:
    object_families = _dict_items(pack.canonicalization.get("object_families"))
    if not object_families:
        warnings.append("generated domain pack has empty object_families; downstream will use neutral canonicalization")
    for family in object_families:
        confidence = _float_value(family.get("merge_confidence"), default=0.0)
        aliases = _list_values(family.get("aliases"))
        evidence_required = bool(family.get("evidence_required", False))
        family_id = _clean_text(family.get("family_id") or family.get("name") or "<unknown>")
        if confidence >= 0.7 and not aliases and not evidence_required:
            warnings.append(f"object_family {family_id} needs aliases or evidence_required for high-confidence merge")
    checks["canonicalization"] = "passed"

def _check_scoring_rules(
    pack: DomainPack,
    errors: List[str],
    warnings: List[str],
    checks: Dict[str, Any],
) -> None:
    weak_signal_rules = getattr(pack, "weak_signal_rules", {})
    if not isinstance(weak_signal_rules, dict):
        return
    scoring_adjustments = _dict_items(weak_signal_rules.get("scoring_adjustments"))

    total_positive_delta = 0.0
    for idx, rule in enumerate(scoring_adjustments):
        match_terms = _list_values(rule.get("match_terms"))
        req_evidence = _list_values(rule.get("required_evidence_fields"))
        rule_id = _clean_text(rule.get("rule_id", ""))
        score_delta = _float_value(rule.get("score_delta"), default=0.0)

        # 1. 拦截无条件加分规则
        if not match_terms and not req_evidence and score_delta > 0:
            errors.append(f"scoring rule '{rule_id}' is an unconditional positive rule (no match_terms or required_evidence_fields)")

        # 2. 拦截不规范的 Rule ID
        if rule_id.startswith("rule_00"):
            if match_terms:
                new_id = f"rule_{'_'.join(match_terms[:2])}"
                rule["rule_id"] = new_id
                warnings.append(f"auto-renamed meaningless rule_id '{rule_id}' to '{new_id}'")
            else:
                warnings.append(f"scoring rule has meaningless rule_id '{rule_id}'")

        # 3. 统计总加分
        if score_delta > 0:
            total_positive_delta += score_delta

    if total_positive_delta > 2.0:
        warnings.append(f"total positive score_delta ({total_positive_delta}) exceeds recommended maximum of 2.0")

    checks["scoring_rules"] = "passed" if not any("scoring rule" in item for item in errors) else "failed"

def _check_object_overlap(
    pack: DomainPack,
    errors: List[str],
    warnings: List[str],
    checks: Dict[str, Any],
) -> None:
    candidate_formation = getattr(pack, "candidate_formation", {})
    obs_scopes = getattr(pack, "observation_scopes", {})
    if not isinstance(candidate_formation, dict) or not isinstance(obs_scopes, dict):
        return

    tech_objects = set(_list_values(candidate_formation.get("technical_object_types")))
    scope_echo_terms = set(_list_values(obs_scopes.get("scope_echo_terms")))
    main_scope = _clean_text(obs_scopes.get("main_scope"))

    if main_scope:
        scope_echo_terms.add(main_scope)

    if not tech_objects:
        return

    overlap = tech_objects & scope_echo_terms
    normalized_scope_terms = {term.lower() for term in scope_echo_terms if term}
    normalized_main_scope = main_scope.lower() if main_scope else ""
    exact_main_scope_hits = [obj for obj in tech_objects if normalized_main_scope and obj.lower() == normalized_main_scope]
    exact_scope_echo_hits = [
        obj for obj in tech_objects
        if obj.lower() in normalized_scope_terms and obj not in exact_main_scope_hits
    ]
    if exact_main_scope_hits:
        errors.append(
            "technical_object_types contains exact main_scope term(s): "
            + ", ".join(sorted(exact_main_scope_hits))
        )
    if exact_scope_echo_hits:
        errors.append(
            "technical_object_types contains exact scope_echo_terms: "
            + ", ".join(sorted(exact_scope_echo_hits))
        )
    overlap_ratio = len(overlap) / len(tech_objects)

    if overlap_ratio > 0.5:
        errors.append(f"technical_object_types heavily overlaps ({overlap_ratio:.0%}) with scope_echo_terms/main_scope")
    elif overlap_ratio > 0:
        warnings.append(f"technical_object_types has significant overlap ({overlap_ratio:.0%}) with scope_echo_terms")

    # Check if technical objects are only broad names
    broad_objs = {"系统", "技术", "方法", "方案", "平台", "架构", "应用"}
    pure_broad = all(obj.lower() in broad_objs or obj in scope_echo_terms for obj in tech_objects)
    if pure_broad and tech_objects:
        errors.append("technical_object_types contains only broad domain names or shell terms")

    checks["object_overlap"] = "passed" if not any("overlap" in item for item in errors) else "failed"

def _check_dry_run_gate(
    pack: DomainPack,
    report: DomainPackDryRunReport | None,
    errors: List[str],
    warnings: List[str],
    *,
    require_dry_run: bool,
    shell_candidate_ratio_threshold: float,
) -> Dict[str, Any]:
    if report is None:
        if require_dry_run:
            errors.append("dry-run report is required before Domain Pack enters main flow")
        return {}

    metrics = {
        "sample_doc_count": report.sample_doc_count,
        "candidate_count": report.candidate_count,
        "specific_candidate_count": report.specific_candidate_count,
        "shell_candidate_ratio": report.shell_candidate_ratio,
        "off_domain_leakage_terms": list(report.off_domain_leakage_terms),
        "manual_override": report.manual_override,
    }
    if report.domain_pack_hash != pack.domain_pack_hash:
        errors.append("dry-run report domain_pack_hash does not match Domain Pack hash")
    if report.sample_doc_count <= 0:
        errors.append("dry-run report must include at least one sample document")
    if report.candidate_count <= 0:
        errors.append("dry-run report must include at least one candidate")
    if report.shell_candidate_ratio > shell_candidate_ratio_threshold:
        errors.append(
            "dry-run shell_candidate_ratio "
            f"{report.shell_candidate_ratio:.2f} exceeds threshold {shell_candidate_ratio_threshold:.2f}"
        )
    if report.off_domain_leakage_terms and not report.manual_override:
        errors.append(
            "dry-run found off-domain leakage terms without manual_override: "
            + ", ".join(report.off_domain_leakage_terms)
        )
    if report.candidate_count > 0 and report.specific_candidate_count == 0:
        warnings.append("dry-run did not produce specific candidates")
    if not report.top_invalid_candidate_reasons:
        warnings.append("dry-run report has no invalid candidate reason statistics")
    return metrics


def _coerce_dry_run_report(
    report: DomainPackDryRunReport | Dict[str, Any] | None,
) -> DomainPackDryRunReport | None:
    if report is None:
        return None
    if isinstance(report, DomainPackDryRunReport):
        return report
    if isinstance(report, dict):
        return DomainPackDryRunReport.from_dict(report)
    raise TypeError("dry_run_report must be DomainPackDryRunReport, dict, or None")


def _dict_items(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _list_values(value: Any) -> List[str]:
    if isinstance(value, list):
        return [_clean_text(item) for item in value if _clean_text(item)]
    if isinstance(value, str) and value.strip():
        return [_clean_text(value)]
    return []


def _is_list(value: Any) -> bool:
    return isinstance(value, list)


def _is_broad_keyword(value: str) -> bool:
    normalized = _clean_text(value).lower()
    return normalized in BROAD_KEYWORDS or len(normalized) <= 2


def _float_value(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clean_text(value: Any) -> str:
    return str(value or "").strip()
