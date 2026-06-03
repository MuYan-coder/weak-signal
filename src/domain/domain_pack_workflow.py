from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Sequence

from .domain_pack_dry_run import DomainPackDryRunReport, run_domain_pack_dry_run
from .domain_pack_generator import DEFAULT_MEMORY_DIR, DomainPackGenerationRequest, DomainPackGenerator
from .domain_pack_validator import DomainPackValidationReport, validate_domain_pack
from .models import DomainPack


@dataclass
class DomainPackWorkflowResult:
    status: str
    domain_pack: DomainPack
    validation_report: DomainPackValidationReport
    dry_run_report: DomainPackDryRunReport | None = None
    basic_validation_report: DomainPackValidationReport | None = None
    manual_review_required: bool = True
    review_hints: list[str] = field(default_factory=list)


class DomainPackWorkflow:
    def __init__(
        self,
        *,
        generator: Any | None = None,
        memory_dir: Path | str = DEFAULT_MEMORY_DIR,
    ):
        self.memory_dir = Path(memory_dir)
        self.generator = generator or DomainPackGenerator(memory_dir=self.memory_dir)

    def prepare_domain_pack(
        self,
        request: DomainPackGenerationRequest,
        *,
        sample_documents: Sequence[Dict[str, Any]] | None = None,
        candidate_records: Sequence[Dict[str, Any]] | None = None,
        human_review_approved: bool = False,
        refresh_cache: bool = False,
        manual_override: bool = False,
    ) -> DomainPackWorkflowResult:
        pack = self.generator.generate(request, refresh_cache=refresh_cache)

        basic_validation = validate_domain_pack(
            pack,
            dry_run_report=None,
            require_dry_run=False,
        )
        if not basic_validation.is_valid:
            return DomainPackWorkflowResult(
                status="blocked",
                domain_pack=pack,
                validation_report=basic_validation,
                basic_validation_report=basic_validation,
                dry_run_report=None,
                manual_review_required=True,
                review_hints=_review_hints(pack, basic_validation, None),
            )

        dry_run_report = run_domain_pack_dry_run(
            pack,
            sample_documents=sample_documents or request.sample_documents,
            candidate_records=candidate_records,
            memory_dir=self.memory_dir,
            manual_override=manual_override,
        )
        final_validation = validate_domain_pack(
            pack,
            dry_run_report=dry_run_report,
            require_dry_run=True,
        )

        if not final_validation.is_valid:
            status = "blocked"
            manual_review_required = True
        elif not human_review_approved:
            status = "pending_review"
            manual_review_required = True
        else:
            status = "ready"
            manual_review_required = False

        return DomainPackWorkflowResult(
            status=status,
            domain_pack=pack,
            validation_report=final_validation,
            basic_validation_report=basic_validation,
            dry_run_report=dry_run_report,
            manual_review_required=manual_review_required,
            review_hints=_review_hints(pack, final_validation, dry_run_report),
        )


def _review_hints(
    pack: DomainPack,
    validation_report: DomainPackValidationReport,
    dry_run_report: DomainPackDryRunReport | None,
) -> list[str]:
    reporting = pack.reporting if isinstance(pack.reporting, dict) else {}
    hints = []
    for value in reporting.get("review_hints", []) or []:
        if isinstance(value, str) and value.strip():
            hints.append(value.strip())
    hints.extend(validation_report.warnings)
    if dry_run_report is not None:
        hints.extend(dry_run_report.recommended_pack_changes)
    return _dedupe(hints)


def _dedupe(values: Sequence[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        marker = value.lower()
        if marker in seen:
            continue
        seen.add(marker)
        result.append(value)
    return result
