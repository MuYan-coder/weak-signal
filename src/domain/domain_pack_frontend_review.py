from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .domain_pack_dry_run import DomainPackDryRunReport
from .domain_pack_source_counts import SourceCountRecommendation, recommend_source_counts
from .domain_pack_validator import DomainPackValidationReport, validate_domain_pack
from .models import DomainPack


@dataclass(frozen=True)
class DomainPackReviewState:
    domain_pack_id: str
    domain_pack_version: str
    domain_pack_hash: str
    validation_report: DomainPackValidationReport
    recommendation: SourceCountRecommendation
    confirmed: bool

    @property
    def final_counts(self) -> dict[str, int]:
        return self.recommendation.final_counts

    @property
    def total_count(self) -> int:
        return self.recommendation.total_count

    @property
    def can_start_analysis(self) -> bool:
        return bool(self.confirmed and self.validation_report.is_valid and self.total_count > 0)


def build_domain_pack_review_state(
    pack: DomainPack,
    *,
    available_counts: Mapping[str, Any] | None = None,
    explicit_counts: Mapping[str, Any] | None = None,
    confirmed: bool = False,
    total_sample_size: int | None = None,
    source_types: Iterable[str] | None = None,
    dry_run_report: DomainPackDryRunReport | dict[str, Any] | None = None,
    require_dry_run: bool = False,
) -> DomainPackReviewState:
    validation_report = validate_domain_pack(
        pack,
        dry_run_report=dry_run_report,
        require_dry_run=require_dry_run,
    )
    recommendation = recommend_source_counts(
        pack,
        available_counts=available_counts or {},
        total_sample_size=total_sample_size,
        explicit_counts=explicit_counts,
        source_types=source_types,
    )
    return DomainPackReviewState(
        domain_pack_id=pack.pack_id,
        domain_pack_version=pack.domain_pack_version,
        domain_pack_hash=pack.domain_pack_hash,
        validation_report=validation_report,
        recommendation=recommendation,
        confirmed=bool(confirmed),
    )
