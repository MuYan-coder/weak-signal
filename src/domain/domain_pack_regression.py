from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple


PHASE8_REQUIRED_DIMENSIONS: Tuple[str, ...] = (
    "humanoid_preset_regression",
    "fictional_domain_robot_leakage",
    "new_domain_generation_validation",
    "domain_pack_dry_run_calibration",
    "pipeline_domain_metadata",
    "resume_and_report_domain_recovery",
    "domain_cache_isolation",
    "source_type_weights_to_counts",
    "domain_pack_hash_stability",
    "candidate_specificity",
    "weak_signal_explanation_completeness",
    "object_family_toggle",
    "frontend_domain_pack_review_flow",
)


@dataclass(frozen=True)
class Phase8RegressionDimension:
    dimension_id: str
    acceptance_metric: str
    test_modules: Tuple[str, ...]


@dataclass(frozen=True)
class Phase8RegressionReport:
    dimensions: Tuple[Phase8RegressionDimension, ...]
    missing_dimensions: Tuple[str, ...]

    @property
    def covered_dimensions(self) -> Tuple[str, ...]:
        return tuple(item.dimension_id for item in self.dimensions)

    @property
    def is_complete(self) -> bool:
        return not self.missing_dimensions


PHASE8_REGRESSION_MATRIX: Tuple[Phase8RegressionDimension, ...] = (
    Phase8RegressionDimension(
        "humanoid_preset_regression",
        "humanoid_robot preset loads legacy-sensitive rules and enables object family registry.",
        ("tests.test_domain_pack_loader", "tests.test_no_default_robot_leakage"),
    ),
    Phase8RegressionDimension(
        "fictional_domain_robot_leakage",
        "Non-robot domains do not inject humanoid, robot, embodied, or world-model defaults without evidence.",
        ("tests.test_no_default_robot_leakage",),
    ),
    Phase8RegressionDimension(
        "new_domain_generation_validation",
        "Generated packs are structurally complete, cached, saved outside source config, and validated before use.",
        ("tests.test_domain_pack_generator", "tests.test_domain_pack_stage3_quality"),
    ),
    Phase8RegressionDimension(
        "domain_pack_dry_run_calibration",
        "Dry-run records candidate specificity, shell ratio, off-domain leakage, invalid reasons, and review gate status.",
        ("tests.test_domain_pack_stage3_quality",),
    ),
    Phase8RegressionDimension(
        "pipeline_domain_metadata",
        "Pipeline outputs attach domain_pack_id, domain_pack_version, domain_pack_hash, and runtime mode.",
        ("tests.test_domain_context_pipeline",),
    ),
    Phase8RegressionDimension(
        "resume_and_report_domain_recovery",
        "run_from_events and regenerate_report_from_result recover Domain Pack snapshots or metadata.",
        ("tests.test_domain_context_pipeline",),
    ),
    Phase8RegressionDimension(
        "domain_cache_isolation",
        "Event and document-level cache paths include domain_pack_hash for different packs.",
        ("tests.test_domain_context_pipeline",),
    ),
    Phase8RegressionDimension(
        "source_type_weights_to_counts",
        "source_type_weights and count_policy resolve through the unified SourceQuery.counts recommendation function.",
        ("tests.test_domain_pack_source_counts",),
    ),
    Phase8RegressionDimension(
        "domain_pack_hash_stability",
        "domain_pack_hash ignores non-semantic runtime metadata while preserving semantic differences.",
        ("tests.test_domain_pack_loader",),
    ),
    Phase8RegressionDimension(
        "candidate_specificity",
        "Candidate formation records Domain Pack rule hits and rejects shell-only or insufficiently specific candidates.",
        ("tests.test_no_default_robot_leakage", "tests.test_domain_pack_stage3_quality"),
    ),
    Phase8RegressionDimension(
        "weak_signal_explanation_completeness",
        "Weak-signal scoring explanation includes Domain Pack rule contribution and reasons.",
        ("tests.test_no_default_robot_leakage",),
    ),
    Phase8RegressionDimension(
        "object_family_toggle",
        "Object-family canonicalization is disabled by default and enabled only by explicit Domain Pack settings.",
        ("tests.test_no_default_robot_leakage",),
    ),
    Phase8RegressionDimension(
        "frontend_domain_pack_review_flow",
        "Frontend review state requires valid pack, explicit user confirmation, and confirmed counts before analysis.",
        ("tests.test_domain_pack_source_counts",),
    ),
)


def evaluate_phase8_regression_matrix(
    required_dimensions: Iterable[str] = PHASE8_REQUIRED_DIMENSIONS,
) -> Phase8RegressionReport:
    required = tuple(required_dimensions)
    matrix_by_id = {item.dimension_id: item for item in PHASE8_REGRESSION_MATRIX}
    dimensions = tuple(matrix_by_id[item] for item in required if item in matrix_by_id)
    missing = tuple(item for item in required if item not in matrix_by_id)
    return Phase8RegressionReport(dimensions=dimensions, missing_dimensions=missing)
