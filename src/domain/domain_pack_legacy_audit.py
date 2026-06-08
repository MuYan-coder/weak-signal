from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


REQUIRED_STAGE9_DOCS = ("README.md", "ARCHITECTURE.md", "PROJECT_OVERVIEW.md")
REQUIRED_DOCUMENTATION_TOPICS = ("Domain Pack", "生成", "复核", "复用", "版本")


@dataclass(frozen=True)
class Stage9LegacyCleanupReport:
    documentation_checked: bool
    legacy_rules_are_preset_guarded: bool
    neutral_fallback_checked: bool
    prompt_is_domain_neutral: bool
    failures: List[str] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        return not self.failures


def evaluate_stage9_legacy_cleanup(root: Path | str | None = None) -> Stage9LegacyCleanupReport:
    """Audit the stage 9 cleanup contract without running a full analysis."""

    project_root = Path(root or Path.cwd())
    failures: List[str] = []

    documentation_checked = _check_documentation(project_root, failures)
    legacy_rules_are_preset_guarded = _check_legacy_preset_guards(project_root, failures)
    neutral_fallback_checked = _check_neutral_fallback(project_root, failures)
    prompt_is_domain_neutral = _check_generator_prompt(project_root, failures)

    return Stage9LegacyCleanupReport(
        documentation_checked=documentation_checked,
        legacy_rules_are_preset_guarded=legacy_rules_are_preset_guarded,
        neutral_fallback_checked=neutral_fallback_checked,
        prompt_is_domain_neutral=prompt_is_domain_neutral,
        failures=failures,
    )


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _check_documentation(root: Path, failures: List[str]) -> bool:
    ok = True
    combined_docs = []
    for relative in REQUIRED_STAGE9_DOCS:
        path = root / relative
        if not path.exists():
            failures.append(f"missing required documentation: {relative}")
            ok = False
            continue
        combined_docs.append(_read_text(path))

    combined_text = "\n".join(combined_docs)
    for topic in REQUIRED_DOCUMENTATION_TOPICS:
        if topic not in combined_text:
            failures.append(f"documentation missing Domain Pack topic: {topic}")
            ok = False
    return ok


def _check_legacy_preset_guards(root: Path, failures: List[str]) -> bool:
    checks = {
        "src/extraction/tech_lexicon.py": [
            'pack_id == "humanoid_robot"',
            "use_legacy_robot_rules=True",
        ],
        "src/extraction/candidate_former.py": [
            "use_legacy_robot_rules",
        ],
        "src/scoring/topic_refiner.py": [
            'pack_id == "humanoid_robot"',
        ],
        "src/validation/object_family_canonicalizer.py": [
            "object_family_enabled",
        ],
    }

    ok = True
    for relative, markers in checks.items():
        text = _read_text(root / relative)
        for marker in markers:
            if marker not in text:
                failures.append(f"legacy rule guard missing in {relative}: {marker}")
                ok = False
    return ok


def _check_neutral_fallback(root: Path, failures: List[str]) -> bool:
    loader_text = _read_text(root / "src/domain/domain_pack_loader.py")
    pipeline_text = _read_text(root / "src/core/pipeline.py")
    ok = True

    if 'NEUTRAL_PACK_ID = "neutral"' not in loader_text:
        failures.append("domain pack loader must define neutral pack id")
        ok = False
    if "pack_ref: Union[str, Path, None] = NEUTRAL_PACK_ID" not in loader_text:
        failures.append("domain pack loader must fall back to neutral pack")
        ok = False
    if '"humanoid_robot"' in loader_text and 'preset_id = "neutral"' not in loader_text:
        failures.append("domain pack loader references humanoid_robot without neutral default guard")
        ok = False
    if "使用 neutral pack" not in pipeline_text:
        failures.append("pipeline recovery path must document neutral fallback")
        ok = False
    return ok


def _check_generator_prompt(root: Path, failures: List[str]) -> bool:
    generator_text = _read_text(root / "src/domain/domain_pack_generator.py")
    ok = True
    if "humanoid_preset_pattern_summary" in generator_text:
        failures.append("generator prompt still exposes humanoid preset pattern naming")
        ok = False
    if "reference_candidate_pattern_summary" not in generator_text:
        failures.append("generator prompt must expose domain-neutral reference candidate patterns")
        ok = False
    return ok
