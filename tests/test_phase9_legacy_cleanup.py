import unittest
from pathlib import Path

from src.domain.domain_pack_generator import (
    DomainPackGenerationRequest,
    build_candidate_formation_prompt,
)
from src.domain.domain_pack_legacy_audit import evaluate_stage9_legacy_cleanup


class Phase9LegacyCleanupTest(unittest.TestCase):
    def test_stage9_cleanup_audit_passes_documented_acceptance_criteria(self):
        report = evaluate_stage9_legacy_cleanup(Path(__file__).resolve().parents[1])

        self.assertTrue(report.is_complete, report.failures)
        self.assertEqual([], report.failures)
        self.assertTrue(report.documentation_checked)
        self.assertTrue(report.legacy_rules_are_preset_guarded)

    def test_candidate_prompt_uses_domain_neutral_reference_patterns(self):
        request = DomainPackGenerationRequest(
            field_id="battery_materials",
            field_name="电池材料",
            keywords=["固态电解质"],
        )
        prompt = build_candidate_formation_prompt(
            request,
            domain_model={"domain_boundary": "电池材料及其制备、性能与应用边界。"},
        )

        self.assertIn("reference_candidate_pattern_summary", prompt)
        self.assertNotIn("humanoid_preset_pattern_summary", prompt)


if __name__ == "__main__":
    unittest.main()
