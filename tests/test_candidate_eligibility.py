import unittest

import pandas as pd

from src.scoring.candidate_eligibility import (
    apply_candidate_eligibility,
    evaluate_candidate_eligibility,
)
from tests.test_space_manufacturing_domain_policy import _space_context


class CandidateEligibilityTest(unittest.TestCase):
    def _space_row(self):
        return {
            "display_candidate_name": "太空3D打印冷焊工艺",
            "raw_phrase": "太空3D打印结构件冷焊工艺",
            "raw_candidate_text": "太空3D打印结构件冷焊工艺",
            "candidate_stage": "formed_candidate",
            "topic_granularity": "fine_grained_topic",
            "display_tier": "weak_signal",
            "is_scope_internal_candidate": True,
            "has_mechanism_core": True,
            "has_non_scope_constraint": True,
            "survives_without_scope": True,
            "scope_shell_heavy": False,
            "mechanism_core": "冷焊工艺",
            "object_modifier_tokens": ["太空3D打印"],
            "task_constraint_tokens": ["在轨装配"],
            "source_count": 2,
            "cluster_evidence_count": 2,
        }

    def test_space_domain_context_supplies_technical_anchor_terms(self):
        profile = evaluate_candidate_eligibility(
            self._space_row(),
            domain_context=_space_context(),
        )

        self.assertIn(profile.candidate_eligibility, {"eligible", "candidate_monitoring"})
        self.assertNotIn(
            "domain_pack_status_empty_without_technical_envelope",
            profile.eligibility_reason_codes,
        )
        self.assertTrue(profile.has_technical_anchor)

    def test_scope_only_space_policy_phrase_is_not_score_applicable(self):
        row = {
            **self._space_row(),
            "display_candidate_name": "太空制造技术方案",
            "raw_phrase": "太空制造技术方案",
            "raw_candidate_text": "太空制造技术方案",
            "candidate_stage": "formed_candidate",
            "topic_granularity": "scope_internal_candidate",
            "display_tier": "hotspot",
            "has_mechanism_core": True,
            "has_non_scope_constraint": False,
            "survives_without_scope": False,
            "scope_shell_heavy": True,
            "generic_core_only": True,
            "mechanism_core": "方案",
            "object_modifier_tokens": ["太空制造"],
            "task_constraint_tokens": [],
        }

        profile = evaluate_candidate_eligibility(row, domain_context=_space_context())

        self.assertEqual(profile.score_applicability, "not_applicable")
        self.assertEqual(profile.candidate_eligibility, "not_eligible")
        self.assertIn("technical_envelope_missing", profile.eligibility_reason_codes)

    def test_robot_defaults_do_not_count_as_space_domain_technical_anchors(self):
        row = {
            **self._space_row(),
            "display_candidate_name": "机械臂夹爪控制工艺",
            "raw_phrase": "机械臂夹爪控制工艺",
            "raw_candidate_text": "机械臂夹爪控制工艺",
            "mechanism_core": "控制",
            "object_modifier_tokens": ["机械臂", "夹爪"],
            "task_constraint_tokens": ["抓取"],
        }

        profile = evaluate_candidate_eligibility(row, domain_context=_space_context())

        self.assertEqual(profile.score_applicability, "not_applicable")
        self.assertFalse(profile.has_technical_anchor)
        self.assertIn("missing_domain_technical_anchor", profile.eligibility_reason_codes)

    def test_apply_candidate_eligibility_writes_contract_columns(self):
        result = apply_candidate_eligibility(
            pd.DataFrame([self._space_row()]),
            domain_context=_space_context(),
        )

        self.assertEqual(result.loc[0, "score_applicability"], "applicable")
        self.assertIn(result.loc[0, "candidate_eligibility"], {"eligible", "candidate_monitoring"})
        self.assertTrue(bool(result.loc[0, "candidate_has_technical_anchor"]))


if __name__ == "__main__":
    unittest.main()
