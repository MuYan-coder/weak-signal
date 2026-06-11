import unittest

import pandas as pd

from src.scoring.scorer import _is_scope_shell_constraint, score_all_candidates
from tests.test_space_manufacturing_domain_policy import _space_context


class ScoringScoreSuppressionTest(unittest.TestCase):
    def _space_candidate_with_legacy_shell_token(self):
        return {
            "id": "space-valid-score",
            "display_candidate_name": "太空3D打印冷焊工艺",
            "tech_name": "太空3D打印冷焊工艺",
            "raw_phrase": "太空3D打印冷焊工艺",
            "raw_candidate_text": "太空3D打印冷焊工艺",
            "candidate_stage": "formed_candidate_strong",
            "topic_granularity": "",
            "display_tier": "weak_signal",
            "is_scope_internal_candidate": True,
            "has_mechanism_core": True,
            "has_non_scope_constraint": True,
            "mechanism_core": "冷焊工艺",
            "object_modifier_tokens": ["太空3D打印"],
            "task_constraint_tokens": ["control"],
            "data_modifier_tokens": [],
            "method_modifier_tokens": [],
            "source_count": 2,
            "cluster_evidence_count": 2,
            "org_count": 2,
            "total_mentions": 2,
            "mention_dates": [],
            "evidence_items": [
                {
                    "title": "太空3D打印冷焊工艺早期验证",
                    "snippet": "团队验证太空3D打印冷焊工艺支持在轨结构成形。",
                    "raw_candidate_text": "太空3D打印冷焊工艺",
                }
            ],
        }

    def test_space_context_does_not_treat_humanoid_shell_token_as_scope_shell(self):
        self.assertTrue(_is_scope_shell_constraint("control", "task_constraint_tokens"))
        self.assertFalse(
            _is_scope_shell_constraint(
                "control",
                "task_constraint_tokens",
                domain_context=_space_context(),
            )
        )

    def test_space_candidate_not_reclassified_by_humanoid_scope_shell_terms(self):
        scored = score_all_candidates(
            pd.DataFrame([self._space_candidate_with_legacy_shell_token()]),
            domain_context=_space_context(),
        )

        self.assertFalse(bool(scored.loc[0, "scope_shell_heavy"]))
        self.assertEqual(scored.loc[0, "scope_shell_reason"], "")
        self.assertEqual(scored.loc[0, "topic_granularity"], "fine_grained_topic")
        self.assertEqual(scored.loc[0, "display_tier"], "weak_signal")
        self.assertEqual(scored.loc[0, "score_applicability"], "applicable")
        self.assertGreater(float(scored.loc[0, "weak_signal_raw_score"]), 0.0)


if __name__ == "__main__":
    unittest.main()
