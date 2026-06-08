import unittest

from src.domain import DomainPack
from src.domain.domain_pack_frontend_review import build_domain_pack_review_state
from src.domain.domain_pack_source_counts import recommend_source_counts


def _pack(weights=None, count_policy=None):
    return DomainPack.from_dict(
        {
            "schema_version": "domain_pack_v1",
            "pack_id": "battery_materials",
            "pack_name": "Battery Materials",
            "pack_version": "generated.v1",
            "source": {
                "mode": "llm_generated",
                "based_on_user_input": {
                    "field_id": "battery_materials",
                    "field_name": "电池材料",
                    "keywords": ["电池材料"],
                    "synonyms": [],
                    "exclude_terms": [],
                },
            },
            "domain_identity": {
                "field_id": "battery_materials",
                "field_name": "电池材料",
            },
            "search_strategy": {
                "core_keywords": ["电池材料"],
                "synonyms": [],
                "english_terms": ["battery materials"],
                "exclude_terms": [],
                "source_type_weights": weights
                or {
                    "paper": 2.0,
                    "patent": 1.0,
                    "news": 1.0,
                    "report": 0.0,
                    "policy": 0.0,
                },
                "count_policy": count_policy or {"mode": "recommend_only"},
            },
            "candidate_formation": {
                "technical_object_types": ["正极材料"],
                "mechanism_types": ["界面钝化"],
                "generic_terms": ["技术", "系统"],
                "shell_terms": ["技术", "方法"],
                "valid_candidate_patterns": [],
                "invalid_candidate_patterns": [
                    {
                        "pattern_id": "shell_only_rejection",
                        "reject_terms": ["技术", "方法"],
                        "max_specific_slot_count": 0,
                    }
                ],
            },
        }
    )


class DomainPackSourceCountsTest(unittest.TestCase):
    def test_recommends_counts_from_domain_pack_weights_and_available_limits(self):
        result = recommend_source_counts(
            _pack(),
            available_counts={
                "paper": 100,
                "patent": 100,
                "news": 5,
                "report": 100,
                "policy": 100,
            },
            total_sample_size=40,
        )

        self.assertEqual(
            result.final_counts,
            {"paper": 23, "patent": 12, "news": 5, "report": 0, "policy": 0},
        )
        self.assertEqual(result.rows_by_source["news"].limit_reason, "available_cap")
        self.assertEqual(result.rows_by_source["report"].limit_reason, "zero_weight")

    def test_count_policy_min_max_are_applied_outside_web_app(self):
        result = recommend_source_counts(
            _pack(
                weights={"paper": 1.0, "patent": 1.0, "news": 1.0},
                count_policy={
                    "mode": "recommend_only",
                    "min_per_source": {"paper": 12},
                    "max_per_source": {"patent": 5},
                },
            ),
            available_counts={"paper": 100, "patent": 100, "news": 100},
            total_sample_size=30,
        )

        self.assertEqual(result.final_counts["paper"], 12)
        self.assertEqual(result.final_counts["patent"], 5)
        self.assertEqual(result.final_counts["news"], 13)
        self.assertEqual(result.rows_by_source["paper"].limit_reason, "policy_min")
        self.assertEqual(result.rows_by_source["patent"].limit_reason, "policy_max")

    def test_user_explicit_counts_take_priority_over_recommendations(self):
        result = recommend_source_counts(
            _pack(),
            available_counts={"paper": 100, "patent": 100, "news": 100},
            total_sample_size=40,
            explicit_counts={"paper": 3, "patent": 7, "news": 0},
        )

        self.assertEqual(result.final_counts, {"paper": 3, "patent": 7, "news": 0})
        self.assertEqual(result.rows_by_source["paper"].limit_reason, "user_explicit")
        self.assertEqual(result.total_count, 10)

    def test_frontend_review_state_requires_valid_pack_confirmation_and_nonzero_counts(self):
        pending = build_domain_pack_review_state(
            _pack(),
            available_counts={"paper": 100, "patent": 100, "news": 100},
            explicit_counts={"paper": 4, "patent": 0, "news": 0},
            confirmed=False,
        )
        confirmed = build_domain_pack_review_state(
            _pack(),
            available_counts={"paper": 100, "patent": 100, "news": 100},
            explicit_counts={"paper": 4, "patent": 0, "news": 0},
            confirmed=True,
        )

        self.assertFalse(pending.can_start_analysis)
        self.assertTrue(confirmed.can_start_analysis)
        self.assertEqual(confirmed.final_counts, {"paper": 4, "patent": 0, "news": 0})
        self.assertTrue(confirmed.validation_report.is_valid)


if __name__ == "__main__":
    unittest.main()
