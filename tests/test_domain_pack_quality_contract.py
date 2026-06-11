import unittest

from src.domain import DomainPack
from src.domain.domain_pack_validator import validate_domain_pack


class DomainPackQualityContractTest(unittest.TestCase):
    def test_space_manufacturing_pack_has_specific_candidate_terms(self):
        payload = {
            "schema_version": "domain_pack_v1",
            "pack_id": "space_manufacturing",
            "pack_name": "太空制造",
            "domain_pack_version": "2026-06-11-test",
            "domain_identity": {
                "field_id": "space_manufacturing",
                "field_name": "太空制造",
                "domain_boundary": "太空制造",
                "core_keywords": ["太空制造", "太空3D打印"],
                "english_terms": ["space manufacturing"],
                "excluded_topics": ["人形机器人", "具身智能"],
            },
            "search_strategy": {
                "query_templates": [],
                "core_keywords": ["太空3D打印", "微重力制造"],
                "synonyms": ["space manufacturing"],
                "must_have_terms": [],
                "nice_to_have_terms": [],
                "exclude_terms": [],
                "english_terms": ["space manufacturing"],
                "source_weights": {},
                "query_expansion_rules": [],
            },
            "observation_scopes": {
                "main_scope": "太空制造",
                "sub_scopes": ["微重力制造"],
                "scope_aliases": ["space manufacturing"],
                "scope_echo_terms": ["太空制造", "space manufacturing"],
                "off_domain_anchor_terms": ["人形机器人", "humanoid robot"],
            },
            "candidate_formation": {
                "technical_object_types": ["太空3D打印", "微重力增材制造", "在轨装配"],
                "mechanism_types": ["冷焊工艺", "additive manufacturing"],
                "task_or_performance_types": ["在轨装配", "结构成形"],
                "data_or_method_types": ["轨道验证", "微重力实验"],
                "scene_or_application_types": ["空间站", "近地轨道"],
                "generic_terms": ["太空制造", "技术"],
                "shell_terms": ["方法", "系统"],
                "valid_candidate_patterns": [
                    {
                        "pattern_id": "space_object_mechanism",
                        "required_slots": ["technical_object", "mechanism"],
                        "min_required_slot_count": 2,
                        "evidence_required": True,
                    }
                ],
                "invalid_candidate_patterns": [
                    {
                        "pattern_id": "space_scope_only",
                        "reject_terms": ["太空制造"],
                        "max_specific_slot_count": 1,
                    }
                ],
                "minimum_specificity_rule": {
                    "min_non_shell_slots": 2,
                    "require_evidence_span": True,
                    "allow_scope_only_candidate": False,
                },
            },
            "weak_signal_rules": {},
            "canonicalization": {},
            "evidence_rules": {},
            "reporting": {},
        }

        report = validate_domain_pack(DomainPack.from_dict(payload), require_dry_run=False)

        self.assertEqual(report.errors, [])
        self.assertEqual(report.checks.get("candidate_formation"), "passed")


if __name__ == "__main__":
    unittest.main()
