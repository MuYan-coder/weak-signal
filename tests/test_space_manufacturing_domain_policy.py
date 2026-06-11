import unittest

import pandas as pd

from src.domain import DomainContext, DomainPack
from src.extraction.candidate_former import build_candidate_forms
from src.scoring.candidate_eligibility import evaluate_candidate_eligibility
from src.scoring.scorer import score_all_candidates


def _space_pack() -> DomainPack:
    return DomainPack.from_dict(
        {
            "schema_version": "domain_pack_v1",
            "pack_id": "space_manufacturing",
            "pack_name": "太空制造 Domain Pack",
            "pack_version": "generated.v1",
            "source": {
                "mode": "llm_generated",
                "model": "fake-model",
                "prompt_version": "domain_pack_generator_v3",
                "based_on_user_input": {
                    "field_id": "space_manufacturing",
                    "field_name": "太空制造",
                    "keywords": ["太空制造", "在轨制造", "微重力制造"],
                    "synonyms": ["space manufacturing", "in-space manufacturing"],
                    "exclude_terms": ["人形机器人", "具身智能", "灵巧手", "夹爪"],
                },
            },
            "domain_identity": {
                "field_id": "space_manufacturing",
                "field_name": "太空制造",
                "domain_boundary": "微重力、在轨和航天场景下的材料沉积、结构成形、装配、维修与制造工艺。",
                "out_of_scope_domains": ["人形机器人", "具身智能", "灵巧手", "夹爪"],
            },
            "search_strategy": {
                "core_keywords": ["太空制造", "在轨制造", "微重力制造"],
                "synonyms": ["space manufacturing", "in-space manufacturing"],
                "english_terms": [
                    "space manufacturing",
                    "orbital additive manufacturing",
                    "microgravity manufacturing",
                    "on-orbit assembly",
                ],
                "exclude_terms": ["人形机器人", "具身智能", "灵巧手", "夹爪"],
            },
            "observation_scopes": {
                "main_scope": "太空制造",
                "sub_scopes": ["在轨制造", "微重力制造", "在轨装配"],
                "scope_aliases": [
                    "太空制造",
                    "在轨制造",
                    "空间制造",
                    "微重力制造",
                    "space manufacturing",
                    "in-space manufacturing",
                    "orbital manufacturing",
                    "microgravity manufacturing",
                ],
                "scope_echo_terms": ["太空制造", "space manufacturing"],
                "off_domain_anchor_terms": ["人形机器人", "具身智能", "灵巧手", "夹爪"],
            },
            "candidate_formation": {
                "technical_object_types": [
                    "太空3D打印",
                    "微重力增材制造",
                    "在轨装配",
                    "冷焊",
                    "orbital additive manufacturing",
                    "microgravity manufacturing",
                ],
                "mechanism_types": ["冷焊工艺", "additive manufacturing", "cold welding"],
                "task_or_performance_types": [
                    "在轨维修",
                    "在轨建造",
                    "结构成形",
                    "缺陷控制",
                    "材料沉积",
                    "on-orbit assembly",
                ],
                "data_or_method_types": ["冷焊工艺", "additive manufacturing", "cold welding"],
                "scene_or_application_types": ["太空制造", "在轨装配", "on-orbit assembly"],
                "generic_terms": [
                    "太空制造",
                    "space manufacturing",
                    "技术",
                    "方法",
                    "系统",
                    "平台",
                    "方案",
                    "产业",
                    "应用",
                    "能力",
                    "发展",
                    "解决方案",
                ],
                "shell_terms": [
                    "太空制造",
                    "space manufacturing",
                    "技术",
                    "方法",
                    "系统",
                    "平台",
                    "方案",
                    "产业",
                    "应用",
                    "能力",
                    "发展",
                    "解决方案",
                ],
                "valid_candidate_patterns": [
                    {
                        "pattern_id": "space_object_mechanism",
                        "required_slots": ["technical_object", "mechanism"],
                        "optional_slots": ["performance", "evidence_span"],
                        "min_required_slot_count": 2,
                        "evidence_required": True,
                    },
                    {
                        "pattern_id": "space_scene_mechanism",
                        "required_slots": ["scene", "mechanism"],
                        "optional_slots": ["performance", "evidence_span"],
                        "min_required_slot_count": 2,
                        "evidence_required": True,
                    },
                ],
                "invalid_candidate_patterns": [
                    {
                        "pattern_id": "space_scope_only",
                        "reject_terms": ["太空制造", "space manufacturing", "技术", "方案"],
                        "max_specific_slot_count": 1,
                    },
                    {
                        "pattern_id": "humanoid_only",
                        "reject_terms": ["人形机器人", "具身智能", "灵巧手", "夹爪"],
                        "max_specific_slot_count": 1,
                    },
                ],
                "minimum_specificity_rule": {
                    "min_non_shell_slots": 2,
                    "require_evidence_span": True,
                    "allow_scope_only_candidate": False,
                },
            },
        }
    )


def _space_context() -> DomainContext:
    return DomainContext.from_pack(_space_pack())


class SpaceManufacturingDomainPolicyTest(unittest.TestCase):
    def test_space_cold_welding_candidate_does_not_leak_humanoid_terms(self):
        events = pd.DataFrame(
            [
                {
                    "id": "space-paper-cold-weld",
                    "subject": "航天材料团队",
                    "action": "提出",
                    "technology": ["太空3D打印", "冷焊工艺"],
                    "technical_object": "太空3D打印结构件",
                    "mechanism": "冷焊工艺",
                    "task": "在轨建造结构成形",
                    "scene": "在轨装配",
                    "evidence_span": "团队提出太空3D打印结构件冷焊工艺，用于在轨建造和结构成形。",
                    "observation_scopes": ["太空制造"],
                    "candidate_units": [
                        {
                            "raw_phrase": "太空3D打印结构件冷焊工艺",
                            "raw_candidate_text": "太空3D打印结构件冷焊工艺",
                            "mechanism_core": "冷焊工艺",
                            "mechanism_core_tokens": ["冷焊工艺"],
                            "object_modifier_tokens": ["太空3D打印结构件"],
                            "task_constraint_tokens": ["在轨建造", "结构成形"],
                            "scene_tokens": ["在轨装配"],
                            "scope_names": ["太空制造"],
                            "source_extraction_mode": "domain_pack_test",
                        }
                    ],
                }
            ]
        )
        raw = pd.DataFrame(
            [
                {
                    "id": "space-paper-cold-weld",
                    "source_type": "paper",
                    "title": "太空3D打印结构件冷焊工艺",
                    "text": "机器人操作只是背景，核心是太空3D打印结构件冷焊工艺用于在轨建造和结构成形。",
                    "analysis_tech_field_name": "太空制造",
                }
            ]
        )

        candidates = build_candidate_forms(events, raw, domain_context=_space_context())
        names = " ".join(candidates["display_candidate_name"].fillna("").astype(str).tolist())

        self.assertFalse(candidates.empty)
        self.assertIn("冷焊", names)
        for leaked in ["人形机器人", "具身智能", "灵巧手", "夹爪"]:
            self.assertNotIn(leaked, names)
        self.assertIn("fine_grained_topic", set(candidates["topic_granularity"].astype(str)))

    def test_scope_only_space_policy_phrase_is_not_weak_signal_candidate(self):
        events = pd.DataFrame(
            [
                {
                    "id": "space-policy",
                    "subject": "主管部门",
                    "action": "发布",
                    "technology": ["太空制造"],
                    "technical_object": "太空制造技术方案",
                    "mechanism": "",
                    "task": "产业发展",
                    "evidence_span": "政策提出推进太空制造技术方案和产业发展。",
                    "observation_scopes": ["太空制造"],
                    "candidate_units": [
                        {
                            "raw_phrase": "太空制造技术方案",
                            "raw_candidate_text": "太空制造技术方案",
                            "mechanism_core": "",
                            "mechanism_core_tokens": [],
                            "object_modifier_tokens": ["太空制造"],
                            "task_constraint_tokens": ["产业发展"],
                            "scope_names": ["太空制造"],
                            "source_extraction_mode": "domain_pack_test",
                        }
                    ],
                }
            ]
        )
        raw = pd.DataFrame(
            [
                {
                    "id": "space-policy",
                    "source_type": "policy",
                    "title": "太空制造技术方案",
                    "text": "政策提出推进太空制造技术方案和产业发展。",
                    "analysis_tech_field_name": "太空制造",
                }
            ]
        )

        candidates = build_candidate_forms(events, raw, domain_context=_space_context())
        rows = candidates[candidates["raw_phrase"].astype(str) == "太空制造技术方案"]

        self.assertNotIn("formed_candidate_strong", set(rows["candidate_stage"].astype(str)))
        self.assertFalse(
            (
                (rows["display_tier"].astype(str) == "weak_signal")
                & (rows["raw_phrase"].astype(str) == "太空制造技术方案")
            ).any()
        )

    def test_space_candidate_eligibility_and_scoring_use_domain_context(self):
        row = {
            "id": "space-score",
            "display_candidate_name": "太空3D打印冷焊工艺",
            "tech_name": "太空3D打印冷焊工艺",
            "candidate_stage": "formed_candidate_strong",
            "raw_phrase": "太空3D打印冷焊工艺",
            "normalized_candidate_text": "太空3D打印冷焊工艺",
            "canonical_candidate_name_en": "space 3d printing cold welding process",
            "primary_scope": "太空制造",
            "scope_name": "太空制造",
            "scope_names": ["太空制造"],
            "mechanism_core": "冷焊工艺",
            "mechanism_core_tokens": ["冷焊工艺"],
            "object_modifier_tokens": ["太空3D打印"],
            "task_constraint_tokens": ["在轨建造", "结构成形"],
            "scene_tokens": ["在轨装配"],
            "source_types": ["paper"],
            "source_count": 1,
            "org_count": 1,
            "total_mentions": 2,
            "cluster_evidence_count": 2,
            "has_mechanism_core": True,
            "has_non_scope_constraint": True,
            "is_scope_internal_candidate": True,
            "is_scope_echo": False,
            "topic_granularity": "fine_grained_topic",
            "display_tier": "weak_signal",
            "mention_dates": ["2026-05-01", "2026-05-20"],
            "evidence_items": [
                {
                    "source_type": "paper",
                    "title": "太空3D打印冷焊工艺早期验证",
                    "snippet": "团队验证太空3D打印冷焊工艺支持在轨结构成形。",
                    "raw_candidate_text": "太空3D打印冷焊工艺",
                }
            ],
            "raw_phrase_example": "太空3D打印冷焊工艺",
        }

        profile = evaluate_candidate_eligibility(row, domain_context=_space_context())
        scored = score_all_candidates(pd.DataFrame([row]), domain_context=_space_context())

        self.assertIn(profile["status"], {"eligible", "candidate_monitoring"})
        self.assertNotEqual(scored.loc[0, "score_applicability"], "not_applicable")
        self.assertGreater(float(scored.loc[0, "weak_signal_raw_score"]), 0)


if __name__ == "__main__":
    unittest.main()
