import unittest

import pandas as pd

from src.domain import DomainContext, DomainPack, load_domain_context
from src.extraction.candidate_former import build_candidate_forms
from src.extraction.event_extractor import process_events
from src.extraction.tech_lexicon import build_domain_lexicon
from src.scoring.scorer import score_all_candidates
from src.scoring.signal_generator import generate_candidate_outputs
from src.scoring.topic_refiner import _build_prompt, _surface_name_from_evidence


def _battery_pack() -> DomainPack:
    return DomainPack.from_dict(
        {
            "schema_version": "domain_pack_v1",
            "pack_id": "battery_materials",
            "pack_name": "电池材料 Domain Pack",
            "pack_version": "generated.v1",
            "source": {
                "mode": "llm_generated",
                "model": "fake-model",
                "prompt_version": "domain_pack_generator_v1",
                "based_on_user_input": {
                    "field_id": "battery_materials",
                    "field_name": "电池材料",
                    "keywords": ["电池材料", "固态电池"],
                    "synonyms": ["battery materials"],
                    "exclude_terms": ["机器人", "humanoid robot", "world model"],
                },
            },
            "domain_identity": {
                "field_id": "battery_materials",
                "field_name": "电池材料",
                "domain_boundary": "电池材料、电解质、界面和正负极材料改性。",
                "out_of_scope_domains": ["机器人", "具身智能", "世界模型"],
            },
            "search_strategy": {
                "core_keywords": ["电池材料", "固态电池"],
                "synonyms": ["battery materials"],
                "english_terms": ["battery material", "solid-state battery"],
                "exclude_terms": ["机器人", "humanoid robot", "world model"],
            },
            "observation_scopes": {
                "main_scope": "电池材料",
                "sub_scopes": ["固态电池"],
                "scope_aliases": ["电池材料", "固态电池", "battery materials", "solid-state battery"],
                "scope_echo_terms": ["电池材料"],
                "off_domain_anchor_terms": ["机器人", "humanoid robot", "world model"],
            },
            "candidate_formation": {
                "technical_object_types": ["电解质", "正极材料", "钙钛矿薄膜", "cathode material"],
                "mechanism_types": ["界面钝化", "掺杂", "interface passivation"],
                "task_or_performance_types": ["循环稳定性", "能量密度", "cycle stability"],
                "data_or_method_types": ["阻抗谱", "electrochemical impedance"],
                "scene_or_application_types": ["固态电池"],
                "generic_terms": ["材料", "技术", "material", "technology"],
                "shell_terms": ["技术", "方法", "系统", "应用", "technology", "method", "system"],
                "valid_candidate_patterns": [
                    {
                        "pattern_id": "battery_object_mechanism",
                        "required_slots": ["technical_object", "mechanism"],
                        "optional_slots": ["performance", "evidence_span"],
                        "min_required_slot_count": 2,
                        "evidence_required": True,
                    }
                ],
                "invalid_candidate_patterns": [
                    {
                        "pattern_id": "battery_shell_only",
                        "reject_terms": ["材料", "技术", "方法", "系统"],
                        "max_specific_slot_count": 1,
                    }
                ],
                "minimum_specificity_rule": {
                    "min_non_shell_slots": 2,
                    "require_evidence_span": True,
                    "allow_scope_only_candidate": False,
                },
            },
        }
    )


def _battery_context() -> DomainContext:
    return DomainContext.from_pack(_battery_pack())


class NoDefaultRobotLeakageTest(unittest.TestCase):
    def test_non_robot_domain_lexicon_does_not_inject_robot_defaults(self):
        lexicon = build_domain_lexicon(_battery_pack())

        self.assertNotIn("humanoid robot", lexicon.observation_scopes)
        self.assertNotIn("world model", lexicon.observation_scopes)
        self.assertEqual(lexicon.detect_supported_observation_scopes("robot planning world model"), [])
        self.assertIn("电池材料", lexicon.detect_supported_observation_scopes("固态电池电解质界面钝化"))
        self.assertIn("界面钝化", lexicon.extract_mechanism_core_tokens("电解质界面钝化提升循环稳定性"))

    def test_domain_lexicon_filters_off_domain_technologies_and_preserves_sub_scopes(self):
        payload = _battery_pack().to_dict()
        payload["observation_scopes"]["main_scope"] = "battery materials"
        payload["observation_scopes"]["sub_scopes"] = ["solid-state battery"]
        payload["observation_scopes"]["scope_aliases"] = ["battery materials", "solid-state battery"]
        payload["search_strategy"]["core_keywords"] = ["battery materials"]
        payload["domain_identity"]["field_name"] = "battery materials"
        pack = DomainPack.from_dict(payload)
        lexicon = build_domain_lexicon(pack)

        normalized = lexicon.normalize_technologies(
            ["world model", "humanoid robot", "cathode material"],
            fallback_text="solid-state battery cathode material",
        )
        scopes = lexicon.detect_supported_observation_scopes("solid-state battery cathode material")

        self.assertIn("cathode material", normalized)
        self.assertNotIn("world model", normalized)
        self.assertNotIn("humanoid robot", normalized)
        self.assertIn("solid-state battery", lexicon.observation_scopes)
        self.assertIn("solid-state battery", scopes)

    def test_process_events_uses_domain_pack_scope_and_excludes_robot_defaults(self):
        raw = pd.DataFrame(
            [
                {
                    "id": "paper-battery",
                    "source_type": "paper",
                    "title": "固态电池电解质界面钝化",
                    "text": "研究团队提出固态电池电解质界面钝化方法，降低阻抗并提升循环稳定性。",
                    "analysis_tech_field_name": "电池材料",
                }
            ]
        )

        events = process_events(
            raw,
            use_api=False,
            refresh_cache=True,
            cache_path=None,
            domain_context=_battery_context(),
        )

        joined_scope = " ".join(events["observation_scopes"].astype(str).tolist())
        joined_tech = " ".join(events["technology"].astype(str).tolist())
        self.assertIn("电池材料", joined_scope)
        self.assertIn("电解质", joined_tech)
        self.assertNotIn("humanoid robot", joined_scope)
        self.assertNotIn("world model", joined_scope)

    def test_candidate_forms_use_domain_pack_rules_and_record_rule_hits(self):
        events = pd.DataFrame(
            [
                {
                    "id": "paper-battery",
                    "subject": "研究团队",
                    "action": "提出",
                    "technology": ["电解质"],
                    "technical_object": "电解质",
                    "mechanism": "界面钝化",
                    "task": "提升循环稳定性",
                    "evidence_span": "研究团队提出固态电池电解质界面钝化方法，降低阻抗并提升循环稳定性。",
                    "observation_scopes": ["电池材料"],
                    "candidate_units": [
                        {
                            "raw_phrase": "电解质界面钝化",
                            "raw_candidate_text": "电解质界面钝化",
                            "mechanism_core": "界面钝化",
                            "mechanism_core_tokens": ["界面钝化"],
                            "object_modifier_tokens": ["电解质"],
                            "task_constraint_tokens": ["循环稳定性"],
                            "scope_names": ["电池材料"],
                            "source_extraction_mode": "domain_pack_test",
                        }
                    ],
                }
            ]
        )
        raw = pd.DataFrame(
            [
                {
                    "id": "paper-battery",
                    "source_type": "paper",
                    "title": "固态电池电解质界面钝化",
                    "text": "研究团队提出固态电池电解质界面钝化方法，降低阻抗并提升循环稳定性。",
                    "analysis_tech_field_name": "电池材料",
                }
            ]
        )

        candidates = build_candidate_forms(events, raw, domain_context=_battery_context())

        names = " ".join(candidates["display_candidate_name"].fillna("").astype(str).tolist())
        rule_ids = " ".join(candidates["domain_pack_candidate_rule_ids"].fillna("").astype(str).tolist())
        reasons = " ".join(candidates["domain_pack_candidate_reason"].fillna("").astype(str).tolist())
        self.assertIn("电解质", names)
        self.assertIn("界面钝化", names)
        self.assertIn("battery_object_mechanism", rule_ids)
        self.assertIn("technical_object", reasons)
        self.assertNotIn("机器人", names)

    def test_candidate_forms_reject_domain_pack_shell_only_candidate(self):
        events = pd.DataFrame(
            [
                {
                    "id": "paper-shell",
                    "subject": "研究团队",
                    "action": "提出",
                    "technology": ["电池材料"],
                    "technical_object": "材料",
                    "mechanism": "",
                    "task": "",
                    "evidence_span": "研究团队提出一种材料技术。",
                    "observation_scopes": ["电池材料"],
                    "candidate_units": [
                        {
                            "raw_phrase": "材料技术",
                            "raw_candidate_text": "材料技术",
                            "mechanism_core": "",
                            "mechanism_core_tokens": [],
                            "object_modifier_tokens": ["材料"],
                            "task_constraint_tokens": [],
                            "scope_names": ["电池材料"],
                            "source_extraction_mode": "domain_pack_test",
                        }
                    ],
                }
            ]
        )
        raw = pd.DataFrame(
            [
                {
                    "id": "paper-shell",
                    "source_type": "paper",
                    "title": "材料技术",
                    "text": "研究团队提出一种材料技术。",
                    "analysis_tech_field_name": "电池材料",
                }
            ]
        )

        candidates = build_candidate_forms(events, raw, domain_context=_battery_context())

        shell_rows = candidates[candidates["raw_phrase"].astype(str) == "材料技术"]
        self.assertFalse(shell_rows.empty)
        self.assertTrue(set(shell_rows["domain_pack_candidate_status"].astype(str)) <= {"rejected"})
        self.assertIn("battery_shell_only", " ".join(shell_rows["domain_pack_candidate_rule_ids"].astype(str)))

    def test_candidate_trace_requires_real_evidence_for_domain_pack_patterns(self):
        events = pd.DataFrame(
            [
                {
                    "id": "paper-no-evidence",
                    "technology": ["cathode material"],
                    "technical_object": "cathode material",
                    "mechanism": "interface passivation",
                    "task": "",
                    "evidence_span": "",
                    "observation_scopes": ["鐢垫睜鏉愭枡"],
                    "candidate_units": [
                        {
                            "raw_phrase": "cathode material interface passivation",
                            "raw_candidate_text": "cathode material interface passivation",
                            "mechanism_core": "interface passivation",
                            "mechanism_core_tokens": ["interface passivation"],
                            "object_modifier_tokens": ["cathode material"],
                            "task_constraint_tokens": [],
                            "scope_names": ["鐢垫睜鏉愭枡"],
                            "source_extraction_mode": "domain_pack_test",
                        }
                    ],
                }
            ]
        )
        raw = pd.DataFrame([{"id": "paper-no-evidence", "source_type": "paper", "title": "", "text": ""}])

        candidates = build_candidate_forms(events, raw, domain_context=_battery_context())
        rows = candidates[candidates["raw_phrase"].astype(str) == "cathode material interface passivation"]

        self.assertFalse(rows.empty)
        self.assertNotIn("accepted", set(rows["domain_pack_candidate_status"].astype(str)))
        self.assertNotIn("battery_object_mechanism", " ".join(rows["domain_pack_candidate_rule_ids"].astype(str)))

    def test_data_or_method_hint_does_not_count_twice_or_accept_without_valid_pattern(self):
        events = pd.DataFrame(
            [
                {
                    "id": "paper-data-only",
                    "technology": ["cathode material"],
                    "technical_object": "cathode material",
                    "mechanism": "",
                    "task": "",
                    "evidence_span": "cathode material was measured with electrochemical impedance",
                    "observation_scopes": ["鐢垫睜鏉愭枡"],
                    "candidate_units": [
                        {
                            "raw_phrase": "cathode material electrochemical impedance",
                            "raw_candidate_text": "cathode material electrochemical impedance",
                            "mechanism_core": "",
                            "mechanism_core_tokens": [],
                            "object_modifier_tokens": ["cathode material"],
                            "data_modifier_tokens": ["electrochemical impedance"],
                            "task_constraint_tokens": [],
                            "scope_names": ["鐢垫睜鏉愭枡"],
                            "source_extraction_mode": "domain_pack_test",
                        }
                    ],
                }
            ]
        )
        raw = pd.DataFrame(
            [
                {
                    "id": "paper-data-only",
                    "source_type": "paper",
                    "title": "cathode material impedance measurement",
                    "text": "cathode material was measured with electrochemical impedance",
                }
            ]
        )

        candidates = build_candidate_forms(events, raw, domain_context=_battery_context())
        rows = candidates[candidates["raw_phrase"].astype(str) == "cathode material electrochemical impedance"]
        slot_hits = rows.iloc[0]["domain_pack_candidate_slot_hits"]

        self.assertFalse(rows.empty)
        self.assertNotIn("accepted", set(rows["domain_pack_candidate_status"].astype(str)))
        self.assertEqual(slot_hits["method"], [])

    def test_valid_domain_pack_pattern_without_mechanism_can_form_candidate(self):
        payload = _battery_pack().to_dict()
        payload["candidate_formation"]["valid_candidate_patterns"] = [
            {
                "pattern_id": "battery_object_performance",
                "required_slots": ["technical_object", "performance"],
                "min_required_slot_count": 2,
                "evidence_required": True,
            }
        ]
        pack = DomainPack.from_dict(payload)
        context = DomainContext.from_pack(pack)
        events = pd.DataFrame(
            [
                {
                    "id": "paper-performance",
                    "technology": ["cathode material"],
                    "technical_object": "cathode material",
                    "mechanism": "",
                    "task": "cycle stability",
                    "evidence_span": "cathode material improves cycle stability",
                    "observation_scopes": ["鐢垫睜鏉愭枡"],
                }
            ]
        )
        raw = pd.DataFrame(
            [
                {
                    "id": "paper-performance",
                    "source_type": "paper",
                    "title": "cathode material cycle stability",
                    "text": "cathode material improves cycle stability",
                }
            ]
        )

        candidates = build_candidate_forms(events, raw, domain_context=context)
        rows = candidates[candidates["raw_phrase"].astype(str).str.contains("cathode material", na=False)]

        self.assertFalse(rows.empty)
        self.assertIn("battery_object_performance", " ".join(rows["domain_pack_candidate_rule_ids"].astype(str)))
        self.assertIn("accepted", set(rows["domain_pack_candidate_status"].astype(str)))
        self.assertTrue(any(str(stage).startswith("formed_candidate") for stage in rows["candidate_stage"]))

    def test_scoring_uses_domain_pack_adjustments_and_explains_rule_contribution(self):
        payload = _battery_pack().to_dict()
        payload["weak_signal_rules"] = {
            "early_stage_markers": ["早期", "early"],
            "low_attention_markers": ["高校"],
            "niche_actor_markers": ["研究团队"],
            "cross_domain_markers": [],
            "engineering_trace_markers": ["阻抗谱", "electrochemical impedance"],
            "commercialization_noise_markers": [],
            "policy_or_market_noise_markers": [],
            "scoring_adjustments": [
                {
                    "rule_id": "battery_early_validation",
                    "applies_to": "candidate",
                    "match_terms": ["界面钝化", "阻抗谱"],
                    "required_evidence_fields": ["evidence_items"],
                    "score_delta": 0.8,
                    "max_delta": 1.0,
                    "reason_template": "命中电池材料早期实验验证线索。",
                }
            ],
        }
        context = DomainContext.from_pack(DomainPack.from_dict(payload))
        candidates = pd.DataFrame(
            [
                {
                    "id": "paper-battery-score",
                    "display_candidate_name": "电解质界面钝化",
                    "tech_name": "电解质界面钝化",
                    "candidate_stage": "formed_candidate_strong",
                    "source_types": ["paper"],
                    "source_count": 1,
                    "org_count": 1,
                    "total_mentions": 2,
                    "cluster_evidence_count": 2,
                    "has_mechanism_core": True,
                    "has_non_scope_constraint": True,
                    "is_scope_internal_candidate": True,
                    "topic_granularity": "fine_grained_topic",
                    "display_tier": "weak_signal",
                    "mention_dates": ["2026-05-01", "2026-05-18"],
                    "evidence_items": [
                        {
                            "source_type": "paper",
                            "title": "固态电池电解质界面钝化早期实验验证",
                            "snippet": "研究团队使用阻抗谱验证界面钝化可提升循环稳定性。",
                            "raw_candidate_text": "电解质界面钝化",
                        }
                    ],
                    "raw_phrase_example": "电解质界面钝化",
                    "mechanism_core": "界面钝化",
                }
            ]
        )

        scored = score_all_candidates(candidates, domain_context=context)
        row = scored.iloc[0]

        self.assertGreater(row["domain_pack_scoring_delta"], 0)
        self.assertIn("battery_early_validation", row["domain_pack_scoring_rule_ids"])
        self.assertIn("命中电池材料早期实验验证线索", row["domain_pack_scoring_reason"])
        self.assertIn("领域规则", row["explanation"])

    def test_object_family_is_disabled_for_non_robot_domain_even_without_scope_text(self):
        forms = pd.DataFrame(
            [
                {
                    "id": "paper-robot-looking",
                    "candidate_stage": "formed_candidate_strong",
                    "display_candidate_name": "robot training",
                    "canonical_candidate_name_en": "robot training",
                    "raw_phrase": "robot training",
                    "raw_candidate_text": "robot training",
                    "normalized_candidate_text": "robot training",
                    "mechanism_core": "training",
                    "has_mechanism_core": True,
                    "has_non_scope_constraint": True,
                    "source_type": "paper",
                }
            ]
        )
        raw = pd.DataFrame(
            [
                {
                    "id": "paper-robot-looking",
                    "source_type": "paper",
                    "title": "robot training",
                    "text": "robot training was mentioned in passing.",
                }
            ]
        )

        output = generate_candidate_outputs(forms, raw, domain_context=_battery_context())
        row = output["candidates_df"].iloc[0]

        self.assertEqual(row["family_id"], "")
        self.assertFalse(row["family_matched"])
        self.assertEqual(row["canonical_term"], "robot training")

    def test_humanoid_preset_enables_domain_pack_object_family_registry(self):
        forms = pd.DataFrame(
            [
                {
                    "id": "paper-humanoid-family",
                    "candidate_stage": "formed_candidate_strong",
                    "display_candidate_name": "robot training",
                    "canonical_candidate_name_en": "robot training",
                    "raw_phrase": "robot training",
                    "raw_candidate_text": "robot training",
                    "normalized_candidate_text": "robot training",
                    "mechanism_core": "training",
                    "has_mechanism_core": True,
                    "has_non_scope_constraint": True,
                    "source_type": "paper",
                }
            ]
        )
        raw = pd.DataFrame(
            [
                {
                    "id": "paper-humanoid-family",
                    "source_type": "paper",
                    "title": "robot training",
                    "text": "A robot training method was evaluated for humanoid robot control.",
                }
            ]
        )

        output = generate_candidate_outputs(forms, raw, domain_context=load_domain_context("humanoid_robot"))
        row = output["candidates_df"].iloc[0]

        self.assertEqual(row["family_id"], "of_robot_training")
        self.assertTrue(row["family_matched"])
        self.assertEqual(row["canonical_term"], "robot training")

    def test_topic_refiner_prompt_uses_domain_pack_context(self):
        prompt = _build_prompt(
            [
                {
                    "candidate_cluster_id": "battery-topic",
                    "display_candidate_name": "电解质界面钝化",
                    "signal_type": "weak_signal",
                    "scope_name": "电池材料",
                    "mechanism_core": "界面钝化",
                    "source_count": 1,
                    "source_types": ["paper"],
                    "evidence_items": [
                        {
                            "source_type": "paper",
                            "title": "固态电池电解质界面钝化",
                            "snippet": "电解质界面钝化提升循环稳定性。",
                        }
                    ],
                }
            ],
            domain_context=_battery_context(),
        )

        self.assertIn("电池材料", prompt)
        self.assertIn("电解质", prompt)
        self.assertIn("界面钝化", prompt)
        self.assertIn("循环稳定性", prompt)

    def test_robot_world_model_surface_adjustment_is_guarded_by_humanoid_preset(self):
        row = {
            "scope_name": "world model",
            "mechanism_core": "planning",
            "task_constraint_tokens": ["robot"],
            "object_modifier_tokens": ["robot"],
            "display_candidate_name": "world model planning",
            "raw_phrase_example": "social navigation robot planning",
            "evidence_items": [
                {
                    "title": "World model for robot social navigation planning",
                    "snippet": "The method improves social navigation for robot planning.",
                }
            ],
        }

        battery_name = _surface_name_from_evidence(
            row,
            "small_topic",
            "scene+mechanism",
            "world model planning",
            domain_context=_battery_context(),
        )
        humanoid_name = _surface_name_from_evidence(
            row,
            "small_topic",
            "scene+mechanism",
            "world model planning",
            domain_context=load_domain_context("humanoid_robot"),
        )

        self.assertEqual(battery_name, "world model planning")
        self.assertEqual(humanoid_name, "机器人社交导航规划")


if __name__ == "__main__":
    unittest.main()
