import unittest

import pandas as pd

from src.domain import DomainContext, DomainPack, load_domain_context
from src.extraction.candidate_former import (
    _cluster_specificity_profile,
    _is_generic_method_only_display_name,
    _technical_object_name_from_slots,
    build_candidate_forms,
)
from src.extraction.domain_candidate_policy import build_domain_candidate_policy
from src.extraction.event_extractor import process_events
from src.extraction.tech_lexicon import build_domain_lexicon
from src.scoring.scorer import score_all_candidates
from src.scoring.signal_generator import generate_candidate_outputs
from src.scoring.topic_refiner import _build_prompt, _surface_name_from_evidence
from src.validation.event_quality import (
    build_event_quality_table,
    merge_event_quality_into_candidates,
    merge_event_quality_into_events,
)


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


def _electronic_materials_pack() -> DomainPack:
    return DomainPack.from_dict(
        {
            "schema_version": "domain_pack_v1",
            "pack_id": "new_electronic_materials",
            "pack_name": "新型电子材料 Domain Pack",
            "pack_version": "generated.v1",
            "source": {
                "mode": "llm_generated",
                "model": "fake-model",
                "prompt_version": "domain_pack_generator_v3",
                "based_on_user_input": {
                    "field_id": "new_electronic_materials",
                    "field_name": "新型电子材料",
                    "keywords": ["新型电子材料", "碳化硅", "氮化镓", "二维材料", "钙钛矿"],
                    "synonyms": ["new electronic materials", "SiC", "GaN", "2D materials", "perovskite"],
                    "exclude_terms": ["机器人", "具身智能", "训练技术", "控制技术", "规划技术", "仿真技术"],
                },
            },
            "domain_identity": {
                "field_id": "new_electronic_materials",
                "field_name": "新型电子材料",
                "domain_boundary": "宽禁带半导体、二维材料、钙钛矿、拓扑绝缘体等电子材料及其制备、掺杂、外延和器件性能。",
                "out_of_scope_domains": ["机器人", "具身智能", "通用人工智能训练", "自动驾驶控制"],
            },
            "search_strategy": {
                "core_keywords": ["新型电子材料", "碳化硅", "氮化镓", "二维材料", "钙钛矿", "拓扑绝缘体"],
                "synonyms": ["new electronic materials", "SiC", "GaN", "2D materials", "perovskite"],
                "english_terms": ["wide bandgap semiconductor", "2D material", "perovskite"],
                "exclude_terms": ["机器人", "具身智能", "训练技术", "控制技术", "规划技术", "仿真技术"],
            },
            "observation_scopes": {
                "main_scope": "新型电子材料",
                "sub_scopes": ["宽禁带半导体", "二维材料", "钙钛矿", "拓扑绝缘体"],
                "scope_aliases": ["新型电子材料", "电子材料", "SiC", "GaN", "perovskite", "2D materials"],
                "scope_echo_terms": ["材料", "电子材料", "新材料"],
                "off_domain_anchor_terms": ["机器人", "具身智能", "训练技术", "控制技术", "规划技术", "仿真技术"],
            },
            "candidate_formation": {
                "technical_object_types": ["碳化硅", "氮化镓", "二维材料", "钙钛矿", "拓扑绝缘体", "SiC", "GaN"],
                "mechanism_types": ["化学气相沉积", "外延生长", "掺杂", "界面钝化", "缺陷调控"],
                "task_or_performance_types": ["载流子迁移率", "击穿电压", "缺陷密度", "稳定性"],
                "data_or_method_types": ["化学气相沉积", "原子层沉积", "分子束外延", "光致发光"],
                "scene_or_application_types": ["功率器件", "光电器件", "射频器件"],
                "generic_terms": ["材料", "技术", "方法", "系统", "方案"],
                "shell_terms": ["材料", "技术", "方法", "系统", "应用", "方案"],
                "valid_candidate_patterns": [],
                "invalid_candidate_patterns": [
                    {
                        "pattern_id": "electronic_materials_shell_only",
                        "reject_terms": ["材料", "技术", "方法", "系统", "方案"],
                        "max_specific_slot_count": 0,
                    }
                ],
                "minimum_specificity_rule": {
                    "min_non_shell_slots": 1,
                    "require_evidence_span": True,
                    "allow_scope_only_candidate": False,
                },
            },
        }
    )


def _electronic_materials_context() -> DomainContext:
    return DomainContext.from_pack(_electronic_materials_pack())


def _space_manufacturing_pack() -> DomainPack:
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
                    "exclude_terms": ["人形机器人", "具身智能", "灵巧手", "夹爪", "机械臂", "world model"],
                },
            },
            "domain_identity": {
                "field_id": "space_manufacturing",
                "field_name": "太空制造",
                "domain_boundary": "微重力、在轨和航天场景下的材料沉积、结构成形、装配、维修与制造工艺。",
                "out_of_scope_domains": ["人形机器人", "具身智能", "灵巧手", "夹爪", "机械臂", "world model"],
            },
            "search_strategy": {
                "core_keywords": ["太空制造", "在轨制造", "微重力制造"],
                "synonyms": ["space manufacturing", "in-space manufacturing"],
                "english_terms": ["space manufacturing", "orbital additive manufacturing", "microgravity manufacturing"],
                "exclude_terms": ["人形机器人", "具身智能", "灵巧手", "夹爪", "机械臂", "world model"],
            },
            "observation_scopes": {
                "main_scope": "太空制造",
                "sub_scopes": ["在轨制造", "微重力制造"],
                "scope_aliases": ["太空制造", "在轨制造", "微重力制造", "space manufacturing"],
                "scope_echo_terms": ["太空制造", "space manufacturing"],
                "off_domain_anchor_terms": ["人形机器人", "具身智能", "灵巧手", "夹爪", "机械臂", "world model"],
            },
            "candidate_formation": {
                "technical_object_types": ["太空3D打印", "微重力增材制造", "在轨装配", "冷焊"],
                "mechanism_types": ["冷焊工艺", "additive manufacturing", "cold welding"],
                "task_or_performance_types": ["在轨维修", "在轨建造", "结构成形"],
                "data_or_method_types": ["冷焊工艺", "additive manufacturing"],
                "scene_or_application_types": ["太空制造", "在轨装配"],
                "generic_terms": ["太空制造", "space manufacturing", "技术", "方法", "系统", "平台", "方案"],
                "shell_terms": ["太空制造", "space manufacturing", "技术", "方法", "系统", "平台", "方案"],
                "valid_candidate_patterns": [],
                "invalid_candidate_patterns": [],
                "minimum_specificity_rule": {
                    "min_non_shell_slots": 2,
                    "require_evidence_span": True,
                    "allow_scope_only_candidate": False,
                },
            },
        }
    )


def _space_manufacturing_context() -> DomainContext:
    return DomainContext.from_pack(_space_manufacturing_pack())


def _space_manufacturing_policy():
    return build_domain_candidate_policy(build_domain_lexicon(_space_manufacturing_context()))


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

    def test_non_humanoid_candidate_forms_do_not_leak_manipulator_surfaces(self):
        events = pd.DataFrame(
            [
                {
                    "id": "paper-battery-manipulator-words",
                    "subject": "研究团队",
                    "action": "提出",
                    "technology": ["电解质"],
                    "technical_object": "电解质",
                    "mechanism": "界面钝化",
                    "task": "提升循环稳定性",
                    "evidence_span": "研究团队提出固态电池电解质界面钝化方法，提升循环稳定性。",
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
                    "id": "paper-battery-manipulator-words",
                    "source_type": "paper",
                    "title": "固态电池夹爪抓取装配误报词界面钝化",
                    "text": "固态电池研究文本包含夹爪、抓取和装配等旁路词，但主题是电解质界面钝化。",
                    "analysis_tech_field_name": "电池材料",
                }
            ]
        )

        candidates = build_candidate_forms(events, raw, domain_context=_battery_context())
        rows = candidates[
            (candidates["id"].astype(str) == "paper-battery-manipulator-words")
            & (candidates["raw_phrase"].astype(str) != "电池材料")
            & (~candidates["candidate_stage"].astype(str).eq("scope_overview"))
        ]

        self.assertFalse(rows.empty)
        for _, row in rows.iterrows():
            self.assertEqual(row["object_surface_candidates"], [])
            self.assertEqual(row["preferred_object_surface"], "")
            self.assertEqual(row["display_preferred_object_surface"], "")
            self.assertEqual(row["preferred_task_surface"], "")
            self.assertEqual(row["display_preferred_task_surface"], "")

    def test_fine_grained_candidate_display_tier_remains_weak_signal_even_before_strong_stage(self):
        events = pd.DataFrame(
            [
                {
                    "id": "paper-battery-tier",
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
                    "id": "paper-battery-tier",
                    "source_type": "paper",
                    "title": "固态电池电解质界面钝化",
                    "text": "研究团队提出固态电池电解质界面钝化方法，降低阻抗并提升循环稳定性。",
                    "analysis_tech_field_name": "电池材料",
                }
            ]
        )

        candidates = build_candidate_forms(events, raw, domain_context=_battery_context())
        rows = candidates[candidates["candidate_stage"].astype(str) == "formed_candidate"]

        self.assertFalse(rows.empty)
        self.assertTrue((rows["topic_granularity"].astype(str) == "fine_grained_topic").any())
        self.assertEqual(set(rows["display_tier"].astype(str)), {"weak_signal"})

    def test_candidate_display_name_does_not_repeat_same_process_and_mechanism(self):
        name = _technical_object_name_from_slots(
            {
                "primary_scope": "太空制造",
                "scope_names": ["太空制造"],
                "mechanism_core": "冷焊工艺",
                "mechanism_core_tokens": ["冷焊工艺"],
                "object_modifier_tokens": ["轨道转移飞行器"],
                "task_constraint_tokens": [],
                "data_modifier_tokens": [],
                "method_modifier_tokens": ["冷焊工艺"],
                "scene_tokens": [],
                "domain_lexicon": build_domain_lexicon(_battery_context()),
            }
        )

        self.assertEqual(name, "采用冷焊工艺的轨道转移飞行器")

    def test_shell_invalid_pattern_without_specificity_limit_does_not_reject_specific_candidate(self):
        payload = _battery_pack().to_dict()
        payload["candidate_formation"]["invalid_candidate_patterns"] = [
            {
                "pattern_id": "shell_term_only",
                "reject_terms": ["技术", "方法", "系统", "应用"],
            }
        ]
        context = DomainContext.from_pack(DomainPack.from_dict(payload))
        events = pd.DataFrame(
            [
                {
                    "id": "paper-specific-shell-word",
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
                            "raw_phrase": "电解质界面钝化方法",
                            "raw_candidate_text": "电解质界面钝化方法",
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
                    "id": "paper-specific-shell-word",
                    "source_type": "paper",
                    "title": "固态电池电解质界面钝化方法",
                    "text": "研究团队提出固态电池电解质界面钝化方法，降低阻抗并提升循环稳定性。",
                    "analysis_tech_field_name": "电池材料",
                }
            ]
        )

        candidates = build_candidate_forms(events, raw, domain_context=context)
        rows = candidates[
            candidates["domain_pack_candidate_rule_ids"].fillna("").astype(str).str.contains(
                "battery_object_mechanism", na=False
            )
        ]

        self.assertFalse(rows.empty)
        self.assertIn("accepted", set(rows["domain_pack_candidate_status"].astype(str)))
        self.assertTrue(any(str(stage).startswith("formed_candidate") for stage in rows["candidate_stage"]))
        self.assertNotIn("rejected", set(rows["domain_pack_candidate_status"].astype(str)))

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

    def test_domain_relevance_gate_blocks_off_domain_generic_method_candidates(self):
        events = pd.DataFrame(
            [
                {
                    "id": "paper-off-domain-training",
                    "subject": "研究团队",
                    "action": "提出",
                    "technology": ["训练技术"],
                    "technical_object": "训练技术",
                    "mechanism": "训练",
                    "task": "控制规划",
                    "evidence_span": "机器人控制中的训练技术和规划技术可以提升仿真控制效果。",
                    "observation_scopes": ["新型电子材料"],
                    "candidate_units": [
                        {
                            "raw_phrase": "训练技术",
                            "raw_candidate_text": "训练技术",
                            "mechanism_core": "训练",
                            "mechanism_core_tokens": ["训练"],
                            "object_modifier_tokens": ["控制"],
                            "task_constraint_tokens": ["规划"],
                            "method_modifier_tokens": ["训练技术"],
                            "scope_names": ["新型电子材料"],
                            "source_extraction_mode": "domain_pack_test",
                        }
                    ],
                },
                {
                    "id": "paper-sic-cvd",
                    "subject": "研究团队",
                    "action": "提出",
                    "technology": ["碳化硅"],
                    "technical_object": "碳化硅",
                    "mechanism": "化学气相沉积",
                    "task": "降低缺陷密度",
                    "evidence_span": "研究团队提出碳化硅外延层化学气相沉积工艺，降低缺陷密度并提升击穿电压。",
                    "observation_scopes": ["新型电子材料"],
                    "candidate_units": [
                        {
                            "raw_phrase": "碳化硅化学气相沉积",
                            "raw_candidate_text": "碳化硅化学气相沉积",
                            "mechanism_core": "化学气相沉积",
                            "mechanism_core_tokens": ["化学气相沉积"],
                            "object_modifier_tokens": ["碳化硅"],
                            "task_constraint_tokens": ["缺陷密度"],
                            "method_modifier_tokens": ["化学气相沉积"],
                            "scope_names": ["新型电子材料"],
                            "source_extraction_mode": "domain_pack_test",
                        }
                    ],
                },
            ]
        )
        raw = pd.DataFrame(
            [
                {
                    "id": "paper-off-domain-training",
                    "source_type": "paper",
                    "title": "机器人控制训练技术",
                    "text": "机器人控制中的训练技术和规划技术可以提升仿真控制效果。",
                    "analysis_tech_field_name": "新型电子材料",
                },
                {
                    "id": "paper-sic-cvd",
                    "source_type": "paper",
                    "title": "碳化硅外延层化学气相沉积",
                    "text": "研究团队提出碳化硅外延层化学气相沉积工艺，降低缺陷密度并提升击穿电压。",
                    "analysis_tech_field_name": "新型电子材料",
                },
            ]
        )
        context = _electronic_materials_context()
        quality = build_event_quality_table(events, raw, domain_lexicon=build_domain_lexicon(context))
        enriched_events = merge_event_quality_into_events(events, quality)

        candidates = build_candidate_forms(enriched_events, raw, domain_context=context)
        formed = candidates[candidates["candidate_stage"].astype(str).str.startswith("formed_candidate", na=False)]
        names = " ".join(formed["display_candidate_name"].fillna("").astype(str).tolist())

        for bad_name in ["训练技术", "控制技术", "规划技术", "仿真技术"]:
            self.assertNotIn(bad_name, names)
        self.assertIn("碳化硅", names)
        self.assertIn("化学气相沉积", names)

    def test_long_narrative_candidate_name_is_compressed_or_filtered(self):
        raw_narrative = "钙钛矿薄膜在低温湿法制备过程中通过引入离子液体添加剂实现高稳定性"
        events = pd.DataFrame(
            [
                {
                    "id": "paper-long-name",
                    "subject": "研究团队",
                    "action": "提出",
                    "technology": ["钙钛矿薄膜"],
                    "technical_object": "钙钛矿薄膜",
                    "mechanism": "界面钝化",
                    "task": "提升稳定性",
                    "evidence_span": f"研究团队提出{raw_narrative}，显著提升器件稳定性。",
                    "observation_scopes": ["新型电子材料"],
                    "candidate_units": [
                        {
                            "raw_phrase": raw_narrative,
                            "raw_candidate_text": raw_narrative,
                            "mechanism_core": "界面钝化",
                            "mechanism_core_tokens": ["界面钝化"],
                            "object_modifier_tokens": [raw_narrative],
                            "task_constraint_tokens": ["稳定性"],
                            "method_modifier_tokens": ["低温湿法制备"],
                            "scope_names": ["新型电子材料"],
                            "source_extraction_mode": "domain_pack_test",
                        }
                    ],
                }
            ]
        )
        raw = pd.DataFrame(
            [
                {
                    "id": "paper-long-name",
                    "source_type": "paper",
                    "title": "钙钛矿薄膜界面钝化",
                    "text": f"研究团队提出{raw_narrative}，显著提升器件稳定性。",
                    "analysis_tech_field_name": "新型电子材料",
                }
            ]
        )

        candidates = build_candidate_forms(events, raw, domain_context=_electronic_materials_context())
        formed = candidates[candidates["candidate_stage"].astype(str).str.startswith("formed_candidate", na=False)]

        self.assertFalse(formed.empty)
        for name in formed["display_candidate_name"].fillna("").astype(str):
            self.assertLessEqual(len(name), 32)
            self.assertNotIn("通过", name)
            self.assertNotIn("实现", name)

    def test_filtered_candidate_keeps_single_event_evidence_for_quality_join(self):
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
                    "confidence": 0.8,
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
                    "date": "2026-06-01",
                }
            ]
        )

        candidates = build_candidate_forms(events, raw, domain_context=_battery_context())
        quality = build_event_quality_table(events, raw, domain_lexicon=build_domain_lexicon(_battery_context()))
        enriched = merge_event_quality_into_candidates(candidates, quality)
        shell_rows = enriched[enriched["raw_phrase"].astype(str) == "材料技术"]

        self.assertFalse(shell_rows.empty)
        row = shell_rows.iloc[0]
        self.assertEqual(row["mention_ids"], ["paper-shell"])
        self.assertEqual(row["evidence_items"][0]["id"], "paper-shell")
        self.assertGreater(float(row["candidate_evidence_quality"]), 0)

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

    def test_space_policy_does_not_treat_robot_default_object_slots_as_generic(self):
        policy = _space_manufacturing_policy()
        units = [
            {
                "raw_phrase": label,
                "raw_candidate_text": label,
                "canonical_candidate_name_en": label,
                "object_modifier_tokens": [label],
                "task_constraint_tokens": [],
                "data_modifier_tokens": [],
                "method_modifier_tokens": [],
                "domain_context": _space_manufacturing_context(),
            }
            for label in ["world model", "机械臂", "夹爪", "灵巧手"]
        ]

        for unit in units:
            with self.subTest(unit=unit["raw_phrase"]):
                profile = _cluster_specificity_profile([unit], policy=policy)

                self.assertNotEqual(profile["generic_cluster_risk"], "high")
                self.assertIn("specific_object_slot", profile["specificity_reason"])

    def test_space_policy_does_not_reject_legacy_robot_method_name_with_domain_anchor(self):
        context = _space_manufacturing_context()
        lexicon = build_domain_lexicon(context)
        policy = build_domain_candidate_policy(lexicon)
        unit = {
            "raw_phrase": "训练技术",
            "raw_candidate_text": "训练技术",
            "canonical_candidate_name_en": "训练技术",
            "mechanism_core": "训练",
            "object_modifier_tokens": ["太空3D打印"],
            "task_constraint_tokens": [],
            "data_modifier_tokens": [],
            "method_modifier_tokens": ["训练技术"],
            "domain_context": context,
        }

        self.assertFalse(
            _is_generic_method_only_display_name(
                "训练技术",
                unit,
                domain_lexicon=lexicon,
                policy=policy,
            )
        )

    def test_signal_generation_preserves_formed_candidate_aggregate_counts_for_weak_signals(self):
        forms = pd.DataFrame(
            [
                {
                    "id": "paper-sic-cvd",
                    "candidate_stage": "formed_candidate",
                    "display_candidate_name": "碳化硅化学气相沉积",
                    "canonical_candidate_name_en": "碳化硅 化学气相沉积",
                    "normalized_candidate_text": "碳化硅 化学气相沉积",
                    "raw_phrase": "碳化硅化学气相沉积",
                    "raw_candidate_text": "碳化硅化学气相沉积",
                    "candidate_cluster_id": "cand::sic-cvd",
                    "mechanism_core": "化学气相沉积",
                    "has_mechanism_core": True,
                    "has_non_scope_constraint": True,
                    "is_scope_internal_candidate": True,
                    "is_scope_echo": False,
                    "topic_granularity": "fine_grained_topic",
                    "display_tier": "weak_signal",
                    "cluster_evidence_count": 2,
                    "cluster_item_count": 2,
                    "source_count": 2,
                    "total_mentions": 2,
                    "source_types": ["paper", "patent"],
                    "mention_ids": ["paper-sic-cvd", "patent-sic-cvd"],
                    "mention_dates": ["2026-05-01", "2026-05-18"],
                    "evidence_titles": ["碳化硅外延层化学气相沉积", "碳化硅沉积工艺专利"],
                    "evidence_items": [
                        {"id": "paper-sic-cvd", "source_type": "paper", "title": "碳化硅外延层化学气相沉积"},
                        {"id": "patent-sic-cvd", "source_type": "patent", "title": "碳化硅沉积工艺专利"},
                    ],
                    "object_modifier_tokens": ["碳化硅"],
                    "task_constraint_tokens": ["缺陷密度"],
                    "method_modifier_tokens": ["化学气相沉积"],
                    "scope_names": ["新型电子材料"],
                    "primary_scope": "新型电子材料",
                    "source_type": "paper",
                }
            ]
        )
        raw = pd.DataFrame(
            [
                {
                    "id": "paper-sic-cvd",
                    "source_type": "paper",
                    "title": "碳化硅外延层化学气相沉积",
                    "text": "研究团队提出碳化硅外延层化学气相沉积工艺，降低缺陷密度。",
                    "analysis_tech_field_name": "新型电子材料",
                },
                {
                    "id": "patent-sic-cvd",
                    "source_type": "patent",
                    "title": "碳化硅沉积工艺专利",
                    "text": "专利公开碳化硅化学气相沉积设备和工艺参数。",
                    "analysis_tech_field_name": "新型电子材料",
                },
            ]
        )

        output = generate_candidate_outputs(forms, raw, domain_context=_electronic_materials_context())
        row = output["candidates_df"].iloc[0]

        self.assertEqual(row["signal_type"], "weak_signal")
        self.assertEqual(row["cluster_evidence_count"], 2)
        self.assertEqual(row["source_count"], 2)
        self.assertEqual(row["total_mentions"], 2)
        self.assertEqual(set(row["mention_ids"]), {"paper-sic-cvd", "patent-sic-cvd"})


if __name__ == "__main__":
    unittest.main()
