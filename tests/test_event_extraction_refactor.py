import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from src.core.pipeline import AnalysisPipeline
from src.domain.models import DomainContext, DomainPack
import src.extraction.event_extractor as event_extractor
from src.extraction.candidate_former import (
    _naturalized_topic_name,
    _topic_naturalness_reason,
    build_candidate_forms,
)
from src.extraction.event_extractor import (
    _filter_events_for_doc,
    _weak_signal_batch_prompt,
    _weak_signal_event_prompt,
    process_events,
)
from src.extraction.event_schema import (
    WEAK_SIGNAL_EVENT_SCHEMA_VERSION,
    backfill_events_dataframe,
    event_schema_summary,
    normalize_event_schema,
)
from src.scoring.signal_generator import generate_candidate_outputs
from src.scoring.topic_refiner import refine_research_scored_candidates
from src.validation.event_quality import (
    build_event_quality_table,
    merge_event_quality_into_candidates,
    merge_event_quality_into_events,
)


class EventExtractionRefactorTest(unittest.TestCase):
    def _signal_form_row(self, **overrides):
        row = {
            "id": "doc-1",
            "candidate_stage": "formed_candidate",
            "signal_type": "weak_signal",
            "display_candidate_name": "robot planning control",
            "canonical_candidate_name_en": "robot planning control",
            "raw_phrase": "robot planning control",
            "normalized_candidate_text": "robot planning control",
            "primary_scope": "humanoid robot",
            "scope_name": "humanoid robot",
            "scope_names": ["humanoid robot"],
            "mechanism_core": "planning",
            "mechanism_core_tokens": ["planning"],
            "task_constraint_tokens": ["navigation"],
            "object_modifier_tokens": ["robot"],
            "data_modifier_tokens": [],
            "method_modifier_tokens": [],
            "source_types": ["paper"],
            "source_count": 1,
            "cluster_evidence_count": 1,
            "total_mentions": 1,
            "org_count": 1,
            "is_scope_internal_candidate": True,
            "is_scope_echo": False,
            "has_mechanism_core": True,
            "has_non_scope_constraint": True,
            "topic_granularity": "fine_grained_topic",
            "display_tier": "weak_signal",
            "weak_signal_score": 9.0,
            "hotspot_score": 6.0,
            "score": 9.0,
        }
        row.update(overrides)
        return row

    def _domain_context(
        self,
        *,
        pack_id,
        field_name,
        aliases=None,
        technical_terms=None,
        mechanism_terms=None,
        exclude_terms=None,
    ):
        payload = {
            "schema_version": "domain_pack_v1",
            "pack_id": pack_id,
            "pack_name": field_name,
            "domain_identity": {
                "field_id": pack_id,
                "field_name": field_name,
                "out_of_scope_domains": exclude_terms or [],
            },
            "search_strategy": {
                "core_keywords": technical_terms or [],
                "synonyms": aliases or [],
                "exclude_terms": exclude_terms or [],
            },
            "observation_scopes": {
                "main_scope": field_name,
                "scope_aliases": aliases or [],
                "off_domain_anchor_terms": exclude_terms or [],
            },
            "candidate_formation": {
                "technical_object_types": technical_terms or [],
                "mechanism_types": mechanism_terms or [],
                "generic_terms": ["technology", "method", "system"],
                "shell_terms": ["technology", "method", "system"],
                "valid_candidate_patterns": [
                    {
                        "pattern_id": "object_mechanism",
                        "required_slots": ["technical_object", "mechanism"],
                    }
                ],
                "invalid_candidate_patterns": [
                    {
                        "pattern_id": "shell_only",
                        "reject_terms": ["technology"],
                        "max_specific_slot_count": 0,
                    }
                ],
            },
        }
        return DomainContext.from_pack(DomainPack.from_dict(payload))

    def test_backfill_marks_legacy_and_native_schema(self):
        events = pd.DataFrame(
            [
                {
                    "id": "legacy_doc",
                    "subject": "研究机构",
                    "action": "提出",
                    "technology": ["世界模型"],
                    "scene": "机器人",
                },
                {
                    "id": "native_doc",
                    "event_schema_version": WEAK_SIGNAL_EVENT_SCHEMA_VERSION,
                    "event_id": "native_evt_001",
                    "technical_object": "humanoid robot world model",
                    "mechanism": "planning",
                    "task": "manipulation",
                    "technology": ["world model"],
                },
            ]
        )

        backfilled = backfill_events_dataframe(events)
        summary = event_schema_summary(backfilled)

        self.assertEqual(set(backfilled["event_schema_version"]), {WEAK_SIGNAL_EVENT_SCHEMA_VERSION})
        self.assertIn("legacy_backfill", set(backfilled["schema_migration_mode"]))
        self.assertIn("native", set(backfilled["schema_migration_mode"]))
        self.assertEqual(summary["event_count"], 2)

    def test_event_quality_uses_v2_evidence_dimensions(self):
        raw = pd.DataFrame(
            [
                {
                    "id": "doc1",
                    "source_type": "paper",
                    "title": "Prototype tactile world model for humanoid manipulation",
                    "text": "A novel early prototype uses tactile sensor data and a world model for robot planning.",
                    "date": "2026-01-01",
                    "url": "https://example.com/paper",
                }
            ]
        )
        events = backfill_events_dataframe(
            pd.DataFrame(
                [
                    {
                        "id": "doc1",
                        "event_schema_version": WEAK_SIGNAL_EVENT_SCHEMA_VERSION,
                        "event_id": "doc1_evt_001",
                        "source_type": "paper",
                        "title": "Prototype tactile world model for humanoid manipulation",
                        "technology": ["world model", "robot planning"],
                        "technical_object": "humanoid robot tactile world model",
                        "mechanism": "world model planning",
                        "task": "manipulation",
                        "data_modality": ["tactile sensor data"],
                        "method": ["prototype"],
                        "weak_signal_reason": "early prototype and multimodal sensor trajectory",
                        "uncertainty": "early-stage evidence",
                        "evidence_span": "novel early prototype uses tactile sensor data and a world model for robot planning",
                        "confidence": 0.82,
                    }
                ]
            )
        )

        quality = build_event_quality_table(events, raw)
        enriched_events = merge_event_quality_into_events(events, quality)
        candidates = pd.DataFrame(
            [
                {
                    "candidate_id": "cand1",
                    "mention_ids": ["doc1"],
                    "evidence_items": [{"id": "doc1", "title": "Prototype tactile world model"}],
                    "weak_signal_score": 7.0,
                }
            ]
        )
        enriched_candidates = merge_event_quality_into_candidates(candidates, quality)

        self.assertGreaterEqual(float(quality.loc[0, "evidence_span_score"]), 8.0)
        self.assertGreaterEqual(float(quality.loc[0, "confidence_score"]), 8.0)
        self.assertGreaterEqual(float(quality.loc[0, "foresight_relevance_score"]), 7.0)
        self.assertIn("foresight_relevance_score", enriched_events.columns)
        self.assertGreater(float(enriched_candidates.loc[0, "candidate_evidence_foresight_relevance"]), 0)
        self.assertGreaterEqual(int(enriched_candidates.loc[0, "high_foresight_evidence_count"]), 1)

    def test_pipeline_scores_events_during_extraction_stage(self):
        raw = pd.DataFrame(
            [
                {
                    "id": "doc1",
                    "source_type": "paper",
                    "title": "Prototype tactile world model for humanoid manipulation",
                    "text": "A novel early prototype uses tactile sensor data and a world model for robot planning.",
                    "date": "2026-01-01",
                    "url": "https://example.com/paper",
                }
            ]
        )
        events = backfill_events_dataframe(
            pd.DataFrame(
                [
                    {
                        "id": "doc1",
                        "event_id": "doc1_evt_001",
                        "source_type": "paper",
                        "title": "Prototype tactile world model for humanoid manipulation",
                        "technology": ["world model", "robot planning"],
                        "technical_object": "humanoid robot tactile world model",
                        "mechanism": "world model planning",
                        "task": "manipulation",
                        "data_modality": ["tactile sensor data"],
                        "method": ["prototype"],
                        "weak_signal_reason": "early prototype and multimodal sensor trajectory",
                        "evidence_span": "novel early prototype uses tactile sensor data and a world model for robot planning",
                        "confidence": 0.82,
                    }
                ]
            )
        )

        pipeline = AnalysisPipeline()
        enriched_events = pipeline._score_events_during_extraction(events, raw)

        self.assertIn("event_quality_score", enriched_events.columns)
        self.assertFalse(pipeline.latest_event_quality_df.empty)
        self.assertGreater(float(enriched_events.loc[0, "event_quality_score"]), 0)

    def test_schema_fields_generate_candidate_units(self):
        events = backfill_events_dataframe(
            pd.DataFrame(
                [
                    {
                        "id": "doc1",
                        "event_schema_version": WEAK_SIGNAL_EVENT_SCHEMA_VERSION,
                        "event_id": "doc1_evt_001",
                        "source_extraction_mode": "api",
                        "source_type": "paper",
                        "technology": ["world model"],
                        "technical_object": "humanoid robot tactile world model",
                        "mechanism": "world model planning",
                        "task": "dexterous manipulation",
                        "data_modality": ["tactile sensor data"],
                        "method": ["self-supervised learning"],
                        "observation_scopes": ["humanoid robot"],
                        "evidence_span": "humanoid robot tactile world model for dexterous manipulation",
                        "confidence": 0.8,
                    }
                ]
            )
        )
        raw = pd.DataFrame(
            [
                {
                    "id": "doc1",
                    "source_type": "paper",
                    "title": "Tactile world model",
                    "text": "humanoid robot tactile world model for dexterous manipulation",
                    "date": "2026",
                }
            ]
        )

        candidates = build_candidate_forms(events, raw)

        self.assertFalse(candidates.empty)
        self.assertTrue(
            any(str(value).endswith("_schema") for value in candidates["source_extraction_mode"])
        )

    def test_local_process_events_outputs_v2_schema(self):
        raw = pd.DataFrame(
            [
                {
                    "id": "doc1",
                    "source_type": "paper",
                    "title": "Robot world model prototype",
                    "text": (
                        "In 2026, a university lab proposed a novel humanoid robot "
                        "world model prototype using tactile sensor data for planning."
                    ),
                    "date": "2026",
                }
            ]
        )

        events = process_events(raw, use_api=False, refresh_cache=True, cache_path=None)

        self.assertEqual(events.loc[0, "event_schema_version"], WEAK_SIGNAL_EVENT_SCHEMA_VERSION)
        self.assertEqual(events.loc[0, "source_extraction_mode"], "local")
        self.assertEqual(events.loc[0, "subject"], "未知")
        self.assertTrue(str(events.loc[0, "evidence_span"]))
        self.assertGreater(float(events.loc[0, "confidence"]), 0)

    def test_cached_events_renormalize_subject_with_source_row(self):
        raw = pd.DataFrame(
            [
                {
                    "id": "doc-cache",
                    "source_type": "patent",
                    "title": "双驱动齿条控制的柔性连续体穿刺活检机器人机构",
                    "text": "上海海事大学提出一种柔性连续体穿刺活检机器人机构。",
                    "org": "上海海事大学",
                    "date": "2026",
                }
            ]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "events.json"
            cache_path.write_text(
                json.dumps(
                    {
                        "metadata": {"source_ids": ["doc-cache"]},
                        "events": [
                            {
                                "id": "doc-cache",
                                "subject": "本发明",
                                "action": "高速跑酷导航",
                                "technology": ["机器人"],
                                "scene": "机器人",
                                "time": "2026",
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            events = process_events(raw, use_api=False, refresh_cache=False, cache_path=str(cache_path))

        self.assertEqual(events.loc[0, "subject"], "上海海事大学")
        self.assertEqual(events.loc[0, "action"], "提出")

    def test_pipeline_standardizes_patent_applicant_fields(self):
        pipeline = AnalysisPipeline()
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "patent_sample.csv"
            pd.DataFrame(
                [
                    {
                        "id": "p1",
                        "source_type": "patent",
                        "title_cn": "一种机器人控制方法",
                        "abstract_cn": "本发明公开了一种机器人控制方法。",
                        "applicants_norm": "北京可以科技有限公司",
                        "inventors": "张三; 李四",
                        "public_date": "2026-01-01",
                    }
                ]
            ).to_csv(data_path, index=False, encoding="utf-8-sig")

            raw = pipeline._load_data(data_path, sample_size=0)

        self.assertEqual(raw.loc[0, "org"], "北京可以科技有限公司")
        self.assertEqual(raw.loc[0, "authors"], "张三; 李四")
        self.assertEqual(raw.loc[0, "source_type"], "patent")

    def test_pipeline_cached_events_use_source_metadata_for_subject(self):
        pipeline = AnalysisPipeline()
        raw = pd.DataFrame(
            [
                {
                    "id": "patent:cached",
                    "source_type": "patent",
                    "title": "一种机器人控制方法",
                    "text": "北京可以科技有限公司提出一种机器人控制方法。",
                    "org": "北京可以科技有限公司",
                    "date": "2026-01-01",
                }
            ]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            cache_path = pipeline._event_cache_path(raw, cache_dir)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(
                    {
                        "metadata": {"source_ids": ["patent:cached"]},
                        "events": [
                            {
                                "id": "patent:cached",
                                "subject": "本发明",
                                "action": "高速跑酷导航",
                                "technology": ["机器人"],
                                "scene": "机器人",
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            events = pipeline._extract_events(raw, use_cache=True, cache_dir=cache_dir)

        self.assertEqual(events.loc[0, "subject"], "北京可以科技有限公司")
        self.assertEqual(events.loc[0, "action"], "提出")

    def test_candidate_forms_use_selected_domain_as_generic_scope(self):
        events = pd.DataFrame(
            [
                {
                    "id": "paper:material",
                    "subject": "复旦大学团队",
                    "action": "提出",
                    "technology": ["钙钛矿薄膜"],
                    "technical_object": "钙钛矿薄膜",
                    "mechanism": "界面钝化",
                    "task": "提升稳定性",
                    "evidence_span": "提出一种钙钛矿薄膜界面钝化策略以提升器件稳定性",
                    "observation_scopes": [],
                }
            ]
        )
        raw = pd.DataFrame(
            [
                {
                    "id": "paper:material",
                    "source_type": "paper",
                    "title": "钙钛矿薄膜界面钝化策略",
                    "text": "复旦大学团队提出一种钙钛矿薄膜界面钝化策略以提升器件稳定性。",
                    "analysis_tech_field_name": "新型电子材料",
                    "date": "2026-01-01",
                }
            ]
        )

        candidates = build_candidate_forms(events, raw)

        self.assertFalse(candidates.empty)
        self.assertIn("新型电子材料", set(candidates["scope_name"].astype(str)))
        names = " ".join(candidates["display_candidate_name"].fillna("").astype(str).tolist())
        self.assertIn("钙钛矿", names)
        self.assertNotIn("机器人", names)
        self.assertNotIn("待收口", names)

    def test_candidate_review_hint_stays_out_of_display_name(self):
        unit = {
            "primary_scope": "新型电子材料",
            "scope_names": ["新型电子材料"],
            "mechanism_core": "界面钝化",
            "mechanism_core_tokens": ["界面钝化"],
            "task_constraint_tokens": ["高温稳定性"],
            "object_modifier_tokens": [],
            "data_modifier_tokens": [],
            "method_modifier_tokens": [],
            "scene_tokens": [],
            "has_mechanism_core": True,
            "has_non_scope_constraint": True,
        }

        name = _naturalized_topic_name(unit, "高温稳定性界面钝化技术")
        reason = _topic_naturalness_reason(unit, name)

        self.assertNotIn("待收口", name)
        self.assertIn("收口", reason)

    def test_signal_generation_accepts_source_id_only_raw_data_and_preserves_upstream_signal_type(self):
        forms = pd.DataFrame([self._signal_form_row(id="paper:source-1")])
        raw = pd.DataFrame(
            [
                {
                    "source_id": "paper:source-1",
                    "source_type": "paper",
                    "title": "Robot planning control",
                    "text": "Robot planning control for humanoid navigation.",
                }
            ]
        )

        output = generate_candidate_outputs(forms, raw)
        signals = output["candidates_df"]

        self.assertEqual(len(signals), 1)
        self.assertEqual(signals.loc[0, "signal_type"], "weak_signal")
        self.assertEqual(output["diagnostics"]["id_filled_from_source_id_count"], 1)

    def test_topic_refiner_keeps_rule_technical_name_when_llm_returns_application_surface(self):
        scored = pd.DataFrame(
            [
                self._signal_form_row(
                    id="robot-app-doc",
                    display_candidate_name="机器人视觉控制技术",
                    rule_display_candidate_name="机器人视觉控制技术",
                    topic_summary_name="机器人视觉控制技术",
                    candidate_cluster_id="cluster-robot-visual-control",
                    source_count=2,
                    cluster_evidence_count=3,
                    total_mentions=3,
                    mechanism_core="control",
                    mechanism_core_tokens=["control"],
                    task_constraint_tokens=["inspection"],
                    object_modifier_tokens=["robot"],
                    data_modifier_tokens=["visual"],
                    raw_phrase="robot visual control",
                    raw_phrase_example="robot visual control for inspection",
                )
            ]
        )
        llm_payload = [
            {
                "candidate_key": "cluster-robot-visual-control",
                "refined_topic_name": "机器人巡检应用",
                "small_topic_judgment": "small_topic",
                "small_topic_type": "small_application",
                "small_topic_pattern": "application+mechanism",
                "reason": "应用场景更自然",
            }
        ]

        with patch("src.scoring.topic_refiner.get_provider_and_client", return_value=("mock", object())), patch(
            "src.scoring.topic_refiner.chat_text",
            return_value=(json.dumps(llm_payload, ensure_ascii=False), {"prompt_tokens": 1, "completion_tokens": 1}, None),
        ), patch("src.scoring.topic_refiner.record_call"):
            refined = refine_research_scored_candidates(scored, top_k=5, batch_size=5)

        self.assertEqual(refined.loc[0, "display_candidate_name"], "机器人视觉控制技术")
        self.assertEqual(refined.loc[0, "rule_display_candidate_name"], "机器人视觉控制技术")
        self.assertEqual(refined.loc[0, "llm_refined_topic_name"], "机器人巡检应用")
        self.assertEqual(refined.loc[0, "topic_summary_name"], "机器人巡检应用")

    def test_signal_generation_uses_rule_technical_name_over_application_surface(self):
        forms = pd.DataFrame(
            [
                self._signal_form_row(
                    id="robot-app-doc",
                    display_candidate_name="机器人巡检应用",
                    rule_display_candidate_name="机器人视觉控制技术",
                    topic_summary_name="机器人巡检应用",
                    llm_refined_topic_name="机器人巡检应用",
                    llm_small_topic_type="small_application",
                    llm_small_topic_pattern="application+mechanism",
                    mechanism_core="control",
                    mechanism_core_tokens=["control"],
                    task_constraint_tokens=["inspection"],
                    object_modifier_tokens=["robot"],
                    data_modifier_tokens=["visual"],
                    source_count=2,
                    cluster_evidence_count=3,
                    total_mentions=3,
                    candidate_cluster_id="cluster-robot-visual-control",
                )
            ]
        )
        raw = pd.DataFrame(
            [
                {
                    "id": "robot-app-doc",
                    "source_type": "paper",
                    "title": "Robot visual control for inspection",
                    "text": "Robot visual control technology supports inspection scenarios.",
                    "analysis_tech_field_name": "humanoid robot",
                }
            ]
        )

        output = generate_candidate_outputs(forms, raw)
        signals = output["candidates_df"]

        self.assertEqual(len(signals), 1)
        self.assertEqual(signals.loc[0, "display_candidate_name"], "机器人视觉控制技术")
        self.assertEqual(signals.loc[0, "tech_name"], "机器人视觉控制技术")
        self.assertEqual(signals.loc[0, "final_research_object_name"], "机器人视觉控制技术")
        self.assertEqual(signals.loc[0, "topic_summary_name"], "机器人巡检应用")
        self.assertEqual(signals.loc[0, "signal_type"], "weak_signal")
        self.assertEqual(output["diagnostics"]["application_surface_rewritten_count"], 1)

    def test_application_only_surface_does_not_pass_as_technical_weak_signal(self):
        forms = pd.DataFrame(
            [
                self._signal_form_row(
                    id="robot-only-app-doc",
                    display_candidate_name="机器人巡检应用",
                    canonical_candidate_name_en="robot inspection application",
                    raw_phrase="robot inspection application",
                    normalized_candidate_text="robot inspection application",
                    rule_display_candidate_name="",
                    topic_summary_name="机器人巡检应用",
                    mechanism_core="",
                    mechanism_core_tokens=[],
                    has_mechanism_core=False,
                    has_non_scope_constraint=True,
                    task_constraint_tokens=["inspection"],
                    object_modifier_tokens=["robot"],
                    data_modifier_tokens=[],
                    source_count=2,
                    cluster_evidence_count=3,
                    total_mentions=3,
                )
            ]
        )
        raw = pd.DataFrame(
            [
                {
                    "id": "robot-only-app-doc",
                    "source_type": "news",
                    "title": "Robot inspection application",
                    "text": "Robot inspection application appears in deployment reports.",
                    "analysis_tech_field_name": "humanoid robot",
                }
            ]
        )

        output = generate_candidate_outputs(forms, raw)
        signals = output["candidates_df"]

        self.assertEqual(len(signals), 1)
        self.assertNotEqual(signals.loc[0, "signal_type"], "weak_signal")
        self.assertEqual(signals.loc[0, "technical_name_issue"], "application_only_surface")
        self.assertEqual(output["diagnostics"]["application_only_rejected_count"], 1)

    def test_domain_pack_terms_prevent_robot_domain_from_being_treated_as_off_domain(self):
        forms = pd.DataFrame([self._signal_form_row(id="robot-doc")])
        raw = pd.DataFrame(
            [
                {
                    "id": "robot-doc",
                    "source_type": "paper",
                    "title": "Robot planning control",
                    "text": "Robot planning control for humanoid navigation.",
                    "analysis_tech_field_name": "humanoid robot",
                }
            ]
        )
        domain_context = self._domain_context(
            pack_id="humanoid_robot",
            field_name="humanoid robot",
            aliases=["humanoid robot", "bipedal robot"],
            technical_terms=["humanoid robot", "robot"],
            mechanism_terms=["planning", "control", "navigation"],
        )

        signals = generate_candidate_outputs(forms, raw, domain_context=domain_context)["candidates_df"]

        self.assertEqual(len(signals), 1)
        self.assertEqual(signals.loc[0, "display_candidate_name"], "robot planning control")
        self.assertEqual(signals.loc[0, "signal_type"], "weak_signal")

    def test_domain_pack_scope_aliases_match_space_manufacturing_variants(self):
        forms = pd.DataFrame(
            [
                self._signal_form_row(
                    id="space-doc",
                    display_candidate_name="orbital additive manufacturing",
                    canonical_candidate_name_en="orbital additive manufacturing",
                    raw_phrase="orbital additive manufacturing",
                    normalized_candidate_text="orbital additive manufacturing",
                    primary_scope="orbital manufacturing",
                    scope_name="orbital manufacturing",
                    scope_names=["orbital manufacturing"],
                    mechanism_core="additive manufacturing",
                    mechanism_core_tokens=["additive manufacturing"],
                    task_constraint_tokens=["on-orbit assembly"],
                    object_modifier_tokens=["space station"],
                )
            ]
        )
        raw = pd.DataFrame(
            [
                {
                    "id": "space-doc",
                    "source_type": "paper",
                    "title": "Orbital additive manufacturing",
                    "text": "A microgravity additive manufacturing route supports on-orbit assembly.",
                    "analysis_tech_field_name": "space manufacturing",
                }
            ]
        )
        domain_context = self._domain_context(
            pack_id="space_manufacturing",
            field_name="space manufacturing",
            aliases=["orbital manufacturing", "in-space manufacturing", "microgravity manufacturing"],
            technical_terms=[
                "space manufacturing",
                "orbital manufacturing",
                "microgravity manufacturing",
                "space station",
            ],
            mechanism_terms=["additive manufacturing", "on-orbit assembly"],
        )

        signals = generate_candidate_outputs(forms, raw, domain_context=domain_context)["candidates_df"]

        self.assertEqual(len(signals), 1)
        self.assertEqual(signals.loc[0, "analysis_tech_field_name"], "space manufacturing")

    def test_signal_generation_filters_to_selected_analysis_domain(self):
        forms = pd.DataFrame(
            [
                {
                    "id": "material-doc",
                    "candidate_stage": "formed_candidate_strong",
                    "display_candidate_name": "钙钛矿薄膜界面钝化",
                    "canonical_candidate_name_en": "perovskite interface passivation",
                    "raw_phrase": "钙钛矿薄膜 界面钝化",
                    "normalized_candidate_text": "钙钛矿薄膜界面钝化",
                    "primary_scope": "新型电子材料",
                    "scope_name": "新型电子材料",
                    "scope_names": ["新型电子材料"],
                    "mechanism_core": "界面钝化",
                    "mechanism_core_tokens": ["界面钝化"],
                    "task_constraint_tokens": ["提升稳定性"],
                    "object_modifier_tokens": ["钙钛矿薄膜"],
                    "data_modifier_tokens": [],
                    "method_modifier_tokens": ["掺杂改性"],
                    "source_types": ["paper"],
                    "source_count": 1,
                    "cluster_evidence_count": 1,
                    "is_scope_internal_candidate": True,
                    "is_scope_echo": False,
                    "has_mechanism_core": True,
                    "has_non_scope_constraint": True,
                    "weak_signal_score": 8.0,
                },
                {
                    "id": "robot-doc",
                    "candidate_stage": "formed_candidate_strong",
                    "display_candidate_name": "机器人规划技术",
                    "canonical_candidate_name_en": "robot planning",
                    "raw_phrase": "robot planning",
                    "normalized_candidate_text": "robot planning",
                    "primary_scope": "人形机器人",
                    "scope_name": "人形机器人",
                    "scope_names": ["人形机器人"],
                    "mechanism_core": "planning",
                    "mechanism_core_tokens": ["planning"],
                    "task_constraint_tokens": ["navigation"],
                    "object_modifier_tokens": ["robot"],
                    "data_modifier_tokens": [],
                    "method_modifier_tokens": [],
                    "source_types": ["paper"],
                    "source_count": 1,
                    "cluster_evidence_count": 1,
                    "is_scope_internal_candidate": True,
                    "is_scope_echo": False,
                    "has_mechanism_core": True,
                    "has_non_scope_constraint": True,
                    "weak_signal_score": 9.0,
                },
                {
                    "id": "generic-planning-doc",
                    "candidate_stage": "formed_candidate_strong",
                    "display_candidate_name": "规划技术",
                    "canonical_candidate_name_en": "planning",
                    "raw_phrase": "planning",
                    "normalized_candidate_text": "planning",
                    "primary_scope": "新型电子材料",
                    "scope_name": "新型电子材料",
                    "scope_names": ["新型电子材料"],
                    "mechanism_core": "planning",
                    "mechanism_core_tokens": ["planning"],
                    "task_constraint_tokens": ["planning"],
                    "object_modifier_tokens": [],
                    "data_modifier_tokens": [],
                    "method_modifier_tokens": [],
                    "source_types": ["paper"],
                    "source_count": 1,
                    "cluster_evidence_count": 1,
                    "is_scope_internal_candidate": True,
                    "is_scope_echo": False,
                    "has_mechanism_core": True,
                    "has_non_scope_constraint": True,
                    "weak_signal_score": 7.5,
                },
            ]
        )
        raw = pd.DataFrame(
            [
                {
                    "id": "material-doc",
                    "source_type": "paper",
                    "title": "钙钛矿薄膜界面钝化材料",
                    "text": "钙钛矿薄膜界面钝化材料提升器件稳定性。",
                    "analysis_tech_field_name": "新型电子材料",
                },
                {
                    "id": "robot-doc",
                    "source_type": "paper",
                    "title": "Robot planning",
                    "text": "Robot planning for navigation.",
                    "analysis_tech_field_name": "新型电子材料",
                },
                {
                    "id": "generic-planning-doc",
                    "source_type": "paper",
                    "title": "Planning control policy",
                    "text": "Planning control policy for navigation.",
                    "analysis_tech_field_name": "新型电子材料",
                },
            ]
        )

        signals = generate_candidate_outputs(forms, raw)["candidates_df"]

        names = " ".join(signals["display_candidate_name"].fillna("").astype(str).tolist())
        self.assertIn("钙钛矿", names)
        self.assertNotIn("机器人", names)
        self.assertNotIn("规划技术", names)
        self.assertEqual(set(signals["analysis_tech_field_name"].astype(str)), {"新型电子材料"})

    def test_non_robot_domain_disables_robot_family_normalization(self):
        forms = pd.DataFrame(
            [
                {
                    "id": "material-planning",
                    "candidate_stage": "formed_candidate",
                    "display_candidate_name": "材料工艺规划",
                    "canonical_candidate_name_en": "planning",
                    "raw_phrase": "planning",
                    "normalized_candidate_text": "planning",
                    "primary_scope": "新型电子材料",
                    "scope_name": "新型电子材料",
                    "scope_names": ["新型电子材料"],
                    "mechanism_core": "planning",
                    "mechanism_core_tokens": ["planning"],
                    "task_constraint_tokens": ["工艺优化"],
                    "object_modifier_tokens": ["薄膜材料"],
                    "data_modifier_tokens": [],
                    "method_modifier_tokens": [],
                    "source_types": ["paper"],
                    "source_count": 1,
                    "cluster_evidence_count": 1,
                    "is_scope_internal_candidate": True,
                    "has_mechanism_core": True,
                    "has_non_scope_constraint": True,
                }
            ]
        )
        raw = pd.DataFrame(
            [
                {
                    "id": "material-planning",
                    "source_type": "paper",
                    "title": "薄膜材料工艺规划",
                    "text": "面向薄膜材料的工艺规划与稳定性优化。",
                    "analysis_tech_field_name": "新型电子材料",
                }
            ]
        )

        signals = generate_candidate_outputs(forms, raw)["candidates_df"]

        self.assertFalse(bool(signals.loc[0, "family_matched"]))
        self.assertEqual(signals.loc[0, "family_id"], "")
        self.assertNotIn("robot", str(signals.loc[0, "canonical_term"]).lower())

    def test_configured_local_loading_does_not_fallback_to_default_samples(self):
        pipeline = AnalysisPipeline()
        source_config = {"counts": {"missing_source": 3}}

        loaded = pipeline._load_data(None, sample_size=3, source_config=source_config)

        self.assertTrue(loaded.empty)

    def test_placeholder_values_are_cleaned_across_event_fields(self):
        normalized = normalize_event_schema(
            {
                "id": "string",
                "event_id": "string",
                "subject": "上海交通大学团队",
                "action": "提出",
                "technology": ["string"],
                "scene": "string",
                "time": "text",
                "technical_object": "string",
                "mechanism": "占位符",
                "task": "待填",
                "data_modality": ["string", "3D"],
                "method": "placeholder, 掺杂改性",
                "capability_change": "sample",
                "problem_solved": "example",
                "weak_signal_reason": "string",
                "weak_signal_reasons": ["string", "具体材料对象、机制和性能任务同时出现"],
                "candidate_units": [
                    {
                        "mechanism_core": "string",
                        "mechanism_core_tokens": ["string", "界面钝化"],
                        "has_mechanism_core": True,
                        "object_like_score": 0.7,
                    }
                ],
            },
            source_row={"id": "doc-real", "title": "钙钛矿薄膜界面钝化材料"},
        )

        self.assertEqual(normalized["id"], "doc-real")
        self.assertNotEqual(normalized["event_id"], "string")
        self.assertEqual(normalized["technology"], ["未知"])
        self.assertEqual(normalized["scene"], "未知")
        self.assertEqual(normalized["time"], "未知")
        self.assertEqual(normalized["technical_object"], "")
        self.assertEqual(normalized["mechanism"], "")
        self.assertEqual(normalized["task"], "")
        self.assertEqual(normalized["data_modality"], ["3D"])
        self.assertEqual(normalized["method"], ["掺杂改性"])
        self.assertEqual(normalized["weak_signal_reason"], "具体材料对象、机制和性能任务同时出现")
        self.assertEqual(normalized["candidate_units"][0]["mechanism_core_tokens"], ["界面钝化"])
        self.assertNotIn("mechanism_core", normalized["candidate_units"][0])

    def test_llm_prompt_does_not_use_string_placeholders_in_schema_example(self):
        prompts = [
            _weak_signal_event_prompt("上海交通大学团队提出钙钛矿薄膜界面钝化材料。"),
            _weak_signal_batch_prompt("1. 上海交通大学团队提出钙钛矿薄膜界面钝化材料。", 1),
        ]
        forbidden = [
            '"technical_object": "string"',
            '"mechanism": "string"',
            '"task": "string"',
            '"capability_change": "string"',
            '"weak_signal_reason": "string"',
        ]
        for prompt in prompts:
            for bad_value in forbidden:
                self.assertNotIn(bad_value, prompt)

    def test_empty_batch_retry_uses_batch_extraction_model_for_single_calls(self):
        model_name = "Qwen/Qwen2.5-14B-Instruct"
        responses = iter(
            [
                "[]",
                json.dumps(
                    [
                        {
                            "event_id": "evt_1",
                            "subject": "上海交通大学团队",
                            "action": "提出",
                            "technology": ["钙钛矿薄膜"],
                            "technical_object": "钙钛矿薄膜界面钝化材料",
                            "mechanism": "界面钝化",
                            "task": "提升薄膜稳定性",
                            "capability_change": "增强器件稳定性",
                            "weak_signal_reason": "早期材料路线出现",
                            "uncertainty": "仍需验证",
                            "evidence_span": "提出钙钛矿薄膜界面钝化材料",
                            "confidence": 0.82,
                        }
                    ],
                    ensure_ascii=False,
                ),
            ]
        )
        called_models = []

        def fake_chat_text(*args, **kwargs):
            called_models.append(kwargs["model"])
            return next(responses), {"prompt_tokens": 0, "completion_tokens": 0}, {}

        with patch.object(event_extractor, "get_provider_and_client", return_value=("api", object())):
            with patch.object(event_extractor, "chat_text", side_effect=fake_chat_text):
                events = event_extractor._extract_batch_worker(
                    ["上海交通大学团队提出钙钛矿薄膜界面钝化材料。"],
                    10,
                    1,
                    1,
                    model_name,
                )

        self.assertEqual(called_models, [model_name, model_name])
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["_source_text_index"], 10)

    def test_event_filter_applies_confidence_and_max_limit(self):
        previous_max = os.environ.get("EVENT_EXTRACTION_MAX_EVENTS_PER_DOC")
        previous_min = os.environ.get("EVENT_EXTRACTION_MIN_CONFIDENCE")
        try:
            os.environ["EVENT_EXTRACTION_MAX_EVENTS_PER_DOC"] = "2"
            os.environ["EVENT_EXTRACTION_MIN_CONFIDENCE"] = "0.5"
            events = [
                {"event_id": "e1", "confidence": 0.9, "evidence_span": "a"},
                {"event_id": "e2", "confidence": 0.2, "evidence_span": "b"},
                {"event_id": "e3", "confidence": 0.8, "evidence_span": "c"},
                {"event_id": "e4", "confidence": 0.7, "evidence_span": "d"},
            ]

            selected, filtered, trimmed = _filter_events_for_doc(events)

            self.assertEqual([event["event_id"] for event in selected], ["e1", "e3"])
            self.assertEqual(filtered, 1)
            self.assertEqual(trimmed, 1)
        finally:
            if previous_max is None:
                os.environ.pop("EVENT_EXTRACTION_MAX_EVENTS_PER_DOC", None)
            else:
                os.environ["EVENT_EXTRACTION_MAX_EVENTS_PER_DOC"] = previous_max
            if previous_min is None:
                os.environ.pop("EVENT_EXTRACTION_MIN_CONFIDENCE", None)
            else:
                os.environ["EVENT_EXTRACTION_MIN_CONFIDENCE"] = previous_min

    def test_subject_strict_cleaning_and_metadata_fallback(self):
        """subject 只能保留真实主体，代词和技术对象短语回退为未知"""
        for bad_subject in ["我们", "本文", "日本科学家", "Japanese scientists", "运动规划系统", "RefAlign world model"]:
            normalized = normalize_event_schema(
                {
                    "subject": bad_subject,
                    "action": "提出",
                    "technology": ["world model"],
                },
                source_row={
                    "title": "Robot world model prototype",
                    "text": "We propose RefAlign, a robot world model prototype.",
                },
            )
            self.assertEqual(normalized["subject"], "未知")

        valid = normalize_event_schema(
            {
                "subject": "哈工大团队",
                "action": "提出",
                "technology": ["双足控制"],
            },
            source_row={},
        )
        self.assertEqual(valid["subject"], "哈工大团队")

        fallback = normalize_event_schema(
            {
                "subject": "本发明",
                "action": "提出",
                "technology": ["机器人"],
            },
            source_row={
                "org": "上海海事大学",
                "title": "双驱动齿条控制的柔性连续体穿刺活检机器人机构",
                "text": "本发明公开了一种柔性连续体穿刺活检机器人机构。",
            },
        )
        self.assertEqual(fallback["subject"], "上海海事大学")

        context_fallback = normalize_event_schema(
            {
                "subject": "研究人员",
                "action": "提出",
                "technology": ["world model"],
            },
            source_row={
                "title": "东京大学团队提出机器人世界模型",
                "text": "东京大学团队提出一种用于机器人高速移动的世界模型。",
            },
        )
        self.assertEqual(context_fallback["subject"], "东京大学")

    def test_action_cleaning(self):
        """测试对不符合定义的 action（技术类目/名词）进行清洗和回退"""

        # 1. 含有技术词汇且过长的错误 action
        event = {
            "subject": "哈工大团队",
            "action": "全身反应规划控制",
            "technology": ["双足控制"]
        }
        source_row = {
            "title": "一种新型人形机器人运动规划与控制设计方案",
            "text": "本发明设计并实现了一种人形机器人动作生成方法。"
        }
        normalized = normalize_event_schema(event, source_row=source_row)
        # 应该匹配到 title 中的 "设计" (因为 "设计" 是标准研发动词之一)
        self.assertEqual(normalized["action"], "设计")

        # 2. 纯技术词汇无对应动词的 fallback
        event_fall = {
            "subject": "研究人员",
            "action": "运动规划系统",
            "technology": ["运动规划"]
        }
        normalized_fall = normalize_event_schema(event_fall, source_row={})
        self.assertEqual(normalized_fall["action"], "研发")

        event_task_phrase = {
            "subject": "东京大学团队",
            "action": "高速跑酷导航",
            "technology": ["world model"],
        }
        normalized_task = normalize_event_schema(
            event_task_phrase,
            source_row={
                "title": "东京大学团队提出高速跑酷导航世界模型",
                "text": "东京大学团队提出一种支持高速跑酷导航的机器人世界模型。",
            },
        )
        self.assertEqual(normalized_task["action"], "提出")

if __name__ == "__main__":
    unittest.main()
