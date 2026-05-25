import os
import unittest

import pandas as pd

from src.extraction.candidate_former import build_candidate_forms
from src.extraction.event_extractor import _filter_events_for_doc, process_events
from src.extraction.event_schema import (
    WEAK_SIGNAL_EVENT_SCHEMA_VERSION,
    backfill_events_dataframe,
    event_schema_summary,
)
from src.validation.event_quality import (
    build_event_quality_table,
    merge_event_quality_into_candidates,
    merge_event_quality_into_events,
)


class EventExtractionRefactorTest(unittest.TestCase):
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
        self.assertTrue(str(events.loc[0, "evidence_span"]))
        self.assertGreater(float(events.loc[0, "confidence"]), 0)

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


if __name__ == "__main__":
    unittest.main()
