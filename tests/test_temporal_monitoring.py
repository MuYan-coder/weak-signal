import unittest

import pandas as pd

from src.validation.temporal_validator import build_temporal_validation_table


class TemporalMonitoringTests(unittest.TestCase):
    def test_rising_signal_gets_high_monitoring_priority(self):
        candidates = pd.DataFrame(
            [
                {
                    "candidate_id": "c1",
                    "display_candidate_name": "adaptive tactile policy",
                    "evidence_items": [
                        {"id": "e1", "date": "2024-01-15", "source_type": "paper", "org": "Lab A", "event_quality_score": 8.0},
                        {"id": "e2", "date": "2025-03-10", "source_type": "paper", "org": "Lab A", "event_quality_score": 8.5},
                        {"id": "e3", "date": "2025-06-12", "source_type": "news", "org": "Company B", "event_quality_score": 7.8},
                    ],
                    "source_count": 2,
                    "cluster_evidence_count": 3,
                }
            ]
        )

        temporal = build_temporal_validation_table(
            candidates,
            observation_months=12,
            validation_months=12,
            min_dated_evidence=2,
        )

        self.assertEqual(temporal.loc[0, "monitoring_priority"], "high")
        self.assertIn("高频跟踪", temporal.loc[0, "monitoring_action"])
        self.assertTrue(temporal.loc[0, "next_observation_window_start"])
        self.assertTrue(temporal.loc[0, "next_observation_window_end"])

    def test_missing_dates_are_retained_for_date_backfill(self):
        candidates = pd.DataFrame(
            [
                {
                    "candidate_id": "c2",
                    "display_candidate_name": "low data manipulation planner",
                    "evidence_items": [{"id": "e4", "source_type": "report", "event_quality_score": 6.0}],
                }
            ]
        )

        temporal = build_temporal_validation_table(candidates)

        self.assertEqual(temporal.loc[0, "temporal_validation_tier"], "insufficient_date")
        self.assertEqual(temporal.loc[0, "monitoring_priority"], "needs_dates")
        self.assertIn("补齐来源日期", temporal.loc[0, "monitoring_action"])

    def test_validate_temporal_filters_only_weak_signals(self):
        from src.core.pipeline import AnalysisPipeline
        pipeline = AnalysisPipeline()

        # Create a candidate DataFrame with mixed stages/signal types
        candidates = pd.DataFrame([
            {
                "candidate_id": "c1",
                "display_candidate_name": "weak signal candidate",
                "signal_type": "weak_signal",
                "candidate_stage": "formed_candidate_strong",
                "evidence_items": [
                    {"id": "e1", "date": "2024-01-15", "source_type": "paper", "org": "Lab A", "event_quality_score": 8.0},
                    {"id": "e2", "date": "2025-03-10", "source_type": "paper", "org": "Lab A", "event_quality_score": 8.5},
                ],
                "source_count": 2,
                "cluster_evidence_count": 2,
            },
            {
                "candidate_id": "c2",
                "display_candidate_name": "other candidate",
                "signal_type": "other",
                "candidate_stage": "formed_candidate",
                "evidence_items": [
                    {"id": "e3", "date": "2024-01-15", "source_type": "paper", "org": "Lab A", "event_quality_score": 8.0},
                ],
                "source_count": 1,
                "cluster_evidence_count": 1,
            }
        ])

        events = pd.DataFrame()
        raw_data = pd.DataFrame()

        # Run _validate_temporal
        temporal_df = pipeline._validate_temporal(candidates, events, raw_data)

        # Verify that temporal_df only contains the weak signal candidate c1
        self.assertEqual(len(temporal_df), 1)
        self.assertEqual(temporal_df.loc[0, "candidate_name"], "weak signal candidate")

    def test_weak_signals_archiving_and_trend_comparison(self):
        import shutil
        import tempfile
        from pathlib import Path
        from src.core.pipeline import AnalysisPipeline
        from src.utils.config import Config

        # Mock Config.RESULT_DIR using a temporary directory
        temp_dir = Path(tempfile.mkdtemp())
        original_result_dir = Config.RESULT_DIR
        Config.RESULT_DIR = temp_dir

        try:
            pipeline = AnalysisPipeline()

            # Setup a past run archive
            archive_dir = temp_dir / "weak_signals_archive"
            archive_dir.mkdir(parents=True, exist_ok=True)

            # Save a past weak signals file
            past_run_name = "20260528_100000"
            past_signals = [
                {
                    "display_candidate_name": "stable topic",
                    "total_mentions": 5,
                    "signal_type": "weak_signal",
                    "signal_id": "WS001"
                },
                {
                    "display_candidate_name": "declining topic",
                    "total_mentions": 10,
                    "signal_type": "weak_signal",
                    "signal_id": "WS002"
                },
                {
                    "display_candidate_name": "growing topic",
                    "total_mentions": 2,
                    "signal_type": "weak_signal",
                    "signal_id": "WS003"
                }
            ]
            pd.DataFrame(past_signals).to_json(
                archive_dir / f"weak_signals_{past_run_name}.json",
                orient='records', force_ascii=False, indent=2
            )

            # Current run signals
            current_run_dir = temp_dir / "20260529_100000"
            current_run_dir.mkdir(parents=True, exist_ok=True)

            current_signals = pd.DataFrame([
                {
                    "display_candidate_name": "stable topic",
                    "total_mentions": 5,
                    "signal_type": "weak_signal",
                    "signal_id": "WS001"
                },
                {
                    "display_candidate_name": "declining topic",
                    "total_mentions": 4,  # decreased from 10
                    "signal_type": "weak_signal",
                    "signal_id": "WS002"
                },
                {
                    "display_candidate_name": "growing topic",
                    "total_mentions": 8,  # increased from 2
                    "signal_type": "weak_signal",
                    "signal_id": "WS003"
                },
                {
                    "display_candidate_name": "new topic",
                    "total_mentions": 3,
                    "signal_type": "weak_signal",
                    "signal_id": "WS004"
                }
            ])

            # Execute comparison & archiving
            pipeline._compare_and_archive_weak_signals(current_run_dir, current_signals)

            comp_df = pipeline.latest_weak_signals_comparison_df
            self.assertEqual(len(comp_df), 4)

            # Check trends
            comp_df.set_index("candidate_name", inplace=True)
            self.assertEqual(comp_df.loc["stable topic", "trend"], "持平")
            self.assertEqual(comp_df.loc["declining topic", "trend"], "减弱")
            self.assertEqual(comp_df.loc["growing topic", "trend"], "增长")
            self.assertEqual(comp_df.loc["new topic", "trend"], "新增")

            # Check that files were written
            self.assertTrue((current_run_dir / "weak_signals_comparison.json").exists())
            self.assertTrue((archive_dir / f"weak_signals_{current_run_dir.name}.json").exists())

        finally:
            # Restore Config.RESULT_DIR and cleanup temp dir
            Config.RESULT_DIR = original_result_dir
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    unittest.main()
