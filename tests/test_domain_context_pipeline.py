from pathlib import Path
import json
import tempfile
import unittest

import pandas as pd

from src.core.pipeline import AnalysisPipeline
from src.extraction.event_extractor import _get_doc_level_cache_dir
from src.domain import DomainContext, DomainPack
from src.domain.domain_pack_dry_run import (
    DomainPackDryRunReport,
    save_domain_pack_dry_run_report,
)
from src.domain.domain_pack_generator import save_domain_pack_snapshot
from src.utils.config import Config


def _pack_payload(pack_id: str, field_name: str) -> dict:
    return {
        "schema_version": "domain_pack_v1",
        "pack_id": pack_id,
        "pack_name": f"{field_name} Domain Pack",
        "pack_version": "generated.v1",
        "source": {
            "mode": "llm_generated",
            "model": "fake-model",
            "prompt_version": "domain_pack_generator_v1",
            "based_on_user_input": {
                "field_id": pack_id,
                "field_name": field_name,
                "keywords": [field_name],
                "synonyms": [],
                "exclude_terms": [],
            },
        },
        "domain_identity": {
            "field_id": pack_id,
            "field_name": field_name,
            "domain_boundary": f"{field_name} test boundary",
        },
        "search_strategy": {
            "core_keywords": [field_name],
            "synonyms": [],
            "english_terms": [],
            "exclude_terms": [],
        },
        "candidate_formation": {
            "technical_object_types": [field_name],
            "mechanism_types": ["机制"],
            "generic_terms": ["技术", "系统"],
            "shell_terms": ["技术", "系统"],
            "valid_candidate_patterns": [],
            "invalid_candidate_patterns": [],
        },
        "reporting": {
            "display_labels": {"domain": field_name},
            "review_hints": [],
        },
    }


def _pack(pack_id: str = "battery_materials", field_name: str = "电池材料") -> DomainPack:
    return DomainPack.from_dict(_pack_payload(pack_id, field_name))


def _events_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": "doc1",
                "source_type": "paper",
                "title": "固态电池电解质",
                "subject": "研究团队",
                "action": "提出",
                "object": "电解质界面钝化",
                "evidence_span": "研究团队提出电解质界面钝化方法。",
                "confidence": 0.9,
            }
        ]
    )


def _raw_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": "doc1",
                "source_type": "paper",
                "title": "固态电池电解质",
                "text": "研究团队提出电解质界面钝化方法。",
            }
        ]
    )


def _candidate_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "candidate_id": "cand1",
                "display_candidate_name": "电解质界面钝化",
                "candidate_stage": "formed_candidate_strong",
                "signal_type": "weak_signal",
                "weak_signal_score": 0.8,
                "cluster_evidence_count": 1,
                "source_count": 1,
            }
        ]
    )


def _dry_run_report(pack: DomainPack) -> DomainPackDryRunReport:
    return DomainPackDryRunReport(
        domain_pack_id=pack.pack_id,
        domain_pack_hash=pack.domain_pack_hash,
        sample_doc_count=1,
        candidate_count=1,
        specific_candidate_count=1,
        shell_candidate_ratio=0.0,
        off_domain_leakage_terms=[],
        top_invalid_candidate_reasons={},
        recommended_pack_changes=[],
    )


class StubPipeline(AnalysisPipeline):
    def _score_events_during_extraction(self, events_df, raw_data):
        self.latest_event_quality_df = pd.DataFrame()
        return events_df.copy()

    def _form_candidates(self, events_df, raw_data):
        return _candidate_df()

    def _score_candidates(self, candidate_forms_df):
        scored = candidate_forms_df.copy()
        scored["score"] = 0.8
        return scored

    def _refine_topics(self, scored_df):
        return scored_df.copy()

    def _validate_reverse(self, scored_df, *args):
        self.latest_reverse_validation_df = pd.DataFrame()
        return scored_df.copy()

    def _validate_temporal(self, candidate_df, events_df, raw_data):
        self.latest_temporal_validation_df = pd.DataFrame()
        return pd.DataFrame()

    def _generate_signals(self, candidate_df, raw_data):
        return candidate_df.copy()

    def _generate_report(self, signals_output):
        self.report_artifacts = {"report_generation_mode": "stub"}
        return f"stub report for {self.domain_context.domain_pack_id}"

    def _compare_and_archive_weak_signals(self, result_dir, signals_df):
        self.latest_weak_signals_comparison_df = pd.DataFrame()


class DomainContextPipelineTest(unittest.TestCase):
    def test_event_cache_path_includes_domain_pack_hash(self):
        battery = _pack("battery_materials", "电池材料")
        solar = _pack("solar_materials", "光伏材料")
        first = AnalysisPipeline(domain_context=DomainContext.from_pack(battery))
        second = AnalysisPipeline(domain_context=DomainContext.from_pack(solar))
        raw = _raw_df()

        with tempfile.TemporaryDirectory() as tmpdir:
            first_path = first._event_cache_path(raw, Path(tmpdir))
            second_path = second._event_cache_path(raw, Path(tmpdir))

        self.assertIn(battery.domain_pack_hash, first_path.name)
        self.assertIn(solar.domain_pack_hash, second_path.name)
        self.assertNotEqual(first_path.name, second_path.name)

    def test_doc_level_event_cache_dir_is_isolated_by_domain_pack_hash(self):
        battery = _pack("battery_materials", "电池材料")
        solar = _pack("solar_materials", "光伏材料")
        raw = _raw_df()

        with tempfile.TemporaryDirectory() as tmpdir:
            first_path = AnalysisPipeline(
                domain_context=DomainContext.from_pack(battery)
            )._event_cache_path(raw, Path(tmpdir))
            second_path = AnalysisPipeline(
                domain_context=DomainContext.from_pack(solar)
            )._event_cache_path(raw, Path(tmpdir))
            first_doc_cache = _get_doc_level_cache_dir(first_path)
            second_doc_cache = _get_doc_level_cache_dir(second_path)

        self.assertIn(battery.domain_pack_hash, str(first_doc_cache))
        self.assertIn(solar.domain_pack_hash, str(second_doc_cache))
        self.assertNotEqual(first_doc_cache, second_doc_cache)

    def test_save_results_writes_domain_pack_snapshot_dry_run_snapshot_and_metadata(self):
        pack = _pack()
        pipeline = StubPipeline(domain_context=DomainContext.from_pack(pack))

        with tempfile.TemporaryDirectory() as tmpdir:
            memory_dir = Path(tmpdir) / "memory"
            dry_run_path = save_domain_pack_dry_run_report(_dry_run_report(pack), memory_dir=memory_dir)
            pipeline.domain_pack_dry_run_report_path = dry_run_path
            result_dir = Path(tmpdir) / "result"

            pipeline._save_results(
                result_dir,
                _events_df(),
                _candidate_df(),
                _candidate_df(),
                _candidate_df(),
                _candidate_df(),
                _candidate_df(),
                "report",
                raw_data=_raw_df(),
            )

            events = json.loads((result_dir / "events.json").read_text(encoding="utf-8"))
            signals = json.loads((result_dir / "signals.json").read_text(encoding="utf-8"))
            report_metadata = json.loads((result_dir / "report_metadata.json").read_text(encoding="utf-8"))

            self.assertTrue((result_dir / "domain_pack.yaml").exists())
            self.assertTrue((result_dir / "domain_pack_dry_run.json").exists())
            self.assertEqual(events[0]["domain_pack_id"], pack.pack_id)
            self.assertEqual(events[0]["domain_pack_hash"], pack.domain_pack_hash)
            self.assertEqual(signals[0]["domain_pack_hash"], pack.domain_pack_hash)
            self.assertEqual(report_metadata["domain_pack_hash"], pack.domain_pack_hash)
            self.assertEqual(report_metadata["domain_pack_id"], pack.pack_id)

    def test_run_from_events_recovers_domain_pack_from_source_result_dir(self):
        pack = _pack()

        with tempfile.TemporaryDirectory() as tmpdir:
            original_result_root = Config.RESULT_DIR
            Config.RESULT_DIR = Path(tmpdir) / "results"
            source_result_dir = Path(tmpdir) / "source"
            save_domain_pack_snapshot(pack, result_dir=source_result_dir)
            try:
                pipeline = StubPipeline()
                output = pipeline.run_from_events(
                    events_df=_events_df(),
                    result_name_suffix="from_events_test",
                    source_result_dir=source_result_dir,
                )
                output_domain_hash = output["domain_context"].domain_pack_hash
                saved_snapshot_exists = (Path(output["result_dir"]) / "domain_pack.yaml").exists()
            finally:
                Config.RESULT_DIR = original_result_root

        self.assertEqual(output_domain_hash, pack.domain_pack_hash)
        self.assertTrue(saved_snapshot_exists)

    def test_run_from_events_recovers_domain_pack_from_event_metadata(self):
        pack = _pack()
        events_df = _events_df()
        events_df["domain_pack_id"] = pack.pack_id
        events_df["domain_pack_version"] = pack.domain_pack_version
        events_df["domain_pack_hash"] = pack.domain_pack_hash

        with tempfile.TemporaryDirectory() as tmpdir:
            original_result_root = Config.RESULT_DIR
            Config.RESULT_DIR = Path(tmpdir) / "results"
            try:
                pipeline = StubPipeline()
                output = pipeline.run_from_events(
                    events_df=events_df,
                    result_name_suffix="from_events_metadata_test",
                )
                output_domain_hash = output["domain_context"].domain_pack_hash
                output_source = output["domain_context"].runtime_mode
            finally:
                Config.RESULT_DIR = original_result_root

        self.assertEqual(output_domain_hash, pack.domain_pack_hash)
        self.assertEqual(output_source, "restored_metadata")

    def test_regenerate_report_from_result_restores_domain_pack_and_dry_run_snapshot(self):
        pack = _pack()

        with tempfile.TemporaryDirectory() as tmpdir:
            original_result_root = Config.RESULT_DIR
            Config.RESULT_DIR = Path(tmpdir) / "results"
            source_result_dir = Path(tmpdir) / "source"
            save_domain_pack_snapshot(pack, result_dir=source_result_dir)
            (source_result_dir / "domain_pack_dry_run.json").write_text(
                json.dumps(_dry_run_report(pack).to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            _events_df().to_json(source_result_dir / "events.json", orient="records", force_ascii=False, indent=2)
            _candidate_df().to_json(source_result_dir / "signals.json", orient="records", force_ascii=False, indent=2)
            _candidate_df().to_json(source_result_dir / "candidate_forms.json", orient="records", force_ascii=False, indent=2)
            _candidate_df().to_json(source_result_dir / "scored.json", orient="records", force_ascii=False, indent=2)
            _candidate_df().to_json(source_result_dir / "refined.json", orient="records", force_ascii=False, indent=2)
            _candidate_df().to_json(source_result_dir / "validated.json", orient="records", force_ascii=False, indent=2)

            try:
                pipeline = StubPipeline()
                output = pipeline.regenerate_report_from_result(
                    source_result_dir,
                    result_name_suffix="report_refresh_test",
                )
                output_domain_hash = output["domain_context"].domain_pack_hash
                target_dir = Path(output["result_dir"])
                snapshot_exists = (target_dir / "domain_pack.yaml").exists()
                copied_dry_run = json.loads((target_dir / "domain_pack_dry_run.json").read_text(encoding="utf-8"))
            finally:
                Config.RESULT_DIR = original_result_root

        self.assertEqual(output_domain_hash, pack.domain_pack_hash)
        self.assertTrue(snapshot_exists)
        self.assertEqual(copied_dry_run["domain_pack_hash"], pack.domain_pack_hash)


if __name__ == "__main__":
    unittest.main()
