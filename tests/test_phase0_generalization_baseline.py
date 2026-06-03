from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
BASELINE_DOC = ROOT / "docs" / "generalization" / "phase0_baseline_and_risk_inventory.md"


class Phase0GeneralizationBaselineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = BASELINE_DOC.read_text(encoding="utf-8")

    def test_humanoid_baseline_records_required_metrics(self):
        required_fragments = [
            "result/20260510_192455(v2.8全量抽取版)",
            "事件数：5490",
            "候选数：614",
            "弱信号数：81",
            "被确认弱信号数：20",
            "result/20260525_193629",
            "事件数：333",
            "候选数：483",
            "弱信号数：28",
            "被确认弱信号数：9",
            "机器人串域过滤效果",
        ]
        for fragment in required_fragments:
            self.assertIn(fragment, self.text)

    def test_domain_hardcoding_inventory_covers_phase0_files(self):
        required_files = [
            "src/extraction/tech_lexicon.py",
            "src/extraction/candidate_former.py",
            "src/scoring/signal_generator.py",
            "src/scoring/topic_refiner.py",
            "web_app.py",
        ]
        for path in required_files:
            self.assertIn(path, self.text)
        self.assertIn("humanoid preset 保留", self.text)
        self.assertIn("通用范式迁移", self.text)

    def test_cross_domain_guard_tests_are_catalogued(self):
        guard_tests = [
            "tests/test_event_extraction_refactor.py::test_candidate_forms_use_selected_domain_as_generic_scope",
            "tests/test_event_extraction_refactor.py::test_signal_generation_filters_to_selected_analysis_domain",
            "tests/test_event_extraction_refactor.py::test_non_robot_domain_disables_robot_family_normalization",
            "tests/test_event_extraction_refactor.py::test_configured_local_loading_does_not_fallback_to_default_samples",
            "tests/test_data_access.py::test_db_cache_path_handles_empty_source_query_id",
        ]
        for test_name in guard_tests:
            self.assertIn(test_name, self.text)


if __name__ == "__main__":
    unittest.main()
