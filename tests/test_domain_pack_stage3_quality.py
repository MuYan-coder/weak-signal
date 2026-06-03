from pathlib import Path
import json
import tempfile
import unittest

from src.domain import DomainPack
from src.domain.domain_pack_dry_run import (
    DomainPackDryRunReport,
    run_domain_pack_dry_run,
)
from src.domain.domain_pack_generator import DomainPackGenerationRequest
from src.domain.domain_pack_validator import validate_domain_pack
from src.domain.domain_pack_workflow import DomainPackWorkflow


def _pack_payload():
    return {
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
                "exclude_terms": ["招聘"],
            },
        },
        "domain_identity": {
            "field_id": "battery_materials",
            "field_name": "电池材料",
            "domain_boundary": "电池材料及其电解质、界面和正负极材料改性。",
            "adjacent_domains": ["整车制造"],
            "out_of_scope_domains": ["招聘"],
        },
        "search_strategy": {
            "core_keywords": ["电池材料", "固态电池"],
            "synonyms": ["battery materials"],
            "english_terms": ["battery materials"],
            "exclude_terms": ["招聘"],
        },
        "observation_scopes": {
            "main_scope": "电池材料",
            "sub_scopes": ["固态电池"],
            "scope_aliases": ["battery materials"],
            "scope_echo_terms": ["电池材料"],
            "off_domain_anchor_terms": ["招聘"],
        },
        "candidate_formation": {
            "technical_object_types": ["电解质", "正极材料", "cathode material"],
            "mechanism_types": ["界面钝化", "掺杂", "interface passivation"],
            "task_or_performance_types": ["循环稳定性", "cycle stability"],
            "data_or_method_types": ["阻抗谱", "electrochemical impedance"],
            "scene_or_application_types": ["固态电池"],
            "generic_terms": ["材料", "技术", "material", "technology"],
            "shell_terms": ["技术", "方法", "系统", "应用", "technology", "method", "system"],
            "valid_candidate_patterns": [
                {
                    "pattern_id": "battery_object_mechanism_task",
                    "description": "材料对象、机制和性能目标同时出现。",
                    "required_slots": ["technical_object", "mechanism"],
                    "optional_slots": ["performance", "evidence_span"],
                    "min_required_slot_count": 2,
                    "evidence_required": True,
                    "weight": 1.0,
                    "reason_template": "候选包含材料对象和机制。",
                }
            ],
            "invalid_candidate_patterns": [
                {
                    "pattern_id": "battery_shell_only",
                    "description": "仅泛化壳词。",
                    "reject_if_slots_only": ["technical_object"],
                    "reject_terms": ["材料", "技术"],
                    "max_specific_slot_count": 1,
                    "reason_template": "候选缺少具体材料对象。",
                }
            ],
            "minimum_specificity_rule": {
                "min_non_shell_slots": 2,
                "require_evidence_span": True,
                "allow_scope_only_candidate": False,
            },
        },
        "weak_signal_rules": {
            "early_stage_markers": ["实验室", "prototype"],
            "low_attention_markers": ["小团队"],
            "niche_actor_markers": ["university lab"],
            "cross_domain_markers": ["interface engineering"],
            "engineering_trace_markers": ["阻抗"],
            "commercialization_noise_markers": ["融资"],
            "policy_or_market_noise_markers": ["补贴"],
            "scoring_adjustments": [
                {
                    "rule_id": "battery_interface_trace",
                    "applies_to": "candidate",
                    "match_terms": ["界面"],
                    "required_evidence_fields": ["evidence_span"],
                    "score_delta": 0.3,
                    "max_delta": 0.8,
                    "reason_template": "界面工程痕迹增强弱信号解释。",
                }
            ],
        },
        "canonicalization": {
            "object_families": [],
            "alias_groups": [],
            "parent_child_terms": [],
            "do_not_merge_rules": [
                {
                    "rule_id": "battery_vs_vehicle",
                    "left_terms": ["电池材料"],
                    "right_terms": ["整车制造"],
                    "reason": "材料方向不等同整车制造。",
                }
            ],
            "object_family_enabled": False,
        },
        "evidence_rules": {
            "strong_evidence_patterns": [],
            "weak_evidence_patterns": [],
            "source_reliability_hints": ["paper", "patent"],
            "traceability_fields": ["title", "text", "evidence_span", "source_type"],
            "evidence_rejection_patterns": ["招聘"],
        },
        "reporting": {
            "display_labels": {"domain": "电池材料"},
            "explanation_templates": [],
            "review_hints": ["复核界面工程是否有证据。"],
        },
    }


def _pack():
    return DomainPack.from_dict(_pack_payload())


def _sample_documents():
    return [
        {
            "id": "doc1",
            "source_type": "paper",
            "title": "固态电池电解质界面钝化",
            "text": "电解质界面钝化降低阻抗并提升循环稳定性。",
        },
        {
            "id": "doc2",
            "source_type": "news",
            "title": "招聘信息",
            "text": "电池企业招聘材料工程师。",
        },
    ]


class FakeGenerator:
    def __init__(self, pack):
        self.pack = pack
        self.calls = 0

    def generate(self, request, refresh_cache=False):
        self.calls += 1
        return self.pack


class DomainPackStage3QualityTest(unittest.TestCase):
    def test_dry_run_outputs_required_quality_metrics_and_saves_report(self):
        pack = _pack()
        candidates = [
            {
                "candidate_text": "固态电池电解质界面钝化",
                "evidence_span": "电解质界面钝化降低阻抗并提升循环稳定性",
            },
            {"candidate_text": "材料技术", "evidence_span": "材料技术"},
            {"candidate_text": "招聘信息", "evidence_span": "招聘材料工程师"},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            report = run_domain_pack_dry_run(
                pack,
                sample_documents=_sample_documents(),
                candidate_records=candidates,
                memory_dir=Path(tmpdir),
            )

            saved_path = Path(tmpdir) / f"{pack.domain_pack_hash}_dry_run.json"
            self.assertTrue(saved_path.exists())
            saved_payload = json.loads(saved_path.read_text(encoding="utf-8"))
            self.assertEqual(saved_payload["domain_pack_hash"], pack.domain_pack_hash)

        self.assertIsInstance(report, DomainPackDryRunReport)
        self.assertEqual(report.sample_doc_count, 2)
        self.assertEqual(report.candidate_count, 3)
        self.assertEqual(report.specific_candidate_count, 1)
        self.assertGreater(report.shell_candidate_ratio, 0)
        self.assertIn("招聘", report.off_domain_leakage_terms)
        self.assertIn("shell_only", report.top_invalid_candidate_reasons)

    def test_validator_requires_dry_run_report_and_blocks_bad_quality(self):
        pack = _pack()

        missing = validate_domain_pack(pack, dry_run_report=None)
        self.assertFalse(missing.is_valid)
        self.assertEqual(missing.gate_status, "blocked")
        self.assertIn("dry-run", " ".join(missing.errors))

        good_report = DomainPackDryRunReport(
            domain_pack_id=pack.pack_id,
            domain_pack_hash=pack.domain_pack_hash,
            sample_doc_count=2,
            candidate_count=2,
            specific_candidate_count=2,
            shell_candidate_ratio=0.0,
            off_domain_leakage_terms=[],
            top_invalid_candidate_reasons={},
            recommended_pack_changes=[],
        )
        valid = validate_domain_pack(pack, dry_run_report=good_report)
        self.assertTrue(valid.is_valid)
        self.assertEqual(valid.gate_status, "passed")

        broad_payload = _pack_payload()
        broad_payload["search_strategy"]["core_keywords"] = ["技术", "系统"]
        broad_pack = DomainPack.from_dict(broad_payload)
        bad_report = DomainPackDryRunReport(
            domain_pack_id=broad_pack.pack_id,
            domain_pack_hash=broad_pack.domain_pack_hash,
            sample_doc_count=2,
            candidate_count=5,
            specific_candidate_count=1,
            shell_candidate_ratio=0.8,
            off_domain_leakage_terms=["招聘"],
            top_invalid_candidate_reasons={"shell_only": 4},
            recommended_pack_changes=["增加更具体的 core_keywords"],
        )
        invalid = validate_domain_pack(broad_pack, dry_run_report=bad_report)
        self.assertFalse(invalid.is_valid)
        self.assertEqual(invalid.gate_status, "blocked")
        self.assertTrue(any("core_keywords" in message for message in invalid.errors))
        self.assertTrue(any("shell_candidate_ratio" in message for message in invalid.errors))

    def test_workflow_orchestrates_generation_dry_run_validation_and_review(self):
        pack = _pack()
        request = DomainPackGenerationRequest(
            field_id="battery_materials",
            field_name="电池材料",
            keywords=["电池材料"],
            synonyms=["battery materials"],
            exclude_terms=["招聘"],
            source_types=["paper", "patent"],
            sample_documents=_sample_documents(),
        )
        candidates = [
            {
                "candidate_text": "固态电池电解质界面钝化",
                "evidence_span": "电解质界面钝化降低阻抗并提升循环稳定性",
            }
        ]
        fake_generator = FakeGenerator(pack)
        with tempfile.TemporaryDirectory() as tmpdir:
            workflow = DomainPackWorkflow(generator=fake_generator, memory_dir=Path(tmpdir))
            result = workflow.prepare_domain_pack(
                request,
                sample_documents=_sample_documents(),
                candidate_records=candidates,
                human_review_approved=True,
            )

            dry_run_path = Path(tmpdir) / f"{pack.domain_pack_hash}_dry_run.json"
            self.assertTrue(dry_run_path.exists())

        self.assertEqual(fake_generator.calls, 1)
        self.assertEqual(result.status, "ready")
        self.assertTrue(result.validation_report.is_valid)
        self.assertEqual(result.domain_pack.pack_id, "battery_materials")

    def test_workflow_keeps_valid_pack_pending_until_human_review(self):
        pack = _pack()
        request = DomainPackGenerationRequest(
            field_id="battery_materials",
            field_name="电池材料",
            keywords=["电池材料"],
            sample_documents=_sample_documents(),
        )
        candidates = [
            {
                "candidate_text": "固态电池电解质界面钝化",
                "evidence_span": "电解质界面钝化降低阻抗并提升循环稳定性",
            }
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            workflow = DomainPackWorkflow(generator=FakeGenerator(pack), memory_dir=Path(tmpdir))
            result = workflow.prepare_domain_pack(
                request,
                sample_documents=_sample_documents(),
                candidate_records=candidates,
                human_review_approved=False,
            )

        self.assertEqual(result.status, "pending_review")
        self.assertTrue(result.manual_review_required)
        self.assertTrue(result.validation_report.is_valid)

    def test_workflow_blocks_when_dry_run_quality_fails(self):
        pack = _pack()
        request = DomainPackGenerationRequest(
            field_id="battery_materials",
            field_name="电池材料",
            keywords=["电池材料"],
            sample_documents=_sample_documents(),
        )
        candidates = [{"candidate_text": "材料技术", "evidence_span": "材料技术"}]
        with tempfile.TemporaryDirectory() as tmpdir:
            workflow = DomainPackWorkflow(generator=FakeGenerator(pack), memory_dir=Path(tmpdir))
            result = workflow.prepare_domain_pack(
                request,
                sample_documents=_sample_documents(),
                candidate_records=candidates,
                human_review_approved=True,
            )

        self.assertEqual(result.status, "blocked")
        self.assertFalse(result.validation_report.is_valid)
        self.assertTrue(any("shell_candidate_ratio" in item for item in result.validation_report.errors))


if __name__ == "__main__":
    unittest.main()
