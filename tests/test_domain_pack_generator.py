import json
from pathlib import Path
import tempfile
import unittest

import yaml

from src.domain import DomainPack, load_domain_pack
from src.domain.domain_pack_generator import (
    DOMAIN_PACK_GENERATOR_PROMPT_VERSION,
    DomainPackGenerationError,
    DomainPackGenerationRequest,
    DomainPackGenerator,
    build_candidate_formation_prompt,
    build_domain_modeling_prompt,
    build_domain_pack_cache_key,
    build_integration_prompt,
    build_weak_signal_rules_prompt,
    save_domain_pack_snapshot,
    save_domain_pack_to_memory,
)


class FakeChat:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    @property
    def call_count(self):
        return len(self.prompts)

    def __call__(self, prompt, **kwargs):
        self.prompts.append((prompt, kwargs))
        if not self.responses:
            raise AssertionError("FakeChat received more calls than expected")
        payload = self.responses.pop(0)
        return json.dumps(payload, ensure_ascii=False), {"prompt_tokens": 10, "completion_tokens": 20}, None


def _request():
    return DomainPackGenerationRequest(
        field_id="battery_materials",
        field_name="电池材料",
        keywords=["电池材料", "固态电池"],
        synonyms=["battery materials"],
        exclude_terms=["招聘"],
        source_types=["paper", "patent"],
        sample_documents=[
            {
                "title": "固态电池电解质界面改性",
                "text": "界面阻抗降低，循环稳定性提升。",
                "source_type": "paper",
            }
        ],
    )


def _responses():
    domain_model = {
        "domain_boundary": "电池材料及其电解质、界面、正负极材料改性。",
        "adjacent_domains": ["整车制造"],
        "out_of_scope_domains": ["招聘"],
        "sub_directions": ["solid-state electrolyte", "interface passivation"],
        "noise_terms": ["招聘"],
        "search_strategy": {
            "core_keywords": ["电池材料"],
            "synonyms": ["battery materials"],
            "english_terms": ["battery materials"],
            "exclude_terms": ["招聘"],
        },
    }
    candidate_formation = {
        "technical_object_types": ["electrolyte", "cathode material"],
        "mechanism_types": ["interface passivation", "doping"],
        "task_or_performance_types": ["cycle stability"],
        "data_or_method_types": ["electrochemical impedance"],
        "scene_or_application_types": ["solid-state battery"],
        "generic_terms": ["material", "technology", "材料", "技术"],
        "shell_terms": ["system", "method", "系统", "方法"],
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
    }
    weak_signal_rules = {
        "early_stage_markers": ["prototype", "实验室"],
        "low_attention_markers": ["小团队"],
        "niche_actor_markers": ["university lab"],
        "cross_domain_markers": ["interface engineering"],
        "engineering_trace_markers": ["impedance"],
        "commercialization_noise_markers": ["融资"],
        "policy_or_market_noise_markers": ["补贴"],
        "scoring_adjustments": [
            {
                "rule_id": "battery_interface_trace",
                "applies_to": "candidate",
                "match_terms": ["interface"],
                "required_evidence_fields": ["evidence_span"],
                "score_delta": 0.3,
                "max_delta": 0.8,
                "reason_template": "界面工程痕迹增强弱信号解释。",
            }
        ],
    }
    final_pack = {
        "schema_version": "domain_pack_v1",
        "pack_id": "battery_materials",
        "pack_name": "电池材料 Domain Pack",
        "pack_version": "generated.v1",
        "source": {
            "mode": "llm_generated",
            "model": "fake-model",
            "prompt_version": DOMAIN_PACK_GENERATOR_PROMPT_VERSION,
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
            "domain_boundary": "电池材料及其电解质、界面、正负极材料改性。",
            "adjacent_domains": ["整车制造"],
            "out_of_scope_domains": ["招聘"],
        },
        "search_strategy": {
            "core_keywords": ["电池材料"],
            "synonyms": ["battery materials"],
            "english_terms": ["battery materials"],
            "exclude_terms": ["招聘"],
        },
        "observation_scopes": {
            "main_scope": "电池材料",
            "sub_scopes": ["solid-state electrolyte"],
            "scope_aliases": ["battery materials"],
            "scope_echo_terms": ["电池材料"],
            "off_domain_anchor_terms": ["招聘"],
        },
        "candidate_formation": candidate_formation,
        "weak_signal_rules": weak_signal_rules,
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
            "traceability_fields": ["title", "text", "evidence_span", "source_type"],
            "strong_evidence_patterns": [],
            "weak_evidence_patterns": [],
            "source_reliability_hints": ["paper", "patent"],
            "evidence_rejection_patterns": ["招聘"],
        },
        "reporting": {
            "display_labels": {"domain": "电池材料"},
            "explanation_templates": [],
            "review_hints": ["复核界面工程是否有证据。"],
        },
    }
    return [domain_model, candidate_formation, weak_signal_rules, final_pack]


class DomainPackGeneratorTest(unittest.TestCase):
    def test_prompt_builders_cover_four_llm_stages_and_schema_slots(self):
        request = _request()
        prompts = [
            build_domain_modeling_prompt(request),
            build_candidate_formation_prompt(request, {"domain_boundary": "电池材料"}),
            build_weak_signal_rules_prompt(
                request,
                {"domain_boundary": "电池材料"},
                {"technical_object_types": ["electrolyte"]},
            ),
            build_integration_prompt(
                request,
                {"domain_boundary": "电池材料"},
                {"technical_object_types": ["electrolyte"]},
                {"early_stage_markers": ["prototype"]},
            ),
        ]

        self.assertIn("阶段一", prompts[0])
        self.assertIn("阶段二", prompts[1])
        self.assertIn("阶段三", prompts[2])
        self.assertIn("domain_pack_v1", prompts[3])
        self.assertIn("technical_object", prompts[3])
        self.assertIn("mechanism", prompts[3])
        self.assertIn("evidence_span", prompts[3])

    def test_generator_creates_structurally_complete_pack_and_saves_runtime_yaml(self):
        request = _request()
        fake_chat = FakeChat(_responses())
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = DomainPackGenerator(
                chat_fn=fake_chat,
                memory_dir=Path(tmpdir),
                model="fake-model",
            )

            pack = generator.generate(request, refresh_cache=True)

            saved_path = Path(tmpdir) / f"{pack.domain_pack_hash}.yaml"
            self.assertIsInstance(pack, DomainPack)
            self.assertEqual(pack.pack_id, "battery_materials")
            self.assertEqual(pack.source["mode"], "llm_generated")
            self.assertEqual(pack.source["model"], "fake-model")
            self.assertEqual(pack.source["prompt_version"], DOMAIN_PACK_GENERATOR_PROMPT_VERSION)
            self.assertIn("candidate_formation", pack.to_dict())
            self.assertIn("weak_signal_rules", pack.to_dict())
            self.assertTrue(saved_path.exists())
            self.assertFalse((Path(tmpdir) / "src" / "config" / "domain_packs").exists())
            self.assertEqual(fake_chat.call_count, 4)

            saved = yaml.safe_load(saved_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["pack_id"], "battery_materials")
            self.assertEqual(saved["domain_pack_hash"], pack.domain_pack_hash)
            reloaded = load_domain_pack(saved_path)
            self.assertEqual(reloaded.pack_id, pack.pack_id)
            self.assertEqual(reloaded.domain_pack_hash, pack.domain_pack_hash)

    def test_cache_key_contains_domain_prompt_version_and_model_and_reuses_pack(self):
        request = _request()
        cache_key = build_domain_pack_cache_key(
            request,
            model="fake-model",
            prompt_version=DOMAIN_PACK_GENERATOR_PROMPT_VERSION,
        )

        self.assertIn("battery_materials", cache_key)
        self.assertIn("fake-model", cache_key)
        self.assertIn(DOMAIN_PACK_GENERATOR_PROMPT_VERSION, cache_key)

        fake_chat = FakeChat(_responses())
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = DomainPackGenerator(
                chat_fn=fake_chat,
                memory_dir=Path(tmpdir),
                model="fake-model",
            )
            first = generator.generate(request, refresh_cache=True)
            second = generator.generate(request, refresh_cache=False)

        self.assertEqual(first.domain_pack_hash, second.domain_pack_hash)
        self.assertEqual(fake_chat.call_count, 4)

    def test_generation_failure_raises_and_does_not_save_pack(self):
        request = _request()

        def bad_chat(prompt, **kwargs):
            return "not-json", {}, None

        with tempfile.TemporaryDirectory() as tmpdir:
            generator = DomainPackGenerator(
                chat_fn=bad_chat,
                memory_dir=Path(tmpdir),
                model="fake-model",
            )

            with self.assertRaises(DomainPackGenerationError):
                generator.generate(request, refresh_cache=True)

            self.assertEqual(list(Path(tmpdir).glob("*.yaml")), [])
            self.assertEqual(list((Path(tmpdir) / "cache").glob("*.json")) if (Path(tmpdir) / "cache").exists() else [], [])

    def test_user_edited_pack_and_result_snapshot_save_outside_source_config(self):
        pack = DomainPack.from_dict(
            {
                "schema_version": "domain_pack_v1",
                "pack_id": "battery_materials",
                "pack_name": "电池材料 Domain Pack",
                "pack_version": "user.v1",
                "source": {
                    "mode": "user_edited",
                    "model": "fake-model",
                    "prompt_version": DOMAIN_PACK_GENERATOR_PROMPT_VERSION,
                    "based_on_user_input": {
                        "field_id": "battery_materials",
                        "field_name": "电池材料",
                        "keywords": ["电池材料"],
                        "synonyms": ["battery materials"],
                        "exclude_terms": [],
                    },
                },
                "domain_identity": {
                    "field_id": "battery_materials",
                    "field_name": "电池材料",
                    "domain_boundary": "用户编辑后的电池材料边界。",
                },
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_dir = Path(tmpdir) / "memory" / "domain_packs"
            result_dir = Path(tmpdir) / "result" / "run_001"

            memory_path = save_domain_pack_to_memory(pack, memory_dir=memory_dir)
            snapshot_path = save_domain_pack_snapshot(pack, result_dir=result_dir)

            self.assertEqual(memory_path.parent, memory_dir)
            self.assertEqual(snapshot_path, result_dir / "domain_pack.yaml")
            self.assertTrue(memory_path.exists())
            self.assertTrue(snapshot_path.exists())
            self.assertFalse((Path(tmpdir) / "src" / "config" / "domain_packs").exists())
            self.assertEqual(yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))["pack_id"], "battery_materials")


if __name__ == "__main__":
    unittest.main()
