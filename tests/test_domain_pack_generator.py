import copy
import json
from pathlib import Path
import tempfile
import unittest

import yaml

from src.domain import DomainPack, load_domain_pack, validate_domain_pack
from src.domain.domain_pack_generator import (
    DOMAIN_PACK_GENERATOR_PROMPT_VERSION,
    DomainPackGenerationError,
    DomainPackGenerationRequest,
    DomainPackGenerator,
    build_candidate_formation_prompt,
    build_domain_modeling_prompt,
    build_domain_pack_cache_key,
    build_integration_prompt,
    is_current_generated_domain_pack,
    build_weak_signal_markers_prompt,
    build_scoring_adjustments_prompt,
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
        if isinstance(payload, str):
            return payload, {"prompt_tokens": 10, "completion_tokens": 20}, None
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
    weak_signal_markers = {
        "early_stage_markers": ["prototype", "实验室"],
        "low_attention_markers": ["小团队"],
        "niche_actor_markers": ["university lab"],
        "cross_domain_markers": ["interface engineering"],
        "engineering_trace_markers": ["impedance"],
        "commercialization_noise_markers": ["融资"],
        "policy_or_market_noise_markers": ["补贴"],
    }
    scoring_adjustments = {
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
        ]
    }
    weak_signal_rules = {**weak_signal_markers, **scoring_adjustments}
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
    return [domain_model, candidate_formation, weak_signal_markers, scoring_adjustments, final_pack]


class DomainPackGeneratorTest(unittest.TestCase):
    def test_prompt_builders_cover_four_llm_stages_and_schema_slots(self):
        request = _request()
        prompts = [
            build_domain_modeling_prompt(request),
            build_candidate_formation_prompt(request, {"domain_boundary": "电池材料"}),
            build_weak_signal_markers_prompt(
                request,
                {"domain_boundary": "电池材料"},
                {"technical_object_types": ["electrolyte"]},
            ),
            build_scoring_adjustments_prompt(
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
        self.assertIn("阶段 3a", prompts[2])
        self.assertIn("阶段 3b", prompts[3])
        self.assertIn("domain_pack_v1", prompts[4])
        self.assertIn("technical_object", prompts[4])
        self.assertIn("mechanism", prompts[4])
        self.assertIn("evidence_span", prompts[4])

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
            self.assertEqual(fake_chat.call_count, 5)

            saved = yaml.safe_load(saved_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["pack_id"], "battery_materials")
            self.assertEqual(saved["domain_pack_hash"], pack.domain_pack_hash)
            reloaded = load_domain_pack(saved_path)
            self.assertEqual(reloaded.pack_id, pack.pack_id)
            self.assertEqual(reloaded.domain_pack_hash, pack.domain_pack_hash)

    def test_generator_preserves_stage_two_candidate_shell_terms_when_integration_clears_them(self):
        request = _request()
        responses = copy.deepcopy(_responses())
        stage_two_candidate_formation = copy.deepcopy(responses[1])
        responses[4]["candidate_formation"] = copy.deepcopy(responses[4]["candidate_formation"])
        responses[4]["candidate_formation"]["generic_terms"] = []
        responses[4]["candidate_formation"]["shell_terms"] = []
        fake_chat = FakeChat(responses)
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = DomainPackGenerator(
                chat_fn=fake_chat,
                memory_dir=Path(tmpdir),
                model="fake-model",
            )

            pack = generator.generate(request, refresh_cache=True)

        self.assertEqual(
            pack.candidate_formation["generic_terms"],
            stage_two_candidate_formation["generic_terms"],
        )
        self.assertEqual(
            pack.candidate_formation["shell_terms"],
            stage_two_candidate_formation["shell_terms"],
        )
        report = validate_domain_pack(pack, require_dry_run=False)
        self.assertNotIn(
            "candidate_formation.generic_terms should include at least two shell-like examples",
            report.warnings,
        )
        self.assertNotIn(
            "candidate_formation.shell_terms should include at least two shell-like examples",
            report.warnings,
        )

    def test_generator_limits_shell_invalid_patterns_to_pure_shell_candidates(self):
        candidate_formation = {
            "generic_terms": ["技术", "方法"],
            "shell_terms": ["技术", "方法"],
            "invalid_candidate_patterns": [
                {
                    "pattern_id": "shell_term_only",
                    "reject_terms": ["技术", "方法", "系统", "应用"],
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            generator = DomainPackGenerator(
                chat_fn=FakeChat([]),
                memory_dir=Path(tmpdir),
                model="fake-model",
            )
            generator._ensure_candidate_formation_safety(candidate_formation)

        invalid_patterns = candidate_formation["invalid_candidate_patterns"]
        shell_pattern = next(item for item in invalid_patterns if item["pattern_id"] == "shell_term_only")
        self.assertEqual(shell_pattern["max_specific_slot_count"], 0)

    def test_generator_retries_weak_signal_markers_when_model_returns_non_structured_text(self):
        request = _request()
        responses = copy.deepcopy(_responses())
        repaired_markers = copy.deepcopy(responses[2])
        responses = [
            responses[0],
            responses[1],
            "好的，下面是弱信号标记词：early stage markers 包括 prototype。",
            repaired_markers,
            responses[3],
            responses[4],
        ]
        fake_chat = FakeChat(responses)
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = DomainPackGenerator(
                chat_fn=fake_chat,
                memory_dir=Path(tmpdir),
                model="fake-model",
            )

            pack = generator.generate(request, refresh_cache=True)

            debug_files = list((Path(tmpdir) / "cache" / "stage_failures").glob("*weak_signal_markers*.txt"))

        self.assertEqual(fake_chat.call_count, 6)
        self.assertEqual(pack.weak_signal_rules["early_stage_markers"], repaired_markers["early_stage_markers"])
        self.assertTrue(debug_files)
        self.assertIn("只返回一个合法 JSON 对象", fake_chat.prompts[3][0])

    def test_generator_normalizes_object_marker_payloads_to_string_lists(self):
        request = _request()
        responses = copy.deepcopy(_responses())
        object_markers = copy.deepcopy(responses[2])
        object_markers["early_stage_markers"] = [
            {"description": "实验室早期验证", "keywords": ["实验室验证", "prototype"]},
        ]
        object_markers["low_attention_markers"] = [
            {"name": "低引用"},
        ]
        responses[2] = object_markers
        responses[4] = copy.deepcopy(responses[4])
        responses[4]["weak_signal_rules"] = None
        fake_chat = FakeChat(responses)
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = DomainPackGenerator(
                chat_fn=fake_chat,
                memory_dir=Path(tmpdir),
                model="fake-model",
            )

            pack = generator.generate(request, refresh_cache=True)

        self.assertEqual(pack.weak_signal_rules["early_stage_markers"], ["实验室验证", "prototype"])
        self.assertEqual(pack.weak_signal_rules["low_attention_markers"], ["低引用"])
        self.assertTrue(all(isinstance(item, str) for item in pack.weak_signal_rules["early_stage_markers"]))

    def test_generator_recovers_missing_specific_candidate_terms_from_request_keywords(self):
        request = _request()
        responses = copy.deepcopy(_responses())
        responses[1] = copy.deepcopy(responses[1])
        responses[1]["technical_object_types"] = []
        responses[1]["mechanism_types"] = []
        responses[4] = copy.deepcopy(responses[4])
        responses[4]["candidate_formation"] = copy.deepcopy(responses[4]["candidate_formation"])
        responses[4]["candidate_formation"]["technical_object_types"] = []
        responses[4]["candidate_formation"]["mechanism_types"] = []
        fake_chat = FakeChat(responses)
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = DomainPackGenerator(
                chat_fn=fake_chat,
                memory_dir=Path(tmpdir),
                model="fake-model",
            )

            pack = generator.generate(request, refresh_cache=True)

        self.assertIn("电池材料", pack.candidate_formation["technical_object_types"])
        report = validate_domain_pack(pack, require_dry_run=False)
        self.assertTrue(report.is_valid, report.errors)

    def test_generator_flattens_grouped_mechanism_types_before_validation(self):
        request = _request()
        responses = copy.deepcopy(_responses())
        responses[1] = copy.deepcopy(responses[1])
        responses[1]["technical_object_types"] = []
        responses[1]["mechanism_types"] = {
            "基础机制": ["界面钝化"],
            "工艺机制": ["掺杂"],
        }
        responses[4] = copy.deepcopy(responses[4])
        responses[4]["candidate_formation"] = copy.deepcopy(responses[4]["candidate_formation"])
        responses[4]["candidate_formation"]["technical_object_types"] = []
        responses[4]["candidate_formation"]["mechanism_types"] = []
        fake_chat = FakeChat(responses)
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = DomainPackGenerator(
                chat_fn=fake_chat,
                memory_dir=Path(tmpdir),
                model="fake-model",
            )

            pack = generator.generate(request, refresh_cache=True)

        self.assertEqual(pack.candidate_formation["mechanism_types"], ["界面钝化", "掺杂"])
        report = validate_domain_pack(pack, require_dry_run=False)
        self.assertTrue(report.is_valid, report.errors)

    def test_generated_pack_current_helper_rejects_old_generated_pack_but_allows_presets(self):
        old_generated = DomainPack.from_dict(
            {
                **copy.deepcopy(_responses()[4]),
                "source": {
                    **copy.deepcopy(_responses()[4]["source"]),
                    "mode": "llm_generated",
                    "prompt_version": "domain_pack_generator_v2",
                },
            }
        )
        preset = DomainPack.from_dict(
            {
                **copy.deepcopy(_responses()[4]),
                "source": {
                    **copy.deepcopy(_responses()[4]["source"]),
                    "mode": "preset",
                    "prompt_version": "preset.v1",
                },
            }
        )

        self.assertFalse(is_current_generated_domain_pack(old_generated))
        self.assertTrue(is_current_generated_domain_pack(preset))

    def test_cached_generated_pack_from_old_prompt_version_is_not_reused(self):
        request = _request()
        old_payload = copy.deepcopy(_responses()[4])
        old_payload["source"]["prompt_version"] = "domain_pack_generator_v2"

        with tempfile.TemporaryDirectory() as tmpdir:
            memory_dir = Path(tmpdir)
            cached_pack_path = save_domain_pack_to_memory(
                DomainPack.from_dict(old_payload),
                memory_dir=memory_dir,
            )
            cache_key = build_domain_pack_cache_key(
                request,
                model="fake-model",
                prompt_version=DOMAIN_PACK_GENERATOR_PROMPT_VERSION,
            )
            cache_path = memory_dir / "cache" / f"{cache_key}.json"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(
                    {
                        "cache_key": cache_key,
                        "prompt_version": "domain_pack_generator_v2",
                        "model": "fake-model",
                        "pack_path": str(cached_pack_path),
                        "stage_payloads": {},
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            generator = DomainPackGenerator(
                chat_fn=FakeChat([]),
                memory_dir=memory_dir,
                model="fake-model",
            )

            cached = generator._load_cached_pack(cache_path)

        self.assertIsNone(cached)

    def test_generator_populates_observation_scopes_from_request_and_domain_model_when_integration_omits_them(self):
        request = _request()
        responses = copy.deepcopy(_responses())
        responses[4] = copy.deepcopy(responses[4])
        responses[4]["observation_scopes"] = {
            "main_scope": "",
            "sub_scopes": [],
            "scope_aliases": [],
            "scope_echo_terms": [],
            "off_domain_anchor_terms": [],
        }
        fake_chat = FakeChat(responses)
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = DomainPackGenerator(
                chat_fn=fake_chat,
                memory_dir=Path(tmpdir),
                model="fake-model",
            )

            pack = generator.generate(request, refresh_cache=True)

        scopes = pack.observation_scopes
        self.assertEqual(scopes["main_scope"], "电池材料")
        self.assertIn("solid-state electrolyte", scopes["sub_scopes"])
        self.assertIn("电池材料", scopes["scope_echo_terms"])
        self.assertIn("battery materials", scopes["scope_aliases"])
        self.assertIn("招聘", scopes["off_domain_anchor_terms"])

    def test_generator_expands_candidate_object_terms_with_request_keywords_and_synonyms(self):
        request = _request()
        responses = copy.deepcopy(_responses())
        responses[1] = copy.deepcopy(responses[1])
        responses[1]["technical_object_types"] = ["electrolyte"]
        responses[4] = copy.deepcopy(responses[4])
        responses[4]["candidate_formation"] = copy.deepcopy(responses[4]["candidate_formation"])
        responses[4]["candidate_formation"]["technical_object_types"] = ["electrolyte"]
        fake_chat = FakeChat(responses)
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = DomainPackGenerator(
                chat_fn=fake_chat,
                memory_dir=Path(tmpdir),
                model="fake-model",
            )

            pack = generator.generate(request, refresh_cache=True)

        object_terms = pack.candidate_formation["technical_object_types"]
        self.assertIn("electrolyte", object_terms)
        self.assertIn("电池材料", object_terms)
        self.assertIn("固态电池", object_terms)
        self.assertIn("battery materials", object_terms)

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
        self.assertEqual(fake_chat.call_count, 5)

    def test_cached_pack_uses_stage_payloads_to_recover_missing_candidate_fields(self):
        request = _request()
        responses = copy.deepcopy(_responses())
        stage_two_candidate_formation = copy.deepcopy(responses[1])
        cached_payload = copy.deepcopy(responses[4])
        cached_payload["candidate_formation"]["generic_terms"] = []
        cached_payload["candidate_formation"]["shell_terms"] = []

        with tempfile.TemporaryDirectory() as tmpdir:
            memory_dir = Path(tmpdir)
            cached_pack_path = save_domain_pack_to_memory(
                DomainPack.from_dict(cached_payload),
                memory_dir=memory_dir,
            )
            cache_key = build_domain_pack_cache_key(
                request,
                model="fake-model",
                prompt_version=DOMAIN_PACK_GENERATOR_PROMPT_VERSION,
            )
            cache_path = memory_dir / "cache" / f"{cache_key}.json"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(
                    {
                        "cache_key": cache_key,
                        "prompt_version": DOMAIN_PACK_GENERATOR_PROMPT_VERSION,
                        "model": "fake-model",
                        "pack_path": str(cached_pack_path),
                        "stage_payloads": {"candidate_formation": stage_two_candidate_formation},
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            fake_chat = FakeChat([])
            generator = DomainPackGenerator(
                chat_fn=fake_chat,
                memory_dir=memory_dir,
                model="fake-model",
            )

            pack = generator.generate(request, refresh_cache=False)

        self.assertEqual(fake_chat.call_count, 0)
        self.assertEqual(
            pack.candidate_formation["generic_terms"],
            stage_two_candidate_formation["generic_terms"],
        )
        self.assertEqual(
            pack.candidate_formation["shell_terms"],
            stage_two_candidate_formation["shell_terms"],
        )

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
