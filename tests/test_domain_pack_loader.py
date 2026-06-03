import tempfile
from pathlib import Path
import unittest

import yaml

from src.domain import (
    DomainContext,
    DomainPack,
    load_domain_context,
    load_domain_pack,
)


class DomainPackLoaderTest(unittest.TestCase):
    def test_domain_pack_model_exposes_required_runtime_metadata(self):
        pack = DomainPack.from_dict(
            {
                "schema_version": "domain_pack_v1",
                "pack_id": "neutral",
                "pack_name": "Neutral Domain Pack",
                "source": {
                    "mode": "preset",
                    "model": "",
                    "prompt_version": "",
                    "based_on_user_input": {
                        "field_id": "neutral",
                        "field_name": "Neutral",
                        "keywords": [],
                        "synonyms": [],
                        "exclude_terms": [],
                    },
                },
                "domain_identity": {
                    "field_id": "neutral",
                    "field_name": "Neutral",
                },
            }
        )

        context = DomainContext.from_pack(pack, runtime_mode="neutral")

        self.assertEqual(pack.schema_version, "domain_pack_v1")
        self.assertEqual(pack.pack_id, "neutral")
        self.assertEqual(pack.domain_identity["field_name"], "Neutral")
        self.assertTrue(pack.domain_pack_hash)
        self.assertEqual(context.domain_pack_id, "neutral")
        self.assertEqual(context.domain_pack_version, pack.domain_pack_version)
        self.assertEqual(context.domain_pack_hash, pack.domain_pack_hash)
        self.assertEqual(context.runtime_mode, "neutral")

    def test_missing_pack_falls_back_to_neutral_context(self):
        pack = load_domain_pack("missing_pack")
        context = load_domain_context("missing_pack")

        self.assertEqual(pack.pack_id, "neutral")
        self.assertEqual(pack.source["mode"], "preset")
        self.assertEqual(context.domain_pack_id, "neutral")
        self.assertEqual(context.runtime_mode, "neutral")
        self.assertEqual(context.source_query["field_id"], "neutral")

    def test_loads_humanoid_robot_preset(self):
        pack = load_domain_pack("humanoid_robot")

        self.assertEqual(pack.pack_id, "humanoid_robot")
        self.assertEqual(pack.domain_pack_version, "preset.v1")
        self.assertIn("humanoid robot", pack.search_strategy["english_terms"])
        self.assertIn("人形机器人", pack.search_strategy["core_keywords"])
        self.assertIn("humanoid robot", pack.observation_scopes["scope_aliases"])
        self.assertIn("机器人", pack.candidate_formation["generic_terms"])
        self.assertTrue(pack.canonicalization["object_family_enabled"])
        self.assertTrue(pack.domain_pack_hash)

    def test_explicit_file_loading_supports_user_generated_pack(self):
        payload = {
            "schema_version": "domain_pack_v1",
            "pack_id": "battery_materials",
            "pack_name": "Battery Materials",
            "pack_version": "generated.v1",
            "source": {
                "mode": "llm_generated",
                "model": "test-model",
                "prompt_version": "domain_pack_prompt_v1",
                "based_on_user_input": {
                    "field_id": "battery_materials",
                    "field_name": "电池材料",
                    "keywords": ["电池材料"],
                    "synonyms": ["battery materials"],
                    "exclude_terms": ["招聘"],
                },
            },
            "domain_identity": {
                "field_id": "battery_materials",
                "field_name": "电池材料",
            },
            "search_strategy": {
                "core_keywords": ["电池材料"],
                "synonyms": ["battery materials"],
                "english_terms": ["battery materials"],
                "exclude_terms": ["招聘"],
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "battery_materials.yaml"
            path.write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")

            pack = load_domain_pack(path)

        self.assertEqual(pack.pack_id, "battery_materials")
        self.assertEqual(pack.domain_pack_version, "generated.v1")
        self.assertEqual(pack.source["mode"], "llm_generated")
        self.assertIn("battery materials", pack.search_strategy["english_terms"])

    def test_domain_pack_hash_excludes_non_semantic_runtime_metadata(self):
        base = {
            "schema_version": "domain_pack_v1",
            "pack_id": "material",
            "pack_name": "Material",
            "source": {
                "mode": "llm_generated",
                "model": "test-model",
                "prompt_version": "domain_pack_prompt_v1",
                "based_on_user_input": {
                    "field_id": "material",
                    "field_name": "新型材料",
                    "keywords": [" 新型材料 ", "新型材料"],
                    "synonyms": ["advanced materials"],
                    "exclude_terms": [],
                },
            },
            "domain_identity": {
                "field_id": "material",
                "field_name": "新型材料",
            },
            "search_strategy": {
                "core_keywords": ["新型材料"],
                "synonyms": ["advanced materials"],
                "english_terms": ["advanced materials"],
                "exclude_terms": [],
            },
        }
        first = dict(base)
        first["generated_at"] = "2026-01-01T00:00:00"
        first["runtime_state"] = {"result_dir": "result/run-a"}
        second = dict(base)
        second["generated_at"] = "2026-02-01T00:00:00"
        second["runtime_state"] = {"result_dir": "result/run-b"}

        first_pack = DomainPack.from_dict(first)
        second_pack = DomainPack.from_dict(second)

        self.assertEqual(first_pack.domain_pack_hash, second_pack.domain_pack_hash)


if __name__ == "__main__":
    unittest.main()
