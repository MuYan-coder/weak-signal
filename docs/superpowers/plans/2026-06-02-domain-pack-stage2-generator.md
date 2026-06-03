# Domain Pack Stage 2 Generator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an offline LLM Domain Pack generator that can create, cache, and save candidate Domain Packs for arbitrary technology domains without wiring them into the pipeline.

**Architecture:** `src/domain/domain_pack_generator.py` owns request modeling, four prompt builders, LLM orchestration, response parsing, cache-key generation, and runtime file persistence. It reuses `DomainPack.from_dict()` for schema completion and hash calculation, writes generated packs to `memory/domain_packs/`, and never writes generated packs into `src/config/domain_packs/`.

**Tech Stack:** Python dataclasses, `json`, `yaml`, `hashlib`, existing `src.utils.llm_client.chat_text`, `unittest`.

---

### Task 1: Generator Request and Prompt Builders

**Files:**
- Create: `tests/test_domain_pack_generator.py`
- Create: `src/domain/domain_pack_generator.py`
- Modify: `src/domain/__init__.py`

- [ ] **Step 1: Write the failing test**

```python
request = DomainPackGenerationRequest(
    field_id="battery_materials",
    field_name="电池材料",
    keywords=["电池材料"],
    synonyms=["battery materials"],
    exclude_terms=["招聘"],
    source_types=["paper", "patent"],
    sample_documents=[{"title": "固态电池电解质界面改性", "text": "界面阻抗降低"}],
)
prompts = [
    build_domain_modeling_prompt(request),
    build_candidate_formation_prompt(request, {"domain_boundary": "电池材料"}),
    build_weak_signal_rules_prompt(request, {"domain_boundary": "电池材料"}, {"technical_object_types": ["electrolyte"]}),
    build_integration_prompt(request, {"domain_boundary": "电池材料"}, {"technical_object_types": ["electrolyte"]}, {"early_stage_markers": ["prototype"]}),
]
assert "阶段一" in prompts[0]
assert "阶段二" in prompts[1]
assert "阶段三" in prompts[2]
assert "domain_pack_v1" in prompts[3]
assert "technical_object" in prompts[3]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_domain_pack_generator -v`
Expected: FAIL with `ImportError` for missing generator API.

- [ ] **Step 3: Write minimal implementation**

Implement `DomainPackGenerationRequest`, `build_domain_modeling_prompt()`, `build_candidate_formation_prompt()`, `build_weak_signal_rules_prompt()`, and `build_integration_prompt()`. Export these symbols from `src/domain/__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_domain_pack_generator -v`
Expected: prompt builder tests pass.

### Task 2: LLM Orchestration and Runtime Saving

**Files:**
- Modify: `tests/test_domain_pack_generator.py`
- Modify: `src/domain/domain_pack_generator.py`

- [ ] **Step 1: Write the failing test**

```python
generator = DomainPackGenerator(chat_fn=fake_chat, memory_dir=tmp_path, model="fake-model")
pack = generator.generate(request, refresh_cache=True)
assert pack.pack_id == "battery_materials"
assert pack.source["mode"] == "llm_generated"
assert (tmp_path / f"{pack.domain_pack_hash}.yaml").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_domain_pack_generator -v`
Expected: FAIL because `DomainPackGenerator.generate()` is not implemented.

- [ ] **Step 3: Write minimal implementation**

Call the four prompt stages in order. Parse each LLM response as JSON or YAML. Normalize the final payload with request metadata, create a `DomainPack`, and save it to `memory_dir/<domain_pack_hash>.yaml`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_domain_pack_generator -v`
Expected: generation and runtime save tests pass.

### Task 3: Cache Reuse and Cache Key Contract

**Files:**
- Modify: `tests/test_domain_pack_generator.py`
- Modify: `src/domain/domain_pack_generator.py`

- [ ] **Step 1: Write the failing test**

```python
cache_key = build_domain_pack_cache_key(request, model="fake-model", prompt_version=DOMAIN_PACK_GENERATOR_PROMPT_VERSION)
assert "battery_materials" in cache_key
assert "fake-model" in cache_key
assert DOMAIN_PACK_GENERATOR_PROMPT_VERSION in cache_key
first_pack = generator.generate(request, refresh_cache=True)
second_pack = generator.generate(request, refresh_cache=False)
assert first_pack.domain_pack_hash == second_pack.domain_pack_hash
assert fake_chat.call_count == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_domain_pack_generator -v`
Expected: FAIL because cache key and cache reuse are not implemented.

- [ ] **Step 3: Write minimal implementation**

Write cache files to `memory_dir/cache/<cache_key>.json` with generated pack path and response metadata. When `refresh_cache=False`, load the existing generated pack instead of calling the LLM.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_domain_pack_generator -v`
Expected: cache key and reuse tests pass.

### Task 4: Failure Safety

**Files:**
- Modify: `tests/test_domain_pack_generator.py`
- Modify: `src/domain/domain_pack_generator.py`

- [ ] **Step 1: Write the failing test**

```python
generator = DomainPackGenerator(chat_fn=lambda *args, **kwargs: ("not-json", {}, None), memory_dir=tmp_path)
with assertRaises(DomainPackGenerationError):
    generator.generate(request, refresh_cache=True)
assert list(tmp_path.glob("*.yaml")) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_domain_pack_generator -v`
Expected: FAIL if generator swallows parse errors or writes files on failure.

- [ ] **Step 3: Write minimal implementation**

Raise `DomainPackGenerationError` when any stage cannot be parsed or the final pack is structurally unusable. Do not save pack or cache files until a `DomainPack` has been constructed.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_domain_pack_generator -v`
Expected: failure-safety tests pass.

### Task 5: Verification

**Files:**
- Test: `tests/test_domain_pack_generator.py`
- Test: `tests/test_domain_pack_loader.py`
- Test: `tests/test_event_extraction_refactor.py`
- Test: `tests/test_data_access.py`
- Test: `tests/test_phase0_generalization_baseline.py`

- [ ] **Step 1: Compile new modules**

Run: `python -m py_compile src/domain/domain_pack_generator.py tests/test_domain_pack_generator.py`
Expected: exit code 0.

- [ ] **Step 2: Run focused tests**

Run: `python -m unittest tests.test_domain_pack_generator -v`
Expected: all stage 2 tests pass.

- [ ] **Step 3: Run stage 0-2 regression set**

Run: `python -m unittest tests.test_domain_pack_generator tests.test_domain_pack_loader tests.test_event_extraction_refactor tests.test_data_access tests.test_phase0_generalization_baseline -v`
Expected: all tests pass.
