# Domain Pack Stage 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the Domain Pack schema, loader, neutral pack, and humanoid robot preset without wiring Domain Pack into the business pipeline yet.

**Architecture:** `src/domain/models.py` owns the typed runtime contract and canonical hash logic. `src/domain/domain_pack_loader.py` owns YAML/JSON loading, preset lookup, neutral fallback, and `DomainContext` creation. Preset files live under `src/config/domain_packs/`; generated/user-edited packs remain supported by loading explicit file paths.

**Tech Stack:** Python dataclasses, `PyYAML`, `json`, `hashlib`, `unittest`.

---

### Task 1: Domain Pack Model Contract

**Files:**
- Create: `tests/test_domain_pack_loader.py`
- Create: `src/domain/models.py`
- Create: `src/domain/__init__.py`

- [ ] **Step 1: Write the failing test**

```python
from src.domain import DomainPack, DomainContext

pack = DomainPack.from_dict({
    "schema_version": "domain_pack_v1",
    "pack_id": "neutral",
    "pack_name": "Neutral Domain Pack",
    "source": {"mode": "preset"},
    "domain_identity": {"field_id": "neutral", "field_name": "Neutral"},
})

assert pack.pack_id == "neutral"
assert pack.domain_identity["field_name"] == "Neutral"
assert pack.domain_pack_hash
context = DomainContext.from_pack(pack, runtime_mode="neutral")
assert context.domain_pack_id == "neutral"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_domain_pack_loader -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.domain'`.

- [ ] **Step 3: Write minimal implementation**

Create dataclasses `DomainPack` and `DomainContext`. `DomainPack.from_dict()` should copy the raw payload, expose top-level required fields, normalize missing sections to dictionaries, and compute a stable canonical hash.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_domain_pack_loader -v`
Expected: PASS for model construction tests.

### Task 2: Loader and Neutral Fallback

**Files:**
- Modify: `tests/test_domain_pack_loader.py`
- Create: `src/domain/domain_pack_loader.py`
- Create: `src/config/domain_packs/neutral.yaml`

- [ ] **Step 1: Write the failing test**

```python
from src.domain import load_domain_pack, load_domain_context

pack = load_domain_pack("missing_pack")
assert pack.pack_id == "neutral"
context = load_domain_context("missing_pack")
assert context.runtime_mode == "neutral"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_domain_pack_loader -v`
Expected: FAIL because `load_domain_pack` is not defined or `neutral.yaml` is missing.

- [ ] **Step 3: Write minimal implementation**

Implement preset path lookup in `src/config/domain_packs/`, explicit YAML/JSON file loading, and neutral fallback when an ID or path cannot be found.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_domain_pack_loader -v`
Expected: PASS for neutral fallback tests.

### Task 3: Humanoid Robot Preset

**Files:**
- Modify: `tests/test_domain_pack_loader.py`
- Create: `src/config/domain_packs/humanoid_robot.yaml`

- [ ] **Step 1: Write the failing test**

```python
pack = load_domain_pack("humanoid_robot")
assert pack.pack_id == "humanoid_robot"
assert "humanoid robot" in pack.search_strategy["english_terms"]
assert "机器人" in pack.candidate_formation["generic_terms"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_domain_pack_loader -v`
Expected: FAIL because `humanoid_robot.yaml` is missing.

- [ ] **Step 3: Write minimal implementation**

Create the preset with the stage 1 schema fields: source, domain identity, search strategy, observation scopes, candidate formation, weak signal rules, canonicalization, evidence rules, and reporting.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_domain_pack_loader -v`
Expected: PASS for humanoid preset tests.

### Task 4: Hash Stability and Explicit File Loading

**Files:**
- Modify: `tests/test_domain_pack_loader.py`
- Modify: `src/domain/models.py`
- Modify: `src/domain/domain_pack_loader.py`

- [ ] **Step 1: Write the failing test**

```python
payload = {
    "schema_version": "domain_pack_v1",
    "pack_id": "material",
    "source": {
        "mode": "llm_generated",
        "model": "test-model",
        "prompt_version": "domain_pack_prompt_v1",
        "based_on_user_input": {
            "field_id": "material",
            "field_name": "新型材料",
            "keywords": ["新型材料"],
            "synonyms": ["advanced materials"],
            "exclude_terms": [],
        },
    },
    "domain_identity": {"field_id": "material", "field_name": "新型材料"},
    "search_strategy": {
        "core_keywords": ["新型材料"],
        "synonyms": ["advanced materials"],
        "english_terms": ["advanced materials"],
        "exclude_terms": [],
    },
}
first = DomainPack.from_dict({**payload, "generated_at": "2026-01-01T00:00:00"})
second = DomainPack.from_dict({**payload, "generated_at": "2026-02-01T00:00:00"})
assert first.domain_pack_hash == second.domain_pack_hash
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_domain_pack_loader -v`
Expected: FAIL if hash includes non-semantic metadata.

- [ ] **Step 3: Write minimal implementation**

Canonicalize hash input by sorting dictionaries, de-duplicating scalar lists, trimming/case-normalizing strings, and excluding `generated_at`, paths, runtime state, and UI state.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_domain_pack_loader -v`
Expected: PASS for hash stability and explicit file loading tests.

### Task 5: Verification

**Files:**
- Test: `tests/test_domain_pack_loader.py`
- Test: `tests/test_event_extraction_refactor.py`
- Test: `tests/test_data_access.py`
- Test: `tests/test_phase0_generalization_baseline.py`

- [ ] **Step 1: Run focused tests**

Run: `python -m unittest tests.test_domain_pack_loader -v`
Expected: all Domain Pack stage 1 tests pass.

- [ ] **Step 2: Run phase 0 and stage 1 regression set**

Run: `python -m unittest tests.test_domain_pack_loader tests.test_event_extraction_refactor tests.test_data_access tests.test_phase0_generalization_baseline -v`
Expected: all tests pass.
