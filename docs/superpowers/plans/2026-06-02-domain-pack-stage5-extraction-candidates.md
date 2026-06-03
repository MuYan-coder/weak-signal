# Domain Pack Stage 5 Extraction and Candidate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make event normalization and candidate formation read the current Domain Pack instead of defaulting to humanoid robot lexicon rules.

**Architecture:** `tech_lexicon.py` exposes a context-aware `DomainLexicon` built from `DomainPack`. `event_extractor.py` accepts `domain_context` or `domain_pack` and uses that lexicon during technology normalization, scope detection, and robot guard decisions. `candidate_former.py` accepts the same context, applies Domain Pack candidate slots, shell terms, and pattern metadata, and records rule hit reasons on candidate rows.

**Tech Stack:** Python, pandas, unittest, existing Domain Pack dataclasses.

---

### Task 1: Context-Aware Lexicon

**Files:**
- Modify: `src/extraction/tech_lexicon.py`
- Test: `tests/test_no_default_robot_leakage.py`

- [x] **Step 1: Write failing test**

```python
def test_non_robot_domain_lexicon_does_not_inject_robot_defaults(self):
    pack = _battery_pack()
    lexicon = build_domain_lexicon(pack)
    self.assertEqual(lexicon.detect_supported_observation_scopes("robot planning world model"), [])
    self.assertIn("电池材料", lexicon.detect_supported_observation_scopes("固态电池电解质界面钝化"))
    self.assertNotIn("humanoid robot", lexicon.observation_scopes)
```

- [x] **Step 2: Verify RED**

Run: `python -m unittest tests.test_no_default_robot_leakage -v`

Expected: FAIL because `build_domain_lexicon` is missing.

- [x] **Step 3: Implement minimal lexicon**

Create `DomainLexicon` with methods for aliases, scope detection, token extraction, technology normalization, and robot preset guard.

- [x] **Step 4: Verify GREEN**

Run: `python -m unittest tests.test_no_default_robot_leakage -v`

Expected: PASS for lexicon behavior.

### Task 2: Event Extraction Context

**Files:**
- Modify: `src/extraction/event_extractor.py`
- Modify: `src/core/pipeline.py`
- Test: `tests/test_no_default_robot_leakage.py`

- [x] **Step 1: Write failing test**

```python
events = process_events(raw, use_api=False, refresh_cache=True, cache_path=None, domain_context=DomainContext.from_pack(pack))
self.assertNotIn("humanoid robot", " ".join(events["observation_scopes"].astype(str)))
self.assertIn("电池材料", " ".join(events["observation_scopes"].astype(str)))
```

- [x] **Step 2: Verify RED**

Run: `python -m unittest tests.test_no_default_robot_leakage -v`

Expected: FAIL because `process_events()` lacks domain context support.

- [x] **Step 3: Implement minimal extraction wiring**

Pass `DomainLexicon` through local and cached event normalization paths. Do not remove existing neutral/humanoid behavior.

- [x] **Step 4: Verify GREEN**

Run: `python -m unittest tests.test_no_default_robot_leakage -v`

Expected: PASS for event extraction behavior.

### Task 3: Candidate Formation Context

**Files:**
- Modify: `src/extraction/candidate_former.py`
- Modify: `src/core/pipeline.py`
- Test: `tests/test_no_default_robot_leakage.py`

- [x] **Step 1: Write failing test**

```python
candidates = build_candidate_forms(events, raw, domain_context=DomainContext.from_pack(pack))
self.assertIn("钝化", " ".join(candidates["display_candidate_name"].astype(str)))
self.assertIn("battery_object_mechanism", " ".join(candidates["domain_pack_candidate_rule_ids"].astype(str)))
self.assertNotIn("机器人", " ".join(candidates["display_candidate_name"].astype(str)))
```

- [x] **Step 2: Verify RED**

Run: `python -m unittest tests.test_no_default_robot_leakage -v`

Expected: FAIL because `build_candidate_forms()` lacks Domain Pack rule trace columns.

- [x] **Step 3: Implement minimal candidate wiring**

Use Domain Pack object/mechanism/task/data/method/shell terms to enrich token extraction, reject shell-only candidates, and write rule hit columns.

- [x] **Step 4: Verify GREEN**

Run: `python -m unittest tests.test_no_default_robot_leakage -v`

Expected: PASS for candidate specificity and rule trace behavior.

### Task 4: Verification

**Files:**
- Verify: `src/extraction/tech_lexicon.py`
- Verify: `src/extraction/event_extractor.py`
- Verify: `src/extraction/candidate_former.py`
- Verify: `src/core/pipeline.py`
- Verify: `tests/test_no_default_robot_leakage.py`

- [x] **Step 1: Compile**

Run: `python -m py_compile src/extraction/tech_lexicon.py src/extraction/event_extractor.py src/extraction/candidate_former.py src/core/pipeline.py tests/test_no_default_robot_leakage.py`

Expected: exit code 0.

- [x] **Step 2: Stage 5 tests**

Run: `python -m unittest tests.test_no_default_robot_leakage -v`

Expected: all tests pass.

- [x] **Step 3: Regression**

Run: `python -m unittest tests.test_no_default_robot_leakage tests.test_domain_context_pipeline tests.test_domain_pack_stage3_quality tests.test_domain_pack_generator tests.test_domain_pack_loader tests.test_event_extraction_refactor tests.test_data_access tests.test_phase0_generalization_baseline -v`

Expected: all selected tests pass.
