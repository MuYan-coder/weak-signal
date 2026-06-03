# Domain Pack Stage 3 Quality Gates Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Domain Pack dry-run, validator, and workflow quality gates so generated packs cannot enter later pipeline stages without schema checks, dry-run evidence, and review status.

**Architecture:** `domain_pack_dry_run.py` computes deterministic small-sample quality metrics and writes dry-run reports. `domain_pack_validator.py` validates schema, structured rule shape, broad keywords, conflicts, and dry-run gates without starting dry-run itself. `domain_pack_workflow.py` is the single orchestration layer that calls generator, validator, dry-run, and review state in order.

**Tech Stack:** Python dataclasses, `json`, existing DomainPack model and stage 2 save helpers, `unittest`.

---

### Task 1: Dry-Run Report

**Files:**
- Create: `tests/test_domain_pack_stage3_quality.py`
- Create: `src/domain/domain_pack_dry_run.py`

- [ ] **Step 1: Write the failing test**

```python
report = run_domain_pack_dry_run(pack, sample_documents=docs, candidate_records=candidates, memory_dir=tmpdir)
assert report.sample_doc_count == 2
assert report.candidate_count == 3
assert report.specific_candidate_count == 1
assert report.shell_candidate_ratio > 0
assert "招聘" in report.off_domain_leakage_terms
assert (tmpdir / f"{pack.domain_pack_hash}_dry_run.json").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_domain_pack_stage3_quality -v`
Expected: FAIL with missing `src.domain.domain_pack_dry_run`.

- [ ] **Step 3: Write minimal implementation**

Implement `DomainPackDryRunReport`, `run_domain_pack_dry_run()`, and `save_domain_pack_dry_run_report()`. Use candidate records when provided; otherwise derive light candidate records from sample text by matching Domain Pack terms.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_domain_pack_stage3_quality -v`
Expected: dry-run report tests pass.

### Task 2: Validator

**Files:**
- Modify: `tests/test_domain_pack_stage3_quality.py`
- Create: `src/domain/domain_pack_validator.py`

- [ ] **Step 1: Write the failing test**

```python
missing = validate_domain_pack(pack, dry_run_report=None)
assert not missing.is_valid
assert "dry-run" in " ".join(missing.errors)

valid = validate_domain_pack(pack, dry_run_report=good_report)
assert valid.is_valid

bad = validate_domain_pack(broad_pack, dry_run_report=bad_report)
assert not bad.is_valid
assert bad.gate_status == "blocked"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_domain_pack_stage3_quality -v`
Expected: FAIL with missing validator API.

- [ ] **Step 3: Write minimal implementation**

Implement `DomainPackValidationReport` and `validate_domain_pack()`. Check required sections, list fields, nonempty keywords and identity, generic/shell term minimums, broad-only keywords, valid/invalid pattern overlap, object family alias evidence, dry-run existence, shell ratio threshold, and off-domain leakage.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_domain_pack_stage3_quality -v`
Expected: validator tests pass.

### Task 3: Workflow

**Files:**
- Modify: `tests/test_domain_pack_stage3_quality.py`
- Create: `src/domain/domain_pack_workflow.py`
- Modify: `src/domain/__init__.py`

- [ ] **Step 1: Write the failing test**

```python
workflow = DomainPackWorkflow(generator=fake_generator, memory_dir=tmpdir)
result = workflow.prepare_domain_pack(request, sample_documents=docs, human_review_approved=True)
assert result.status == "ready"
assert fake_generator.calls == 1
assert result.validation_report.is_valid
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_domain_pack_stage3_quality -v`
Expected: FAIL with missing workflow API.

- [ ] **Step 3: Write minimal implementation**

Implement `DomainPackWorkflowResult` and `DomainPackWorkflow.prepare_domain_pack()`. It should call generator, run basic validation without dry-run, run dry-run, run final validation, then mark `ready`, `pending_review`, or `blocked`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_domain_pack_stage3_quality -v`
Expected: workflow tests pass.

### Task 4: Verification

**Files:**
- Test: `tests/test_domain_pack_stage3_quality.py`
- Regression: stage 0-2 tests

- [ ] **Step 1: Compile new modules**

Run: `python -m py_compile src/domain/domain_pack_dry_run.py src/domain/domain_pack_validator.py src/domain/domain_pack_workflow.py tests/test_domain_pack_stage3_quality.py`
Expected: exit code 0.

- [ ] **Step 2: Run focused tests**

Run: `python -m unittest tests.test_domain_pack_stage3_quality -v`
Expected: all stage 3 tests pass.

- [ ] **Step 3: Run stage 0-3 regression set**

Run: `python -m unittest tests.test_domain_pack_stage3_quality tests.test_domain_pack_generator tests.test_domain_pack_loader tests.test_event_extraction_refactor tests.test_data_access tests.test_phase0_generalization_baseline -v`
Expected: all tests pass.
