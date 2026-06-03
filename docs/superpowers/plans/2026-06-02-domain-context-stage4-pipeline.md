# Domain Context Stage 4 Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Thread one Domain Pack through pipeline execution, result snapshots, event cache isolation, historical event continuation, and report regeneration.

**Architecture:** `AnalysisPipeline` owns a `DomainContext` and exposes explicit recovery helpers. Pipeline result saving uses the existing shared Domain Pack snapshot writer and copies dry-run reports without duplicating serialization logic. Cache file names include `domain_pack_hash` so different packs never reuse event extraction cache.

**Tech Stack:** Python dataclasses, pandas, unittest, existing `src.domain` Domain Pack APIs.

---

### Task 1: Pipeline Domain Context Tests

**Files:**
- Create: `tests/test_domain_context_pipeline.py`
- Modify: `src/core/pipeline.py`

- [ ] **Step 1: Write the failing tests**

```python
class DomainContextPipelineTest(unittest.TestCase):
    def test_event_cache_path_includes_domain_pack_hash(self):
        pipeline = AnalysisPipeline(domain_context=DomainContext.from_pack(_pack("battery")))
        other = AnalysisPipeline(domain_context=DomainContext.from_pack(_pack("solar")))
        raw = pd.DataFrame([{"id": "doc1", "source_type": "paper", "title": "t", "text": "x"}])
        with tempfile.TemporaryDirectory() as tmpdir:
            first_path = pipeline._event_cache_path(raw, Path(tmpdir))
            second_path = other._event_cache_path(raw, Path(tmpdir))
        self.assertIn(pipeline.domain_context.domain_pack_hash, first_path.name)
        self.assertNotEqual(first_path.name, second_path.name)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_domain_context_pipeline -v`

Expected: FAIL because `AnalysisPipeline.__init__` does not accept `domain_context`.

- [ ] **Step 3: Implement minimal context ownership**

Add `domain_context` and `domain_pack_ref` constructor parameters, load neutral when omitted, and expose `_domain_metadata()`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_domain_context_pipeline -v`

Expected: PASS for cache path behavior.

### Task 2: Result Snapshot Tests

**Files:**
- Modify: `tests/test_domain_context_pipeline.py`
- Modify: `src/core/pipeline.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_save_results_writes_domain_pack_snapshot_and_metadata_columns(self):
    pipeline = AnalysisPipeline(domain_context=DomainContext.from_pack(_pack("battery")))
    dry_run = DomainPackDryRunReport(
        domain_pack_id=pipeline.domain_context.domain_pack_id,
        domain_pack_hash=pipeline.domain_context.domain_pack_hash,
        sample_doc_count=1,
        candidate_count=1,
        specific_candidate_count=1,
        shell_candidate_ratio=0.0,
        off_domain_leakage_terms=[],
        top_invalid_candidate_reasons={},
        recommended_pack_changes=[],
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        save_domain_pack_dry_run_report(dry_run, memory_dir=Path(tmpdir) / "memory")
        pipeline.domain_pack_dry_run_report_path = Path(tmpdir) / "memory" / f"{pipeline.domain_context.domain_pack_hash}_dry_run.json"
        result_dir = Path(tmpdir) / "result"
        pipeline._save_results(result_dir, events_df, candidates_df, scored_df, refined_df, validated_df, signals_output, "report", raw_data=raw_df)
        self.assertTrue((result_dir / "domain_pack.yaml").exists())
        self.assertTrue((result_dir / "domain_pack_dry_run.json").exists())
        events = json.loads((result_dir / "events.json").read_text(encoding="utf-8"))
        self.assertEqual(events[0]["domain_pack_hash"], pipeline.domain_context.domain_pack_hash)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_domain_context_pipeline -v`

Expected: FAIL because `_save_results()` does not write Domain Pack snapshot or metadata columns.

- [ ] **Step 3: Implement snapshot saving**

Use `save_domain_pack_snapshot()` for `domain_pack.yaml`, copy `domain_pack_dry_run.json` when available, and add Domain Pack metadata columns before writing result dataframes.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_domain_context_pipeline -v`

Expected: PASS for snapshot and metadata behavior.

### Task 3: Historical Recovery Tests

**Files:**
- Modify: `tests/test_domain_context_pipeline.py`
- Modify: `src/core/pipeline.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_run_from_events_recovers_domain_pack_from_result_dir_snapshot(self):
    pipeline = AnalysisPipeline()
    with tempfile.TemporaryDirectory() as tmpdir:
        source_result_dir = Path(tmpdir) / "source"
        save_domain_pack_snapshot(pack, result_dir=source_result_dir)
        output = pipeline.run_from_events(events_df=events_df, raw_data_path=None, source_result_dir=source_result_dir)
    self.assertEqual(output["domain_context"].domain_pack_hash, pack.domain_pack_hash)
```

```python
def test_regenerate_report_from_result_restores_domain_pack_snapshot(self):
    source_result_dir = _write_minimal_result_dir(pack)
    output = pipeline.regenerate_report_from_result(source_result_dir)
    self.assertEqual(output["domain_context"].domain_pack_hash, pack.domain_pack_hash)
    self.assertTrue((output["result_dir"] / "domain_pack.yaml").exists())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m unittest tests.test_domain_context_pipeline -v`

Expected: FAIL because `source_result_dir` and Domain Pack recovery do not exist.

- [ ] **Step 3: Implement recovery helpers**

Add `_resolve_domain_context()`, `_restore_domain_context_from_result_dir()`, and `_restore_domain_context_from_frames()`. Update `run_from_events()` and `regenerate_report_from_result()` to use them and return `domain_context`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m unittest tests.test_domain_context_pipeline -v`

Expected: PASS for recovery behavior.

### Task 4: Verification

**Files:**
- Verify: `src/core/pipeline.py`
- Verify: `tests/test_domain_context_pipeline.py`

- [ ] **Step 1: Compile changed code**

Run: `python -m py_compile src/core/pipeline.py tests/test_domain_context_pipeline.py`

Expected: exit code 0.

- [ ] **Step 2: Run phase 4 tests**

Run: `python -m unittest tests.test_domain_context_pipeline -v`

Expected: all phase 4 tests pass.

- [ ] **Step 3: Run phase 0-4 regression**

Run: `python -m unittest tests.test_domain_context_pipeline tests.test_domain_pack_stage3_quality tests.test_domain_pack_generator tests.test_domain_pack_loader tests.test_event_extraction_refactor tests.test_data_access tests.test_phase0_generalization_baseline -v`

Expected: all selected regression tests pass.
