# Space Manufacturing Domain Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove humanoid robot preset leakage from space manufacturing candidate formation and weak-signal scoring while preserving the humanoid preset behavior behind an explicit legacy/domain profile.

**Architecture:** Introduce a small `DomainCandidatePolicy` adapter that reads `DomainLexicon` and Domain Pack sections and answers candidate-formation questions such as scope labels, generic/shell terms, surface patterns, technical anchors, and legacy humanoid mode. `candidate_former.py`, `candidate_eligibility.py`, and `scorer.py` will use this policy instead of direct humanoid constants on non-humanoid Domain Packs. Existing humanoid constants remain available only when the active domain is the humanoid preset or legacy robot rules are enabled.

**Tech Stack:** Python, pandas, unittest, existing Domain Pack dataclasses and `DomainLexicon`.

---

## Scope

This plan fixes the path where space manufacturing runs are polluted by humanoid robot defaults:

```text
candidate_former.py hard-coded humanoid rules
  -> display name / cluster signature / scope-shell fields
  -> candidate_eligibility.py eligibility gate
  -> scorer.py resolved granularity and score suppression
  -> weak-signal output
```

The plan does not rewrite event extraction, LLM prompts, or the Domain Pack generator. It only changes runtime candidate policy and the downstream gates that consume candidate fields.

## File Structure

- Create: `src/extraction/domain_candidate_policy.py`
  - Owns domain-aware candidate policy methods.
  - Provides a legacy humanoid profile while making the default non-humanoid profile neutral.
- Modify: `src/extraction/candidate_former.py`
  - Builds a policy from the active `domain_context` or `domain_pack`.
  - Uses policy for scope labels, generic object labels, method-only display checks, surface hints, title topic anchors, cluster specificity, and bridge logic gating.
- Modify: `src/scoring/candidate_eligibility.py`
  - Accepts `domain_context` and derives technical anchors from Domain Pack candidate terms.
- Modify: `src/scoring/scorer.py`
  - Passes `domain_context` to eligibility.
  - Uses domain-aware scope-shell fallback instead of hard-coded humanoid shell tokens.
- Modify: `src/core/pipeline.py`
  - Passes `self.domain_context` into candidate eligibility application.
- Test: `tests/test_space_manufacturing_domain_policy.py`
  - New focused regression tests for space manufacturing contamination and eligibility.
- Test: `tests/test_no_default_robot_leakage.py`
  - Keep broad leakage regression intact and run it as a required regression suite after candidate formation changes.
- Test: `tests/test_candidate_eligibility.py`
  - Add direct Domain Pack-aware eligibility cases.
- Test: `tests/test_scoring_score_suppression.py`
  - Add score-suppression case where a valid space manufacturing candidate is not downgraded by humanoid shell logic.

---

### Task 1: Space Manufacturing Leakage Tests

**Files:**
- Create: `tests/test_space_manufacturing_domain_policy.py`
- Verify: `src/extraction/candidate_former.py`
- Verify: `src/scoring/candidate_eligibility.py`
- Verify: `src/scoring/scorer.py`

- [ ] **Step 1: Write the failing test module**

Create `tests/test_space_manufacturing_domain_policy.py` with these tests:

```python
import unittest

import pandas as pd

from src.domain.models import DomainContext, DomainPack
from src.extraction.candidate_former import build_candidate_forms
from src.scoring.candidate_eligibility import evaluate_candidate_eligibility
from src.scoring.scorer import score_all_candidates


def _space_pack() -> DomainPack:
    payload = {
        "schema_version": "domain_pack_v1",
        "pack_id": "space_manufacturing",
        "pack_name": "太空制造",
        "domain_pack_version": "2026-06-11-test",
        "domain_identity": {
            "field_id": "space_manufacturing",
            "field_name": "太空制造",
            "domain_boundary": "太空制造关注微重力、在轨、月面或空间站环境中的制造、装配、材料成形与验证。",
            "core_keywords": ["太空制造", "太空3D打印", "微重力制造", "在轨装配", "冷焊"],
            "english_terms": ["space manufacturing", "in-space manufacturing", "orbital manufacturing", "microgravity manufacturing"],
            "excluded_topics": ["人形机器人", "具身智能", "机械臂控制", "灵巧手抓取"],
        },
        "search_strategy": {"query_templates": [], "must_have_terms": [], "nice_to_have_terms": [], "exclude_terms": [], "source_weights": {}, "query_expansion_rules": []},
        "observation_scopes": {
            "main_scope": "太空制造",
            "sub_scopes": ["微重力制造", "在轨制造", "空间站制造"],
            "scope_aliases": ["space manufacturing", "in-space manufacturing", "orbital manufacturing", "microgravity manufacturing"],
            "scope_echo_terms": ["太空制造", "space manufacturing", "in-space manufacturing"],
            "off_domain_anchor_terms": ["人形机器人", "具身智能", "灵巧手", "夹爪", "humanoid robot", "dexterous hand"],
        },
        "candidate_formation": {
            "technical_object_types": ["太空3D打印", "微重力增材制造", "在轨装配", "冷焊", "空间站制造", "月壤建造", "orbital additive manufacturing", "microgravity manufacturing"],
            "mechanism_types": ["冷焊工艺", "additive manufacturing", "cold welding", "laser sintering", "electron beam melting", "microgravity solidification"],
            "task_or_performance_types": ["在轨维修", "在轨建造", "结构成形", "缺陷控制", "材料沉积", "on-orbit assembly"],
            "data_or_method_types": ["微重力实验", "真空热循环", "轨道验证", "parabolic flight test"],
            "scene_or_application_types": ["空间站", "近地轨道", "月面基地", "in orbit", "space station"],
            "generic_terms": ["太空制造", "space manufacturing", "技术", "方法", "系统", "平台", "方案"],
            "shell_terms": ["产业", "应用", "能力", "发展", "解决方案", "technology", "system", "method"],
            "valid_candidate_patterns": [
                {"pattern_id": "space_object_mechanism", "required_slots": ["technical_object", "mechanism"], "min_required_slot_count": 2, "evidence_required": True},
                {"pattern_id": "space_scene_mechanism", "required_slots": ["scene", "mechanism"], "min_required_slot_count": 2, "evidence_required": True}
            ],
            "invalid_candidate_patterns": [
                {"pattern_id": "space_scope_only", "reject_terms": ["太空制造", "space manufacturing"], "max_specific_slot_count": 1},
                {"pattern_id": "humanoid_only", "reject_terms": ["人形机器人", "具身智能", "灵巧手", "夹爪", "humanoid robot", "dexterous hand"], "max_specific_slot_count": 1}
            ],
            "minimum_specificity_rule": {"min_non_shell_slots": 2, "require_evidence_span": True, "allow_scope_only_candidate": False},
        },
        "weak_signal_rules": {
            "early_stage_markers": ["首次", "验证", "prototype", "demonstration"],
            "low_attention_markers": ["小规模", "实验室", "pilot"],
            "niche_actor_markers": ["大学", "初创"],
            "cross_domain_markers": ["地面制造", "航空制造", "材料科学"],
            "engineering_trace_markers": ["样机", "测试", "轨道验证"],
            "commercialization_noise_markers": ["融资", "估值", "市场规模"],
            "policy_or_market_noise_markers": ["政策", "补贴", "产业规划"],
            "scoring_adjustments": [],
        },
        "canonicalization": {"object_families": [], "alias_groups": [], "parent_child_terms": [], "do_not_merge_rules": [], "object_family_enabled": False},
        "evidence_rules": {"strong_evidence_patterns": [], "weak_evidence_patterns": [], "source_reliability_hints": [], "traceability_fields": [], "evidence_rejection_patterns": []},
        "reporting": {"display_labels": {"space_manufacturing": "太空制造"}, "explanation_templates": [], "review_hints": []},
    }
    return DomainPack.from_dict(payload)


def _space_context() -> DomainContext:
    return DomainContext.from_pack(_space_pack())


class SpaceManufacturingDomainPolicyTest(unittest.TestCase):
    def test_candidate_formation_does_not_use_humanoid_surface_terms(self):
        events = pd.DataFrame([
            {
                "id": "space-cold-weld",
                "technology": ["太空3D打印"],
                "technical_object": "轨道转移飞行器结构件",
                "mechanism": "冷焊工艺",
                "task": "在轨装配",
                "evidence_span": "团队验证太空3D打印结构件的冷焊工艺，可用于在轨装配。",
                "observation_scopes": ["太空制造"],
                "candidate_units": [
                    {
                        "raw_phrase": "太空3D打印结构件冷焊工艺",
                        "raw_candidate_text": "太空3D打印结构件冷焊工艺",
                        "mechanism_core": "冷焊工艺",
                        "mechanism_core_tokens": ["冷焊工艺"],
                        "object_modifier_tokens": ["太空3D打印", "轨道转移飞行器结构件"],
                        "task_constraint_tokens": ["在轨装配"],
                        "method_modifier_tokens": ["冷焊工艺"],
                        "scene_tokens": ["空间站"],
                        "scope_names": ["太空制造"],
                        "source_extraction_mode": "domain_pack_test",
                    }
                ],
            }
        ])
        raw = pd.DataFrame([
            {
                "id": "space-cold-weld",
                "source_type": "paper",
                "title": "太空3D打印结构件冷焊工艺验证",
                "text": "团队验证太空3D打印结构件的冷焊工艺，可用于在轨装配。文中背景提到机器人操作，但主题不是人形机器人。",
                "analysis_tech_field_name": "太空制造",
            }
        ])

        candidates = build_candidate_forms(events, raw, domain_context=_space_context())
        formed = candidates[candidates["candidate_stage"].astype(str).str.startswith("formed_candidate", na=False)]
        names = " ".join(formed["display_candidate_name"].fillna("").astype(str).tolist())

        self.assertFalse(formed.empty)
        self.assertIn("冷焊", names)
        self.assertNotIn("人形机器人", names)
        self.assertNotIn("具身智能", names)
        self.assertNotIn("灵巧手", names)
        self.assertNotIn("夹爪", names)
        self.assertTrue((formed["topic_granularity"].astype(str) == "fine_grained_topic").any())

    def test_scope_only_space_manufacturing_is_not_weak_signal_ready(self):
        events = pd.DataFrame([
            {
                "id": "space-shell",
                "technology": ["太空制造"],
                "technical_object": "太空制造",
                "mechanism": "",
                "task": "",
                "evidence_span": "地方发布太空制造产业发展方案。",
                "observation_scopes": ["太空制造"],
                "candidate_units": [
                    {
                        "raw_phrase": "太空制造技术方案",
                        "raw_candidate_text": "太空制造技术方案",
                        "mechanism_core": "",
                        "mechanism_core_tokens": [],
                        "object_modifier_tokens": ["太空制造"],
                        "task_constraint_tokens": [],
                        "scope_names": ["太空制造"],
                        "source_extraction_mode": "domain_pack_test",
                    }
                ],
            }
        ])
        raw = pd.DataFrame([
            {"id": "space-shell", "source_type": "policy", "title": "太空制造产业发展方案", "text": "地方发布太空制造产业发展方案。", "analysis_tech_field_name": "太空制造"}
        ])

        candidates = build_candidate_forms(events, raw, domain_context=_space_context())
        shell = candidates[candidates["raw_phrase"].astype(str) == "太空制造技术方案"]

        self.assertFalse(shell.empty)
        self.assertFalse((shell["candidate_stage"].astype(str) == "formed_candidate_strong").any())
        self.assertFalse((shell["display_tier"].astype(str) == "weak_signal").any())

    def test_space_domain_anchors_make_eligibility_and_scoring_domain_aware(self):
        row = {
            "id": "space-score",
            "display_candidate_name": "太空3D打印冷焊工艺",
            "raw_phrase": "太空3D打印结构件冷焊工艺",
            "raw_candidate_text": "太空3D打印结构件冷焊工艺",
            "candidate_stage": "formed_candidate",
            "topic_granularity": "fine_grained_topic",
            "display_tier": "weak_signal",
            "is_scope_internal_candidate": True,
            "has_mechanism_core": True,
            "has_non_scope_constraint": True,
            "survives_without_scope": True,
            "scope_shell_heavy": False,
            "mechanism_core": "冷焊工艺",
            "mechanism_core_tokens": ["冷焊工艺"],
            "object_modifier_tokens": ["太空3D打印"],
            "task_constraint_tokens": ["在轨装配"],
            "source_count": 2,
            "cluster_evidence_count": 2,
            "org_count": 2,
            "total_mentions": 2,
            "mention_dates": [],
            "evidence_items": [],
        }

        profile = evaluate_candidate_eligibility(row, domain_context=_space_context())
        scored = score_all_candidates(pd.DataFrame([row]), domain_context=_space_context()).iloc[0]

        self.assertIn(profile.candidate_eligibility, {"eligible", "candidate_monitoring"})
        self.assertNotEqual(scored["score_applicability"], "not_applicable")
        self.assertGreater(float(scored["weak_signal_raw_score"]), 0.0)
```

- [ ] **Step 2: Run the focused test to verify RED**

Run:

```powershell
python -m unittest tests.test_space_manufacturing_domain_policy -v
```

Expected: FAIL before implementation because `evaluate_candidate_eligibility()` does not accept `domain_context`, and candidate formation still uses hard-coded humanoid display/shell rules.

- [ ] **Step 3: Commit the RED test**

Run:

```powershell
git add tests/test_space_manufacturing_domain_policy.py
git commit -m "test: capture space manufacturing domain policy regressions"
```

Expected: commit succeeds with only the new test file staged.

---

### Task 2: Add Domain Candidate Policy Adapter

**Files:**
- Create: `src/extraction/domain_candidate_policy.py`
- Test: `tests/test_space_manufacturing_domain_policy.py`

- [ ] **Step 1: Create the policy module**

Create `src/extraction/domain_candidate_policy.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, List, Sequence, Tuple

from .tech_lexicon import normalize_signal_phrase


SurfacePattern = Tuple[str, Tuple[str, ...]]

LEGACY_HUMANOID_PACK_IDS = {"humanoid_robot", "humanoid_robot_preset", "legacy_humanoid_robot"}


def _dedupe(values: Iterable[Any]) -> List[str]:
    seen = set()
    result: List[str] = []
    for value in values or []:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        result.append(text)
        seen.add(text)
    return result


def _section(domain_lexicon: Any, name: str) -> dict:
    pack = getattr(domain_lexicon, "domain_pack", None)
    if pack is not None:
        value = getattr(pack, name, {})
        return value if isinstance(value, dict) else {}
    context = getattr(domain_lexicon, "domain_context", None)
    pack = getattr(context, "domain_pack", None)
    value = getattr(pack, name, {}) if pack is not None else {}
    return value if isinstance(value, dict) else {}


def _identity(domain_lexicon: Any) -> dict:
    return _section(domain_lexicon, "domain_identity")


@dataclass(frozen=True)
class DomainCandidatePolicy:
    domain_lexicon: Any
    legacy_scope_labels: dict[str, str] = field(default_factory=dict)
    legacy_generic_method_display_names: set[str] = field(default_factory=set)
    legacy_generic_object_labels: set[str] = field(default_factory=set)
    legacy_generic_tech_object_slots: set[str] = field(default_factory=set)
    legacy_surface_object_patterns: Sequence[SurfacePattern] = field(default_factory=tuple)
    legacy_surface_task_patterns: Sequence[SurfacePattern] = field(default_factory=tuple)
    legacy_title_topic_anchors: Sequence[str] = field(default_factory=tuple)

    @property
    def pack_id(self) -> str:
        identity = _identity(self.domain_lexicon)
        return str(identity.get("field_id") or getattr(self.domain_lexicon, "pack_id", "") or "").strip()

    @property
    def is_legacy_humanoid(self) -> bool:
        return bool(
            getattr(self.domain_lexicon, "use_legacy_robot_rules", False)
            or self.pack_id in LEGACY_HUMANOID_PACK_IDS
        )

    def scope_label(self, scope: str) -> str:
        text = str(scope or "").strip()
        if not text:
            return ""
        reporting = _section(self.domain_lexicon, "reporting")
        labels = reporting.get("display_labels", {}) if isinstance(reporting.get("display_labels", {}), dict) else {}
        if text in labels:
            return str(labels[text]).strip() or text
        if self.is_legacy_humanoid:
            return self.legacy_scope_labels.get(text, text)
        identity = _identity(self.domain_lexicon)
        field_name = str(identity.get("field_name", "")).strip()
        obs = _section(self.domain_lexicon, "observation_scopes")
        if text == obs.get("main_scope") and field_name:
            return field_name
        return text

    def generic_terms(self) -> List[str]:
        formation = _section(self.domain_lexicon, "candidate_formation")
        return _dedupe(formation.get("generic_terms", []))

    def shell_terms(self) -> List[str]:
        formation = _section(self.domain_lexicon, "candidate_formation")
        return _dedupe(formation.get("shell_terms", []))

    def technical_anchor_terms(self) -> List[str]:
        formation = _section(self.domain_lexicon, "candidate_formation")
        obs = _section(self.domain_lexicon, "observation_scopes")
        identity = _identity(self.domain_lexicon)
        return _dedupe(
            list(formation.get("technical_object_types", []))
            + list(formation.get("mechanism_types", []))
            + list(formation.get("task_or_performance_types", []))
            + list(formation.get("data_or_method_types", []))
            + list(formation.get("scene_or_application_types", []))
            + list(obs.get("technical_object", []))
            + list(identity.get("core_keywords", []))
            + list(identity.get("english_terms", []))
        )

    def generic_method_display_names(self) -> set[str]:
        if self.is_legacy_humanoid:
            return set(self.legacy_generic_method_display_names)
        return {normalize_signal_phrase(value) for value in self.generic_terms() + self.shell_terms()}

    def generic_object_labels(self) -> set[str]:
        if self.is_legacy_humanoid:
            return set(self.legacy_generic_object_labels)
        return set(self.generic_terms() + self.shell_terms())

    def generic_tech_object_slots(self) -> set[str]:
        if self.is_legacy_humanoid:
            return set(self.legacy_generic_tech_object_slots)
        return set(self.generic_terms() + self.shell_terms())

    def surface_object_patterns(self) -> Sequence[SurfacePattern]:
        return tuple(self.legacy_surface_object_patterns) if self.is_legacy_humanoid else tuple()

    def surface_task_patterns(self) -> Sequence[SurfacePattern]:
        return tuple(self.legacy_surface_task_patterns) if self.is_legacy_humanoid else tuple()

    def title_topic_anchors(self) -> Sequence[str]:
        if self.is_legacy_humanoid:
            return tuple(self.legacy_title_topic_anchors)
        formation = _section(self.domain_lexicon, "candidate_formation")
        return tuple(
            _dedupe(
                list(formation.get("technical_object_types", []))
                + list(formation.get("mechanism_types", []))
                + list(formation.get("task_or_performance_types", []))
                + list(formation.get("scene_or_application_types", []))
            )
        )

    def is_generic_or_shell(self, value: str) -> bool:
        text = str(value or "").strip()
        if not text:
            return True
        if hasattr(self.domain_lexicon, "is_generic_or_shell"):
            return bool(self.domain_lexicon.is_generic_or_shell(text))
        normalized = normalize_signal_phrase(text)
        terms = {normalize_signal_phrase(item) for item in self.generic_terms() + self.shell_terms()}
        return normalized in terms


def build_domain_candidate_policy(
    domain_lexicon: Any,
    *,
    legacy_scope_labels: dict[str, str] | None = None,
    legacy_generic_method_display_names: set[str] | None = None,
    legacy_generic_object_labels: set[str] | None = None,
    legacy_generic_tech_object_slots: set[str] | None = None,
    legacy_surface_object_patterns: Sequence[SurfacePattern] = (),
    legacy_surface_task_patterns: Sequence[SurfacePattern] = (),
    legacy_title_topic_anchors: Sequence[str] = (),
) -> DomainCandidatePolicy:
    return DomainCandidatePolicy(
        domain_lexicon=domain_lexicon,
        legacy_scope_labels=legacy_scope_labels or {},
        legacy_generic_method_display_names=legacy_generic_method_display_names or set(),
        legacy_generic_object_labels=legacy_generic_object_labels or set(),
        legacy_generic_tech_object_slots=legacy_generic_tech_object_slots or set(),
        legacy_surface_object_patterns=tuple(legacy_surface_object_patterns),
        legacy_surface_task_patterns=tuple(legacy_surface_task_patterns),
        legacy_title_topic_anchors=tuple(legacy_title_topic_anchors),
    )
```

- [ ] **Step 2: Compile the new module**

Run:

```powershell
python -m py_compile src/extraction/domain_candidate_policy.py
```

Expected: exit code 0.

- [ ] **Step 3: Commit the policy module**

Run:

```powershell
git add src/extraction/domain_candidate_policy.py
git commit -m "feat: add domain candidate policy adapter"
```

Expected: commit succeeds with only the policy module staged.

---

### Task 3: Gate Humanoid Surface and Label Logic in Candidate Formation

**Files:**
- Modify: `src/extraction/candidate_former.py`
- Test: `tests/test_space_manufacturing_domain_policy.py`
- Test: `tests/test_no_default_robot_leakage.py`

- [ ] **Step 1: Import and build the policy**

In `src/extraction/candidate_former.py`, add:

```python
from .domain_candidate_policy import build_domain_candidate_policy
```

Add a local builder near `build_candidate_forms`:

```python
def _candidate_policy(domain_lexicon):
    return build_domain_candidate_policy(
        domain_lexicon,
        legacy_scope_labels=SCOPE_LABELS,
        legacy_generic_method_display_names=GENERIC_METHOD_ONLY_DISPLAY_NAMES,
        legacy_generic_object_labels=TECH_OBJECT_GENERIC_LABELS,
        legacy_generic_tech_object_slots=GENERIC_TECH_OBJECT_SLOTS,
        legacy_surface_object_patterns=DISPLAY_OBJECT_SURFACE_PATTERNS,
        legacy_surface_task_patterns=DISPLAY_TASK_SURFACE_PATTERNS,
        legacy_title_topic_anchors=STRONG_TITLE_TOPIC_ANCHORS,
    )
```

Inside `build_candidate_forms()`:

```python
domain_lexicon = build_domain_lexicon(domain_context or domain_pack)
policy = _candidate_policy(domain_lexicon)
```

- [ ] **Step 2: Replace scope labels with policy labels**

Add helper:

```python
def _scope_label(scope, policy=None):
    if policy is not None:
        return policy.scope_label(scope)
    return SCOPE_LABELS.get(scope, scope)
```

Replace the `SCOPE_LABELS.get(scope, scope)` call sites in `build_candidate_forms()` with `_scope_label(scope, policy=policy)`.

Expected behavior: non-humanoid scopes display Domain Pack labels or the raw scope; humanoid preset still displays `人形机器人`, `具身智能`, and `世界模型`.

- [ ] **Step 3: Gate surface hints**

Change `_source_surface_hints()` signature:

```python
def _source_surface_hints(title="", text="", source_type="", policy=None):
    object_patterns = policy.surface_object_patterns() if policy is not None else DISPLAY_OBJECT_SURFACE_PATTERNS
    task_patterns = policy.surface_task_patterns() if policy is not None else DISPLAY_TASK_SURFACE_PATTERNS
    object_candidates = _detect_surface_candidates(combined, object_patterns)
    task_candidates = _detect_surface_candidates(combined, task_patterns)
```

Update its `_unit_row()` call to pass `policy=policy`, and update `_unit_row()` signature:

```python
def _unit_row(event_id, scope_names, unit, source_type="", domain_lexicon=None, policy=None):
```

When `policy.is_legacy_humanoid` is false, the returned object/task surface candidates must not include `双臂`, `灵巧手`, `夹爪`, `机械臂`, `装配`, or `抓取` unless those terms are explicitly supplied by a future Domain Pack policy.

- [ ] **Step 4: Gate title topic anchors**

Change `_title_topic_discriminator()` signature:

```python
def _title_topic_discriminator(unit, policy=None):
    anchors = policy.title_topic_anchors() if policy is not None else STRONG_TITLE_TOPIC_ANCHORS
```

Use `anchors` in the loop instead of `STRONG_TITLE_TOPIC_ANCHORS`.

Change `_theme_group_signature(unit)` to `_theme_group_signature(unit, policy=None)` and pass the policy from `build_candidate_forms()`.

- [ ] **Step 5: Gate manipulator bridge fields**

In `_infer_bridged_object_surface()`, `_bridge_reason()`, and `_bridge_confidence()`, add an optional `policy=None` parameter and return the neutral values for non-humanoid domains:

```python
if policy is not None and not policy.is_legacy_humanoid:
    return ""
```

For `_bridge_confidence()`, return `0.0` in the same condition.

Update the three call sites in `_unit_row()` to pass `policy=policy`.

- [ ] **Step 6: Run focused candidate tests**

Run:

```powershell
python -m unittest tests.test_space_manufacturing_domain_policy tests.test_no_default_robot_leakage -v
```

Expected: the new space manufacturing candidate-name contamination test passes, and existing no-default-robot leakage tests keep passing.

- [ ] **Step 7: Commit candidate surface gating**

Run:

```powershell
git add src/extraction/candidate_former.py tests/test_space_manufacturing_domain_policy.py
git commit -m "fix: gate humanoid candidate surfaces by domain policy"
```

Expected: commit succeeds with candidate formation changes and the test file staged.

---

### Task 4: Make Candidate Specificity and Cluster Risk Domain-Aware

**Files:**
- Modify: `src/extraction/candidate_former.py`
- Test: `tests/test_space_manufacturing_domain_policy.py`
- Test: `tests/test_no_default_robot_leakage.py`

- [ ] **Step 1: Add policy-aware generic helpers**

Add these helpers near the existing generic constants:

```python
def _generic_object_labels(policy=None):
    return set(policy.generic_object_labels()) if policy is not None else TECH_OBJECT_GENERIC_LABELS


def _generic_tech_object_slots(policy=None):
    return set(policy.generic_tech_object_slots()) if policy is not None else GENERIC_TECH_OBJECT_SLOTS


def _generic_method_display_names(policy=None):
    if policy is not None:
        return set(policy.generic_method_display_names())
    return {normalize_signal_phrase(item) for item in GENERIC_METHOD_ONLY_DISPLAY_NAMES}
```

- [ ] **Step 2: Wire policy into specificity functions**

Change these signatures and call chains:

```python
def _compose_tech_object_slot(unit, policy=None):
def _build_technical_object_slots(unit, policy=None):
def _specific_anchor_strength(unit, policy=None):
def _is_specific_slot_value(dim, value, policy=None):
def _candidate_second_anchor_variants(items, policy=None):
def _cluster_specificity_profile(items, policy=None):
def _split_generic_cluster_groups(cluster_groups, policy=None):
def _representative_candidate_score(unit, policy=None):
def _select_cluster_representative(items, policy=None):
def _scope_shell_profile(unit, policy=None):
```

Inside those functions, replace direct reads from `TECH_OBJECT_GENERIC_LABELS`, `GENERIC_TECH_OBJECT_SLOTS`, and `GENERIC_METHOD_ONLY_DISPLAY_NAMES` with the helper functions from Step 1.

- [ ] **Step 3: Preserve existing behavior when no policy is passed**

Every updated function must default to the existing constants when `policy is None`. This keeps internal tests that import private helpers directly stable.

- [ ] **Step 4: Pass policy from the cluster path**

In `build_candidate_forms()`:

```python
cluster_groups = _split_generic_cluster_groups(cluster_groups, policy=policy)
cluster_groups = _tfidf_split_heterogeneous_clusters(cluster_groups, policy=policy)
cluster_profile = _cluster_specificity_profile(items, policy=policy)
selection_bundle = _select_cluster_representative(items, policy=policy)
cluster_scope_shell_profile = _scope_shell_profile(representative, policy=policy)
```

Also pass `policy` into `_theme_group_signature(row, policy=policy)` and `_unit_row(..., policy=policy)`.

- [ ] **Step 5: Make method-only display rejection domain-aware**

Change `_is_generic_method_only_display_name()` to use the policy names:

```python
def _is_generic_method_only_display_name(name, unit, domain_lexicon=None, policy=None):
    generic_names = _generic_method_display_names(policy)
```

In non-humanoid domains, only reject names that are generic/shell for the active Domain Pack and lack a Domain Pack technical anchor.

- [ ] **Step 6: Run focused formation tests**

Run:

```powershell
python -m unittest tests.test_space_manufacturing_domain_policy tests.test_no_default_robot_leakage tests.test_event_extraction_refactor -v
```

Expected: space manufacturing formed candidates remain fine-grained; shell-only candidates do not become weak-signal candidates; existing humanoid and non-robot leakage tests pass.

- [ ] **Step 7: Commit domain-aware candidate specificity**

Run:

```powershell
git add src/extraction/candidate_former.py tests/test_space_manufacturing_domain_policy.py
git commit -m "fix: make candidate specificity domain-aware"
```

Expected: commit succeeds with candidate formation specificity changes.

---

### Task 5: Domain-Aware Candidate Eligibility

**Files:**
- Modify: `src/scoring/candidate_eligibility.py`
- Modify: `src/core/pipeline.py`
- Modify: `src/scoring/scorer.py`
- Test: `tests/test_candidate_eligibility.py`
- Test: `tests/test_space_manufacturing_domain_policy.py`

- [ ] **Step 1: Update eligibility function signatures**

Change:

```python
def evaluate_candidate_eligibility(row: Dict[str, Any], *, phase: str = "candidate_form_ready") -> CandidateEligibilityProfile:
```

to:

```python
def evaluate_candidate_eligibility(
    row: Dict[str, Any],
    *,
    phase: str = "candidate_form_ready",
    domain_context: Any = None,
) -> CandidateEligibilityProfile:
```

Change:

```python
def apply_candidate_eligibility(df, *, phase: str = "candidate_form_ready"):
```

to:

```python
def apply_candidate_eligibility(df, *, phase: str = "candidate_form_ready", domain_context: Any = None):
```

- [ ] **Step 2: Add Domain Pack technical anchors**

In `candidate_eligibility.py`, add:

```python
def _domain_anchor_terms(domain_context: Any = None) -> set[str]:
    if domain_context is None:
        return set()
    try:
        from src.extraction.tech_lexicon import build_domain_lexicon
        from src.extraction.domain_candidate_policy import build_domain_candidate_policy
    except Exception:
        return set()
    lexicon = build_domain_lexicon(domain_context)
    policy = build_domain_candidate_policy(lexicon)
    return {term for term in policy.technical_anchor_terms() if term}
```

Change `_has_technical_anchor(row)` to `_has_technical_anchor(row, domain_context=None)` and merge `TECHNICAL_ANCHOR_TERMS` with `_domain_anchor_terms(domain_context)`.

Update `_technical_envelope(row)` to `_technical_envelope(row, domain_context=None)` and call `_has_technical_anchor(row, domain_context=domain_context)`.

- [ ] **Step 3: Pass domain context inside eligibility evaluation**

In `evaluate_candidate_eligibility()`:

```python
has_technical_anchor = _has_technical_anchor(row, domain_context=domain_context)
technical_envelope = _technical_envelope(row, domain_context=domain_context)
```

In `apply_candidate_eligibility()`:

```python
profiles = [
    evaluate_candidate_eligibility(row.to_dict(), phase=phase, domain_context=domain_context).to_dict()
    for _, row in df.iterrows()
]
```

- [ ] **Step 4: Thread domain context from pipeline and scorer**

In `src/core/pipeline.py`:

```python
return apply_candidate_eligibility(candidate_df, phase=phase, domain_context=self.domain_context)
```

In `src/scoring/scorer.py`, replace both calls:

```python
apply_candidate_eligibility(scored_df, phase="scoring_ready", domain_context=domain_context)
```

- [ ] **Step 5: Add direct eligibility test**

Append to `tests/test_candidate_eligibility.py`:

```python
def test_space_domain_context_supplies_technical_anchor_terms(self):
    from tests.test_space_manufacturing_domain_policy import _space_context

    row = {
        "display_candidate_name": "太空3D打印冷焊工艺",
        "raw_phrase": "太空3D打印结构件冷焊工艺",
        "candidate_stage": "formed_candidate",
        "topic_granularity": "fine_grained_topic",
        "display_tier": "weak_signal",
        "is_scope_internal_candidate": True,
        "has_mechanism_core": True,
        "has_non_scope_constraint": True,
        "survives_without_scope": True,
        "scope_shell_heavy": False,
        "mechanism_core": "冷焊工艺",
        "object_modifier_tokens": ["太空3D打印"],
        "task_constraint_tokens": ["在轨装配"],
        "source_count": 2,
        "cluster_evidence_count": 2,
    }

    profile = evaluate_candidate_eligibility(row, domain_context=_space_context())

    self.assertIn(profile.candidate_eligibility, {"eligible", "candidate_monitoring"})
    self.assertNotIn("domain_pack_status_empty_without_technical_envelope", profile.eligibility_reason_codes)
```

- [ ] **Step 6: Run eligibility tests**

Run:

```powershell
python -m unittest tests.test_candidate_eligibility tests.test_space_manufacturing_domain_policy -v
```

Expected: all tests pass; space manufacturing eligibility does not depend on robot/model/material default anchors.

- [ ] **Step 7: Commit eligibility wiring**

Run:

```powershell
git add src/scoring/candidate_eligibility.py src/core/pipeline.py src/scoring/scorer.py tests/test_candidate_eligibility.py tests/test_space_manufacturing_domain_policy.py
git commit -m "fix: make candidate eligibility domain-aware"
```

Expected: commit succeeds with eligibility and wiring changes.

---

### Task 6: Domain-Aware Scoring Scope-Shell Fallback

**Files:**
- Modify: `src/scoring/scorer.py`
- Test: `tests/test_scoring_score_suppression.py`
- Test: `tests/test_space_manufacturing_domain_policy.py`

- [ ] **Step 1: Add scoring policy helpers**

In `src/scoring/scorer.py`, add:

```python
def _domain_shell_terms(domain_context=None):
    if domain_context is None:
        return set()
    try:
        from src.extraction.tech_lexicon import build_domain_lexicon
        from src.extraction.domain_candidate_policy import build_domain_candidate_policy
    except Exception:
        return set()
    policy = build_domain_candidate_policy(build_domain_lexicon(domain_context))
    return {str(item).strip() for item in policy.generic_terms() + policy.shell_terms() if str(item).strip()}
```

Change:

```python
def _is_scope_shell_constraint(value, key):
```

to:

```python
def _is_scope_shell_constraint(value, key, domain_context=None):
```

Use Domain Pack shell/generic terms first:

```python
token = str(value or "").strip()
domain_shells = _domain_shell_terms(domain_context)
if domain_shells:
    return token in domain_shells
```

Then keep the existing humanoid fallback constants for calls without `domain_context`.

- [ ] **Step 2: Pass domain context into scoring scope-shell profile**

Change:

```python
def _scope_shell_profile(row):
```

to:

```python
def _scope_shell_profile(row, domain_context=None):
```

Within it:

```python
non_shell_pairs = [
    (value, key)
    for value, key in filtered
    if not _is_scope_shell_constraint(value, key, domain_context=domain_context)
]
```

Change `_resolved_topic_granularity(row)` and `_resolved_display_tier(row)` to accept `domain_context=None`, and pass it from `_prepare_scored_candidates(candidates_df, domain_context=None)`.

- [ ] **Step 3: Update scoring calls**

Inside `_prepare_scored_candidates()`:

```python
scored_df["topic_granularity"] = scored_df.apply(lambda row: _resolved_topic_granularity(row, domain_context=domain_context), axis=1)
scored_df["display_tier"] = scored_df.apply(lambda row: _resolved_display_tier(row, domain_context=domain_context), axis=1)
profiles = scored_df.apply(lambda row: _scope_shell_profile(row, domain_context=domain_context), axis=1)
apply_candidate_eligibility(scored_df, phase="scoring_ready", domain_context=domain_context)
```

Inside scoring loops that call `_scope_shell_profile(row)`, pass `domain_context=domain_context` when the surrounding function has it.

- [ ] **Step 4: Add scoring suppression regression**

Append to `tests/test_scoring_score_suppression.py`:

```python
def test_space_candidate_not_reclassified_by_humanoid_scope_shell_terms(self):
    from tests.test_space_manufacturing_domain_policy import _space_context

    candidates = pd.DataFrame([
        {
            "id": "space-valid-score",
            "display_candidate_name": "太空3D打印冷焊工艺",
            "candidate_stage": "formed_candidate",
            "topic_granularity": "fine_grained_topic",
            "display_tier": "weak_signal",
            "is_scope_internal_candidate": True,
            "source_count": 2,
            "org_count": 2,
            "total_mentions": 2,
            "cluster_evidence_count": 2,
            "has_mechanism_core": True,
            "has_non_scope_constraint": True,
            "survives_without_scope": True,
            "scope_shell_heavy": False,
            "mechanism_core": "冷焊工艺",
            "mechanism_core_tokens": ["冷焊工艺"],
            "object_modifier_tokens": ["太空3D打印"],
            "task_constraint_tokens": ["在轨装配"],
            "data_modifier_tokens": [],
            "method_modifier_tokens": ["冷焊工艺"],
            "mention_dates": [],
            "evidence_items": [],
        }
    ])

    scored = score_all_candidates(candidates, domain_context=_space_context())
    row = scored.iloc[0]

    self.assertEqual(row["topic_granularity"], "fine_grained_topic")
    self.assertEqual(row["display_tier"], "weak_signal")
    self.assertFalse(bool(row["scope_shell_heavy"]))
    self.assertGreater(float(row["weak_signal_raw_score"]), 0.0)
```

- [ ] **Step 5: Run scoring tests**

Run:

```powershell
python -m unittest tests.test_scoring_score_suppression tests.test_space_manufacturing_domain_policy -v
```

Expected: scoring does not reclassify valid space manufacturing candidates as humanoid scope shells.

- [ ] **Step 6: Commit scoring fallback**

Run:

```powershell
git add src/scoring/scorer.py tests/test_scoring_score_suppression.py tests/test_space_manufacturing_domain_policy.py
git commit -m "fix: make scoring scope shell logic domain-aware"
```

Expected: commit succeeds with scoring changes and tests.

---

### Task 7: Space Manufacturing Pack Calibration Contract

**Files:**
- Modify: `tests/test_domain_pack_quality_contract.py`
- Modify: `src/domain/domain_pack_validator.py`
- Verify: `memory/domain_packs/*.yaml`

- [ ] **Step 1: Add validation expectations for non-humanoid packs**

Extend `tests/test_domain_pack_quality_contract.py` with a test that ensures a space manufacturing pack has enough candidate specificity terms:

```python
def test_space_manufacturing_pack_has_specific_candidate_terms(self):
    payload = {
        "schema_version": "domain_pack_v1",
        "pack_id": "space_manufacturing",
        "pack_name": "太空制造",
        "domain_pack_version": "2026-06-11-test",
        "domain_identity": {
            "field_id": "space_manufacturing",
            "field_name": "太空制造",
            "domain_boundary": "太空制造",
            "core_keywords": ["太空制造", "太空3D打印"],
            "english_terms": ["space manufacturing"],
            "excluded_topics": ["人形机器人", "具身智能"],
        },
        "search_strategy": {"query_templates": [], "must_have_terms": [], "nice_to_have_terms": [], "exclude_terms": [], "source_weights": {}, "query_expansion_rules": []},
        "observation_scopes": {
            "main_scope": "太空制造",
            "sub_scopes": ["微重力制造"],
            "scope_aliases": ["space manufacturing"],
            "scope_echo_terms": ["太空制造", "space manufacturing"],
            "off_domain_anchor_terms": ["人形机器人", "humanoid robot"],
        },
        "candidate_formation": {
            "technical_object_types": ["太空3D打印", "微重力增材制造", "在轨装配"],
            "mechanism_types": ["冷焊工艺", "additive manufacturing"],
            "task_or_performance_types": ["在轨装配", "结构成形"],
            "data_or_method_types": ["轨道验证", "微重力实验"],
            "scene_or_application_types": ["空间站", "近地轨道"],
            "generic_terms": ["太空制造", "技术"],
            "shell_terms": ["方法", "系统"],
            "valid_candidate_patterns": [{"pattern_id": "space_object_mechanism", "required_slots": ["technical_object", "mechanism"], "min_required_slot_count": 2, "evidence_required": True}],
            "invalid_candidate_patterns": [{"pattern_id": "space_scope_only", "reject_terms": ["太空制造"], "max_specific_slot_count": 1}],
            "minimum_specificity_rule": {"min_non_shell_slots": 2, "require_evidence_span": True, "allow_scope_only_candidate": False},
        },
        "weak_signal_rules": {"early_stage_markers": [], "low_attention_markers": [], "niche_actor_markers": [], "cross_domain_markers": [], "engineering_trace_markers": [], "commercialization_noise_markers": [], "policy_or_market_noise_markers": [], "scoring_adjustments": []},
        "canonicalization": {"object_families": [], "alias_groups": [], "parent_child_terms": [], "do_not_merge_rules": [], "object_family_enabled": False},
        "evidence_rules": {"strong_evidence_patterns": [], "weak_evidence_patterns": [], "source_reliability_hints": [], "traceability_fields": [], "evidence_rejection_patterns": []},
        "reporting": {"display_labels": {"space_manufacturing": "太空制造"}, "explanation_templates": [], "review_hints": []},
    }
    report = validate_domain_pack(DomainPack.from_dict(payload), require_dry_run=False)
    self.assertEqual(report.errors, [])
```

- [ ] **Step 2: Verify validator already accepts the contract**

Run:

```powershell
python -m unittest tests.test_domain_pack_quality_contract -v
```

Expected: PASS if current validator is sufficient. If it fails, update `_check_candidate_formation()` so it accepts space manufacturing technical/mechanism specificity without requiring robot-oriented terms.

- [ ] **Step 3: Document runtime pack hygiene**

Append this exact note to `.planning/space-manufacturing-generalization/findings.md` during Task 7:

```markdown
Runtime pack hygiene: generated space manufacturing packs in `memory/domain_packs/` are runtime artifacts. Source-level regression coverage uses inline test packs so behavior does not depend on a specific cached hash.
```

- [ ] **Step 4: Commit pack contract test**

Run:

```powershell
git add tests/test_domain_pack_quality_contract.py src/domain/domain_pack_validator.py
git commit -m "test: lock space manufacturing domain pack specificity contract"
```

Expected: commit succeeds. If `src/domain/domain_pack_validator.py` was unchanged, stage only the test file.

---

### Task 8: Full Verification and Regression Sweep

**Files:**
- Verify: `src/extraction/domain_candidate_policy.py`
- Verify: `src/extraction/candidate_former.py`
- Verify: `src/scoring/candidate_eligibility.py`
- Verify: `src/scoring/scorer.py`
- Verify: `src/core/pipeline.py`
- Verify: `tests/test_space_manufacturing_domain_policy.py`

- [ ] **Step 1: Compile touched modules**

Run:

```powershell
python -m py_compile src/extraction/domain_candidate_policy.py src/extraction/candidate_former.py src/scoring/candidate_eligibility.py src/scoring/scorer.py src/core/pipeline.py tests/test_space_manufacturing_domain_policy.py
```

Expected: exit code 0.

- [ ] **Step 2: Run focused test suite**

Run:

```powershell
python -m unittest tests.test_space_manufacturing_domain_policy tests.test_no_default_robot_leakage tests.test_candidate_eligibility tests.test_scoring_score_suppression -v
```

Expected: all tests pass.

- [ ] **Step 3: Run Domain Pack regression suite**

Run:

```powershell
python -m unittest tests.test_domain_context_pipeline tests.test_domain_pack_quality_contract tests.test_domain_pack_generator tests.test_event_extraction_refactor tests.test_signal_generation_eligibility_contract tests.test_result_contract_latest_failures -v
```

Expected: all tests pass.

- [ ] **Step 4: Run whitespace and diff checks**

Run:

```powershell
git diff --check
git diff --stat
```

Expected: `git diff --check` exits 0; `git diff --stat` only lists files from this plan.

- [ ] **Step 5: Commit final verification notes if plan files changed**

Run:

```powershell
git add docs/superpowers/plans/2026-06-11-space-manufacturing-domain-policy.md .planning/space-manufacturing-generalization/task_plan.md .planning/space-manufacturing-generalization/findings.md .planning/space-manufacturing-generalization/progress.md
git commit -m "docs: plan space manufacturing domain policy refactor"
```

Expected: commit succeeds if documentation files are meant to be committed. If plan files are not committed in this repo, leave them unstaged and note that decision in `progress.md`.

---

## Acceptance Criteria

- Space manufacturing candidate names do not include humanoid-only labels such as `人形机器人`, `具身智能`, `灵巧手`, `夹爪`, `机械臂`, `humanoid robot`, or `dexterous hand` unless the space Domain Pack explicitly names them as valid technical objects.
- Space manufacturing technical candidates such as `太空3D打印冷焊工艺`, `微重力增材制造`, and `在轨装配制造` can form candidates with `topic_granularity=fine_grained_topic`.
- Scope-only or policy-only surfaces such as `太空制造技术方案` do not become `formed_candidate_strong` or `display_tier=weak_signal`.
- `candidate_eligibility` can recognize space manufacturing technical anchors from Domain Pack terms.
- `score_all_candidates(..., domain_context=space_context)` does not reclassify valid space candidates using humanoid shell tokens.
- Humanoid preset behavior remains available when the active pack is `humanoid_robot` or `DomainLexicon.use_legacy_robot_rules` is true.
- Existing battery/electronic-materials no-default-robot leakage tests remain green.
