from __future__ import annotations

from collections import Counter
import hashlib
import math
import os
import re

import pandas as pd

from .event_schema import clean_event_list, clean_event_text
from .tech_lexicon import (
    BARE_MECHANISM_CORES,
    build_domain_lexicon,
    detect_supported_observation_scopes,
    extract_data_modifier_tokens,
    extract_mechanism_core_tokens,
    extract_method_modifier_tokens,
    extract_object_modifier_tokens,
    extract_scene_tokens,
    extract_task_constraint_tokens,
    has_non_scope_constraint,
    normalize_proxy_token,
    normalize_signal_phrase,
)


MECHANISM_LABELS = {
    "training": "训练",
    "planning": "规划",
    "simulation": "仿真",
    "calibration": "校准",
    "compression": "压缩",
    "memory": "记忆",
    "tactile": "触觉",
    "reasoning": "推理",
    "control": "控制",
    "grounding": "具身落地",
    "alignment": "对齐",
    "policy": "策略",
    "retrieval": "检索",
    "distillation": "蒸馏",
}

METHOD_ONLY_SCOPE_MECHANISM_LABELS = {
    ("world model", "simulation"): "世界模型驱动仿真",
    ("world model", "control"): "世界模型驱动控制",
    ("world model", "planning"): "世界模型驱动规划",
    ("embodied intelligence", "training"): "具身训练机制",
    ("embodied intelligence", "control"): "具身控制机制",
}

CONSTRAINT_LABELS = {
    "robot": "机器人",
    "robotics": "机器人",
    "manipulator": "机械臂",
    "manipulators": "机械臂",
    "gripper": "夹爪",
    "grippers": "夹爪",
    "manipulation": "操控",
    "assembly": "装配",
    "navigation": "导航",
    "driving": "驾驶",
    "agent": "智能体",
    "embodied": "具身",
    "interactive": "交互",
    "decision-making": "决策",
    "video": "视频",
    "multimodal": "多模态",
    "temporal": "时序",
    "3d": "三维",
    "egocentric": "第一视角",
    "visual": "视觉",
    "sensor": "传感",
    "perception": "感知",
    "trajectory": "轨迹",
    "training": "训练",
    "planning": "规划",
    "reasoning": "推理",
    "control": "控制",
    "causal": "因果",
    "dynamic": "动态",
    "sparse": "稀疏",
    "retrieval-based": "检索式",
    "reinforcement": "强化",
    "policy-guided": "策略引导",
}

SCOPE_LABELS = {
    "humanoid robot": "人形机器人",
    "embodied intelligence": "具身智能",
    "world model": "世界模型",
}

GENERIC_TASK_TOKENS = {"interactive", "embodied", "agent", "benchmark", "evaluation", "control"}
GENERIC_OBJECT_TOKENS = {"agent", "environment", "memory", "control"}
GENERIC_DATA_TOKENS = {"simulation"}
GENERIC_METHOD_TOKENS = {"reinforcement", "policy-guided", "causal", "dynamic", "sparse", "retrieval-based"}
DISPLAY_SUPPRESS_TOKENS = {"benchmark", "evaluation", "environment", "policy"}
DISPLAY_SOFT_TASK_TOKENS = {"interactive", "agent"}
DISPLAY_SOFT_OBJECT_TOKENS = {"policy", "memory"}
DISPLAY_SOFT_DATA_TOKENS = {"trajectory", "egocentric", "sensor"}
DISPLAY_WEAK_SCOPE_TOKENS = {"embodied", "interactive", "agent"}
WORLD_MODEL_WEAK_LABELS = {"智能体", "机器人", "视觉", "视频", "驾驶", "多模态", "三维", "交互", "导航", "操控", "决策"}
PATENT_FRIENDLY_CONSTRAINT_KEYS = {"task_constraint_tokens", "object_modifier_tokens", "data_modifier_tokens"}
DISPLAY_SHELL_WORDS = ("机制", "范式", "体系", "能力")
DISPLAY_NARRATIVE_MARKERS = (
    "通过", "利用", "可以", "能够", "实现", "显著", "从而",
    "取得", "用于", "针对", "提出", "发布", "开发", "形成",
)
GENERIC_METHOD_ONLY_DISPLAY_NAMES = {
    "训练技术", "控制技术", "规划技术", "仿真技术",
    "训练方法", "控制方法", "规划方法", "仿真方法",
}
SCOPE_SHELL_TASK_TOKENS = {"agent", "embodied", "interactive", "control"}
SCOPE_SHELL_OBJECT_TOKENS = {"agent", "environment", "control", "memory", "policy"}
SCOPE_SHELL_DATA_TOKENS = {"simulation", "video", "visual"}
SCOPE_SHELL_METHOD_TOKENS = {"reinforcement", "policy-guided", "retrieval-based", "causal", "dynamic", "sparse"}
SCOPE_DISAMBIGUATION_TOKENS = {"agent", "embodied", "interactive", "robot", "video", "visual"}
ANCHOR_PRIORITY = {
    "task_constraint_tokens": ["driving", "navigation", "manipulation", "grasping", "robot", "agent", "embodied", "interactive", "control"],
    "object_modifier_tokens": ["robot", "agent", "environment", "perception", "memory", "control"],
    "data_modifier_tokens": ["video", "sensor", "trajectory", "visual", "3d", "multimodal", "temporal", "simulation"],
    "method_modifier_tokens": ["causal", "dynamic", "sparse", "retrieval-based", "policy-guided", "reinforcement"],
}
ANCHOR_ALIAS_MAP = {
    "robotics": "robot",
    "robots": "robot",
    "agents": "agent",
    "visuals": "visual",
    "videos": "video",
    "sensors": "sensor",
    "trajectories": "trajectory",
}
GENERIC_SCOPE_FILLER_TOKENS = {"embodied", "interactive", "agent", "environment", "control", "simulation"}
TECH_OBJECT_GENERIC_LABELS = {"具身", "交互", "智能体", "环境", "控制", "仿真", "世界模型", "具身智能"}
PROCESS_METHOD_PREFIXES = {"强化", "因果", "动态", "稀疏", "检索式", "策略引导"}
GENERIC_TECH_OBJECT_SLOTS = {"机器人", "世界模型", "具身智能", "智能体", "环境", "控制", "规划", "训练", "推理", "仿真", "视频", "多模态", "三维", "传感", "轨迹"}
GENERIC_CAPABILITY_SLOTS = {"控制", "规划", "训练", "推理", "仿真", "记忆", "触觉", "校准", "压缩", "对齐", "检索", "蒸馏", "策略"}
GENERIC_APPLICATION_SLOTS = {"世界模型", "具身智能"}
PRIMARY_GENERIC_OBJECT_LABELS = {"机器人", "智能体", "环境"}
MIN_SECOND_ANCHOR_COUNT = 2
MIN_SECOND_ANCHOR_RATIO = 0.3
TECHNICAL_READY_HEADWORDS = {
    "模块", "系统", "控制器", "规划器", "采集系统", "执行器", "传感器", "芯片",
    "电机", "夹爪", "灵巧手", "机械臂", "模型头", "数据集", "材料", "磁钢", "薄膜",
}
TECHNICAL_DRAFT_HINTS = {
    "机器人", "机械臂", "灵巧手", "夹爪", "导航", "感知", "触觉", "视频", "三维", "轨迹", "传感",
}
TECH_OBJECT_REFINEMENT_HINTS = {
    "导航": "导航机器人",
    "感知": "感知机器人",
    "决策": "决策机器人",
    "机械臂": "机械臂机器人",
}
DISPLAY_OBJECT_SURFACE_PATTERNS = [
    ("双臂", ("双臂", "dual arm", "dual-arm", "bimanual", "双机械臂")),
    ("灵巧手", ("灵巧手", "dexterous hand", "dexterous-hand", "multi-finger hand", "五指灵巧手")),
    ("夹爪", ("夹爪", "gripper", "夹持器", "抓手")),
    # “robot arm / manipulator” 作为 generic anchor 存在，不在这里直接视为显式 subtype。
    ("机械臂", ("机械臂", "单臂")),
]
DISPLAY_TASK_SURFACE_PATTERNS = [
    ("装配", ("装配", "assembly", "assemble", "assembling", "装联", "装调", "插接")),
    ("抓取", ("抓取", "grasping", "grasp", "抓握", "夹取", "pick and place", "pick-and-place")),
]
MANIPULATOR_DISPLAY_SURFACES = {"双臂", "灵巧手", "夹爪", "机械臂", "机械臂机器人"}
BRIDGED_OBJECT_SURFACES = {"双臂", "灵巧手", "夹爪"}


def _clean_shell_dominated_term(term, domain_lexicon):
    t = str(term or "").strip()
    if not t:
        return ""
    if domain_lexicon.is_generic_or_shell(t):
        return ""
    shell_words = ["性能", "应用", "效果", "数据", "方法", "技术", "场景", "系统", "performance", "application", "effect", "data", "method", "technology", "scene", "system"]
    changed = True
    while changed:
        changed = False
        for sw in shell_words:
            if t.lower().endswith(sw.lower()) and len(t) > len(sw):
                t = t[:-len(sw)].strip("-")
                changed = True
            if t.lower().startswith(sw.lower()) and len(t) > len(sw):
                t = t[len(sw):].strip("-")
                changed = True
    if not t or domain_lexicon.is_generic_or_shell(t) or len(t) <= 1:
        return ""
    return t


def _dedupe_preserve_order(values):
    seen = set()
    deduped = []
    for value in values or []:
        text = str(value).strip()
        if not text or text in seen:
            continue
        deduped.append(text)
        seen.add(text)
    return deduped


def _normalize_surface_hint_text(text):
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _detect_surface_candidates(text, patterns):
    haystack = _normalize_surface_hint_text(text)
    if not haystack:
        return []
    matched = []
    for label, aliases in patterns:
        if any(_normalize_surface_hint_text(alias) in haystack for alias in aliases):
            matched.append(label)
    return matched


def _source_surface_hints(title="", text="", source_type=""):
    title_s = str(title or "").strip()
    text_s = str(text or "").strip()
    combined = " ".join(part for part in [title_s, text_s] if part)
    object_candidates = _detect_surface_candidates(combined, DISPLAY_OBJECT_SURFACE_PATTERNS)
    task_candidates = _detect_surface_candidates(combined, DISPLAY_TASK_SURFACE_PATTERNS)
    # Round 39 P1 修复：news 正文长（2-3k 字），常包含"装配/抓取"这类
    # 无关的 side-mention（例如工博会综述里一句"智能装配检测机器人"）。
    # 这类 substring 命中会被 cluster 代表传播到整簇，污染 patent 成员的
    # preferred_task_surface。对 news 源只信任 title 里的任务面命中。
    if str(source_type or "").strip().lower() == "news":
        title_task_candidates = _detect_surface_candidates(title_s, DISPLAY_TASK_SURFACE_PATTERNS) if title_s else []
        task_candidates = title_task_candidates
    preferred_task_surface = task_candidates[0] if task_candidates else ""
    preferred_object_surface = ""
    if preferred_task_surface == "装配":
        preferred_order = ["双臂", "机械臂", "灵巧手", "夹爪"]
    elif preferred_task_surface == "抓取":
        preferred_order = ["灵巧手", "夹爪", "双臂", "机械臂"]
    else:
        preferred_order = ["双臂", "灵巧手", "夹爪", "机械臂"]
    for label in preferred_order:
        if label in object_candidates:
            preferred_object_surface = label
            break
    return {
        "object_surface_candidates": object_candidates,
        "preferred_object_surface": preferred_object_surface,
        "preferred_task_surface": preferred_task_surface,
    }


def _surface_hint_context(unit, extra_text=""):
    parts = [
        str(unit.get("source_title", "")).strip(),
        str(unit.get("source_text", "")).strip(),
        str(extra_text or "").strip(),
        str(unit.get("raw_phrase", "")).strip(),
        str(unit.get("relation_summary", "")).strip(),
        str(unit.get("relation_target", "")).strip(),
        str(unit.get("relation_task", "")).strip(),
        str(unit.get("relation_data_modality", "")).strip(),
        str(unit.get("relation_method", "")).strip(),
        str(unit.get("canonical_candidate_name_en", "")).strip(),
    ]
    return " ".join(part for part in parts if part)


def _normalize_surface_candidates(raw_value):
    if isinstance(raw_value, list):
        return _dedupe_preserve_order(raw_value)
    if isinstance(raw_value, str) and raw_value.strip():
        return _dedupe_preserve_order(part.strip() for part in raw_value.split("|"))
    return []


def _merge_cluster_surface_hints(items):
    object_candidates = []
    task_candidates = []
    for item in items:
        object_candidates.extend(_normalize_surface_candidates(item.get("object_surface_candidates", [])))
        preferred_object = str(item.get("preferred_object_surface", "")).strip()
        if preferred_object:
            object_candidates.append(preferred_object)
        preferred_task = str(item.get("preferred_task_surface", "")).strip()
        if preferred_task:
            task_candidates.append(preferred_task)
    object_candidates = _dedupe_preserve_order(object_candidates)
    task_candidates = _dedupe_preserve_order(task_candidates)
    preferred_task_surface = task_candidates[0] if task_candidates else ""
    preferred_object_surface = ""
    if preferred_task_surface == "装配":
        preferred_order = ["双臂", "机械臂", "灵巧手", "夹爪"]
    elif preferred_task_surface == "抓取":
        preferred_order = ["灵巧手", "夹爪", "双臂", "机械臂"]
    else:
        preferred_order = ["双臂", "灵巧手", "夹爪", "机械臂"]
    for label in preferred_order:
        if label in object_candidates:
            preferred_object_surface = label
            break
    return {
        "object_surface_candidates": object_candidates,
        "preferred_object_surface": preferred_object_surface,
        "preferred_task_surface": preferred_task_surface,
    }


def _normalized_token_set(values):
    return {
        normalize_proxy_token(value)
        for value in _dedupe_preserve_order(values)
        if str(value or "").strip()
    }


def _unit_context_text(unit):
    parts = [
        unit.get("source_title", ""),
        unit.get("source_text", ""),
        unit.get("raw_candidate_text", ""),
        unit.get("raw_phrase", ""),
        unit.get("relation_summary", ""),
        unit.get("relation_target", ""),
        unit.get("relation_task", ""),
        unit.get("relation_data_modality", ""),
        unit.get("relation_method", ""),
    ]
    return _normalize_surface_hint_text(" ".join(str(part or "").strip() for part in parts if str(part or "").strip()))


def _context_contains_any(text, aliases):
    haystack = _normalize_surface_hint_text(text)
    if not haystack:
        return False
    return any(_normalize_surface_hint_text(alias) in haystack for alias in aliases)


def _context_match_count(text, aliases):
    haystack = _normalize_surface_hint_text(text)
    if not haystack:
        return 0
    return sum(1 for alias in aliases if _normalize_surface_hint_text(alias) in haystack)


def _infer_bridged_object_surface(unit):
    if str(unit.get("preferred_object_surface", "")).strip():
        return ""
    if _normalize_surface_candidates(unit.get("object_surface_candidates", [])):
        return ""

    context = _unit_context_text(unit)
    object_tokens = _normalized_token_set(unit.get("object_modifier_tokens", []))
    task_tokens = _normalized_token_set(unit.get("task_constraint_tokens", []))
    data_tokens = _normalized_token_set(unit.get("data_modifier_tokens", []))
    method_tokens = _normalized_token_set(unit.get("method_modifier_tokens", []))

    has_generic_manipulator_anchor = bool(
        {"manipulator", "robot"} & object_tokens and "manipulation" in task_tokens
        or "manipulator" in object_tokens
        or _context_contains_any(context, ("manipulator", "robot arm", "robotic arm", "end-effector", "robotic manipulation"))
    )
    if not has_generic_manipulator_anchor:
        return ""

    has_assembly = "assembly" in task_tokens or _context_contains_any(context, ("assembly", "assemble", "assembling", "装配", "装联"))
    has_grasp_context = bool(
        "manipulation" in task_tokens
        or _context_contains_any(context, ("grasp", "grasping", "抓取", "抓握", "picking", "precision picking", "in-hand", "reorientation"))
    )
    has_tactile = bool(
        "tactile" in object_tokens
        or "tactile" in data_tokens
        or _context_contains_any(context, ("tactile", "haptic", "触觉", "contact-state", "contact state"))
    )
    has_end_effector = _context_contains_any(context, ("end-effector", "end effector", "multi-finger", "multi finger", "multi-contact", "multi contact"))

    dual_arm_cues = (
        "coordinated", "coordination", "synchronized", "synchronised",
        "two-side", "two side", "paired motion", "bimanual", "dual-arm",
        "cooperative", "协同",
    )
    if has_assembly and _context_match_count(context, dual_arm_cues) >= 2:
        return "双臂"

    dexterous_cues = (
        "multi-finger", "multi finger", "multi-contact", "multi contact",
        "in-hand", "reorientation", "finger gait", "precision handling",
    )
    if has_tactile and has_grasp_context and _context_match_count(context, dexterous_cues) >= 2:
        return "灵巧手"

    gripper_cues = (
        "grasp correction", "contact stability", "precision picking", "release",
        "force feedback", "contact adjustment", "pose adjustment", "contact patch",
        "small-object handling", "picking stability",
    )
    if (
        has_end_effector
        and has_grasp_context
        and _context_match_count(context, gripper_cues) >= 2
        and ("reinforcement" not in method_tokens)
    ):
        return "夹爪"

    return ""


def _bridge_reason(unit):
    """返回桥接推断的原因，用于旁路诊断"""
    context = _unit_context_text(unit)
    object_tokens = _normalized_token_set(unit.get("object_modifier_tokens", []))
    task_tokens = _normalized_token_set(unit.get("task_constraint_tokens", []))
    data_tokens = _normalized_token_set(unit.get("data_modifier_tokens", []))
    method_tokens = _normalized_token_set(unit.get("method_modifier_tokens", []))

    has_generic_manipulator_anchor = bool(
        {"manipulator", "robot"} & object_tokens and "manipulation" in task_tokens
        or "manipulator" in object_tokens
        or _context_contains_any(context, ("manipulator", "robot arm", "robotic arm", "end-effector", "robotic manipulation"))
    )
    if not has_generic_manipulator_anchor:
        return "no_generic_anchor"

    has_assembly = "assembly" in task_tokens or _context_contains_any(context, ("assembly", "assemble", "assembling", "装配", "装联"))
    has_grasp_context = bool(
        "manipulation" in task_tokens
        or _context_contains_any(context, ("grasp", "grasping", "抓取", "抓握", "picking", "precision picking", "in-hand", "reorientation"))
    )
    has_tactile = bool(
        "tactile" in object_tokens
        or "tactile" in data_tokens
        or _context_contains_any(context, ("tactile", "haptic", "触觉", "contact-state", "contact state"))
    )
    has_end_effector = _context_contains_any(context, ("end-effector", "end effector", "multi-finger", "multi finger", "multi-contact", "multi contact"))

    dual_arm_cues = (
        "coordinated", "coordination", "synchronized", "synchronised",
        "two-side", "two side", "paired motion", "bimanual", "dual-arm",
        "cooperative", "协同",
    )
    if has_assembly and _context_match_count(context, dual_arm_cues) >= 2:
        return "dual_arm:assembly+coordination_cues"

    dexterous_cues = (
        "multi-finger", "multi finger", "multi-contact", "multi contact",
        "in-hand", "reorientation", "finger gait", "precision handling",
    )
    if has_tactile and has_grasp_context and _context_match_count(context, dexterous_cues) >= 2:
        return "dexterous:tactile+grasp+dexterous_cues"

    gripper_cues = (
        "grasp correction", "contact stability", "precision picking", "release",
        "force feedback", "contact adjustment", "pose adjustment", "contact patch",
        "small-object handling", "picking stability",
    )
    if (
        has_end_effector
        and has_grasp_context
        and _context_match_count(context, gripper_cues) >= 2
        and ("reinforcement" not in method_tokens)
    ):
        return "gripper:end_effector+grasp+gripper_cues"

    return "no_matching_pattern"


def _bridge_confidence(unit):
    candidate = _infer_bridged_object_surface(unit)
    if not candidate:
        return 0.0

    context = _unit_context_text(unit)
    object_tokens = _normalized_token_set(unit.get("object_modifier_tokens", []))
    task_tokens = _normalized_token_set(unit.get("task_constraint_tokens", []))
    data_tokens = _normalized_token_set(unit.get("data_modifier_tokens", []))

    has_generic_anchor = bool(
        {"manipulator", "robot"} & object_tokens and "manipulation" in task_tokens
        or "manipulator" in object_tokens
        or _context_contains_any(context, ("manipulator", "robot arm", "robotic arm", "end-effector", "robotic manipulation"))
    )

    if candidate == "双臂":
        cue_count = _context_match_count(
            context,
            (
                "coordinated", "coordination", "synchronized", "synchronised",
                "two-side", "two side", "paired motion", "bimanual", "dual-arm",
                "cooperative", "协同",
            ),
        )
        score = 0.55 + 0.1 * cue_count + 0.1 * int("assembly" in task_tokens)
    elif candidate == "灵巧手":
        cue_count = _context_match_count(
            context,
            (
                "multi-finger", "multi finger", "multi-contact", "multi contact",
                "in-hand", "reorientation", "finger gait", "precision handling",
            ),
        )
        score = 0.55 + 0.1 * cue_count + 0.1 * int("tactile" in object_tokens or "tactile" in data_tokens)
    else:
        cue_count = _context_match_count(
            context,
            (
                "grasp correction", "contact stability", "precision picking", "release",
                "force feedback", "contact adjustment", "pose adjustment", "contact patch",
                "small-object handling", "picking stability",
            ),
        )
        score = 0.55 + 0.1 * cue_count + 0.1 * int(_context_contains_any(context, ("end-effector", "end effector")))

    if has_generic_anchor:
        score += 0.05
    return round(min(score, 0.95), 2)


def _display_title_object_surface(unit):
    preferred = str(unit.get("display_preferred_object_surface", "")).strip()
    if preferred:
        return preferred
    return str(unit.get("preferred_object_surface", "")).strip()


def _display_title_task_surface(unit):
    preferred = str(unit.get("display_preferred_task_surface", "")).strip()
    if preferred:
        return preferred
    return str(unit.get("preferred_task_surface", "")).strip()


def _display_surface_hint_from_title(unit):
    title_hints = _source_surface_hints(
        str(unit.get("source_title", "")).strip(),
        _surface_hint_context(unit),
    )
    preferred = str(title_hints.get("preferred_object_surface", "")).strip()
    if preferred not in BRIDGED_OBJECT_SURFACES:
        return ""

    object_tokens = _normalized_token_set(unit.get("object_modifier_tokens", []))
    task_tokens = _normalized_token_set(unit.get("task_constraint_tokens", []))
    relation_target = normalize_proxy_token(unit.get("relation_target", ""))
    relation_task = normalize_proxy_token(unit.get("relation_task", ""))
    local_context = _normalize_surface_hint_text(
        " ".join(
            str(part or "").strip()
            for part in [
                unit.get("raw_phrase", ""),
                unit.get("relation_summary", ""),
                unit.get("relation_target", ""),
                unit.get("relation_task", ""),
            ]
            if str(part or "").strip()
        )
    )

    has_manipulator_anchor = bool(
        "manipulator" in object_tokens
        or "gripper" in object_tokens
        or relation_target in {"manipulator", "gripper"}
        or _context_contains_any(local_context, ("manipulator", "robot arm", "robotic arm", "end-effector", "robotic manipulation"))
        or ("robot" in object_tokens and ("manipulation" in task_tokens or "assembly" in task_tokens))
    )
    if not has_manipulator_anchor:
        return ""

    if preferred == "双臂":
        # Round 39 P1 修复：原先允许 coordination/coordinated/bimanual/dual-arm
        # 任一命中就认定 dual_arm_task，导致大量"双臂协同搬运/双臂模仿学习"被错贴
        # 为 assembly 场景。现仅保留真正的 assembly 同义词。
        has_dual_arm_task = bool(
            "assembly" in task_tokens
            or relation_task == "assembly"
            or _context_contains_any(local_context, ("assembly", "assemble", "assembling", "装配", "装联"))
        )
        return preferred if has_dual_arm_task else ""

    has_fine_manipulation_task = bool(
        "manipulation" in task_tokens
        or relation_task == "manipulation"
        or _context_contains_any(local_context, ("manipulation", "grasp", "grasping", "抓取", "抓握", "tactile", "触觉"))
    )
    return preferred if has_fine_manipulation_task else ""


def _display_task_hint_from_title(unit):
    object_surface = str(unit.get("display_preferred_object_surface", "")).strip()
    if not object_surface:
        return ""
    title_hints = _source_surface_hints(
        str(unit.get("source_title", "")).strip(),
        _surface_hint_context(unit),
    )
    preferred_task = str(title_hints.get("preferred_task_surface", "")).strip()
    if preferred_task not in {"抓取", "装配"}:
        relation_task = normalize_proxy_token(unit.get("relation_task", ""))
        local_context = _normalize_surface_hint_text(
            " ".join(
                str(part or "").strip()
                for part in [
                    unit.get("raw_phrase", ""),
                    unit.get("relation_summary", ""),
                    unit.get("relation_target", ""),
                    unit.get("relation_task", ""),
                    unit.get("relation_data_modality", ""),
                    unit.get("relation_method", ""),
                    unit.get("canonical_candidate_name_en", ""),
                ]
                if str(part or "").strip()
            )
        )
        if object_surface == "双臂":
            # Round 39 P1 修复：去掉 coordination/coordinated/bimanual/dual-arm
            # 四个误触发 token（它们描述的是"双臂配置"而非"装配任务"），
            # 只保留真正的 assembly 同义词。
            has_assembly_cue = bool(
                relation_task == "assembly"
                or _context_contains_any(local_context, ("assembly", "assemble", "assembling", "装配", "装联"))
            )
            return "装配" if has_assembly_cue else ""
        if object_surface in {"灵巧手", "夹爪"}:
            has_grasp_cue = bool(
                relation_task == "manipulation"
                or _context_contains_any(
                    local_context,
                    (
                        "grasp", "grasping", "抓取", "抓握", "manipulation", "precision picking",
                        "contact stability", "in-hand", "reorientation", "tactile", "触觉",
                    ),
                )
            )
            has_task_tail = _context_contains_any(local_context, ("control", "planning", "training", "simulation", "控制", "规划", "训练", "仿真"))
            has_tactile_only_bias = _context_contains_any(local_context, ("tactile", "触觉")) and not _context_contains_any(
                local_context,
                ("grasp", "grasping", "抓取", "抓握", "precision picking", "in-hand", "reorientation"),
            )
            return "抓取" if has_grasp_cue and has_task_tail and not has_tactile_only_bias else ""
        return ""
    return preferred_task


def _is_manipulator_display_unit(unit, object_phrase):
    object_tokens = {normalize_proxy_token(value) for value in _dedupe_preserve_order(unit.get("object_modifier_tokens", []))}
    tech_object_slot = str(_build_technical_object_slots(unit).get("tech_object_slot", "")).strip()
    return bool(
        {"manipulator", "gripper"} & object_tokens
        or any(label in str(object_phrase or "") for label in MANIPULATOR_DISPLAY_SURFACES)
        or tech_object_slot in MANIPULATOR_DISPLAY_SURFACES
    )


def _preferred_object_phrase(unit, object_phrase, capability_phrase=""):
    # 只恢复显式 object surface 保真；bridge 候选仍停留在旁路字段。
    preferred_object_surface = _display_title_object_surface(unit)
    if not preferred_object_surface or not _is_manipulator_display_unit(unit, object_phrase):
        return str(object_phrase or "").strip()
    if preferred_object_surface == "机械臂" and "机器人" in str(object_phrase or ""):
        return "机械臂机器人"
    return preferred_object_surface


def _apply_preferred_task_surface(unit, capability_phrase):
    # bridge/surface hint 仅保留在旁路字段，不再直接改主榜能力表达。
    return str(capability_phrase or "").strip()


def _contains_ascii_letters(text):
    return bool(re.search(r"[A-Za-z]", str(text or "")))


def _label_for_token(token, prefer_zh=False):
    raw = str(token or "").strip()
    if not raw:
        return ""
    normalized = normalize_proxy_token(raw)
    label = (
        CONSTRAINT_LABELS.get(raw)
        or CONSTRAINT_LABELS.get(raw.lower())
        or CONSTRAINT_LABELS.get(normalized)
        or MECHANISM_LABELS.get(raw)
        or MECHANISM_LABELS.get(raw.lower())
        or MECHANISM_LABELS.get(normalized)
    )
    if label:
        return label
    if not _contains_ascii_letters(raw):
        return raw
    if not _contains_ascii_letters(normalized):
        return normalized
    return "" if prefer_zh else normalized


def _strong_slot_labels(values, key):
    labels = []
    for value in _dedupe_preserve_order(values):
        if _is_generic_constraint(value, key) or _is_scope_shell_constraint(value, key):
            continue
        label = _label_for_token(value, prefer_zh=True)
        if label:
            labels.append(label)
    return _dedupe_preserve_order(labels)


def _fallback_slot_labels(values):
    labels = []
    for value in _dedupe_preserve_order(values):
        label = _label_for_token(value, prefer_zh=True)
        if label:
            labels.append(label)
    return _dedupe_preserve_order(labels)


def _compose_tech_object_slot(unit):
    domain_lexicon = unit.get("domain_lexicon")
    if not domain_lexicon:
        domain_lexicon = build_domain_lexicon(unit.get("domain_pack") or unit.get("domain_context"))

    def _clean(val):
        return _clean_shell_dominated_term(val, domain_lexicon)

    explicit_surface = str(unit.get("preferred_object_surface", "")).strip()
    if explicit_surface:
        return _clean(explicit_surface)

    object_labels = _strong_slot_labels(unit.get("object_modifier_tokens", []), "object_modifier_tokens")
    task_labels = _strong_slot_labels(unit.get("task_constraint_tokens", []), "task_constraint_tokens")
    data_labels = _strong_slot_labels(unit.get("data_modifier_tokens", []), "data_modifier_tokens")

    preferred_object_labels = [label for label in object_labels if label not in PRIMARY_GENERIC_OBJECT_LABELS]
    primary_object = preferred_object_labels[0] if preferred_object_labels else (object_labels[0] if object_labels else "")
    primary_task = task_labels[0] if task_labels else ""
    primary_data = data_labels[0] if data_labels else ""

    res = ""
    if primary_object and primary_task and primary_task not in primary_object:
        res = f"{primary_object}{primary_task}"
    elif primary_object:
        res = primary_object
    elif primary_task:
        res = primary_task
    elif primary_data:
        res = primary_data
    else:
        fallback_objects = _fallback_slot_labels(unit.get("object_modifier_tokens", []))
        fallback_tasks = _fallback_slot_labels(unit.get("task_constraint_tokens", []))
        for label in fallback_objects + fallback_tasks:
            if label and label not in TECH_OBJECT_GENERIC_LABELS:
                res = label
                break

    return _clean(res)


def _refine_tech_object_slot(unit, tech_object_slot, capability_slot="", process_slot="", application_slot=""):
    tech_object_slot = str(tech_object_slot or "").strip()
    if tech_object_slot != "机器人":
        return tech_object_slot

    ranked_hints = []
    object_labels = _strong_slot_labels(unit.get("object_modifier_tokens", []), "object_modifier_tokens")
    task_labels = _strong_slot_labels(unit.get("task_constraint_tokens", []), "task_constraint_tokens")
    for label in object_labels + task_labels:
        if label in TECH_OBJECT_REFINEMENT_HINTS:
            ranked_hints.append(label)
    if application_slot in TECH_OBJECT_REFINEMENT_HINTS:
        ranked_hints.append(application_slot)

    for hint in _dedupe_preserve_order(ranked_hints):
        refined = TECH_OBJECT_REFINEMENT_HINTS.get(hint, "")
        if refined:
            return refined

    return tech_object_slot


def _compose_capability_slot(unit, tech_object_slot=""):
    mechanism_label = _label_for_token(unit.get("mechanism_core", ""), prefer_zh=True)
    if not mechanism_label:
        return ""
    task_labels = _strong_slot_labels(unit.get("task_constraint_tokens", []), "task_constraint_tokens")
    for label in task_labels:
        if label and label not in tech_object_slot and label != mechanism_label:
            return f"{label}{mechanism_label}"
    return mechanism_label


def _compose_process_slot(unit):
    domain_lexicon = unit.get("domain_lexicon")
    if not domain_lexicon:
        domain_lexicon = build_domain_lexicon(unit.get("domain_pack") or unit.get("domain_context"))

    data_labels = _strong_slot_labels(unit.get("data_modifier_tokens", []), "data_modifier_tokens")
    method_labels = _strong_slot_labels(unit.get("method_modifier_tokens", []), "method_modifier_tokens")
    mechanism_label = _label_for_token(unit.get("mechanism_core", ""), prefer_zh=True)
    primary_data = "-".join(data_labels[:2]) if len(data_labels) >= 2 else (data_labels[0] if data_labels else "")

    res = ""
    if primary_data and mechanism_label:
        res = f"{primary_data}驱动"
    elif method_labels:
        method_label = method_labels[0]
        if method_label in PROCESS_METHOD_PREFIXES and mechanism_label:
            res = f"{method_label}{mechanism_label}"
        else:
            res = f"{method_label}方法"
    elif primary_data:
        res = primary_data

    if not res:
        return ""
    temp = res
    for suffix in ["驱动", "方法"]:
        if temp.endswith(suffix):
            temp = temp[:-len(suffix)]
    if not temp or domain_lexicon.is_generic_or_shell(temp) or domain_lexicon.is_generic_or_shell(res):
        return ""
    return res


def _compose_carrier_slot(unit):
    data_labels = _strong_slot_labels(unit.get("data_modifier_tokens", []), "data_modifier_tokens")
    if data_labels:
        return "-".join(data_labels[:2]) if len(data_labels) >= 2 else data_labels[0]
    fallback_data = _fallback_slot_labels(unit.get("data_modifier_tokens", []))
    return fallback_data[0] if fallback_data else ""


def _compose_application_slot(unit, tech_object_slot=""):
    profile = _scope_shell_profile(unit)
    task_labels = _strong_slot_labels(unit.get("task_constraint_tokens", []), "task_constraint_tokens")
    for label in task_labels:
        if label and label not in tech_object_slot:
            return label
    primary_scope = str(unit.get("primary_scope", "")).strip()
    if not primary_scope:
        return ""
    if bool(profile.get("scope_shell_heavy", False)) and not bool(profile.get("survives_without_scope", False)):
        return SCOPE_LABELS.get(primary_scope, "")
    if not tech_object_slot:
        return SCOPE_LABELS.get(primary_scope, "")
    return ""


def _build_technical_object_slots(unit):
    tech_object_slot = _compose_tech_object_slot(unit)
    capability_slot = _compose_capability_slot(unit, tech_object_slot=tech_object_slot)
    process_slot = _compose_process_slot(unit)
    carrier_slot = _compose_carrier_slot(unit)
    application_slot = _compose_application_slot(unit, tech_object_slot=tech_object_slot)
    tech_object_slot = _refine_tech_object_slot(
        unit,
        tech_object_slot,
        capability_slot=capability_slot,
        process_slot=process_slot,
        application_slot=application_slot,
    )

    profile = _scope_shell_profile(unit)
    score = 0
    reasons = []
    if tech_object_slot:
        score += 2
        reasons.append("具备对象主语")
    else:
        reasons.append("缺少稳定对象主语")
    if capability_slot:
        score += 1
        reasons.append("具备能力槽位")
    if process_slot:
        score += 1
        reasons.append("具备过程/工艺槽位")
    if carrier_slot:
        score += 1
        reasons.append("具备载体槽位")
    if application_slot:
        reasons.append("具备应用指向")
    if bool(profile.get("scope_shell_heavy", False)):
        score -= 1
        reasons.append("scope壳层偏重")
    if not bool(profile.get("survives_without_scope", False)):
        score -= 1
        reasons.append("去scope后稳定性弱")
    score = max(0, min(score, 5))

    return {
        "tech_object_slot": tech_object_slot,
        "capability_slot": capability_slot,
        "process_slot": process_slot,
        "carrier_slot": carrier_slot,
        "application_slot": application_slot,
        "object_like_score": score,
        "technical_objectness_reason": "；".join(_dedupe_preserve_order(reasons)),
    }


def _specific_anchor_strength(unit):
    slots = _build_technical_object_slots(unit)
    score = 0
    reasons = []
    tech_object_slot = slots.get("tech_object_slot", "")
    capability_slot = slots.get("capability_slot", "")
    process_slot = slots.get("process_slot", "")
    carrier_slot = slots.get("carrier_slot", "")
    application_slot = slots.get("application_slot", "")

    if tech_object_slot and tech_object_slot not in GENERIC_TECH_OBJECT_SLOTS:
        score += 2
        reasons.append("specific_object_slot")
    elif tech_object_slot:
        reasons.append("generic_object_slot")

    if capability_slot and capability_slot not in GENERIC_CAPABILITY_SLOTS:
        score += 2
        reasons.append("specific_capability_slot")
    elif capability_slot:
        reasons.append("generic_capability_slot")

    if process_slot:
        score += 2
        reasons.append("has_process_slot")
        if "-" in process_slot:
            score += 1
            reasons.append("compound_process_slot")
    if carrier_slot:
        score += 1
        reasons.append("has_carrier_slot")
        if "-" in carrier_slot:
            score += 1
            reasons.append("compound_carrier_slot")
    if application_slot and application_slot not in GENERIC_APPLICATION_SLOTS:
        score += 1
        reasons.append("specific_application_slot")
    elif application_slot:
        reasons.append("generic_application_slot")
    return min(score, 6), "；".join(reasons), slots


def _is_specific_slot_value(dim, value):
    value = str(value or "").strip()
    if not value:
        return False
    if dim == "tech_object_slot":
        return value not in GENERIC_TECH_OBJECT_SLOTS
    if dim == "capability_slot":
        return value not in GENERIC_CAPABILITY_SLOTS
    if dim == "application_slot":
        return value not in GENERIC_APPLICATION_SLOTS
    if dim in {"process_slot", "carrier_slot"}:
        return True
    return True


def _stable_anchor_threshold(item_count):
    return max(MIN_SECOND_ANCHOR_COUNT, int(math.ceil(item_count * MIN_SECOND_ANCHOR_RATIO)))


def _candidate_second_anchor_variants(items):
    dims = ["tech_object_slot", "process_slot", "carrier_slot", "application_slot"]
    counters = {dim: Counter() for dim in dims}
    item_count = max(len(items), 1)
    threshold = _stable_anchor_threshold(item_count)

    for item in items:
        slots = _build_technical_object_slots(item)
        for dim in dims:
            value = str(slots.get(dim, "")).strip()
            if not value or not _is_specific_slot_value(dim, value):
                continue
            counters[dim][value] += 1

    variants = []
    for dim in dims:
        for value, count in counters[dim].most_common():
            ratio = count / item_count
            variants.append(
                {
                    "dimension": dim,
                    "value": value,
                    "count": count,
                    "ratio": ratio,
                    "stable": bool(count >= threshold),
                    "stable_rank": (count, ratio),
                }
            )
    return variants, threshold


def _cluster_specificity_profile(items):
    if not items:
        return {
            "generic_cluster_risk": "unknown",
            "cluster_object_specificity": "unknown",
            "specific_anchor_strength": 0,
            "specificity_reason": "empty_cluster",
            "split_recommended": False,
            "split_dimension": "",
            "can_refine_further": "no",
            "refinement_ceiling_reason": "empty_cluster",
            "second_anchor_stability": "none",
            "second_anchor_candidates_summary": "",
            "refine_split_dimension": "",
        }

    strengths = []
    reasons = []
    object_slots = set()
    process_slots = set()
    carrier_slots = set()
    application_slots = set()
    capability_slots = set()

    for item in items:
        strength, reason, slots = _specific_anchor_strength(item)
        strengths.append(strength)
        if reason:
            reasons.append(reason)
        if slots.get("tech_object_slot"):
            object_slots.add(slots["tech_object_slot"])
        if slots.get("process_slot"):
            process_slots.add(slots["process_slot"])
        if slots.get("carrier_slot"):
            carrier_slots.add(slots["carrier_slot"])
        if slots.get("application_slot"):
            application_slots.add(slots["application_slot"])
        if slots.get("capability_slot"):
            capability_slots.add(slots["capability_slot"])

    max_strength = max(strengths) if strengths else 0
    avg_strength = sum(strengths) / max(len(strengths), 1)
    specific_process_count = len([value for value in process_slots if value])
    specific_carrier_count = len([value for value in carrier_slots if value])
    specific_application_count = len([value for value in application_slots if value and value not in GENERIC_APPLICATION_SLOTS])
    generic_object_only = all(value in GENERIC_TECH_OBJECT_SLOTS for value in object_slots) if object_slots else True

    if max_strength >= 4 or specific_process_count >= 2 or specific_carrier_count >= 2:
        specificity = "high"
    elif max_strength >= 2 or avg_strength >= 1.5 or specific_application_count >= 1:
        specificity = "medium"
    else:
        specificity = "low"

    if generic_object_only and specificity == "low":
        risk = "high"
    elif generic_object_only and specificity == "medium":
        risk = "medium"
    else:
        risk = "low"

    split_candidates = {
        "process_slot": specific_process_count,
        "carrier_slot": specific_carrier_count,
        "application_slot": specific_application_count,
    }
    split_dimension = max(split_candidates, key=split_candidates.get)
    split_recommended = bool(risk in {"high", "medium"} and split_candidates.get(split_dimension, 0) >= 2)

    reason_parts = [
        f"max_strength={max_strength}",
        f"avg_strength={avg_strength:.2f}",
        f"generic_object_only={generic_object_only}",
        f"process_variants={specific_process_count}",
        f"carrier_variants={specific_carrier_count}",
        f"application_variants={specific_application_count}",
    ]
    if split_recommended:
        reason_parts.append(f"split_by={split_dimension}")
    if reasons:
        reason_parts.append("slot_evidence=" + "；".join(_dedupe_preserve_order(reasons)[:4]))

    second_anchor_variants, second_anchor_threshold = _candidate_second_anchor_variants(items)
    stable_second_anchor_entries = [entry for entry in second_anchor_variants if entry["stable"]]
    stable_entries_by_dim = {}
    for entry in stable_second_anchor_entries:
        stable_entries_by_dim.setdefault(entry["dimension"], []).append(entry)
    second_anchor_stability = "none"
    if stable_second_anchor_entries:
        best_ratio = max(entry["ratio"] for entry in stable_second_anchor_entries)
        second_anchor_stability = "high" if best_ratio >= 0.5 else "medium"
    elif second_anchor_variants:
        second_anchor_stability = "low"

    second_anchor_candidates_summary = "；".join(
        f"{entry['dimension']}={entry['value']}({entry['count']}/{len(items)},{entry['ratio']:.2f})"
        for entry in second_anchor_variants[:4]
    )

    can_refine_further = "no"
    refinement_ceiling_reason = "当前对象已接近稳定粒度"
    refine_split_dimension = ""

    if specificity == "medium":
        stable_process_or_carrier = [
            entry for entry in stable_second_anchor_entries if entry["dimension"] in {"process_slot", "carrier_slot"}
        ]
        stable_compound_entries = [
            entry for entry in stable_process_or_carrier if "-" in entry["value"]
        ]
        multi_variant_dims = [
            dim for dim, values in stable_entries_by_dim.items()
            if dim in {"process_slot", "carrier_slot", "application_slot"} and len(values) >= 2
        ]

        current_specific_dims = 0
        if not generic_object_only and object_slots:
            current_specific_dims += 1
        if specific_process_count >= 1:
            current_specific_dims += 1
        elif specific_carrier_count >= 1:
            current_specific_dims += 1
        elif specific_application_count >= 1:
            current_specific_dims += 1

        latent_specific_object_entries = [
            entry for entry in stable_second_anchor_entries
            if entry["dimension"] == "tech_object_slot"
            if generic_object_only or len(object_slots) >= 2
        ]

        if latent_specific_object_entries:
            best_entry = sorted(latent_specific_object_entries, key=lambda item: item["stable_rank"], reverse=True)[0]
            can_refine_further = "yes"
            refine_split_dimension = best_entry["dimension"]
            refinement_ceiling_reason = (
                f"存在稳定第二锚点：{best_entry['dimension']}={best_entry['value']}"
                f"({best_entry['count']}/{len(items)})，可继续细分"
            )
        elif stable_compound_entries:
            best_entry = sorted(stable_compound_entries, key=lambda item: item["stable_rank"], reverse=True)[0]
            can_refine_further = "yes"
            refine_split_dimension = best_entry["dimension"]
            refinement_ceiling_reason = (
                f"存在复合第二锚点：{best_entry['dimension']}={best_entry['value']}"
                f"({best_entry['count']}/{len(items)}) 可支撑继续细分"
            )
        elif multi_variant_dims:
            refine_split_dimension = multi_variant_dims[0]
            best_entry = sorted(stable_entries_by_dim[refine_split_dimension], key=lambda item: item["stable_rank"], reverse=True)[0]
            can_refine_further = "yes"
            refinement_ceiling_reason = (
                f"同一维度存在多个稳定子锚点：{refine_split_dimension}，当前最佳拆分点="
                f"{best_entry['value']}({best_entry['count']}/{len(items)})"
            )
        elif second_anchor_variants:
            best_entry = second_anchor_variants[0]
            if (
                not generic_object_only
                and all(entry["dimension"] == "tech_object_slot" for entry in second_anchor_variants)
            ):
                refinement_ceiling_reason = "当前对象已稳定到具体 object+capability 粒度，但缺少额外 process/carrier 第二锚点"
            elif current_specific_dims <= 1 and generic_object_only:
                refinement_ceiling_reason = "主证据仅支持单一 robot+carrier/process+capability 粒度，缺少额外稳定第二锚点"
            else:
                refinement_ceiling_reason = (
                    f"第二锚点存在但不稳定：{best_entry['dimension']}={best_entry['value']}"
                    f"({best_entry['count']}/{len(items)}，阈值>={second_anchor_threshold})"
                )
        else:
            refinement_ceiling_reason = "主证据仅支持当前粒度，未形成稳定第二锚点"
    elif specificity == "high":
        refinement_ceiling_reason = "当前对象已具备较稳定的细对象骨架"
    elif specificity == "low":
        refinement_ceiling_reason = "当前对象过粗，优先处理泛对象而非继续细分"

    return {
        "generic_cluster_risk": risk,
        "cluster_object_specificity": specificity,
        "specific_anchor_strength": round(max_strength, 2),
        "specificity_reason": "；".join(reason_parts),
        "split_recommended": split_recommended,
        "split_dimension": split_dimension if split_recommended else "",
        "can_refine_further": can_refine_further,
        "refinement_ceiling_reason": refinement_ceiling_reason,
        "second_anchor_stability": second_anchor_stability,
        "second_anchor_candidates_summary": second_anchor_candidates_summary,
        "refine_split_dimension": refine_split_dimension,
    }


def _split_generic_cluster_groups(cluster_groups):
    refined = {}
    for signature, items in cluster_groups.items():
        profile = _cluster_specificity_profile(items)
        split_dimension = ""
        if profile.get("can_refine_further") == "yes" and profile.get("refine_split_dimension"):
            split_dimension = profile["refine_split_dimension"]
        elif profile["split_recommended"]:
            split_dimension = profile["split_dimension"]

        if not split_dimension:
            refined[signature] = items
            continue

        subgroup_map = {}
        for item in items:
            slots = _build_technical_object_slots(item)
            split_value = str(slots.get(split_dimension, "")).strip()
            if not split_value:
                split_value = "generic_anchor"
            subgroup_signature = f"{signature} || split:{split_dimension}:{split_value}"
            subgroup_map.setdefault(subgroup_signature, []).append({**item, **slots})

        # 只有真的分成多个非空子簇时才生效，否则退回原 cluster
        if len(subgroup_map) >= 2:
            refined.update(subgroup_map)
        else:
            refined[signature] = items
    return refined


# ---------------------------------------------------------------------------
# M2 / A4：TF-IDF 事后拆分，用于拆散"同 signature 但 title 主题异质"的大桶
#
# 触发条件（三重严防，确保 v2/supp550 零误伤）：
#   1. cluster 成员数 ≥ _TFIDF_MIN_SIZE (默认 4)
#   2. 成员间 char_wb TF-IDF pairwise median 相似度 < _TFIDF_MEDIAN_THRESHOLD
#   3. Agglomerative 拆分结果必须 ≥ 2 个子簇（否则保留原桶）
#
# v2 main cluster evidence count 多为 2-9；进门槛的 ≥4 cluster 同源性很高
# (median sim 预计 > 0.15)，不会触发拆分。仅对 scope3k 的大杂烩桶生效。
#
# env flag: WS_TFIDF_SPLIT=1 (默认开)；=0 时完全 bypass。
# ---------------------------------------------------------------------------
_WS_TFIDF_SPLIT = os.environ.get("WS_TFIDF_SPLIT", "1") == "1"
_TFIDF_MIN_SIZE = int(os.environ.get("WS_TFIDF_MIN_SIZE", "4"))
_TFIDF_MEDIAN_THRESHOLD = float(os.environ.get("WS_TFIDF_MEDIAN_THRESHOLD", "0.08"))
_TFIDF_SPLIT_DISTANCE = float(os.environ.get("WS_TFIDF_SPLIT_DISTANCE", "0.85"))


def _tfidf_cluster_doc(item):
    """组装 TF-IDF 输入文本：title + raw_phrase + text 前 500 字。"""
    parts = [
        str(item.get("source_title", "") or "").strip(),
        str(item.get("raw_phrase", "") or "").strip(),
        str(item.get("source_text", "") or "").strip()[:500],
    ]
    doc = " ".join(p for p in parts if p)
    return doc or "_"


def _tfidf_split_heterogeneous_clusters(cluster_groups):
    """对 size ≥ _TFIDF_MIN_SIZE 且 pairwise median TF-IDF 低的桶应用拆分。

    只拆不合：任何 cluster 成员集合只会被拆成若干非空子集，不与其他 cluster 合并。
    失败回退：sklearn 不可用 / 任何异常 / 单簇 → 保留原 cluster 原样。
    """
    if not _WS_TFIDF_SPLIT or not cluster_groups:
        return cluster_groups
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
    except ImportError:
        return cluster_groups

    refined = {}
    for signature, items in cluster_groups.items():
        if len(items) < _TFIDF_MIN_SIZE:
            refined[signature] = items
            continue

        # M2 精准守门：只对已被系统判为 "generic_cluster_risk=high" 的通用空壳
        # cluster 应用 TF-IDF 拆分。
        #
        # 理由（v2 M2 回归分析）：中文专利文本 char n-gram 相似度对"同主题但
        # 技术描述各异"的 cluster（如 21 条灵巧手抓取控制专利，讲不同的传感/
        # 结构/算法）会算出很低 median，这些是合法的同质主案例，绝不能拆。
        # generic_cluster_risk=high 的 cluster 才是 scope 空壳（感知机器人
        # 控制技术 / 机器人控制技术 等），拆它们不会伤害具体技术主案例。
        cluster_profile = _cluster_specificity_profile(items)
        if cluster_profile.get("generic_cluster_risk") != "high":
            refined[signature] = items
            continue

        docs = [_tfidf_cluster_doc(item) for item in items]
        try:
            vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
            X = vec.fit_transform(docs)
        except Exception:
            refined[signature] = items
            continue
        if X.shape[1] == 0:
            refined[signature] = items
            continue

        try:
            sim = cosine_similarity(X)
            lower = sim[np.tril_indices(len(items), -1)]
            median = float(np.median(lower)) if lower.size else 1.0
        except Exception:
            refined[signature] = items
            continue

        if median >= _TFIDF_MEDIAN_THRESHOLD:
            for it in items:
                it["cluster_split_applied"] = False
                it["cluster_tfidf_median_similarity"] = round(median, 4)
            refined[signature] = items
            continue

        try:
            dist = 1.0 - sim
            np.fill_diagonal(dist, 0.0)
            dist = np.clip(dist, 0.0, 2.0)
            labels = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=_TFIDF_SPLIT_DISTANCE,
                metric="precomputed",
                linkage="average",
            ).fit(dist).labels_
        except Exception:
            for it in items:
                it["cluster_split_applied"] = False
                it["cluster_tfidf_median_similarity"] = round(median, 4)
            refined[signature] = items
            continue

        unique_labels = sorted(set(labels.tolist()))
        if len(unique_labels) < 2:
            for it in items:
                it["cluster_split_applied"] = False
                it["cluster_tfidf_median_similarity"] = round(median, 4)
            refined[signature] = items
            continue

        for sub_id in unique_labels:
            sub_items = [items[i] for i in range(len(items)) if labels[i] == sub_id]
            if not sub_items:
                continue
            reason = f"tfidf_median={median:.3f}<{_TFIDF_MEDIAN_THRESHOLD}"
            for it in sub_items:
                it["cluster_split_applied"] = True
                it["cluster_split_reason"] = reason
                it["cluster_tfidf_median_similarity"] = round(median, 4)
            new_sig = f"{signature} || tfidf_split:{sub_id}"
            refined[new_sig] = sub_items

    return refined


def _normalize_process_anchor(process_slot, carrier_slot):
    process_slot = str(process_slot or "").strip()
    carrier_slot = str(carrier_slot or "").strip()
    if process_slot.endswith("驱动"):
        process_slot = process_slot[:-2]
    if process_slot.endswith("方法"):
        process_slot = process_slot[:-2]
    if process_slot and carrier_slot and process_slot == carrier_slot:
        return carrier_slot
    return process_slot or carrier_slot


def _technical_subject_anchor(unit):
    slots = _build_technical_object_slots(unit)
    tech_object_slot = str(slots.get("tech_object_slot", "")).strip()
    if not tech_object_slot or tech_object_slot in GENERIC_TECH_OBJECT_SLOTS:
        return ""
    return tech_object_slot


def _technical_process_anchor(unit):
    slots = _build_technical_object_slots(unit)
    return _normalize_process_anchor(slots.get("process_slot", ""), slots.get("carrier_slot", ""))


def _technical_item_profile(unit):
    slots = _build_technical_object_slots(unit)
    subject_anchor = _technical_subject_anchor(unit)
    process_anchor = _technical_process_anchor(unit)
    capability_slot = str(slots.get("capability_slot", "")).strip()
    application_slot = str(slots.get("application_slot", "")).strip()
    generic_risk = str(unit.get("generic_cluster_risk", "")).strip()
    specificity = str(unit.get("cluster_object_specificity", "")).strip()
    can_refine_further = str(unit.get("can_refine_further", "")).strip()

    subject_is_ready_head = any(word in subject_anchor for word in TECHNICAL_READY_HEADWORDS)
    subject_has_draft_hint = any(word in subject_anchor for word in TECHNICAL_DRAFT_HINTS)
    generic_robot_like = not subject_anchor

    if generic_robot_like:
        pattern = "generic_object_carrier_capability"
    elif process_anchor and capability_slot:
        pattern = "subject_process_capability"
    elif capability_slot:
        pattern = "subject_capability"
    else:
        pattern = "subject_only"

    if generic_robot_like:
        stage = "stitched_research_object"
        reason = "缺少稳定技术主语，当前更像 generic object 与 carrier/capability 的规则拼接"
    elif can_refine_further == "yes":
        stage = "technical_item_draft"
        reason = "已出现技术主语，但 cluster 仍可继续按第二锚点细分，暂不视为稳定技术条目"
    elif subject_is_ready_head and process_anchor and capability_slot and generic_risk == "low":
        stage = "technical_item_ready"
        reason = "具备稳定技术主语，并包含过程/载体与能力结构，接近技术清单式对象"
    elif subject_is_ready_head and capability_slot:
        stage = "technical_item_draft"
        reason = "具备技术主语，但过程/载体结构仍不够完整，属于技术条目雏形"
    elif subject_has_draft_hint and (process_anchor or capability_slot):
        stage = "technical_item_draft"
        reason = "已有较明确的技术主语，但仍偏研究对象命名，尚未达到清单式技术条目"
    elif specificity == "high" and subject_anchor:
        stage = "technical_item_draft"
        reason = "对象骨架较稳定，但技术主语仍偏场景/能力化表达，先归为技术条目雏形"
    else:
        stage = "stitched_research_object"
        reason = "对象仍主要依赖规则槽位拼接，未形成清晰的技术主语层级"

    return {
        "technical_item_stage": stage,
        "technical_item_reason": reason,
        "technical_subject_anchor": subject_anchor,
        "technical_process_anchor": process_anchor,
        "technical_item_pattern": pattern,
    }


def _technical_object_name_from_slots(unit, include_suffix=True):
    domain_lexicon = unit.get("domain_lexicon")
    if not domain_lexicon:
        domain_lexicon = build_domain_lexicon(unit.get("domain_pack") or unit.get("domain_context"))

    slots = _build_technical_object_slots(unit)
    tech_object_slot = slots["tech_object_slot"]
    capability_slot = slots["capability_slot"]
    process_slot = slots["process_slot"]
    carrier_slot = slots["carrier_slot"]
    application_slot = slots["application_slot"]

    non_empty_slots = [str(slots.get(k, "")).strip() for k in ["tech_object_slot", "capability_slot", "process_slot", "carrier_slot", "application_slot"]]
    non_empty_slots = [s for s in non_empty_slots if s]
    if non_empty_slots and all(domain_lexicon.is_generic_or_shell(s) for s in non_empty_slots):
        return ""

    item_profile = _technical_item_profile({**unit, **slots})

    if not tech_object_slot and not capability_slot:
        return ""

    def _compound_parts(text):
        raw = str(text or "").strip()
        if not raw:
            return []
        return [part.strip() for part in raw.split("-") if part.strip()]

    def _dedupe_compound(text):
        parts = _compound_parts(text)
        if not parts:
            return str(text or "").strip()
        return "-".join(_dedupe_preserve_order(parts))

    def _subtract_compound(text, reference):
        parts = _compound_parts(text)
        if not parts:
            return str(text or "").strip()
        reference_text = str(reference or "").strip()
        reference_parts = set(_compound_parts(reference))
        kept = [
            part for part in parts
            if part not in reference_parts
            and part not in reference_text
        ]
        return "-".join(kept) if kept else ""

    def _strip_prefix_overlap(text, reference):
        raw = str(text or "").strip()
        ref = str(reference or "").strip()
        if not raw or not ref:
            return raw
        if raw.startswith(ref):
            trimmed = raw[len(ref):].strip("-")
            return trimmed or raw
        ref_parts = _compound_parts(ref)
        if ref_parts and any(raw.startswith(part) for part in ref_parts):
            for part in ref_parts:
                if raw.startswith(part):
                    trimmed = raw[len(part):].strip("-")
                    return trimmed or raw
        return raw

    subject_anchor = item_profile.get("technical_subject_anchor", "")
    process_anchor = item_profile.get("technical_process_anchor", "")
    item_stage = item_profile.get("technical_item_stage", "")

    if subject_anchor and item_stage in {"technical_item_draft", "technical_item_ready"}:
        capability_phrase = capability_slot
        if application_slot and capability_phrase.startswith(application_slot):
            capability_phrase = capability_phrase[len(application_slot):] or capability_phrase
        capability_phrase = _apply_preferred_task_surface(unit, capability_phrase)
        display_task_surface = _display_title_task_surface(unit)
        subject_anchor = _preferred_object_phrase(unit, subject_anchor, capability_phrase=capability_phrase)
        capability_phrase = _strip_prefix_overlap(capability_phrase, subject_anchor)
        capability_phrase = re.sub(r"(控制|规划|训练|推理|仿真)\1$", r"\1", capability_phrase)
        process_phrase = _dedupe_compound(process_anchor)
        if process_phrase and process_phrase in subject_anchor:
            process_phrase = ""
        if process_phrase:
            process_phrase = _subtract_compound(process_phrase, subject_anchor)
        if process_phrase and capability_phrase:
            process_phrase = _subtract_compound(process_phrase, capability_phrase)
            capability_phrase = _strip_prefix_overlap(capability_phrase, process_phrase.replace("-", ""))
        capability_phrase = _dedupe_compound(capability_phrase)
        tail_phrase = f"{process_phrase}{capability_phrase}"
        if (
            display_task_surface
            and display_task_surface not in subject_anchor
            and display_task_surface not in tail_phrase
            and any(token in tail_phrase for token in ("控制", "规划", "训练", "仿真"))
        ):
            tail_phrase = f"{display_task_surface}{tail_phrase}"
        item_name = f"{subject_anchor}{process_phrase}{capability_phrase}"
        if tail_phrase:
            item_name = f"{subject_anchor}{tail_phrase}"
        item_name = re.sub(r"(操控)控制$", r"\1", item_name)
        item_name = re.sub(r"(导航)规划$", r"\1规划", item_name)
        item_name = item_name.strip()
        if include_suffix and item_name and not item_name.endswith(("技术", "方法", "系统", "模块")):
            if any(item_name.endswith(word) for word in ("训练", "规划", "控制", "推理", "仿真", "校准", "压缩", "蒸馏", "对齐", "操控")):
                item_name = f"{item_name}技术"
        if item_name:
            return _compact_display_candidate_label(item_name, {**unit, **slots})

    prefix = ""
    if process_slot.endswith("驱动") and carrier_slot:
        if carrier_slot and tech_object_slot and carrier_slot in tech_object_slot:
            prefix = ""
        else:
            prefix = f"基于{_dedupe_compound(carrier_slot)}的"
    elif process_slot.endswith("方法"):
        prefix = f"采用{process_slot[:-2]}的"
    elif process_slot and process_slot not in {capability_slot, carrier_slot}:
        prefix = f"{_dedupe_compound(process_slot)}"

    object_phrase = _dedupe_compound(tech_object_slot)
    if not object_phrase and application_slot:
        object_phrase = application_slot

    capability_phrase = _dedupe_compound(capability_slot)
    capability_phrase = _apply_preferred_task_surface(unit, capability_phrase)
    if capability_phrase and object_phrase and capability_phrase.startswith(object_phrase):
        capability_phrase = capability_phrase[len(object_phrase):]
    object_phrase = _preferred_object_phrase(unit, object_phrase, capability_phrase=capability_phrase)
    if object_phrase and prefix:
        prefix_anchor = prefix.removeprefix("基于").removeprefix("采用").removesuffix("的")
        reduced_object_phrase = _subtract_compound(object_phrase, prefix_anchor)
        if reduced_object_phrase != object_phrase:
            object_phrase = reduced_object_phrase
            if not object_phrase and prefix.startswith("基于"):
                prefix = prefix_anchor
    if prefix:
        prefix_anchor = prefix.removeprefix("基于").removeprefix("采用").removesuffix("的")
        compact_prefix_anchor = re.sub(r"[\s_\-]+", "", prefix_anchor)
        compact_capability = re.sub(r"[\s_\-]+", "", capability_phrase)
        if compact_prefix_anchor and compact_prefix_anchor == compact_capability:
            capability_phrase = ""
        else:
            capability_phrase = _strip_prefix_overlap(capability_phrase, prefix_anchor.replace("-", "")).strip()
    display_application_slot = str(application_slot or "").strip()
    preferred_task_surface = _display_title_task_surface(unit)
    if preferred_task_surface and preferred_task_surface not in object_phrase and preferred_task_surface not in capability_phrase:
        display_application_slot = preferred_task_surface
    if (
        display_application_slot
        and capability_phrase
        and display_application_slot not in object_phrase
        and display_application_slot not in {"世界模型", "具身智能"}
        and not capability_phrase.startswith(display_application_slot)
        and capability_phrase != display_application_slot
    ):
        capability_phrase = f"{display_application_slot}{capability_phrase}"

    prefix_anchor = prefix.removeprefix("基于").removeprefix("采用").removesuffix("的") if prefix else ""
    if (
        prefix.startswith("基于")
        and object_phrase in PRIMARY_GENERIC_OBJECT_LABELS
        and capability_phrase
    ):
        subject_phrase = _refine_tech_object_slot(
            unit,
            object_phrase,
            capability_slot=capability_phrase,
            process_slot=process_slot,
            application_slot=application_slot,
        )
        subject_phrase = _dedupe_compound(subject_phrase)
        subject_process = _dedupe_compound(prefix_anchor)
        if subject_process:
            subject_process = _subtract_compound(subject_process, subject_phrase)
            subject_process = _subtract_compound(subject_process, capability_phrase)
        capability_phrase = _strip_prefix_overlap(capability_phrase, subject_process)
        object_phrase = subject_phrase or object_phrase
        prefix = subject_process

    name = f"{prefix}{object_phrase}{capability_phrase}"
    name = name.replace("--", "-")
    name = re.sub(r"(技术){2,}$", r"\1", name)
    name = re.sub(r"(训练|规划|控制|推理|仿真|校准|压缩|蒸馏|对齐)\1$", r"\1", name)
    name = re.sub(r"([一-龥]+)-\1", r"\1", name)
    name = name.strip()
    if include_suffix and name and not name.endswith(("技术", "方法", "系统", "模块")):
        if any(name.endswith(word) for word in ("训练", "规划", "控制", "推理", "仿真", "校准", "压缩", "蒸馏", "对齐")):
            name = f"{name}技术"
    if name and domain_lexicon.is_generic_or_shell(name):
        return ""
    return _compact_display_candidate_label(name, {**unit, **slots})


def _stable_object_key_basis(unit):
    signature = str(unit.get("cluster_signature_key", "")).strip() or str(unit.get("premerge_signature_key", "")).strip()
    if signature:
        return f"signature::{signature}"

    slots = _build_technical_object_slots(unit)
    parts = []
    for key in ["primary_scope", "tech_object_slot", "process_slot", "capability_slot", "carrier_slot", "application_slot", "mechanism_core"]:
        value = str(slots.get(key, "") if key in slots else unit.get(key, "")).strip()
        if value:
            parts.append(f"{key}={value}")

    if parts:
        return " || ".join(parts)

    fallback = (
        str(unit.get("constraint_signature", "")).strip()
        or str(unit.get("canonical_candidate_name_en", "")).strip()
        or str(unit.get("normalized_candidate_text", "")).strip()
        or str(unit.get("raw_phrase", "")).strip()
        or str(unit.get("display_candidate_name", "")).strip()
    )
    return f"fallback::{fallback}" if fallback else ""


def _stable_object_id_from_basis(basis):
    text = str(basis or "").strip()
    if not text:
        return ""
    return f"obj::{hashlib.md5(text.encode('utf-8')).hexdigest()[:10]}"


def _strip_narrative_display_tail(text, max_chars=32):
    value = str(text or "").strip()
    if not value:
        return ""
    value = re.sub(r"[，。；;:：]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    for marker in DISPLAY_NARRATIVE_MARKERS:
        idx = value.find(marker)
        if idx >= 4:
            value = value[:idx].strip()
            break
    value = value.replace("过程中", "过程").replace("过程中的", "过程")
    value = value.strip(" ，,。.;；:：")
    if len(value) > max_chars:
        value = value[:max_chars].strip(" ，,。.;；:：")
    return value


def _compact_display_candidate_label(text, unit=None, max_chars=32):
    value = str(text or "").strip()
    if not value:
        return ""
    needs_compaction = len(value) > max_chars or any(marker in value for marker in DISPLAY_NARRATIVE_MARKERS)
    if not needs_compaction:
        return value

    unit = unit or {}
    components = []
    for key in ["tech_object_slot", "process_slot", "capability_slot"]:
        component = _strip_narrative_display_tail(unit.get(key, ""), max_chars=18)
        if component and component not in "".join(components):
            components.append(component)
    if components:
        compact = "".join(_dedupe_preserve_order(components)).strip()
        compact = _strip_narrative_display_tail(compact, max_chars=max_chars)
        if compact:
            return compact
    return _strip_narrative_display_tail(value, max_chars=max_chars)


def _is_generic_method_only_display_name(name, unit, domain_lexicon=None):
    text = str(name or "").strip()
    if not text:
        return False
    normalized = normalize_signal_phrase(text)
    generic_names = {normalize_signal_phrase(item) for item in GENERIC_METHOD_ONLY_DISPLAY_NAMES}
    mechanism = str((unit or {}).get("mechanism_core", "")).strip()
    mechanism_label = MECHANISM_LABELS.get(mechanism, _label_for_token(mechanism, prefer_zh=True))
    mechanism_only = mechanism_label and text in {mechanism_label, f"{mechanism_label}技术", f"{mechanism_label}方法"}
    if normalized not in generic_names and not mechanism_only:
        return False
    if domain_lexicon is None:
        domain_lexicon = build_domain_lexicon((unit or {}).get("domain_pack") or (unit or {}).get("domain_context"))
    if domain_lexicon is None or domain_lexicon.use_legacy_robot_rules:
        return False
    context = _surface_hint_context(unit or {}, extra_text=text)
    return not domain_lexicon.has_domain_anchor(context)


def _stable_object_label(unit):
    slots = _build_technical_object_slots(unit)
    subject = str(slots.get("tech_object_slot", "")).strip() or str(slots.get("application_slot", "")).strip()
    process = str(slots.get("process_slot", "")).strip()
    capability = str(slots.get("capability_slot", "")).strip()

    segments = []
    for segment in [subject, process, capability]:
        text = str(segment or "").strip()
        if not text:
            continue
        if segments and (text in "".join(segments) or "".join(segments).endswith(text)):
            continue
        segments.append(text)

    label = "".join(segments).strip()
    label = re.sub(r"(控制|规划|训练|推理|仿真)\1$", r"\1", label)
    if label and not label.endswith(("技术", "方法", "系统", "模块")):
        if any(label.endswith(word) for word in ("训练", "规划", "控制", "推理", "仿真", "操控", "装配", "抓取", "触觉", "导航")):
            label = f"{label}技术"

    if label:
        return _compact_display_candidate_label(label, {**unit, **slots})

    return _compact_display_candidate_label(
        str(unit.get("display_candidate_name", "")).strip()
        or str(unit.get("topic_summary_name", "")).strip()
        or str(unit.get("canonical_candidate_name_en", "")).strip()
        or str(unit.get("normalized_candidate_text", "")).strip(),
        {**unit, **slots},
    )


def _sanitize_display_candidate_name(name, unit):
    text = str(name or "").strip()
    if not text:
        return ""
    # 这里不激进删词；壳词问题通过 display_candidate_name_issue 暴露给诊断表。
    for shell_word in DISPLAY_SHELL_WORDS:
        text = text.replace(f"{shell_word}{shell_word}", shell_word)
    text = _compact_display_candidate_label(text, unit)
    source_type = str(unit.get("source_type", "")).strip().lower()
    if source_type != "patent":
        return text
    if not _contains_ascii_letters(text):
        return text
    mechanism = str(unit.get("mechanism_core", "")).strip()
    mechanism_label = MECHANISM_LABELS.get(mechanism, _label_for_token(mechanism, prefer_zh=True))
    selected = _ordered_display_constraints(unit)
    labels = []
    for value, _ in selected:
        label = _label_for_token(value, prefer_zh=True)
        if label and label not in labels:
            labels.append(label)
    if mechanism_label:
        labels.append(mechanism_label)
    return "".join(_dedupe_preserve_order(labels)) if labels else text


def _display_candidate_name_issue(name, unit):
    text = str(name or "").strip()
    if not text:
        return "missing_name"
    issues = []
    mechanism = str(unit.get("mechanism_core", "")).strip()
    mechanism_label = MECHANISM_LABELS.get(mechanism, _label_for_token(mechanism, prefer_zh=True))
    if mechanism_label and text == mechanism_label:
        issues.append("bare_mechanism")
    if any(word in text for word in DISPLAY_SHELL_WORDS):
        issues.append("upper_shell_word")
    profile = _scope_shell_profile(unit)
    if bool(profile.get("scope_shell_heavy", False)):
        issues.append("scope_shell_heavy")
    if not bool(profile.get("survives_without_scope", False)):
        issues.append("scope_dependent")
    # 使用 domain lexicon 检测名称是否被 shell/generic 词主导（Layer 4 防护）
    domain_lexicon = unit.get("domain_lexicon")
    if domain_lexicon is None:
        domain_lexicon = build_domain_lexicon(unit.get("domain_pack") or unit.get("domain_context"))
    if domain_lexicon and not domain_lexicon.use_legacy_robot_rules:
        if domain_lexicon.is_generic_or_shell(text):
            issues.append("shell_term_dominated")
        elif mechanism_label and text.endswith(mechanism_label):
            core = text[: -len(mechanism_label)].strip()
            if core and domain_lexicon.is_generic_or_shell(core):
                issues.append("core_is_shell")
    return "；".join(_dedupe_preserve_order(issues))


def _normalize_scope_names(raw_value):
    if isinstance(raw_value, list):
        return _dedupe_preserve_order(raw_value)
    if isinstance(raw_value, str) and raw_value.strip():
        return _dedupe_preserve_order(item.strip() for item in raw_value.split(","))
    return []


def _analysis_scope_from_context(context):
    for field in ["analysis_tech_field_name", "tech_field_name", "analysis_domain", "selected_domain"]:
        text = clean_event_text((context or {}).get(field, ""))
        if text and text != "未知":
            return text
    return ""


def _freeform_schema_token(value, *, allow_generic_action=False, max_len=90):
    text = clean_event_text(value)
    if not text or text == "未知":
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[\[\]{}<>]", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" ，,。.;；:：")
    if not text or len(text) > max_len:
        return ""
    generic_actions = {
        "提出", "研发", "研制", "设计", "发布", "开源", "测试", "验证",
        "propose", "proposed", "develop", "developed", "design", "designed",
        "release", "released", "test", "tested", "validate", "validated",
    }
    if not allow_generic_action and text.lower() in generic_actions:
        return ""
    return text


def _normalize_candidate_units(raw_value):
    if isinstance(raw_value, list):
        return [item for item in raw_value if isinstance(item, dict)]
    return []


def _event_list_field(event, field):
    return clean_event_list(event.get(field))


def _event_text_field(event, field):
    return clean_event_text(event.get(field))


def _schema_field_text(event):
    parts = [
        _event_text_field(event, "technical_object"),
        _event_text_field(event, "mechanism"),
        _event_text_field(event, "task"),
        _event_text_field(event, "capability_change"),
        _event_text_field(event, "problem_solved"),
        _event_text_field(event, "novelty_signal"),
        _event_text_field(event, "cross_domain_signal"),
        _event_text_field(event, "scene"),
        _event_text_field(event, "action"),
        " ".join(str(item) for item in _event_list_field(event, "technology")),
        " ".join(str(item) for item in _event_list_field(event, "data_modality")),
        " ".join(str(item) for item in _event_list_field(event, "method")),
    ]
    return " ".join(part for part in parts if str(part or "").strip())


def _has_schema_candidate_fields(event):
    text_fields = [
        "technical_object",
        "mechanism",
        "task",
        "capability_change",
        "problem_solved",
        "novelty_signal",
        "cross_domain_signal",
    ]
    list_fields = [
        "data_modality",
        "method",
        "mechanism_core_tokens",
        "task_constraint_tokens",
        "object_modifier_tokens",
        "data_modifier_tokens",
        "method_modifier_tokens",
    ]
    return any(_event_text_field(event, field) for field in text_fields) or any(
        _event_list_field(event, field) for field in list_fields
    )


def _normalize_schema_token(value):
    token = normalize_proxy_token(value)
    return token or str(value or "").strip()


def _schema_tokens_from_fields(event, explicit_field, extractor, *source_fields):
    values = []
    values.extend(_event_list_field(event, explicit_field))
    texts = [_event_text_field(event, field) for field in source_fields]
    values.extend(extractor(*texts))
    return _dedupe_preserve_order(
        _normalize_schema_token(value)
        for value in values
        if str(value or "").strip()
    )


def _schema_object_tokens(event, domain_lexicon=None):
    values = []
    values.extend(_event_list_field(event, "object_modifier_tokens"))
    technical_object = _event_text_field(event, "technical_object")
    extractor = (
        domain_lexicon.extract_object_modifier_tokens
        if domain_lexicon is not None
        else extract_object_modifier_tokens
    )
    values.extend(
        extractor(
            technical_object,
            _event_text_field(event, "task"),
            _event_text_field(event, "scene"),
            " ".join(str(item) for item in _event_list_field(event, "technology")),
        )
    )
    if technical_object and not values:
        values.append(technical_object)
    return _dedupe_preserve_order(
        _normalize_schema_token(value)
        for value in values
        if str(value or "").strip()
    )


def _schema_candidate_raw_text(
    technical_object,
    object_tokens,
    data_tokens,
    method_tokens,
    mechanism_tokens,
    task_tokens,
):
    raw_parts = []
    raw_parts.extend(object_tokens[:2])
    raw_parts.extend(data_tokens[:2])
    raw_parts.extend(method_tokens[:2])
    raw_parts.extend(mechanism_tokens[:2])
    raw_parts.extend(task_tokens[:2])
    if technical_object:
        raw_parts.insert(0, technical_object)
    return " ".join(_dedupe_preserve_order(raw_parts)).strip()


def _schema_relation_summary(object_tokens, task_tokens, data_tokens, method_tokens, mechanism_tokens):
    parts = []
    if data_tokens:
        parts.append(f"data={data_tokens[0]}")
    if object_tokens:
        parts.append(f"object={object_tokens[0]}")
    if task_tokens:
        parts.append(f"task={task_tokens[0]}")
    if method_tokens:
        parts.append(f"method={method_tokens[0]}")
    if mechanism_tokens:
        parts.append(f"mechanism={mechanism_tokens[0]}")
    return " | ".join(parts)


def _schema_candidate_units(event, observation_scopes, domain_lexicon=None):
    event = event.to_dict() if hasattr(event, "to_dict") else dict(event or {})
    if not _has_schema_candidate_fields(event):
        return []
    technical_object = _event_text_field(event, "technical_object")
    schema_text = _schema_field_text(event)
    analysis_scope = _freeform_schema_token(event.get("analysis_tech_field_name"), max_len=80)
    scope_names = (
        _normalize_scope_names(observation_scopes)
        or (
            domain_lexicon.detect_supported_observation_scopes(schema_text)
            if domain_lexicon is not None
            else detect_supported_observation_scopes(schema_text)
        )
        or ([analysis_scope] if analysis_scope else [])
    )
    if not scope_names:
        return []
    mechanism_tokens = _schema_tokens_from_fields(
        event,
        "mechanism_core_tokens",
        domain_lexicon.extract_mechanism_core_tokens if domain_lexicon is not None else extract_mechanism_core_tokens,
        "mechanism",
        "action",
        "technical_object",
        "evidence_span",
    )
    if not mechanism_tokens:
        mechanism_fallback = (
            _freeform_schema_token(event.get("mechanism"))
            or _freeform_schema_token(event.get("capability_change"))
            or _freeform_schema_token(event.get("action"), allow_generic_action=True, max_len=24)
        )
        if mechanism_fallback:
            mechanism_tokens = [mechanism_fallback]
    task_tokens = _schema_tokens_from_fields(
        event,
        "task_constraint_tokens",
        domain_lexicon.extract_task_constraint_tokens if domain_lexicon is not None else extract_task_constraint_tokens,
        "task",
        "problem_solved",
        "capability_change",
        "scene",
        "technical_object",
    )
    object_tokens = _schema_object_tokens(event, domain_lexicon=domain_lexicon)
    if not object_tokens:
        for tech in _event_list_field(event, "technology"):
            tech_token = _freeform_schema_token(tech)
            if tech_token and tech_token not in scope_names:
                object_tokens = [tech_token]
                break
    data_tokens = _schema_tokens_from_fields(
        event,
        "data_modifier_tokens",
        domain_lexicon.extract_data_modifier_tokens if domain_lexicon is not None else extract_data_modifier_tokens,
        "technical_object",
        "task",
        "evidence_span",
    )
    data_tokens = _dedupe_preserve_order(data_tokens + [_normalize_schema_token(item) for item in _event_list_field(event, "data_modality")])
    method_tokens = _schema_tokens_from_fields(
        event,
        "method_modifier_tokens",
        domain_lexicon.extract_method_modifier_tokens if domain_lexicon is not None else extract_method_modifier_tokens,
        "mechanism",
        "technical_object",
        "evidence_span",
    )
    method_tokens = _dedupe_preserve_order(method_tokens + [_normalize_schema_token(item) for item in _event_list_field(event, "method")])
    scene_tokens = _dedupe_preserve_order(
        (domain_lexicon.extract_scene_tokens if domain_lexicon is not None else extract_scene_tokens)(
            _event_text_field(event, "scene"),
            _event_text_field(event, "task"),
            _event_text_field(event, "technical_object"),
        )
    )
    evidence_present = bool(_event_text_field(event, "evidence_span"))
    if not mechanism_tokens and not (
        domain_lexicon is not None
        and not domain_lexicon.use_legacy_robot_rules
        and domain_lexicon.valid_candidate_pattern_matches(
            task_tokens=task_tokens,
            object_tokens=object_tokens,
            data_tokens=data_tokens,
            scene_tokens=scene_tokens,
            mechanism_tokens=mechanism_tokens,
            method_tokens=method_tokens,
            evidence_present=evidence_present,
        )
    ):
        return []

    has_anchor = bool(object_tokens or task_tokens or data_tokens or method_tokens or technical_object)
    if not has_anchor:
        return []

    raw_candidate_text = _schema_candidate_raw_text(
        technical_object,
        object_tokens,
        data_tokens,
        method_tokens,
        mechanism_tokens,
        task_tokens,
    )
    if not raw_candidate_text:
        return []

    source_mode = _event_text_field(event, "source_extraction_mode") or "schema"
    schema_source_mode = source_mode if source_mode.endswith("_schema") else f"{source_mode}_schema"
    if schema_source_mode == "schema_schema":
        schema_source_mode = "schema"
    relation_summary = _schema_relation_summary(
        object_tokens,
        task_tokens,
        data_tokens,
        method_tokens,
        mechanism_tokens,
    )
    return [
        {
            "raw_phrase": raw_candidate_text,
            "raw_candidate_text": raw_candidate_text,
            "raw_phrase_type": "schema_object_mechanism" if mechanism_tokens else "schema_domain_pack_pattern",
            "mechanism_core": mechanism_tokens[0] if mechanism_tokens else "",
            "secondary_mechanism_cores": mechanism_tokens[1:],
            "scope_names": scope_names,
            "mechanism_core_tokens": mechanism_tokens,
            "task_constraint_tokens": task_tokens,
            "object_modifier_tokens": object_tokens,
            "data_modifier_tokens": data_tokens,
            "method_modifier_tokens": method_tokens,
            "scene_tokens": scene_tokens,
            "action_tokens": mechanism_tokens,
            "is_scope_echo": False,
            "has_mechanism_core": bool(mechanism_tokens),
            "has_task_constraint": bool(task_tokens),
            "scope_context_supported": True,
            "has_non_scope_constraint": True,
            "generic_core_only": False,
            "source_extraction_mode": schema_source_mode,
            "scope_match_mode": _event_text_field(event, "scope_match_mode") or "schema_field",
            "evidence_span": _event_text_field(event, "evidence_span"),
            "relation_target": object_tokens[0] if object_tokens else technical_object,
            "relation_task": task_tokens[0] if task_tokens else "",
            "relation_data_modality": data_tokens[0] if data_tokens else "",
            "relation_method": method_tokens[0] if method_tokens else "",
            "relation_summary": relation_summary,
            "relation_signature": relation_summary,
        }
    ]


def _merge_candidate_units(schema_units, rule_units):
    merged = []
    seen = set()
    for unit in list(schema_units or []) + list(rule_units or []):
        if not isinstance(unit, dict):
            continue
        signature = (
            str(unit.get("raw_candidate_text") or unit.get("raw_phrase") or "").strip().lower(),
            tuple(_normalize_scope_names(unit.get("scope_names", []))),
            tuple(_dedupe_preserve_order(unit.get("mechanism_core_tokens", []))),
            tuple(_dedupe_preserve_order(unit.get("task_constraint_tokens", []))),
            tuple(_dedupe_preserve_order(unit.get("object_modifier_tokens", []))),
            tuple(_dedupe_preserve_order(unit.get("data_modifier_tokens", []))),
            tuple(_dedupe_preserve_order(unit.get("method_modifier_tokens", []))),
        )
        if signature in seen:
            continue
        merged.append(unit)
        seen.add(signature)
    return merged


def _candidate_units_for_event(event, observation_scopes, domain_lexicon=None):
    schema_units = _schema_candidate_units(event, observation_scopes, domain_lexicon=domain_lexicon)
    rule_units = _normalize_candidate_units(event.get("candidate_units", []))
    return _merge_candidate_units(schema_units, rule_units)


def _pick_constraint(values, mechanism_core, generic_tokens):
    values = _dedupe_preserve_order(values)
    mechanism_core = str(mechanism_core or "").strip()
    filtered = [
        value for value in values
        if value
        and value != mechanism_core
        and value not in generic_tokens
    ]
    if filtered:
        return filtered[0]
    fallback = [value for value in values if value and value != mechanism_core]
    if fallback:
        return fallback[0]
    return ""


def _canonical_anchor_token(value):
    token = normalize_proxy_token(value)
    token = ANCHOR_ALIAS_MAP.get(token, token)
    return token


def _ordered_anchor_values(values, key):
    cleaned = []
    seen = set()
    priorities = ANCHOR_PRIORITY.get(key, [])
    normalized_values = []
    for value in _dedupe_preserve_order(values):
        token = _canonical_anchor_token(value)
        if not token or token in seen:
            continue
        normalized_values.append(token)
        seen.add(token)
    if not normalized_values:
        return []
    for preferred in priorities:
        if preferred in normalized_values and preferred not in cleaned:
            cleaned.append(preferred)
    for value in normalized_values:
        if value not in cleaned:
            cleaned.append(value)
    return cleaned


def _anchor_bundle(unit):
    bundle = {
        "task": "",
        "object": "",
        "data": "",
        "method": "",
    }
    for output_key, source_key in [
        ("task", "task_constraint_tokens"),
        ("object", "object_modifier_tokens"),
        ("data", "data_modifier_tokens"),
        ("method", "method_modifier_tokens"),
    ]:
        ordered = _ordered_anchor_values(unit.get(source_key, []), source_key)
        mechanism = str(unit.get("mechanism_core", "")).strip()
        ordered = [value for value in ordered if value and value != mechanism]
        if output_key != "data":
            ordered = [value for value in ordered if value not in DISPLAY_SUPPRESS_TOKENS]
        bundle[output_key] = ordered[0] if ordered else ""

    if bundle["task"] == bundle["object"]:
        bundle["object"] = ""
    if bundle["task"] in GENERIC_SCOPE_FILLER_TOKENS and bundle["object"]:
        bundle["task"] = ""
    if bundle["object"] in GENERIC_SCOPE_FILLER_TOKENS and bundle["task"]:
        bundle["object"] = ""
    if bundle["data"] == "simulation" and (bundle["task"] or bundle["object"]):
        bundle["data"] = ""
    return bundle


def _primary_constraint(unit):
    mechanism_core = str(unit.get("mechanism_core", "")).strip()
    for key, generic_tokens in [
        ("task_constraint_tokens", GENERIC_TASK_TOKENS),
        ("object_modifier_tokens", GENERIC_OBJECT_TOKENS),
        ("data_modifier_tokens", GENERIC_DATA_TOKENS),
    ]:
        value = _pick_constraint(unit.get(key, []), mechanism_core, generic_tokens)
        if value:
            return value, key
    value = _pick_constraint(unit.get("method_modifier_tokens", []), mechanism_core, GENERIC_METHOD_TOKENS)
    if value:
        return value, "method_modifier_tokens"
    return "", ""


def _ordered_constraints(unit):
    mechanism_core = str(unit.get("mechanism_core", "")).strip()
    data_values = set(_dedupe_preserve_order(unit.get("data_modifier_tokens", [])))
    ordered = []
    for key, generic_tokens in [
        ("task_constraint_tokens", GENERIC_TASK_TOKENS),
        ("object_modifier_tokens", GENERIC_OBJECT_TOKENS),
        ("data_modifier_tokens", GENERIC_DATA_TOKENS),
        ("method_modifier_tokens", GENERIC_METHOD_TOKENS),
    ]:
        values = _dedupe_preserve_order(unit.get(key, []))
        strong_values = [
            value for value in values
            if value and value != mechanism_core and value not in generic_tokens
        ]
        weak_values = [
            value for value in values
            if value and value != mechanism_core and value in generic_tokens
        ]
        if key == "task_constraint_tokens" and data_values:
            strong_values = [value for value in strong_values if value not in data_values]
            weak_values = [value for value in weak_values if value not in data_values]
        for value in strong_values + weak_values:
            if (value, key) not in ordered:
                ordered.append((value, key))
    return ordered


def _selected_constraints(unit, max_items=2):
    ordered = _ordered_constraints(unit)
    primary_strong = []
    primary_weak = []
    fallback = []
    seen_values = set()
    for value, key in ordered:
        if value in seen_values:
            continue
        seen_values.add(value)
        if key == "method_modifier_tokens":
            fallback.append((value, key))
        elif (
            (key == "task_constraint_tokens" and value in GENERIC_TASK_TOKENS)
            or (key == "object_modifier_tokens" and value in GENERIC_OBJECT_TOKENS)
            or (key == "data_modifier_tokens" and value in GENERIC_DATA_TOKENS)
        ):
            primary_weak.append((value, key))
        else:
            primary_strong.append((value, key))
    selected = primary_strong[:max_items]
    if len(selected) < max_items:
        selected.extend(primary_weak[: max_items - len(selected)])
    if len(selected) < max_items and not selected:
        selected.extend(fallback[: max_items - len(selected)])
    elif len(selected) < max_items and not any(key == "data_modifier_tokens" for _, key in selected):
        data_values = [item for item in (primary_strong + primary_weak) if item[1] == "data_modifier_tokens" and item not in selected]
        selected.extend(data_values[: max_items - len(selected)])
    return selected[:max_items]


def _preferred_display_constraints(unit, max_items=2):
    selected = _selected_constraints(unit, max_items=max_items + 1)
    non_method = [(value, key) for value, key in selected if key != "method_modifier_tokens"]
    if non_method:
        strong_non_method = [(value, key) for value, key in non_method if not _is_generic_constraint(value, key)]
        if strong_non_method:
            return strong_non_method[:max_items]
        return non_method[:max_items]
    return selected[:max_items]


def _is_generic_constraint(value, key):
    return (
        (key == "task_constraint_tokens" and value in GENERIC_TASK_TOKENS)
        or (key == "object_modifier_tokens" and value in GENERIC_OBJECT_TOKENS)
        or (key == "data_modifier_tokens" and value in GENERIC_DATA_TOKENS)
        or (key == "method_modifier_tokens" and value in GENERIC_METHOD_TOKENS)
    )


def _is_soft_display_constraint(value, key):
    return (
        (key == "task_constraint_tokens" and value in DISPLAY_SOFT_TASK_TOKENS)
        or (key == "object_modifier_tokens" and value in DISPLAY_SOFT_OBJECT_TOKENS)
        or (key == "data_modifier_tokens" and value in DISPLAY_SOFT_DATA_TOKENS)
    )


def _is_scope_shell_constraint(value, key):
    return (
        (key == "task_constraint_tokens" and value in SCOPE_SHELL_TASK_TOKENS)
        or (key == "object_modifier_tokens" and value in SCOPE_SHELL_OBJECT_TOKENS)
        or (key == "data_modifier_tokens" and value in SCOPE_SHELL_DATA_TOKENS)
        or (key == "method_modifier_tokens" and value in SCOPE_SHELL_METHOD_TOKENS)
    )


def _scope_shell_profile(unit):
    selected_constraints = _ordered_constraints(unit)
    non_scope_constraints = [
        (value, key)
        for value, key in selected_constraints
        if not _is_generic_constraint(value, key)
    ]
    non_scope_constraint_count = len({value for value, _ in non_scope_constraints})
    specific_anchor_constraints = [
        (value, key)
        for value, key in non_scope_constraints
        if key in {"task_constraint_tokens", "object_modifier_tokens", "data_modifier_tokens"}
        and not _is_scope_shell_constraint(value, key)
    ]
    task_object_anchor_constraints = [
        (value, key)
        for value, key in specific_anchor_constraints
        if key in {"task_constraint_tokens", "object_modifier_tokens"}
    ]
    data_anchor_constraints = [
        (value, key)
        for value, key in specific_anchor_constraints
        if key == "data_modifier_tokens"
    ]
    survives_without_scope = bool(non_scope_constraint_count >= 2 and specific_anchor_constraints)
    scope_shell_heavy = False
    scope_shell_reason = ""
    if bool(unit.get("has_mechanism_core", False)) and bool(unit.get("has_non_scope_constraint", False)):
        if not survives_without_scope:
            scope_shell_heavy = True
            display_name = str(unit.get("display_candidate_name", "") or unit.get("topic_summary_name", "")).strip()
            scope_labels = [label for label in SCOPE_LABELS.values() if label]
            if display_name and any(label in display_name for label in scope_labels):
                scope_shell_reason = "scope_token_in_name"
            elif data_anchor_constraints and not task_object_anchor_constraints:
                scope_shell_reason = "only_data_method_anchor"
            elif non_scope_constraint_count < 2:
                scope_shell_reason = "non_scope_constraints_insufficient"
            elif not specific_anchor_constraints:
                scope_shell_reason = "constraints_too_generic_without_scope"
            else:
                scope_shell_reason = "scope_dependent_after_strip"
    return {
        "non_scope_constraint_count": non_scope_constraint_count,
        "survives_without_scope": survives_without_scope,
        "scope_shell_heavy": scope_shell_heavy,
        "scope_shell_reason": scope_shell_reason,
    }


def _strong_constraint_ready(unit):
    selected = _selected_constraints(unit, max_items=2)
    if not selected:
        return False
    strong_categories = {
        key
        for value, key in selected
        if not (
            (key == "task_constraint_tokens" and value in GENERIC_TASK_TOKENS)
            or (key == "object_modifier_tokens" and value in GENERIC_OBJECT_TOKENS)
            or (key == "data_modifier_tokens" and value in GENERIC_DATA_TOKENS)
            or (key == "method_modifier_tokens" and value in GENERIC_METHOD_TOKENS)
        )
    }
    if len(strong_categories) >= 2:
        return True
    if len(strong_categories) == 1:
        return any(key != "method_modifier_tokens" for _, key in selected)
    return False


def _compression_mode(unit):
    _, source_key = _primary_constraint(unit)
    mapping = {
        "object_modifier_tokens": "object_task_mechanism",
        "task_constraint_tokens": "object_task_mechanism",
        "data_modifier_tokens": "data_mechanism",
        "method_modifier_tokens": "method_mechanism",
    }
    return mapping.get(source_key, "generic")


def _normalize_constraint_values(values, key):
    tokens = [normalize_proxy_token(value) for value in _dedupe_preserve_order(values)]
    tokens = [token for token in tokens if token]
    if not tokens:
        return []

    def _matches(token, *aliases):
        alias_set = set()
        for alias in aliases:
            alias_text = str(alias or "").strip()
            if not alias_text:
                continue
            alias_set.add(alias_text)
            alias_set.add(normalize_proxy_token(alias_text))
        return token in alias_set

    def _drop_if_present(items, *aliases):
        return [item for item in items if not _matches(item, *aliases)]

    def _has_any(items, *aliases):
        return any(_matches(item, *aliases) for item in items)

    if key == "task_constraint_tokens":
        if len(tokens) >= 2:
            tokens = _drop_if_present(tokens, "robot", "agent", "embodied", "interactive") or tokens
        if _has_any(tokens, "navigation"):
            tokens = _drop_if_present(tokens, "robot") or tokens
        if _has_any(tokens, "manipulation"):
            tokens = _drop_if_present(tokens, "robot") or tokens
        if _has_any(tokens, "control") and _has_any(tokens, "robot"):
            tokens = _drop_if_present(tokens, "robot") or tokens
        if _has_any(tokens, "control") and _has_any(tokens, "manipulation"):
            tokens = _drop_if_present(tokens, "manipulation") or tokens

    elif key == "object_modifier_tokens":
        if _has_any(tokens, "manipulator") and _has_any(tokens, "robot"):
            tokens = _drop_if_present(tokens, "robot")
        if _has_any(tokens, "navigation") and _has_any(tokens, "robot"):
            tokens = _drop_if_present(tokens, "robot") or tokens
        if _has_any(tokens, "manipulator") and _has_any(tokens, "policy"):
            tokens = _drop_if_present(tokens, "policy") or tokens
        if _has_any(tokens, "policy") and _has_any(tokens, "robot") and len(tokens) <= 2:
            tokens = _drop_if_present(tokens, "policy", "robot")
            tokens.append(normalize_proxy_token("manipulator"))
        if _has_any(tokens, "robot") and _has_any(tokens, "perception"):
            tokens = _drop_if_present(tokens, "perception")
        if _has_any(tokens, "control") and _has_any(tokens, "policy"):
            tokens = _drop_if_present(tokens, "policy")

        tactile_context = _has_any(tokens, "tactile")
        if tactile_context:
            tokens = _drop_if_present(tokens, "policy", "robot", "perception")
            if _has_any(tokens, "manipulator"):
                tokens = _drop_if_present(tokens, "manipulator")
                if not _has_any(tokens, "control"):
                    tokens.append(normalize_proxy_token("control"))
            if _has_any(tokens, "control") and not _has_any(tokens, "tactile"):
                tokens.append(normalize_proxy_token("tactile"))

        object_order = {
            "navigation": 10,
            "manipulator": 20,
            "control": 30,
            "tactile": 40,
            "perception": 50,
            "policy": 60,
            "robot": 70,
        }

        def _object_order(token):
            for alias, rank in object_order.items():
                if _matches(token, alias):
                    return rank
            return 999

        tokens = sorted(tokens, key=lambda token: (_object_order(token), token))

    elif key == "data_modifier_tokens":
        token_set = set(tokens)
        if _has_any(tokens, "multimodal"):
            tokens = ["multimodal"]
        elif _has_any(tokens, "video"):
            tokens = ["video"]
        elif _has_any(tokens, "3d"):
            tokens = ["3d"]
        elif _has_any(tokens, "trajectory"):
            tokens = ["trajectory"]
        elif _has_any(tokens, "visual"):
            tokens = ["visual"]
        elif _has_any(tokens, "sensor"):
            tokens = ["sensor"]
        elif _has_any(tokens, "temporal"):
            tokens = ["temporal"]
        elif _has_any(tokens, "simulation"):
            tokens = ["simulation"]

    return _dedupe_preserve_order(tokens)


def _constraint_signature(unit, normalized=False):
    parts = []
    for prefix, key in [
        ("task", "task_constraint_tokens"),
        ("object", "object_modifier_tokens"),
        ("data", "data_modifier_tokens"),
        ("method", "method_modifier_tokens"),
    ]:
        values = _normalize_constraint_values(unit.get(key, []), key) if normalized else _dedupe_preserve_order(unit.get(key, []))
        if values:
            parts.append(f"{prefix}:{'|'.join(values)}")
    return " || ".join(parts)


def _normalized_grouping_view(unit):
    normalized = dict(unit)
    for key in [
        "task_constraint_tokens",
        "object_modifier_tokens",
        "data_modifier_tokens",
        "method_modifier_tokens",
    ]:
        normalized[key] = _normalize_constraint_values(unit.get(key, []), key)
    return normalized


def _source_premerge_signature(unit):
    normalized = _normalized_grouping_view(unit)
    parts = [
        f"mechanism:{str(normalized.get('mechanism_core', '')).strip()}",
        f"scope:{str(normalized.get('primary_scope', '')).strip()}",
        _constraint_signature(normalized, normalized=False),
    ]
    return " || ".join(part for part in parts if part)


def _refresh_unit_row_fields(row):
    refreshed = dict(row)
    refreshed["canonical_candidate_name_en"] = _canonical_candidate_name_en(refreshed)
    refreshed["normalized_candidate_text"] = refreshed["canonical_candidate_name_en"]
    refreshed["constraint_signature"] = _constraint_signature(refreshed, normalized=False)
    refreshed["normalized_constraint_signature"] = _constraint_signature(refreshed, normalized=True)
    refreshed["compression_mode"] = _compression_mode(refreshed)
    refreshed["internal_candidate_label"] = _internal_candidate_label(
        refreshed,
        refreshed["canonical_candidate_name_en"],
        refreshed["constraint_signature"],
    )
    slot_fields = _build_technical_object_slots(refreshed)
    refreshed.update(slot_fields)
    return refreshed


def _premerge_source_units(rows):
    if not rows:
        return rows

    source_rows = {}
    for row in rows:
        source_rows.setdefault(str(row.get("id", "")).strip(), []).append(row)

    grouped = {}
    for row in rows:
        signature = _source_premerge_signature(row)
        key = (str(row.get("id", "")).strip(), signature)
        grouped.setdefault(key, []).append(row)

    merged_rows = []
    for (_, signature), items in grouped.items():
        source_id = str(items[0].get("id", "")).strip()
        source_items = source_rows.get(source_id, [])
        source_signatures = [_source_premerge_signature(item) for item in source_items]
        distinct_source_signatures = _dedupe_preserve_order(source_signatures)
        if len(source_items) <= 1:
            block_reason = "single_unit_source"
        elif len(distinct_source_signatures) == len(source_items):
            block_reason = "no_signature_match_within_source"
        else:
            block_reason = "merged_elsewhere"

        if len(items) == 1:
            item = dict(items[0])
            item["premerge_signature_key"] = signature
            item["source_premerge_group_size"] = 1
            item["source_premerge_applied"] = False
            item["source_premerge_original_unit_count"] = 1
            item["source_premerge_block_reason"] = block_reason
            item["source_premerge_before_signatures"] = " ||| ".join(distinct_source_signatures)
            item["source_premerge_before_unit_count"] = len(source_items)
            item["normalized_constraint_signature"] = _constraint_signature(item, normalized=True)
            merged_rows.append(item)
            continue

        representative = max(
            items,
            key=lambda row: (
                int(row.get("non_scope_constraint_count", 0) or 0),
                len(_dedupe_preserve_order(row.get("object_modifier_tokens", []))),
                len(_dedupe_preserve_order(row.get("data_modifier_tokens", []))),
                len(str(row.get("raw_phrase", "")).strip()),
            ),
        )
        merged = dict(representative)
        for key in [
            "task_constraint_tokens",
            "object_modifier_tokens",
            "data_modifier_tokens",
            "method_modifier_tokens",
            "scene_tokens",
            "mechanism_core_tokens",
            "secondary_mechanism_cores",
            "display_candidate_aliases",
        ]:
            merged[key] = _dedupe_preserve_order(
                value
                for item in items
                for value in (item.get(key, []) or [])
            )
        merged["template_variant_count"] = len(
            _dedupe_preserve_order(item.get("raw_phrase", "") for item in items if item.get("raw_phrase"))
        )
        merged["premerge_signature_key"] = signature
        merged["source_premerge_group_size"] = len(items)
        merged["source_premerge_applied"] = True
        merged["source_premerge_original_unit_count"] = len(items)
        merged["source_premerge_block_reason"] = ""
        merged["source_premerge_before_signatures"] = " ||| ".join(distinct_source_signatures)
        merged["source_premerge_before_unit_count"] = len(source_items)
        merged = _refresh_unit_row_fields(merged)
        merged_rows.append(merged)

    return merged_rows


def _canonical_candidate_name_en(unit):
    mechanism = str(unit.get("mechanism_core", "")).strip()
    if not mechanism:
        return ""
    anchors = _anchor_bundle(unit)
    selected = [anchors["task"] or anchors["object"]]
    if anchors["data"] and anchors["data"] not in selected:
        selected.append(anchors["data"])
    elif anchors["object"] and anchors["object"] not in selected:
        selected.append(anchors["object"])
    selected = [value for value in selected if value]
    if not selected:
        selected = [value for value, _ in _selected_constraints(unit, max_items=2)]
    if not selected:
        return mechanism
    return " ".join(selected + [mechanism])


# --- round1 patch: title-topic discriminator for cluster signature ---
# 目的：当多条事件的 slot (tech_object/capability/process/carrier) 恰好相同，
# 但源标题讲的其实是完全不同的技术主题时，给签名追加一个强区分锚点，
# 防止"机械臂/控制/传感驱动"这类通用槽位把异质事件碰撞到同一 cluster。
_BASE_TITLE_TOPIC_ANCHORS = (
    # 工艺/制造
    "磨抛", "抛光", "打磨", "喷涂", "焊接", "装配", "铺丝", "铺带",
    "切削", "打孔", "清洁",
    # 具体产品/形态
    "开发板", "开发公板", "开发套件", "芯片组", "公板",
    "人形机器人", "双足机器人", "四足机器人", "移动机器人",
    "末端执行器", "末端", "夹具", "夹爪",
    "灵巧手", "五指灵巧手", "仿生手", "软体手", "仿人手", "机械手",
    "双臂", "双机械臂", "单臂",
    # 具体任务/场景
    "抓取", "搬运", "铺丝轨迹", "跟踪", "导航", "装配", "模仿学习",
    # 特定数据/感知
    "点云", "骨骼", "关节", "触觉", "力触觉", "视触觉",
    # 商业/公司动态（通常与技术主题无关，应单独成簇）
    "融资", "估值", "发布会", "IPO", "挂牌", "签约", "收购",
    "开发公板",
)

# M1 / A2-H：人形主线英文 anchor 扩展（仅扩 humanoid 相关；非人形 robotics 不加）
# 匹配逻辑在 _title_topic_discriminator 中已改为 case-insensitive，
# 因此这里统一用小写形式登记。
_HUMANOID_EN_TITLE_TOPIC_ANCHORS = (
    # 产品/形态
    "humanoid robots", "humanoid robot", "humanoid",
    "dexterous hand", "dexterous manipulation",
    "dual-arm", "dual arm", "bimanual",
    "exoskeleton",
    "end-effector", "end effector",
    # 任务/方法
    # 注：不加裸"vla"—会误中 "nvlabs" 等；用长形式匹配足够
    "vision-language-action", "vision language action", "openvla",
    "video world model", "world models", "world model", "world-model",
    "imitation learning", "behavior cloning",
    "teleoperation",
    "manipulation",
    "grasping",
    # 数据/感知
    "tactile sensor", "tactile", "haptic",
)

# env flag：默认关闭 — M1 试验证明：即使 CJK-gated，ext anchor 仍会
# 通过 v2 内部的英文 paper 间接拆散中文主案例（如"双臂机器人控制技术"
# 的 2 条 evidence 里 1 条 paper 被拆走 → main 消失）。
# 因此 49A M1 revert 为默认关闭，保留为后续实验开关，不做主线路径。
_WS_TITLE_ANCHOR_EXT = os.environ.get("WS_TITLE_ANCHOR_EXT", "0") == "1"

STRONG_TITLE_TOPIC_ANCHORS = (
    _BASE_TITLE_TOPIC_ANCHORS + _HUMANOID_EN_TITLE_TOPIC_ANCHORS
    if _WS_TITLE_ANCHOR_EXT
    else _BASE_TITLE_TOPIC_ANCHORS
)


def _title_topic_discriminator(unit):
    """从 source_title / raw_phrase 里找一个强区分锚点。

    - 只取 STRONG_TITLE_TOPIC_ANCHORS 里的词
    - 优先 source_title 命中，其次 raw_phrase
    - 命中多个时，取最长匹配（更具体）
    - 未命中时返回 ""（签名不追加，保持与历史行为一致）
    """
    title = str(unit.get("source_title", "") or "").strip()
    text = str(unit.get("source_text", "") or "").strip()
    raw_phrase = str(unit.get("raw_phrase", "") or "").strip()
    # M1 revert：完全恢复 pre-M1 行为（case-sensitive substring match，
    # first-source-with-hit wins，longest tie-break）。
    # STRONG_TITLE_TOPIC_ANCHORS 在默认 flag=0 下 = _BASE_TITLE_TOPIC_ANCHORS，
    # 所以行为与 M0 一致。若实验需要 humanoid 英文 anchor，
    # 设 WS_TITLE_ANCHOR_EXT=1 启用（会改变 cluster 结构，需验收）。
    hits = []
    for source_text_value in (title, raw_phrase, text):
        if not source_text_value:
            continue
        for anchor in STRONG_TITLE_TOPIC_ANCHORS:
            if anchor in source_text_value:
                hits.append(anchor)
        if hits:
            break
    if not hits:
        return ""
    deduped = list(dict.fromkeys(hits))
    indexed = list(enumerate(deduped))
    indexed.sort(key=lambda pair: (-len(pair[1]), pair[0]))
    return indexed[0][1]


def _theme_group_signature(unit):
    """对象层聚合签名的小修版。

    目标不是重写聚合，而是在原有字符串签名框架上，把主导因子从
    `scope + mechanism` 轻量推向 `tech_object/capability/process/carrier`。

    round1 补丁：在通用槽位签名之外追加 source_title 级别的强区分锚点，
    避免异质事件因为 slot 相同而被碰撞到同一 cluster。
    """
    normalized_unit = _normalized_grouping_view(unit)
    primary_scope = str(normalized_unit.get("primary_scope", "")).strip()
    mechanism = str(normalized_unit.get("mechanism_core", "")).strip()
    slots = _build_technical_object_slots(normalized_unit)
    anchors = _anchor_bundle(normalized_unit)
    profile = _scope_shell_profile(normalized_unit)

    signature_parts = []
    if slots.get("tech_object_slot"):
        signature_parts.append(f"object:{slots['tech_object_slot']}")
    if slots.get("capability_slot"):
        signature_parts.append(f"capability:{slots['capability_slot']}")
    elif mechanism:
        signature_parts.append(f"mechanism:{mechanism}")
    if slots.get("process_slot"):
        signature_parts.append(f"process:{slots['process_slot']}")
    elif slots.get("carrier_slot"):
        signature_parts.append(f"carrier:{slots['carrier_slot']}")

    if len(signature_parts) <= 1:
        primary_anchor = anchors["task"] or anchors["object"]
        if primary_anchor:
            signature_parts.append(f"anchor:{primary_anchor}")
        if anchors["data"]:
            signature_parts.append(f"data:{anchors['data']}")

    if (
        primary_scope
        and (
            bool(profile.get("scope_shell_heavy", False))
            or not slots.get("tech_object_slot")
            or not bool(profile.get("survives_without_scope", False))
        )
    ):
        signature_parts.append(f"scope:{primary_scope}")

    if len(signature_parts) <= 1:
        relation_target = _canonical_anchor_token(unit.get("relation_target", ""))
        relation_task = _canonical_anchor_token(unit.get("relation_task", ""))
        relation_data = _canonical_anchor_token(unit.get("relation_data_modality", ""))
        for prefix, value in [("target", relation_target), ("task", relation_task), ("data", relation_data)]:
            if value and value != mechanism and f"{prefix}:{value}" not in signature_parts:
                signature_parts.append(f"{prefix}:{value}")

    if len(signature_parts) <= 1:
        canonical_name = str(normalized_unit.get("canonical_candidate_name_en", "")).strip()
        if canonical_name:
            signature_parts.append(f"canonical:{canonical_name}")

    # round1 patch: 追加 source_title 强区分锚点，减少异质事件碰撞。
    # 只在锚点未被任何现有 slot 覆盖时才追加，避免与 object/capability/process 重复。
    title_discriminator = _title_topic_discriminator(unit)
    if title_discriminator:
        existing_slot_values = " ".join(
            str(slots.get(key, "") or "")
            for key in (
                "tech_object_slot",
                "capability_slot",
                "process_slot",
                "carrier_slot",
                "application_slot",
            )
        )
        if title_discriminator not in existing_slot_values:
            signature_parts.append(f"title_topic:{title_discriminator}")
    return " || ".join(part for part in signature_parts if part)


def _representative_candidate_score(unit):
    slots = _build_technical_object_slots(unit)
    profile = _scope_shell_profile(unit)
    score = 0
    reasons = []

    object_like_score = int(slots.get("object_like_score", 0) or 0)
    score += object_like_score * 10
    reasons.append(f"object_like={object_like_score}")

    if slots.get("tech_object_slot"):
        score += 12
        reasons.append("has_tech_object_slot")
    else:
        reasons.append("missing_tech_object_slot")
    if slots.get("capability_slot"):
        score += 8
        reasons.append("has_capability_slot")
    if slots.get("process_slot"):
        score += 6
        reasons.append("has_process_slot")
    if slots.get("carrier_slot"):
        score += 4
        reasons.append("has_carrier_slot")
    if slots.get("application_slot") and slots.get("application_slot") not in {"世界模型", "具身智能"}:
        score += 3
        reasons.append("has_specific_application_slot")
    if "-" in str(slots.get("process_slot", "")):
        score += 5
        reasons.append("compound_process_slot")
    if "-" in str(slots.get("carrier_slot", "")):
        score += 4
        reasons.append("compound_carrier_slot")

    topic_granularity = str(unit.get("topic_granularity", "")).strip()
    if topic_granularity == "fine_grained_topic":
        score += 10
        reasons.append("fine_grained_topic")
    elif topic_granularity == "scope_internal_candidate":
        score -= 4
        reasons.append("scope_internal_candidate")

    non_scope_constraint_count = int(unit.get("non_scope_constraint_count", 0) or 0)
    score += min(non_scope_constraint_count, 4) * 3
    reasons.append(f"non_scope_constraints={non_scope_constraint_count}")
    strong_object_count = len(_strong_slot_labels(unit.get("object_modifier_tokens", []), "object_modifier_tokens"))
    strong_data_count = len(_strong_slot_labels(unit.get("data_modifier_tokens", []), "data_modifier_tokens"))
    if strong_object_count >= 2:
        score += 4
        reasons.append("multi_specific_object_tokens")
    if strong_data_count >= 2:
        score += 4
        reasons.append("multi_specific_data_tokens")

    if bool(profile.get("survives_without_scope", False)):
        score += 10
        reasons.append("survives_without_scope")
    else:
        score -= 8
        reasons.append("not_survives_without_scope")

    if bool(profile.get("scope_shell_heavy", False)):
        score -= 12
        reasons.append("scope_shell_heavy")

    display_issue = _display_candidate_name_issue(
        _technical_object_name_from_slots({**unit, **slots}, include_suffix=True)
        or str(unit.get("display_candidate_name", "")).strip(),
        {**unit, **profile},
    )
    if "bare_mechanism" in display_issue:
        score -= 8
        reasons.append("bare_mechanism_penalty")
    if "scope_dependent" in display_issue:
        score -= 4
        reasons.append("scope_dependent_penalty")

    canonical_name = str(unit.get("canonical_candidate_name_en", "")).strip()
    if canonical_name and canonical_name == str(unit.get("mechanism_core", "")).strip():
        score -= 10
        reasons.append("canonical_equals_mechanism")

    return score, "；".join(reasons), slots


def _select_cluster_representative(items):
    scored_items = []
    for idx, item in enumerate(items):
        score, reason, slots = _representative_candidate_score(item)
        scored_items.append(
            {
                "index": idx,
                "item": {**item, **slots},
                "score": score,
                "reason": reason,
                "sort_key": (
                    score,
                    int(bool(slots.get("process_slot"))),
                    int(bool(slots.get("carrier_slot"))),
                    int(bool(slots.get("tech_object_slot"))),
                    int(bool(item.get("survives_without_scope", False))),
                    int(not bool(item.get("scope_shell_heavy", False))),
                    len(str(item.get("canonical_candidate_name_en", "")).strip()),
                    len(str(item.get("raw_phrase", "")).strip()),
                    -idx,
                ),
            }
        )
    scored_items = sorted(scored_items, key=lambda x: x["sort_key"], reverse=True)
    selected = scored_items[0]
    legacy_item = items[0]
    alternatives = []
    for candidate in scored_items[1:4]:
        alt = candidate["item"]
        alt_name = _technical_object_name_from_slots(alt, include_suffix=True) or str(alt.get("canonical_candidate_name_en", "")).strip()
        alternatives.append(f"{alt_name or '未成形'}[{candidate['score']}]")
    selection_summary = "；".join(alternatives)
    return {
        "selected_item": selected["item"],
        "selected_index": selected["index"],
        "selected_score": selected["score"],
        "selected_reason": selected["reason"],
        "legacy_item": legacy_item,
        "legacy_index": 0,
        "legacy_name": _technical_object_name_from_slots(legacy_item, include_suffix=True) or str(legacy_item.get("canonical_candidate_name_en", "")).strip(),
        "selected_name": _technical_object_name_from_slots(selected["item"], include_suffix=True) or str(selected["item"].get("canonical_candidate_name_en", "")).strip(),
        "switched_from_legacy": bool(selected["index"] != 0),
        "selection_competitors_summary": selection_summary,
    }


def _ordered_display_constraints(unit):
    selected = _preferred_display_constraints(unit, max_items=2)
    if len(selected) >= 2:
        has_data = any(key == "data_modifier_tokens" for _, key in selected)
        has_task_or_object = any(key in {"task_constraint_tokens", "object_modifier_tokens"} for _, key in selected)
        if has_data and has_task_or_object:
            data_part = [item for item in selected if item[1] == "data_modifier_tokens"]
            other_part = [item for item in selected if item[1] != "data_modifier_tokens"]
            selected = data_part + other_part
    strong_items = [item for item in selected if not _is_generic_constraint(*item)]
    cleaned = []
    for value, key in selected:
        if value in DISPLAY_SUPPRESS_TOKENS and strong_items:
            continue
        if _is_soft_display_constraint(value, key) and any(not _is_soft_display_constraint(v, k) for v, k in strong_items if v != value):
            continue
        cleaned.append((value, key))
    return cleaned or selected


def _should_insert_scope_label(primary_scope, selected_constraints):
    if not primary_scope or not selected_constraints:
        return False
    if primary_scope == "embodied intelligence" and any(value in {"embodied", "interactive"} for value, _ in selected_constraints):
        return False
    non_method_constraints = [item for item in selected_constraints if item[1] != "method_modifier_tokens"]
    strong_non_method_constraints = [item for item in non_method_constraints if not _is_generic_constraint(*item)]
    if len(strong_non_method_constraints) >= 2:
        return False
    if any(value in SCOPE_DISAMBIGUATION_TOKENS for value, _ in non_method_constraints):
        return True
    if not non_method_constraints:
        return False
    if len(non_method_constraints) == 1:
        return True
    if all(key == "data_modifier_tokens" for _, key in non_method_constraints):
        return True
    return False


def _infer_topic_granularity(unit, display_name):
    if bool(unit.get("is_observation_scope", False)):
        return "generic_or_failed"
    if bool(unit.get("is_scope_echo", False)) or bool(unit.get("generic_core_only", False)):
        return "generic_or_failed"
    if not bool(unit.get("has_mechanism_core", False)) or not bool(unit.get("has_non_scope_constraint", False)):
        return "generic_or_failed"
    candidate_stage = str(unit.get("candidate_stage", "")).strip()
    if candidate_stage not in {"formed_candidate", "formed_candidate_strong"}:
        return "generic_or_failed"

    profile = _scope_shell_profile(unit)
    non_scope_constraint_count = int(profile["non_scope_constraint_count"])
    if bool(profile["survives_without_scope"]) or non_scope_constraint_count >= 3:
        return "fine_grained_topic"
    if bool(profile["scope_shell_heavy"]):
        return "scope_internal_candidate"

    name = str(display_name or "").strip()
    if any(shell in name for shell in ["驱动", "机制", "范式", "体系"]):
        return "scope_internal_candidate"
    return "scope_internal_candidate"


def _infer_topic_granularity_reason(unit, display_name):
    if bool(unit.get("is_observation_scope", False)):
        return "observation_scope"
    if bool(unit.get("is_scope_echo", False)):
        return "scope_echo"
    if bool(unit.get("generic_core_only", False)):
        return "generic_mechanism_repeated"
    if not bool(unit.get("has_mechanism_core", False)):
        return "missing_mechanism_core"
    if not bool(unit.get("has_non_scope_constraint", False)):
        return "missing_task_object_anchor"
    candidate_stage = str(unit.get("candidate_stage", "")).strip()
    if candidate_stage not in {"formed_candidate", "formed_candidate_strong"}:
        return "not_formed_candidate"

    profile = _scope_shell_profile({**unit, "display_candidate_name": display_name})
    if bool(profile["survives_without_scope"]):
        return "survives_without_scope"

    selected_constraints = _ordered_constraints(unit)
    task_object_anchor = [
        (value, key)
        for value, key in selected_constraints
        if key in {"task_constraint_tokens", "object_modifier_tokens"}
        and not _is_generic_constraint(value, key)
        and not _is_scope_shell_constraint(value, key)
    ]
    data_anchor = [
        (value, key)
        for value, key in selected_constraints
        if key == "data_modifier_tokens"
        and not _is_generic_constraint(value, key)
        and not _is_scope_shell_constraint(value, key)
    ]
    mechanism = str(unit.get("mechanism_core", "")).strip()
    repeated = [
        value
        for value, _ in selected_constraints
        if mechanism and normalize_proxy_token(value) == normalize_proxy_token(mechanism)
    ]
    name = str(display_name or "").strip()
    if repeated:
        return "generic_mechanism_repeated"
    if data_anchor and not task_object_anchor:
        return "data_only_constraint"
    if profile.get("scope_shell_heavy") or any(label in name for label in SCOPE_LABELS.values()):
        return "scope_dependent_name"
    return "missing_task_object_anchor"


def _infer_display_tier(topic_granularity, is_observation_scope=False):
    if is_observation_scope:
        return "demo"
    if topic_granularity == "fine_grained_topic":
        return "weak_signal"
    if topic_granularity == "scope_internal_candidate":
        return "hotspot"
    return "manual_review"


def _internal_candidate_label(unit, canonical_name_en, constraint_signature):
    relation_signature = str(unit.get("relation_signature", "")).strip()
    relation_summary = str(unit.get("relation_summary", "")).strip()
    pieces = [f"canonical={canonical_name_en}" if canonical_name_en else "", f"constraints={constraint_signature}" if constraint_signature else ""]
    if relation_signature:
        pieces.append(relation_signature)
    elif relation_summary:
        pieces.append(f"relation={relation_summary}")
    return " || ".join([part for part in pieces if part])


def _naturalized_topic_name(unit, fallback_name):
    profile = _scope_shell_profile(unit)
    slot_name = _technical_object_name_from_slots(unit, include_suffix=True)
    if slot_name:
        return slot_name
    relation_summary = str(unit.get("relation_summary", "")).strip()
    if relation_summary:
        relation_data = _label_for_token(unit.get("relation_data_modality", ""), prefer_zh=True)
        relation_target = _label_for_token(unit.get("relation_target", ""), prefer_zh=True)
        mechanism_label = _label_for_token(unit.get("mechanism_core", ""), prefer_zh=True)
        if relation_target in {"控制", "仿真", "训练", "规划", "推理", "具身", "智能体"}:
            relation_target = ""
        if relation_data in {"控制", "仿真", "训练", "规划", "推理"}:
            relation_data = ""
        if relation_data and mechanism_label and relation_data == mechanism_label:
            relation_data = ""
        if relation_target and mechanism_label and relation_target == mechanism_label:
            relation_target = ""
        if relation_data and relation_target and mechanism_label:
            return f"{relation_data}{relation_target}{mechanism_label}"
        if relation_target and mechanism_label and bool(profile.get("survives_without_scope", False)):
            return f"{relation_target}{mechanism_label}"
    if bool(profile.get("scope_shell_heavy", False)) and not bool(profile.get("survives_without_scope", False)):
        selected_constraints = _ordered_display_constraints(unit)
        task_object_labels = [
            _label_for_token(value, prefer_zh=True)
            for value, key in selected_constraints
            if key in {"task_constraint_tokens", "object_modifier_tokens"}
            and not _is_generic_constraint(value, key)
            and not _is_scope_shell_constraint(value, key)
        ]
        data_labels = [
            _label_for_token(value, prefer_zh=True)
            for value, key in selected_constraints
            if key == "data_modifier_tokens"
            and not _is_generic_constraint(value, key)
            and not _is_scope_shell_constraint(value, key)
        ]
        mechanism_label = _label_for_token(unit.get("mechanism_core", ""), prefer_zh=True)
        labels = _dedupe_preserve_order([label for label in task_object_labels + data_labels if label])
        if labels and mechanism_label:
            return "".join(labels[:2] + [mechanism_label])
        fallback = str(fallback_name or "").strip()
        return fallback
    return str(fallback_name or "").strip()


def _topic_naturalness_reason(unit, topic_name):
    profile = _scope_shell_profile(unit)
    if bool(profile.get("scope_shell_heavy", False)):
        reason = str(profile.get("scope_shell_reason", "")).strip()
        if reason == "only_data_method_anchor":
            return "当前主要依赖数据/方法约束，缺少稳定 task/object 锚点，建议作为待收口线索"
        if reason == "scope_token_in_name":
            return "当前名称仍依赖观察范围词补足语义，建议作为待收口线索"
        if reason == "scope_dependent_after_strip":
            return "去掉观察范围后主题边界不够稳定，建议作为待收口线索"
        return "已形成 scope 内细粒度候选，但当前命名和语义组织仍偏系统压缩标签，建议继续收口"
    if not bool(profile.get("survives_without_scope", False)):
        return "当前已形成较具体候选，但去掉观察范围后主题稳定性一般，建议继续收口或人工复核"
    if int(profile.get("non_scope_constraint_count", 0) or 0) >= 2:
        return "包含明确对象/任务/数据约束，主题较自然"
    return "主题约束较弱，建议人工复核"


def _display_candidate_name(canonical_name_en, unit):
    mechanism = str(unit.get("mechanism_core", "")).strip()
    if not mechanism or canonical_name_en == mechanism:
        return ""
    slot_name = _technical_object_name_from_slots(unit, include_suffix=True)
    if slot_name:
        return _sanitize_display_candidate_name(slot_name, unit)
    selected_constraints = _ordered_display_constraints(unit)
    non_method_constraints = [item for item in selected_constraints if item[1] != "method_modifier_tokens"]
    strong_non_method_constraints = [item for item in non_method_constraints if not _is_generic_constraint(*item)]
    labels = []
    primary_scope = str(unit.get("primary_scope", "")).strip()
    mechanism_label = MECHANISM_LABELS.get(mechanism, _label_for_token(mechanism))

    filtered_constraints = []
    for value, key in selected_constraints:
        if value in DISPLAY_SUPPRESS_TOKENS:
            continue
        if value in DISPLAY_WEAK_SCOPE_TOKENS and (strong_non_method_constraints or any(v not in DISPLAY_WEAK_SCOPE_TOKENS for v, _ in selected_constraints if v != value)):
            continue
        filtered_constraints.append((value, key))
    selected_constraints = filtered_constraints or selected_constraints
    non_method_constraints = [item for item in selected_constraints if item[1] != "method_modifier_tokens"]
    strong_non_method_constraints = [item for item in non_method_constraints if not _is_generic_constraint(*item)]
    profile = _scope_shell_profile(unit)

    if strong_non_method_constraints:
        for value, _ in selected_constraints:
            label = _label_for_token(value)
            if label:
                labels.append(label)
    elif primary_scope == "world model":
        weak_labels = [_label_for_token(value) for value, _ in selected_constraints]
        weak_labels = [label for label in weak_labels if label in WORLD_MODEL_WEAK_LABELS]
        labels.extend(_dedupe_preserve_order(weak_labels[:2]))
    if bool(profile.get("scope_shell_heavy", False)) and not bool(profile.get("survives_without_scope", False)):
        pass
    elif _should_insert_scope_label(primary_scope, selected_constraints):
        if primary_scope == "embodied intelligence" and mechanism == "grounding":
            labels = [label for label in labels if label != SCOPE_LABELS.get(primary_scope, "")]
            return "".join(_dedupe_preserve_order(labels + [mechanism_label]))
        scope_label = SCOPE_LABELS.get(primary_scope, "")
        if scope_label and scope_label not in labels:
            if primary_scope == "world model" and labels:
                labels = labels[:1] + [scope_label]
            else:
                labels.append(scope_label)
    elif not strong_non_method_constraints:
        override_label = METHOD_ONLY_SCOPE_MECHANISM_LABELS.get((primary_scope, mechanism))
        if override_label:
            return override_label
        scope_label = SCOPE_LABELS.get(primary_scope, "")
        if scope_label:
            labels.append(scope_label)
    labels.append(mechanism_label)
    resolved = _sanitize_display_candidate_name("".join(_dedupe_preserve_order(labels)), unit)
    if primary_scope == "world model" and resolved == f"{SCOPE_LABELS.get(primary_scope, '')}{mechanism_label}":
        override_label = METHOD_ONLY_SCOPE_MECHANISM_LABELS.get((primary_scope, mechanism))
        if override_label:
            return override_label
    return resolved


def _patent_friendly_display_name(canonical_name_en, unit):
    primary_scope = str(unit.get("primary_scope", "")).strip()
    mechanism = str(unit.get("mechanism_core", "")).strip()
    mechanism_label = MECHANISM_LABELS.get(mechanism, _label_for_token(mechanism, prefer_zh=True))
    if not mechanism:
        return ""
    slot_name = _technical_object_name_from_slots(unit, include_suffix=True)
    if slot_name:
        return _sanitize_display_candidate_name(slot_name, unit)

    selected_constraints = _ordered_display_constraints(unit)
    strong_constraints = [
        (value, key)
        for value, key in selected_constraints
        if key in PATENT_FRIENDLY_CONSTRAINT_KEYS and value not in DISPLAY_SUPPRESS_TOKENS
    ]
    labels = []
    for value, key in strong_constraints[:2]:
        label = _label_for_token(value, prefer_zh=True)
        if label and label not in labels:
            labels.append(label)

    if not labels:
        return _display_candidate_name(canonical_name_en, unit)

    scope_label = SCOPE_LABELS.get(primary_scope, "")
    if primary_scope == "world model" and scope_label and scope_label not in labels:
        labels = labels[:1] + [scope_label]
    elif primary_scope == "embodied intelligence" and scope_label and scope_label not in labels:
        if not any(label == "具身" for label in labels):
            labels.append(scope_label)

    labels.append(mechanism_label)
    return _sanitize_display_candidate_name("".join(_dedupe_preserve_order(labels)), unit)


def _term_in_text(term, text):
    term_norm = str(term or "").strip().lower()
    text_norm = str(text or "").strip().lower()
    return bool(term_norm and term_norm in text_norm)


def _domain_pack_candidate_trace(unit, domain_lexicon=None):
    if domain_lexicon is None or domain_lexicon.use_legacy_robot_rules:
        return {
            "domain_pack_candidate_rule_ids": "",
            "domain_pack_candidate_reason": "",
            "domain_pack_candidate_status": "",
            "domain_pack_candidate_slot_hits": {},
        }

    text_parts = [
        unit.get("raw_phrase", ""),
        unit.get("raw_candidate_text", ""),
        unit.get("source_title", ""),
        unit.get("source_text", ""),
        unit.get("relation_summary", ""),
        unit.get("relation_target", ""),
        unit.get("relation_task", ""),
        unit.get("relation_data_modality", ""),
        unit.get("relation_method", ""),
    ]
    text = " ".join(str(part or "") for part in text_parts if str(part or "").strip())

    if domain_lexicon._matches_off_domain(text):
        has_in_domain = False
        text_norm = normalize_signal_phrase(text)
        for term in domain_lexicon.domain_specific_terms:
            term_norm = normalize_signal_phrase(term)
            if term_norm and term_norm in text_norm:
                has_in_domain = True
                break
        if not has_in_domain:
            return {
                "domain_pack_candidate_rule_ids": "off_domain_rejection",
                "domain_pack_candidate_reason": "off_domain_leakage",
                "domain_pack_candidate_status": "rejected",
                "domain_pack_candidate_slot_hits": {},
            }

    slot_hits = {
        "technical_object": _dedupe_preserve_order(
            list(unit.get("object_modifier_tokens", []) or [])
            + domain_lexicon.match_terms(domain_lexicon.object_aliases, text)
        ),
        "mechanism": _dedupe_preserve_order(
            list(unit.get("mechanism_core_tokens", []) or [])
            + ([unit.get("mechanism_core")] if unit.get("mechanism_core") else [])
            + domain_lexicon.match_terms(domain_lexicon.mechanism_aliases, text)
        ),
        "performance": _dedupe_preserve_order(
            list(unit.get("task_constraint_tokens", []) or [])
            + domain_lexicon.match_terms(domain_lexicon.task_aliases, text)
        ),
        "data_modality": _dedupe_preserve_order(
            list(unit.get("data_modifier_tokens", []) or [])
            + domain_lexicon.match_terms(domain_lexicon.data_aliases, text)
        ),
        "method": _dedupe_preserve_order(
            list(unit.get("method_modifier_tokens", []) or [])
            + domain_lexicon.match_terms(domain_lexicon.method_aliases, text)
        ),
        "scene": _dedupe_preserve_order(
            list(unit.get("scene_tokens", []) or [])
            + domain_lexicon.match_terms(domain_lexicon.scene_aliases, text)
        ),
    }
    for key, values in list(slot_hits.items()):
        slot_hits[key] = [
            value
            for value in _dedupe_preserve_order(values)
            if value and not domain_lexicon.is_generic_or_shell(value)
        ]
    data_hit_set = set(slot_hits.get("data_modality", []))
    slot_hits["method"] = [value for value in slot_hits.get("method", []) if value not in data_hit_set]

    specific_slots = [key for key, values in slot_hits.items() if values]
    specific_slot_count = len(specific_slots)
    evidence_present = bool(
        str(unit.get("evidence_span", "")).strip()
        or str(unit.get("source_text", "")).strip()
        or str(unit.get("source_title", "")).strip()
    )

    rule_ids = []
    reasons = []
    status = ""
    for pattern in domain_lexicon.invalid_candidate_patterns:
        if not isinstance(pattern, dict):
            continue
        reject_terms = pattern.get("reject_terms", [])
        raw_max_specific = pattern.get("max_specific_slot_count")
        if raw_max_specific is None or str(raw_max_specific).strip() == "":
            max_specific = 0 if reject_terms else 999
        else:
            try:
                max_specific = int(raw_max_specific)
            except (TypeError, ValueError):
                max_specific = 0 if reject_terms else 999
        if any(_term_in_text(term, text) for term in reject_terms or []) and specific_slot_count <= max_specific:
            pattern_id = str(pattern.get("pattern_id", "")).strip()
            if pattern_id:
                rule_ids.append(pattern_id)
            status = "rejected"
            reasons.append(f"invalid_pattern={pattern_id or 'domain_pack_invalid'}")
            reasons.append(f"specific_slot_count={specific_slot_count}")
            break

    min_slots = int(domain_lexicon.minimum_specificity_rule.get("min_non_shell_slots", 0) or 0)
    evidence_required = bool(domain_lexicon.minimum_specificity_rule.get("require_evidence_span", False))
    if not status and evidence_required and not evidence_present:
        status = "rejected"
        reasons.append("evidence=missing")
    if not status and min_slots and specific_slot_count < min_slots:
        status = "rejected"
        reasons.append(f"minimum_specificity={specific_slot_count}/{min_slots}")

    if not status:
        for pattern in domain_lexicon.valid_candidate_patterns:
            if not isinstance(pattern, dict):
                continue
            required_slots = [str(slot or "").strip() for slot in pattern.get("required_slots", []) if str(slot or "").strip()]
            min_required = int(pattern.get("min_required_slot_count", len(required_slots)) or len(required_slots))
            required_hit_count = 0
            for slot in required_slots:
                if slot in {"technical_object", "object"} and slot_hits["technical_object"]:
                    required_hit_count += 1
                elif slot == "mechanism" and slot_hits["mechanism"]:
                    required_hit_count += 1
                elif slot in {"task", "performance"} and slot_hits["performance"]:
                    required_hit_count += 1
                elif slot in {"data_modality", "data"} and slot_hits["data_modality"]:
                    required_hit_count += 1
                elif slot == "method" and slot_hits["method"]:
                    required_hit_count += 1
                elif slot == "scene" and slot_hits["scene"]:
                    required_hit_count += 1
                elif slot == "evidence_span" and evidence_present:
                    required_hit_count += 1
            if required_hit_count >= min_required and (not pattern.get("evidence_required") or evidence_present):
                pattern_id = str(pattern.get("pattern_id", "")).strip()
                if pattern_id:
                    rule_ids.append(pattern_id)
                status = "accepted"
                break

    if (
        not status
        and not domain_lexicon.valid_candidate_patterns
        and (not evidence_required or evidence_present)
        and specific_slot_count >= max(min_slots, 1)
    ):
        status = "accepted"

    for key, values in slot_hits.items():
        if values:
            reasons.append(f"{key}={','.join(values[:3])}")
    if evidence_present:
        reasons.append("evidence=present")

    return {
        "domain_pack_candidate_rule_ids": ";".join(_dedupe_preserve_order(rule_ids)),
        "domain_pack_candidate_reason": "; ".join(_dedupe_preserve_order(reasons)),
        "domain_pack_candidate_status": status,
        "domain_pack_candidate_slot_hits": slot_hits,
    }


def _unit_row(event_id, scope_names, unit, source_type="", domain_lexicon=None):
    mechanism_core_tokens = _dedupe_preserve_order(unit.get("mechanism_core_tokens", []))
    task_constraint_tokens = _dedupe_preserve_order(unit.get("task_constraint_tokens", []))
    object_modifier_tokens = _dedupe_preserve_order(unit.get("object_modifier_tokens", []))
    data_modifier_tokens = _dedupe_preserve_order(unit.get("data_modifier_tokens", []))
    method_modifier_tokens = _dedupe_preserve_order(unit.get("method_modifier_tokens", []))
    scene_tokens = _dedupe_preserve_order(unit.get("scene_tokens", []))
    primary_scope = scope_names[0] if scope_names else ""
    mechanism_core = str(unit.get("mechanism_core", "")).strip() or (mechanism_core_tokens[0] if mechanism_core_tokens else "")
    has_mechanism_core = bool(mechanism_core)
    has_task_constraint = bool(task_constraint_tokens)
    if domain_lexicon is not None and not domain_lexicon.use_legacy_robot_rules:
        has_non_scope_constraint_flag = domain_lexicon.has_non_scope_constraint(
            task_constraint_tokens,
            object_modifier_tokens,
            data_modifier_tokens,
            scene_tokens,
            [mechanism_core],
            method_modifier_tokens,
        )
    else:
        has_non_scope_constraint_flag = has_non_scope_constraint(
            task_constraint_tokens,
            object_modifier_tokens,
            data_modifier_tokens,
            scene_tokens,
            [mechanism_core],
            method_modifier_tokens,
        )
    canonical_name_en = _canonical_candidate_name_en(
        {
            **unit,
            "mechanism_core": mechanism_core,
            "task_constraint_tokens": task_constraint_tokens,
            "object_modifier_tokens": object_modifier_tokens,
            "data_modifier_tokens": data_modifier_tokens,
            "method_modifier_tokens": method_modifier_tokens,
        }
    )
    generic_core_only = bool(
        unit.get("generic_core_only", False)
        or (has_mechanism_core and mechanism_core in BARE_MECHANISM_CORES and not has_non_scope_constraint_flag)
    )
    raw_phrase = str(unit.get("raw_phrase") or unit.get("raw_candidate_text") or "").strip()
    raw_phrase_type = str(unit.get("raw_phrase_type", "")).strip()
    compression_mode = _compression_mode(
        {
            "task_constraint_tokens": task_constraint_tokens,
            "object_modifier_tokens": object_modifier_tokens,
            "data_modifier_tokens": data_modifier_tokens,
            "method_modifier_tokens": method_modifier_tokens,
        }
    )
    constraint_signature = _constraint_signature(
        {
            "task_constraint_tokens": task_constraint_tokens,
            "object_modifier_tokens": object_modifier_tokens,
            "data_modifier_tokens": data_modifier_tokens,
            "method_modifier_tokens": method_modifier_tokens,
        }
    )
    normalized_constraint_signature = _constraint_signature(
        {
            "task_constraint_tokens": task_constraint_tokens,
            "object_modifier_tokens": object_modifier_tokens,
            "data_modifier_tokens": data_modifier_tokens,
            "method_modifier_tokens": method_modifier_tokens,
        },
        normalized=True,
    )
    scope_match_mode = str(unit.get("scope_match_mode", "explicit")).strip() or "explicit"
    source_type = str(source_type or "").strip().lower()
    local_surface_hints = _source_surface_hints(
        str(unit.get("source_title", "")).strip() or raw_phrase,
        _surface_hint_context(
            {
                **unit,
                "raw_phrase": raw_phrase,
            },
            extra_text=raw_phrase,
        ),
        source_type=source_type,
    )
    evidence_present = bool(
        str(unit.get("evidence_span", "")).strip()
        or str(unit.get("source_text", "")).strip()
        or str(unit.get("source_title", "")).strip()
    )
    domain_pack_valid_pattern_ready = bool(
        domain_lexicon is not None
        and not domain_lexicon.use_legacy_robot_rules
        and domain_lexicon.valid_candidate_pattern_matches(
            task_tokens=task_constraint_tokens,
            object_tokens=object_modifier_tokens,
            data_tokens=data_modifier_tokens,
            scene_tokens=scene_tokens,
            mechanism_tokens=mechanism_core_tokens,
            method_tokens=method_modifier_tokens,
            evidence_present=evidence_present,
        )
    )
    if domain_pack_valid_pattern_ready and not canonical_name_en:
        canonical_name_en = raw_phrase

    if bool(unit.get("is_scope_echo", False)):
        stage = "filtered_scope_echo"
        strong_ready_reason = "scope_echo_filtered"
        source_penetration_reason = "filtered_scope_echo"
    elif generic_core_only:
        stage = "unformed_generic_core"
        strong_ready_reason = "generic_core_only"
        source_penetration_reason = "generic_core_only"
    elif (
        (has_mechanism_core or domain_pack_valid_pattern_ready)
        and has_non_scope_constraint_flag
        and canonical_name_en
        and canonical_name_en != mechanism_core
    ):
        stage = "formed_candidate"
        strong_ready_reason = "domain_pack_valid_pattern" if domain_pack_valid_pattern_ready else "needs_cluster_support"
        source_penetration_reason = (
            "domain_pack_pattern_formed"
            if domain_pack_valid_pattern_ready
            else "patent_proxy_formed" if source_type == "patent" and scope_match_mode == "proxy_patent" else "explicit_scope_formed"
        )
    else:
        stage = "filtered_scope_echo"
        strong_ready_reason = "insufficient_structure"
        source_penetration_reason = "insufficient_structure"

    surface_hint_unit = {
        **unit,
        "source_title": str(unit.get("source_title", "")).strip(),
        "source_text": str(unit.get("source_text", "")).strip(),
        "raw_phrase": raw_phrase,
        "object_modifier_tokens": object_modifier_tokens,
        "task_constraint_tokens": task_constraint_tokens,
        "relation_summary": str(unit.get("relation_summary", "")).strip(),
        "relation_target": str(unit.get("relation_target", "")).strip(),
        "relation_task": str(unit.get("relation_task", "")).strip(),
        "relation_data_modality": str(unit.get("relation_data_modality", "")).strip(),
        "relation_method": str(unit.get("relation_method", "")).strip(),
        "canonical_candidate_name_en": canonical_name_en,
    }
    gated_object_surface = _display_surface_hint_from_title(surface_hint_unit)
    gated_object_candidates = [gated_object_surface] if gated_object_surface else []
    granularity_input = {
        **unit,
        "source_title": str(unit.get("source_title", "")).strip(),
        "source_text": str(unit.get("source_text", "")).strip(),
        "object_surface_candidates": gated_object_candidates,
        "preferred_object_surface": gated_object_surface,
        "display_preferred_object_surface": gated_object_surface,
        "preferred_task_surface": local_surface_hints["preferred_task_surface"],
        "display_preferred_task_surface": "",
        "candidate_stage": stage,
        "is_scope_echo": bool(unit.get("is_scope_echo", False)),
        "generic_core_only": generic_core_only,
        "has_mechanism_core": has_mechanism_core,
        "has_non_scope_constraint": has_non_scope_constraint_flag,
        "task_constraint_tokens": task_constraint_tokens,
        "object_modifier_tokens": object_modifier_tokens,
        "data_modifier_tokens": data_modifier_tokens,
        "method_modifier_tokens": method_modifier_tokens,
    }
    granularity_input["display_preferred_task_surface"] = _display_task_hint_from_title(granularity_input)
    if granularity_input["display_preferred_task_surface"]:
        granularity_input["preferred_task_surface"] = granularity_input["display_preferred_task_surface"]
    scope_shell_profile = _scope_shell_profile(granularity_input)
    topic_granularity = _infer_topic_granularity(granularity_input, "")
    topic_granularity_reason = _infer_topic_granularity_reason(granularity_input, "")
    display_tier = _infer_display_tier(topic_granularity)
    internal_label = _internal_candidate_label(granularity_input, canonical_name_en, constraint_signature)
    relation_summary = str(unit.get("relation_summary", "")).strip()
    slot_fields = _build_technical_object_slots(
        {
            **granularity_input,
            "primary_scope": primary_scope,
            "relation_summary": relation_summary,
            "relation_target": str(unit.get("relation_target", "")).strip(),
            "relation_task": str(unit.get("relation_task", "")).strip(),
            "relation_data_modality": str(unit.get("relation_data_modality", "")).strip(),
            "relation_method": str(unit.get("relation_method", "")).strip(),
        }
    )
    domain_pack_trace = _domain_pack_candidate_trace(
        {
            **unit,
            "raw_phrase": raw_phrase,
            "raw_candidate_text": raw_phrase,
            "mechanism_core": mechanism_core,
            "mechanism_core_tokens": mechanism_core_tokens,
            "task_constraint_tokens": task_constraint_tokens,
            "object_modifier_tokens": object_modifier_tokens,
            "data_modifier_tokens": data_modifier_tokens,
            "method_modifier_tokens": method_modifier_tokens,
            "scene_tokens": scene_tokens,
        },
        domain_lexicon=domain_lexicon,
    )

    if domain_pack_trace.get("domain_pack_candidate_status") == "rejected":
        if stage == "formed_candidate":
            stage = "filtered_scope_echo"
            granularity_input["candidate_stage"] = stage

    return {
        "id": event_id,
        "raw_phrase": raw_phrase,
        "raw_candidate_text": raw_phrase,
        "source_title": str(unit.get("source_title", "")).strip(),
        "source_text": str(unit.get("source_text", "")).strip(),
        "raw_phrase_type": raw_phrase_type,
        "source_extraction_mode": str(unit.get("source_extraction_mode", "local")).strip() or "local",
        "mechanism_core": mechanism_core,
        "secondary_mechanism_cores": _dedupe_preserve_order(unit.get("secondary_mechanism_cores", [])),
        "mechanism_core_tokens": mechanism_core_tokens,
        "task_constraint_tokens": task_constraint_tokens,
        "object_modifier_tokens": object_modifier_tokens,
        "data_modifier_tokens": data_modifier_tokens,
        "method_modifier_tokens": method_modifier_tokens,
        "scene_tokens": scene_tokens,
        "normalized_candidate_text": canonical_name_en,
        "canonical_candidate_name_en": canonical_name_en,
        "constraint_signature": constraint_signature,
        "normalized_constraint_signature": normalized_constraint_signature,
        "internal_candidate_label": internal_label,
        "tech_object_slot": slot_fields["tech_object_slot"],
        "capability_slot": slot_fields["capability_slot"],
        "process_slot": slot_fields["process_slot"],
        "carrier_slot": slot_fields["carrier_slot"],
        "application_slot": slot_fields["application_slot"],
        "object_like_score": slot_fields["object_like_score"],
        "technical_objectness_reason": slot_fields["technical_objectness_reason"],
        "cluster_signature_key": "",
        "premerge_signature_key": "",
        "cluster_signature_mode": "",
        "stable_object_key_basis": "",
        "stable_object_id": "",
        "stable_object_label": "",
        "legacy_representative_name": "",
        "representative_candidate_name": "",
        "representative_candidate_score": 0,
        "representative_selection_reason": "",
        "representative_name_source": "",
        "current_representative_name": "",
        "selection_competitors_summary": "",
        "representative_selected_index": 0,
        "representative_switched_from_legacy": False,
        "generic_cluster_risk": "",
        "cluster_object_specificity": "",
        "specific_anchor_strength": 0,
        "specificity_reason": "",
        "can_refine_further": "",
        "refinement_ceiling_reason": "",
        "second_anchor_stability": "",
        "second_anchor_candidates_summary": "",
        "technical_item_stage": "",
        "technical_item_reason": "",
        "technical_subject_anchor": "",
        "technical_process_anchor": "",
        "technical_item_pattern": "",
        "cluster_split_applied": False,
        "source_premerge_group_size": 1,
        "source_premerge_applied": False,
        "source_premerge_original_unit_count": 1,
        "source_premerge_before_unit_count": 1,
        "source_premerge_block_reason": "",
        "source_premerge_before_signatures": "",
        "relation_target": str(unit.get("relation_target", "")).strip(),
        "relation_task": str(unit.get("relation_task", "")).strip(),
        "relation_data_modality": str(unit.get("relation_data_modality", "")).strip(),
        "relation_method": str(unit.get("relation_method", "")).strip(),
        "relation_summary": relation_summary,
        "relation_signature": str(unit.get("relation_signature", "")).strip(),
        "topic_summary_name": "",
        "topic_naturalness_reason": "",
        "compression_mode": compression_mode,
        "display_candidate_name": "",
        "display_candidate_aliases": [canonical_name_en] if canonical_name_en else [],
        "candidate_cluster_id": "",
        "object_surface_candidates": local_surface_hints["object_surface_candidates"],
        "preferred_object_surface": local_surface_hints["preferred_object_surface"],
        "display_preferred_object_surface": granularity_input["display_preferred_object_surface"],
        "preferred_task_surface": local_surface_hints["preferred_task_surface"],
        "display_preferred_task_surface": granularity_input["display_preferred_task_surface"],
        "primary_scope": primary_scope,
        "scope_name": " / ".join(scope_names),
        "scope_names": scope_names,
        "is_observation_scope": False,
        "is_scope_internal_candidate": bool(scope_names),
        "is_scope_echo": bool(unit.get("is_scope_echo", False)),
        "has_mechanism_core": has_mechanism_core,
        "has_task_constraint": has_task_constraint,
        "has_non_scope_constraint": has_non_scope_constraint_flag,
        "generic_core_only": generic_core_only,
        "template_variant_count": 0,
        "alias_count": 1 if canonical_name_en else 0,
        "cluster_evidence_count": 1,
        "cluster_item_count": 1,
        "candidate_stage": stage,
        "topic_granularity": topic_granularity,
        "topic_granularity_reason": topic_granularity_reason,
        "display_tier": display_tier,
        "non_scope_constraint_count": int(scope_shell_profile["non_scope_constraint_count"]),
        "survives_without_scope": bool(scope_shell_profile["survives_without_scope"]),
        "scope_shell_heavy": bool(scope_shell_profile["scope_shell_heavy"]),
        "scope_shell_reason": str(scope_shell_profile["scope_shell_reason"]),
        "display_candidate_name_issue": "",
        "strong_ready_reason": strong_ready_reason,
        "source_penetration_reason": source_penetration_reason,
        "scope_match_mode": scope_match_mode,
        "source_type": source_type,
        # 旁路 bridge 字段：记录桥接推断结果，但不影响主链
        "bridged_object_subtype_candidate": _infer_bridged_object_surface(granularity_input),
        "bridged_object_bridge_reason": _bridge_reason(granularity_input) if _infer_bridged_object_surface(granularity_input) else "",
        "bridged_object_bridge_confidence": _bridge_confidence(granularity_input),
        **domain_pack_trace,
    }


def _safe_event_float(value, default=None):
    try:
        if value is None or pd.isna(value):
            return default
    except Exception:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _event_relevance_text(event_record, context):
    parts = []
    for field in [
        "title", "subject", "action", "technical_object", "mechanism", "task",
        "problem_solved", "capability_change", "evidence_span", "scene",
    ]:
        value = (event_record or {}).get(field, "")
        if isinstance(value, list):
            parts.extend(str(item) for item in value if str(item).strip())
        elif str(value or "").strip():
            parts.append(str(value).strip())
    for field in [
        "technology", "candidate_units", "observation_scopes", "method",
        "data_modality", "weak_signal_reasons",
    ]:
        value = (event_record or {}).get(field, [])
        if isinstance(value, list):
            parts.extend(str(item) for item in value if str(item).strip())
        elif str(value or "").strip():
            parts.append(str(value).strip())
    for field in ["source_title", "source_text"]:
        value = (context or {}).get(field, "")
        if isinstance(value, list):
            parts.extend(str(item) for item in value if str(item).strip())
        elif str(value or "").strip():
            parts.append(str(value).strip())
    return " ".join(parts)


def _event_passes_domain_relevance_gate(event_record, context, domain_lexicon=None):
    if domain_lexicon is None or domain_lexicon.use_legacy_robot_rules:
        return True
    if "tech_relevance_score" not in (event_record or {}):
        return True
    score = _safe_event_float((event_record or {}).get("tech_relevance_score"), default=None)
    if score is None:
        return True

    relevance_text = _event_relevance_text(event_record, context)
    has_domain_anchor = domain_lexicon.has_domain_anchor(relevance_text)
    has_off_domain_anchor = (
        bool(domain_lexicon._matches_off_domain(relevance_text))
        if hasattr(domain_lexicon, "_matches_off_domain")
        else False
    )
    if has_off_domain_anchor and not has_domain_anchor:
        return False
    if score >= 5.0:
        return True
    if has_domain_anchor and score >= 4.0:
        return True
    return False


def build_candidate_forms(events_df: pd.DataFrame, data_df: pd.DataFrame, domain_context=None, domain_pack=None) -> pd.DataFrame:
    domain_lexicon = build_domain_lexicon(domain_context or domain_pack)
    columns = [
        "id", "raw_phrase", "raw_candidate_text", "raw_phrase_type", "source_extraction_mode",
        "mechanism_core", "secondary_mechanism_cores", "mechanism_core_tokens",
        "task_constraint_tokens", "object_modifier_tokens", "data_modifier_tokens",
        "method_modifier_tokens", "scene_tokens", "normalized_candidate_text",
        "canonical_candidate_name_en", "constraint_signature", "normalized_constraint_signature", "internal_candidate_label",
        "tech_object_slot", "capability_slot", "process_slot", "carrier_slot", "application_slot",
        "object_like_score", "technical_objectness_reason",
        "cluster_signature_key", "premerge_signature_key", "cluster_signature_mode", "legacy_representative_name",
        "stable_object_key_basis", "stable_object_id", "stable_object_label",
        "representative_candidate_name", "representative_candidate_score", "representative_selection_reason",
        "representative_name_source", "current_representative_name",
        "selection_competitors_summary", "representative_selected_index", "representative_switched_from_legacy",
        "generic_cluster_risk", "cluster_object_specificity", "specific_anchor_strength", "specificity_reason",
        "can_refine_further", "refinement_ceiling_reason", "second_anchor_stability", "second_anchor_candidates_summary",
        "technical_item_stage", "technical_item_reason", "technical_subject_anchor", "technical_process_anchor", "technical_item_pattern",
        "cluster_split_applied",
        "source_premerge_group_size", "source_premerge_applied", "source_premerge_original_unit_count",
        "source_premerge_before_unit_count", "source_premerge_block_reason", "source_premerge_before_signatures",
        "relation_target", "relation_task", "relation_data_modality", "relation_method",
        "relation_summary", "relation_signature", "topic_summary_name", "topic_naturalness_reason",
        "compression_mode",
        "candidate_cluster_id", "display_candidate_name", "display_candidate_aliases",
        "object_surface_candidates", "preferred_object_surface", "display_preferred_object_surface", "preferred_task_surface", "display_preferred_task_surface",
        "primary_scope", "scope_name", "scope_names", "is_observation_scope",
        "is_scope_internal_candidate", "is_scope_echo", "has_mechanism_core",
        "has_task_constraint", "has_non_scope_constraint", "generic_core_only",
        "template_variant_count", "alias_count", "cluster_evidence_count",
        "cluster_item_count", "candidate_stage", "strong_ready_reason",
        "source_penetration_reason", "scope_match_mode", "source_type",
        "source_types", "source_count", "total_mentions", "org_count", "orgs",
        "mention_ids", "mention_dates", "evidence_titles", "evidence_items",
        "weak_signal_event_count", "weak_signal_event_ratio",
        "low_attention_ratio", "niche_actor_ratio", "non_dominant_ratio",
        "cross_domain_ratio", "traceable_ratio",
        "topic_granularity", "display_tier", "non_scope_constraint_count",
        "survives_without_scope", "scope_shell_heavy", "scope_shell_reason",
        "topic_granularity_reason", "display_candidate_name_issue",
        "domain_pack_candidate_rule_ids", "domain_pack_candidate_reason",
        "domain_pack_candidate_status", "domain_pack_candidate_slot_hits",
        # 旁路 bridge 字段
        "bridged_object_subtype_candidate", "bridged_object_bridge_reason", "bridged_object_bridge_confidence",
    ]
    if events_df is None or events_df.empty or data_df is None or data_df.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    formed_rows = []
    source_type_map = data_df.set_index("id")["source_type"].astype(str).to_dict() if "source_type" in data_df.columns else {}
    source_context_map = {}
    if "id" in data_df.columns:
        for _, record in data_df.drop_duplicates(subset="id", keep="last").iterrows():
            source_context_map[record["id"]] = {
                "source_title": record.get("title", ""),
                "source_text": record.get("text", ""),
                "source_type": record.get("source_type", ""),
                "org": record.get("org", ""),
                "date": record.get("date", ""),
                "url": record.get("url", ""),
                "analysis_tech_field_name": record.get("analysis_tech_field_name", ""),
                "analysis_tech_field_id": record.get("analysis_tech_field_id", ""),
                "analysis_keywords": record.get("analysis_keywords", []),
                "analysis_synonyms": record.get("analysis_synonyms", []),
            }
    event_meta_map = {}
    for _, event in events_df.iterrows():
        event_meta_map[event.get("id")] = event.to_dict()

    candidate_event_records = []
    for _, event in events_df.iterrows():
        event_record = event.to_dict() if hasattr(event, "to_dict") else dict(event or {})
        event_id = event_record.get("id")
        context = source_context_map.get(event_id, {})
        if _event_passes_domain_relevance_gate(event_record, context, domain_lexicon=domain_lexicon):
            candidate_event_records.append(event_record)

    def _metric_source_type(value):
        text = str(value or "").strip()
        mapping = {
            "专利": "patent",
            "文献": "paper",
            "研报": "report",
            "资讯": "news",
            "literature": "paper",
        }
        return mapping.get(text, mapping.get(text.lower(), text.lower() or "unknown"))

    def _event_bool(event_id, key):
        return bool(event_meta_map.get(event_id, {}).get(key, False))

    def _event_score(event_id):
        try:
            return float(event_meta_map.get(event_id, {}).get("weak_signal_event_score", 0.0) or 0.0)
        except Exception:
            return 0.0

    def _context_for_id(event_id):
        return source_context_map.get(event_id, {})

    def _build_metric_payload(items, display_name="", raw_candidate_text=""):
        if not items:
            return {
                "source_types": [],
                "source_count": 0,
                "total_mentions": 0,
                "org_count": 0,
                "orgs": [],
                "mention_ids": [],
                "mention_dates": [],
                "evidence_titles": [],
                "evidence_items": [],
                "weak_signal_event_count": 0,
                "weak_signal_event_ratio": 0.0,
                "low_attention_ratio": 0.0,
                "niche_actor_ratio": 0.0,
                "non_dominant_ratio": 0.0,
                "cross_domain_ratio": 0.0,
                "traceable_ratio": 0.0,
            }

        mention_ids = [item.get("id") for item in items if item.get("id") is not None]
        source_types = []
        orgs = []
        mention_dates = []
        evidence_titles = []
        evidence_items = []
        seen_evidence_ids = set()
        for item in items:
            event_id = item.get("id")
            context = _context_for_id(event_id)
            source_type = _metric_source_type(context.get("source_type") or item.get("source_type"))
            if source_type and source_type not in source_types:
                source_types.append(source_type)
            org = str(context.get("org") or "").strip()
            if org and org not in orgs:
                orgs.append(org)
            date = str(context.get("date") or "").strip()
            if date:
                mention_dates.append(date)
            title = str(context.get("source_title") or "").strip()
            text = str(context.get("source_text") or "").strip()
            if title and title not in evidence_titles:
                evidence_titles.append(title)
            if event_id not in seen_evidence_ids and (title or text):
                seen_evidence_ids.add(event_id)
                evidence_items.append(
                    {
                        "id": str(event_id or ""),
                        "source_type": source_type,
                        "org": org or "Unknown",
                        "date": date,
                        "url": str(context.get("url") or "").strip(),
                        "title": title or display_name,
                        "snippet": text[:300].replace("\n", " "),
                        "text": text[:600],
                        "raw_candidate_text": raw_candidate_text or item.get("raw_candidate_text", ""),
                        "display_candidate_name": display_name,
                    }
                )

        total_mentions = len(items)
        weak_signal_event_count = sum(
            1
            for event_id in mention_ids
            if _event_bool(event_id, "weak_signal_event_candidate") or _event_score(event_id) > 0
        )
        denominator = max(total_mentions, 1)
        return {
            "source_types": source_types,
            "source_count": len(source_types),
            "total_mentions": total_mentions,
            "org_count": len(orgs),
            "orgs": orgs,
            "mention_ids": mention_ids,
            "mention_dates": mention_dates,
            "evidence_titles": evidence_titles[:10],
            "evidence_items": evidence_items[:10],
            "weak_signal_event_count": weak_signal_event_count,
            "weak_signal_event_ratio": round(weak_signal_event_count / denominator, 3),
            "low_attention_ratio": round(sum(1 for event_id in mention_ids if _event_bool(event_id, "low_attention_hint")) / denominator, 3),
            "niche_actor_ratio": round(sum(1 for event_id in mention_ids if _event_bool(event_id, "niche_actor_hint")) / denominator, 3),
            "non_dominant_ratio": round(sum(1 for event_id in mention_ids if _event_bool(event_id, "non_dominant_hint")) / denominator, 3),
            "cross_domain_ratio": round(sum(1 for event_id in mention_ids if _event_bool(event_id, "cross_domain_hint")) / denominator, 3),
            "traceable_ratio": round(sum(1 for event_id in mention_ids if _event_bool(event_id, "traceable_hint")) / denominator, 3),
        }

    scope_items_map = {}
    for event in candidate_event_records:
        event_id = event.get("id")
        source_type = source_type_map.get(event_id, str(event.get("source_type", "")))
        context = source_context_map.get(event_id, {})
        observation_scopes = _normalize_scope_names(event.get("observation_scopes", []))
        if not observation_scopes:
            analysis_scope = _analysis_scope_from_context(context)
            observation_scopes = [analysis_scope] if analysis_scope else []
        for scope in observation_scopes:
            scope_items_map.setdefault(scope, []).append(
                {
                    "id": event_id,
                    "source_type": source_type,
                    "raw_candidate_text": scope,
                }
            )

    for event in candidate_event_records:
        event_id = event.get("id")
        source_type = source_type_map.get(event_id, str(event.get("source_type", "")))
        context = source_context_map.get(event_id, {})
        event_record = dict(event or {})
        observation_scopes = _normalize_scope_names(event_record.get("observation_scopes", []))
        if not observation_scopes:
            analysis_scope = _analysis_scope_from_context(context)
            if analysis_scope:
                observation_scopes = [analysis_scope]
                event_record["observation_scopes"] = observation_scopes
                event_record["scope_match_mode"] = event_record.get("scope_match_mode") or "analysis_field"
                event_record["analysis_tech_field_name"] = analysis_scope
        candidate_units = _candidate_units_for_event(event_record, observation_scopes, domain_lexicon=domain_lexicon)

        for scope in observation_scopes:
            scope_metric_payload = _build_metric_payload(scope_items_map.get(scope, []), display_name=SCOPE_LABELS.get(scope, scope), raw_candidate_text=scope)
            rows.append(
                {
                    "id": event_id,
                    "raw_phrase": scope,
                    "raw_candidate_text": scope,
                    "raw_phrase_type": "scope",
                    "source_extraction_mode": "scope",
                    "mechanism_core": "",
                    "secondary_mechanism_cores": [],
                    "mechanism_core_tokens": [],
                    "task_constraint_tokens": [],
                    "object_modifier_tokens": [],
                    "data_modifier_tokens": [],
                    "method_modifier_tokens": [],
                    "scene_tokens": [],
                    "normalized_candidate_text": scope,
                    "canonical_candidate_name_en": scope,
                    "constraint_signature": "",
                    "normalized_constraint_signature": "",
                    "internal_candidate_label": "",
                    "tech_object_slot": "",
                    "capability_slot": "",
                    "process_slot": "",
                    "carrier_slot": "",
                    "application_slot": "",
                    "object_like_score": 0,
                    "technical_objectness_reason": "观察范围概览项",
                    "cluster_signature_key": f"scope::{scope}",
                    "premerge_signature_key": "",
                    "cluster_signature_mode": "scope_overview",
                    "stable_object_key_basis": f"signature::scope::{scope}",
                    "stable_object_id": _stable_object_id_from_basis(f"signature::scope::{scope}"),
                    "stable_object_label": SCOPE_LABELS.get(scope, scope),
                    "legacy_representative_name": SCOPE_LABELS.get(scope, scope),
                    "representative_candidate_name": SCOPE_LABELS.get(scope, scope),
                    "representative_candidate_score": 0,
                    "representative_selection_reason": "scope_overview",
                    "representative_name_source": "scope_overview",
                    "current_representative_name": SCOPE_LABELS.get(scope, scope),
                    "selection_competitors_summary": "",
                    "representative_selected_index": 0,
                    "representative_switched_from_legacy": False,
                    "generic_cluster_risk": "none",
                    "cluster_object_specificity": "scope_overview",
                    "specific_anchor_strength": 0,
                    "specificity_reason": "scope_overview",
                    "can_refine_further": "no",
                    "refinement_ceiling_reason": "scope_overview",
                    "second_anchor_stability": "none",
                    "second_anchor_candidates_summary": "",
                    "technical_item_stage": "scope_overview",
                    "technical_item_reason": "scope_overview",
                    "technical_subject_anchor": "",
                    "technical_process_anchor": "",
                    "technical_item_pattern": "scope_overview",
                    "cluster_split_applied": False,
                    "source_premerge_group_size": 1,
                    "source_premerge_applied": False,
                    "source_premerge_original_unit_count": 1,
                    "source_premerge_before_unit_count": 1,
                    "source_premerge_block_reason": "",
                    "source_premerge_before_signatures": "",
                    **scope_metric_payload,
                    "relation_target": "",
                    "relation_task": "",
                    "relation_data_modality": "",
                    "relation_method": "",
                    "relation_summary": "",
                    "relation_signature": "",
                    "topic_summary_name": SCOPE_LABELS.get(scope, scope),
                    "topic_naturalness_reason": "观察范围概览项",
                    "compression_mode": "scope_overview",
                    "candidate_cluster_id": f"scope::{scope}",
                    "display_candidate_name": SCOPE_LABELS.get(scope, scope),
                    "display_candidate_aliases": [scope],
                    "object_surface_candidates": [],
                    "preferred_object_surface": "",
                    "display_preferred_object_surface": "",
                    "preferred_task_surface": "",
                    "display_preferred_task_surface": "",
                    "primary_scope": scope,
                    "scope_name": scope,
                    "scope_names": [scope],
                    "is_observation_scope": True,
                    "is_scope_internal_candidate": False,
                    "is_scope_echo": False,
                    "has_mechanism_core": False,
                    "has_task_constraint": False,
                    "has_non_scope_constraint": False,
                    "generic_core_only": False,
                    "template_variant_count": 0,
                    "alias_count": 1,
                    "cluster_evidence_count": 1,
                    "cluster_item_count": 1,
                    "candidate_stage": "scope_overview",
                    "topic_granularity": "generic_or_failed",
                    "topic_granularity_reason": "observation_scope",
                    "display_tier": "demo",
                    "non_scope_constraint_count": 0,
                    "survives_without_scope": False,
                    "scope_shell_heavy": False,
                    "scope_shell_reason": "",
                    "display_candidate_name_issue": "",
                    "strong_ready_reason": "scope_overview",
                    "source_penetration_reason": "scope_overview",
                    "scope_match_mode": str(event_record.get("scope_match_mode", "explicit")).strip() or "explicit",
                    "source_type": source_type,
                    "bridged_object_subtype_candidate": "",
                    "bridged_object_bridge_reason": "",
                    "bridged_object_bridge_confidence": 0.0,
                }
            )

        for unit in candidate_units:
            scope_names = _normalize_scope_names(unit.get("scope_names", observation_scopes))
            row = _unit_row(
                event_id,
                scope_names,
                {
                    **unit,
                    "evidence_span": event_record.get("evidence_span", ""),
                    **source_context_map.get(event_id, {}),
                },
                source_type=source_type,
                domain_lexicon=domain_lexicon,
            )
            if row["candidate_stage"] == "formed_candidate":
                formed_rows.append(row)
            else:
                display_name = (
                    str(row.get("display_candidate_name", "")).strip()
                    or str(row.get("topic_summary_name", "")).strip()
                    or str(row.get("stable_object_label", "")).strip()
                    or str(row.get("normalized_candidate_text", "")).strip()
                    or str(row.get("raw_candidate_text", "")).strip()
                )
                row.update(
                    _build_metric_payload(
                        [row],
                        display_name=display_name,
                        raw_candidate_text=str(row.get("raw_candidate_text", "")).strip(),
                    )
                )
                rows.append(row)

    cluster_groups = {}
    premerged_formed_rows = _premerge_source_units(formed_rows)
    for row in premerged_formed_rows:
        signature = _theme_group_signature(row)
        cluster_groups.setdefault(signature, []).append(row)
    cluster_groups = _split_generic_cluster_groups(cluster_groups)
    # M2/A4: post-hoc TF-IDF-based heterogeneity split (only-split, never-merge)
    cluster_groups = _tfidf_split_heterogeneous_clusters(cluster_groups)

    for signature, items in cluster_groups.items():
        cluster_profile = _cluster_specificity_profile(items)
        cluster_id = f"cand::{hashlib.md5(signature.encode('utf-8')).hexdigest()[:10]}"
        cluster_evidence_count = len({item["id"] for item in items})
        alias_values = _dedupe_preserve_order(item["canonical_candidate_name_en"] for item in items if item["canonical_candidate_name_en"])
        raw_variants = _dedupe_preserve_order(item["raw_phrase"] for item in items if item["raw_phrase"])
        patent_in_cluster = any(str(item.get("source_type", "")).strip().lower() == "patent" for item in items)
        selection_bundle = _select_cluster_representative(items)
        cluster_surface_hints = _merge_cluster_surface_hints(items)
        representative = {
            **selection_bundle["selected_item"],
            **cluster_surface_hints,
            "source_type": "patent" if patent_in_cluster else str(selection_bundle["selected_item"].get("source_type", "")).strip().lower(),
        }
        stable_object_key_basis = _stable_object_key_basis({**representative, "cluster_signature_key": signature})
        stable_object_id = _stable_object_id_from_basis(stable_object_key_basis)
        canonical_name_en = representative.get("canonical_candidate_name_en", "")
        if patent_in_cluster:
            display_candidate_name = _patent_friendly_display_name(canonical_name_en, representative)
        else:
            display_candidate_name = _display_candidate_name(canonical_name_en, representative)
        topic_summary_name = _naturalized_topic_name(representative, display_candidate_name)
        topic_naturalness_reason = _topic_naturalness_reason(representative, topic_summary_name)
        stable_object_label = _stable_object_label({**representative, "display_candidate_name": topic_summary_name or display_candidate_name})
        topic_granularity = _infer_topic_granularity(
            {
                **representative,
                "candidate_stage": "formed_candidate",
                "has_mechanism_core": bool(representative.get("has_mechanism_core", False)),
                "has_non_scope_constraint": bool(representative.get("has_non_scope_constraint", False)),
                "is_scope_echo": bool(representative.get("is_scope_echo", False)),
                "generic_core_only": bool(representative.get("generic_core_only", False)),
            },
            display_candidate_name,
        )
        topic_granularity_reason = _infer_topic_granularity_reason(
            {
                **representative,
                "candidate_stage": "formed_candidate",
                "has_mechanism_core": bool(representative.get("has_mechanism_core", False)),
                "has_non_scope_constraint": bool(representative.get("has_non_scope_constraint", False)),
                "is_scope_echo": bool(representative.get("is_scope_echo", False)),
                "generic_core_only": bool(representative.get("generic_core_only", False)),
            },
            display_candidate_name,
        )
        if cluster_profile["generic_cluster_risk"] == "high":
            topic_granularity = "scope_internal_candidate"
            topic_granularity_reason = "generic_cluster_high_risk"
        display_tier = "manual_review" if cluster_profile["generic_cluster_risk"] == "high" else _infer_display_tier(topic_granularity)
        cluster_scope_shell_profile = _scope_shell_profile(representative)
        item_profile = _technical_item_profile(
            {
                **representative,
                **cluster_profile,
            }
        )
        is_shell_heavy = bool(cluster_scope_shell_profile.get("scope_shell_heavy") or not cluster_scope_shell_profile.get("survives_without_scope"))
        is_strong = bool(
            cluster_evidence_count >= 2
            and display_candidate_name
            and canonical_name_en
            and canonical_name_en != representative["mechanism_core"]
            and _strong_constraint_ready(representative)
            and topic_granularity == "fine_grained_topic"
            and bool(cluster_scope_shell_profile["survives_without_scope"])
            and cluster_profile["generic_cluster_risk"] != "high"
            and not is_shell_heavy
        )
        stage = "formed_candidate_strong" if is_strong else "formed_candidate"
        strong_reason = "stable_constraint_signature" if is_strong else "need_more_evidence"
        cluster_metric_payload = _build_metric_payload(
            items,
            display_name=topic_summary_name or display_candidate_name or canonical_name_en,
            raw_candidate_text=representative.get("raw_candidate_text", ""),
        )
        for item in items:
            resolved_display_name = topic_summary_name or display_candidate_name or canonical_name_en
            generic_method_only_name = _is_generic_method_only_display_name(
                resolved_display_name,
                {**representative, **item},
                domain_lexicon=domain_lexicon,
            )
            display_candidate_name_issue = _display_candidate_name_issue(resolved_display_name, {**item, **cluster_scope_shell_profile})
            if generic_method_only_name:
                display_candidate_name_issue = "；".join(
                    _dedupe_preserve_order(
                        [display_candidate_name_issue, "generic_method_only_without_domain_anchor"]
                    )
                )
            output_stage = "filtered_scope_echo" if generic_method_only_name else stage
            output_topic_granularity = "generic_or_failed" if generic_method_only_name else topic_granularity
            output_display_tier = "manual_review" if generic_method_only_name else (
                "hotspot" if is_shell_heavy and display_tier == "weak_signal"
                else display_tier
            )
            rows.append(
                {
                    **item,
                    "candidate_cluster_id": cluster_id,
                    "display_candidate_name": resolved_display_name,
                    "topic_summary_name": topic_summary_name or resolved_display_name,
                    "topic_naturalness_reason": topic_naturalness_reason,
                    "object_surface_candidates": representative.get("object_surface_candidates", item.get("object_surface_candidates", [])),
                    "preferred_object_surface": representative.get("preferred_object_surface", item.get("preferred_object_surface", "")),
                    "display_preferred_object_surface": representative.get("display_preferred_object_surface", item.get("display_preferred_object_surface", "")),
                    "preferred_task_surface": representative.get("preferred_task_surface", item.get("preferred_task_surface", "")),
                    "display_preferred_task_surface": representative.get("display_preferred_task_surface", item.get("display_preferred_task_surface", "")),
                    "tech_object_slot": representative.get("tech_object_slot", item.get("tech_object_slot", "")),
                    "capability_slot": representative.get("capability_slot", item.get("capability_slot", "")),
                    "process_slot": representative.get("process_slot", item.get("process_slot", "")),
                    "carrier_slot": representative.get("carrier_slot", item.get("carrier_slot", "")),
                    "application_slot": representative.get("application_slot", item.get("application_slot", "")),
                    "object_like_score": representative.get("object_like_score", item.get("object_like_score", 0)),
                    "technical_objectness_reason": representative.get("technical_objectness_reason", item.get("technical_objectness_reason", "")),
                    "cluster_signature_key": signature,
                    "cluster_signature_mode": (
                        "object_slot_key"
                        if representative.get("tech_object_slot") or representative.get("capability_slot")
                        else "anchor_fallback_key"
                    ),
                    "stable_object_key_basis": stable_object_key_basis,
                    "stable_object_id": stable_object_id,
                    "stable_object_label": stable_object_label,
                    "legacy_representative_name": selection_bundle["legacy_name"],
                    "representative_candidate_name": selection_bundle["selected_name"],
                    "representative_candidate_score": selection_bundle["selected_score"],
                    "representative_selection_reason": selection_bundle["selected_reason"],
                    "representative_name_source": "topic_summary_name" if topic_summary_name else "display_candidate_name",
                    "current_representative_name": resolved_display_name,
                    "selection_competitors_summary": selection_bundle["selection_competitors_summary"],
                    "representative_selected_index": selection_bundle["selected_index"],
                    "representative_switched_from_legacy": selection_bundle["switched_from_legacy"],
                    "generic_cluster_risk": cluster_profile["generic_cluster_risk"],
                    "cluster_object_specificity": cluster_profile["cluster_object_specificity"],
                    "specific_anchor_strength": cluster_profile["specific_anchor_strength"],
                    "specificity_reason": cluster_profile["specificity_reason"],
                    "can_refine_further": cluster_profile["can_refine_further"],
                    "refinement_ceiling_reason": cluster_profile["refinement_ceiling_reason"],
                    "second_anchor_stability": cluster_profile["second_anchor_stability"],
                    "second_anchor_candidates_summary": cluster_profile["second_anchor_candidates_summary"],
                    "technical_item_stage": item_profile["technical_item_stage"],
                    "technical_item_reason": item_profile["technical_item_reason"],
                    "technical_subject_anchor": item_profile["technical_subject_anchor"],
                    "technical_process_anchor": item_profile["technical_process_anchor"],
                    "technical_item_pattern": item_profile["technical_item_pattern"],
                    "cluster_split_applied": bool(" || split:" in signature),
        "display_candidate_aliases": alias_values or [canonical_name_en],
                    "alias_count": len(alias_values or [canonical_name_en]),
                    "template_variant_count": max(len(raw_variants) - len(alias_values or [canonical_name_en]), 0),
                    "cluster_evidence_count": cluster_evidence_count,
                    "cluster_item_count": len(items),
                    "candidate_stage": output_stage,
                    "topic_granularity": output_topic_granularity,
                    "topic_granularity_reason": topic_granularity_reason,
                    "display_tier": output_display_tier,
                    "non_scope_constraint_count": int(cluster_scope_shell_profile["non_scope_constraint_count"]),
                    "survives_without_scope": bool(cluster_scope_shell_profile["survives_without_scope"]),
                    "scope_shell_heavy": bool(cluster_scope_shell_profile["scope_shell_heavy"]),
                    "scope_shell_reason": str(cluster_scope_shell_profile["scope_shell_reason"]),
                    "display_candidate_name_issue": display_candidate_name_issue,
                    "strong_ready_reason": strong_reason,
                    **cluster_metric_payload,
                    "source_penetration_reason": (
                        "patent_proxy_clustered"
                        if item.get("source_type") == "patent" and item.get("scope_match_mode") == "proxy_patent"
                        else item.get("source_penetration_reason", "")
                    ),
                }
            )

    forms_df = pd.DataFrame(rows, columns=columns)
    if forms_df.empty:
        return forms_df
    return forms_df.reset_index(drop=True)
