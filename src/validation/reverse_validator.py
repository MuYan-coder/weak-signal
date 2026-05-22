from __future__ import annotations

import json
import re
from collections import defaultdict

import pandas as pd


REVERSE_VALIDATION_COLUMNS = [
    "display_candidate_name",
    "research_surface_name",
    "scope_name",
    "signal_bucket",
    "topic_granularity",
    "mechanism_core",
    "constraint_signature",
    "scope_shell_heavy",
    "survives_without_scope",
    "internal_candidate_label",
    "canonical_candidate_name_en",
    "candidate_cluster_id",
    "source_count",
    "total_mentions",
    "news_count",
    "paper_count",
    "patent_count",
    "raw_candidate_texts",
    "raw_phrases",
    "relation_phrases",
    "source_extraction_modes",
    "evidence_titles",
    "evidence_items",
    "source_types",
    "mention_ids",
    "mention_dates",
    "orgs",
    "reverse_validation_status",
    "source_semantic_consistency",
    "object_semantic_alignment",
    "alignment_confidence",
    "alignment_reason",
    "release_alignment_pass",
    "release_alignment_risk",
    "consistency_reason_breakdown",
    "consistency_reason_note",
    "needs_surface_rewrite",
    "should_go_to_failure_case_section",
    "reverse_validation_note",
]


def _has_nonempty_cluster_ids(df: pd.DataFrame | None) -> bool:
    if df is None or df.empty or "candidate_cluster_id" not in df.columns:
        return False
    return bool(df["candidate_cluster_id"].fillna("").astype(str).str.strip().ne("").any())


ALIGNMENT_BROAD_TITLE_HINTS = {
    "overview",
    "综述",
    "genealogy",
    "族谱",
    "attribution",
    "analysis",
    "benchmark",
    "framework",
    "architecture",
    "bias",
    "survey",
}


ALIGNMENT_CONFLICT_HINTS = {
    "机械臂机器人": {"underwater", "水下", "uuv", "auv", "推进", "propulsion", "vector propulsion"},
}


STRUCTURED_ANCHOR_ALIAS_MAP = {
    "control": {"control", "controller", "controlled", "控制"},
    "training": {"train", "training", "trained", "训练", "learning", "learn"},
    "planning": {"plan", "planning", "planner", "规划"},
    "simulation": {"simulation", "simulate", "simulator", "仿真"},
    "robot": {"robot", "robotic", "robotics", "机器人"},
    "manipulator": {"manipulator", "manipulators", "机械臂", "机械手", "robot arm"},
    "gripper": {"gripper", "夹爪", "夹取器"},
    "dexterous": {"dexterous", "灵巧手", "灵巧"},
    "grasp": {"grasp", "grasping", "抓取", "抓握"},
    "assembly": {"assembly", "装配", "装联"},
    "sensor": {"sensor", "sensing", "传感"},
    "visual": {"visual", "vision", "视觉"},
    "video": {"video", "视频"},
    "3d": {"3d", "three-dimensional", "三维"},
    "trajectory": {"trajectory", "trajectories", "轨迹"},
    "temporal": {"temporal", "time-series", "sequence", "时序"},
    "navigation": {"navigation", "navigate", "导航"},
}


OBJECT_ANCHOR_STOP = {
    "control", "training", "planning", "simulation", "evaluation", "benchmark", "policy",
    "video", "visual", "sensor", "trajectory", "temporal", "3d", "interactive", "embodied",
}


TASK_ANCHOR_STOP = {
    "benchmark", "evaluation", "interactive", "embodied", "robotics", "dynamic", "sparse",
    "causal", "agent",
}


DATA_ANCHOR_STOP = {
    "benchmark", "evaluation", "interactive", "embodied", "control",
}


GENERIC_SHARED_ANCHORS = {"robot", "control", "training", "planning", "simulation", "video", "visual"}


def _safe_list(value):
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    return [value]


def _safe_items(value):
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def _json_dump(value):
    return json.dumps(value, ensure_ascii=False)


def _dedupe_preserve(items):
    seen = set()
    ordered = []
    for item in items:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered


def _most_common_nonempty(items):
    counts = defaultdict(int)
    for item in items:
        text = str(item).strip()
        if not text:
            continue
        counts[text] += 1
    if not counts:
        return ""
    return max(counts.items(), key=lambda kv: (kv[1], len(kv[0])))[0]


def _tokenize(text):
    text = str(text or "").lower()
    if not text:
        return set()
    ascii_tokens = re.findall(r"[a-z0-9]+", text)
    zh_tokens = re.findall(r"[\u4e00-\u9fff]{2,}", text)
    return set(ascii_tokens + zh_tokens)


def _jaccard(left, right):
    if not left or not right:
        return 0.0
    return len(left & right) / max(len(left | right), 1)


def _safe_token_list(value):
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return []
    try:
        loaded = json.loads(text)
    except Exception:
        loaded = None
    if isinstance(loaded, list):
        return [str(item).strip() for item in loaded if str(item).strip()]
    return [item.strip() for item in re.findall(r"[A-Za-z0-9\u4e00-\u9fff\-\+]+", text) if item.strip()]


def _parse_constraint_signature(signature):
    buckets = defaultdict(list)
    for part in str(signature or "").split("||"):
        text = str(part).strip()
        if ":" not in text:
            continue
        key, raw_values = text.split(":", 1)
        bucket = str(key).strip().lower()
        values = [str(item).strip().lower() for item in raw_values.split("|") if str(item).strip()]
        buckets[bucket].extend(values)
    return {key: _dedupe_preserve(values) for key, values in buckets.items()}


def _anchor_stopwords(channel):
    if channel == "object":
        return OBJECT_ANCHOR_STOP
    if channel == "task":
        return TASK_ANCHOR_STOP
    if channel == "data":
        return DATA_ANCHOR_STOP
    return set()


def _normalize_anchor_token(token):
    text = str(token or "").strip().lower()
    if not text:
        return ""
    text = re.sub(r"[\s_]+", " ", text)
    return text.strip()


def _ranked_anchor_tokens(tokens, channel, top_k=4):
    counts = defaultdict(int)
    stopwords = _anchor_stopwords(channel)
    for token in tokens:
        normalized = _normalize_anchor_token(token)
        if not normalized or normalized in stopwords:
            continue
        counts[normalized] += 1
    if not counts:
        return []
    ranked = sorted(counts.items(), key=lambda kv: (kv[1], len(kv[0])), reverse=True)
    max_count = max(counts.values())
    min_support = 1 if max_count == 1 else 2
    return [token for token, count in ranked if count >= min_support][:top_k]


def _structured_anchor_aliases(token, channel):
    normalized = _normalize_anchor_token(token)
    aliases = set()
    if not normalized:
        return aliases
    aliases.add(normalized)
    aliases.update(_tokenize(normalized))
    aliases.update(STRUCTURED_ANCHOR_ALIAS_MAP.get(normalized, set()))
    if channel == "task" and normalized == "control":
        aliases.update({"manipulation", "操控"})
    if channel == "object" and normalized == "robot":
        aliases.update({"platform", "agent"})
    return {str(alias).strip().lower() for alias in aliases if str(alias).strip()}


def _visual_control_anchor_families(text):
    lowered = str(text or "").lower()
    families = set()
    if any(token in lowered for token in ["visual", "vision", "视觉", "机器视觉", "rgb", "rgb-d", "camera", "visuo"]):
        families.add("visual")
    if any(token in lowered for token in ["control", "控制", "controller"]):
        families.add("control")
    if any(token in lowered for token in ["manipulation", "manipulate", "操控", "操作", "grasp", "抓取", "夹取", "mechanical hand", "机械手"]):
        families.add("manipulation")
    if any(token in lowered for token in ["visuomotor", "视觉运动"]):
        families.add("visuomotor")
    if any(token in lowered for token in ["world model", "世界模型", "vla", "vision-language-action"]):
        families.add("upper_surface")
    if any(token in lowered for token in ["welding", "焊接", "inspection", "检测", "巡检", "navigation", "导航", "warehouse", "仓储", "grid", "电网", "spray", "喷涂"]):
        families.add("industrial_noise")
    return families


def _is_visual_control_target(row):
    text = " ".join(
        [
            str(row.get("display_candidate_name", "")).strip(),
            str(row.get("topic_summary_name", "")).strip(),
            str(row.get("final_research_object_name", "")).strip(),
        ]
    ).lower()
    has_visual = any(token in text for token in ["visual", "vision", "视觉", "机器视觉", "visuo"])
    has_control = any(token in text for token in ["control", "控制", "manipulation", "操控", "抓取", "visuomotor"])
    return has_visual and has_control


def _visual_control_shared_anchor_gate(row, forms, avg_score, constraint_diversity):
    # Visual-control-only patch: validate shared narrow anchors before treating all
    # cross-source surface mismatch as fully low consistency.
    if not _is_visual_control_target(row) or avg_score >= 0.12 or constraint_diversity > 3:
        return False

    source_anchor_families = defaultdict(set)
    for form in forms:
        source_type = str(form.get("source_type", "")).strip() or "unknown"
        anchor_text = " ".join(
            _safe_token_list(form.get("task_constraint_tokens", []))
            + _safe_token_list(form.get("object_modifier_tokens", []))
            + _safe_token_list(form.get("data_modifier_tokens", []))
            + _safe_token_list(form.get("scene_tokens", []))
            + [
                str(form.get("display_candidate_name", "")).strip(),
                str(form.get("raw_phrase", "")).strip(),
                str(form.get("relation_summary", "")).strip(),
            ]
        )
        source_anchor_families[source_type].update(_visual_control_anchor_families(anchor_text))

    valid_sources = [families for families in source_anchor_families.values() if families]
    if len(valid_sources) < 2:
        return False

    visual_like_sources = sum(1 for families in valid_sources if "visual" in families)
    control_like_sources = sum(1 for families in valid_sources if families & {"control", "manipulation", "visuomotor"})
    noisy_sources = sum(1 for families in valid_sources if "industrial_noise" in families)

    if visual_like_sources < 2 or control_like_sources < 2:
        return False
    if noisy_sources >= 2:
        return False
    return True


def _evidence_text_by_source(row):
    grouped = defaultdict(list)
    for item in _safe_items(row.get("evidence_items", [])):
        source_type = str(item.get("source_type", "")).strip() or "unknown"
        grouped[source_type].append(
            " ".join(
                part
                for part in [
                    str(item.get("title", "")).strip(),
                    str(item.get("raw_candidate_text", "")).strip(),
                    str(item.get("snippet", "")).strip(),
                ]
                if part
            )
        )
    return grouped


def _cross_source_scores(grouped):
    groups = {key: _tokenize(" ".join(values)) for key, values in grouped.items()}
    keys = list(groups.keys())
    scores = []
    for idx, left_key in enumerate(keys):
        for right_key in keys[idx + 1:]:
            score = _jaccard(groups[left_key], groups[right_key])
            scores.append((left_key, right_key, score, groups[left_key], groups[right_key]))
    return scores


def _evidence_support_text_by_source(row):
    grouped = defaultdict(list)
    for item in _safe_items(row.get("evidence_items", [])):
        source_type = str(item.get("source_type", "")).strip() or "unknown"
        grouped[source_type].append(
            " ".join(
                part
                for part in [
                    str(item.get("title", "")).strip(),
                    str(item.get("snippet", "")).strip(),
                    str(item.get("text", "")).strip(),
                ]
                if part
            )
        )
    return {key: " ".join(values).lower() for key, values in grouped.items()}


def _structured_anchor_inventory(row, forms):
    mechanism_tokens = []
    object_tokens = _safe_token_list(row.get("object_modifier_tokens", []))
    task_tokens = _safe_token_list(row.get("task_constraint_tokens", []))
    data_tokens = _safe_token_list(row.get("data_modifier_tokens", []))

    row_constraint = _parse_constraint_signature(row.get("constraint_signature", ""))
    object_tokens.extend(row_constraint.get("object", []))
    task_tokens.extend(row_constraint.get("task", []))
    data_tokens.extend(row_constraint.get("data", []))

    mechanism = str(row.get("mechanism_core", "")).strip()
    if mechanism:
        mechanism_tokens.append(mechanism)

    for form in forms:
        mechanism = str(form.get("mechanism_core", "")).strip()
        if mechanism:
            mechanism_tokens.append(mechanism)
        object_tokens.extend(_safe_token_list(form.get("object_modifier_tokens", [])))
        task_tokens.extend(_safe_token_list(form.get("task_constraint_tokens", [])))
        data_tokens.extend(_safe_token_list(form.get("data_modifier_tokens", [])))
        parsed = _parse_constraint_signature(form.get("constraint_signature", ""))
        object_tokens.extend(parsed.get("object", []))
        task_tokens.extend(parsed.get("task", []))
        data_tokens.extend(parsed.get("data", []))

    return {
        "mechanism": _ranked_anchor_tokens(mechanism_tokens, "mechanism", top_k=2),
        "object": _ranked_anchor_tokens(object_tokens, "object", top_k=4),
        "task": _ranked_anchor_tokens(task_tokens, "task", top_k=4),
        "data": _ranked_anchor_tokens(data_tokens, "data", top_k=4),
    }


def _structured_source_semantic_features(row, forms, avg_score):
    source_texts = _evidence_support_text_by_source(row)
    if len(source_texts) < 2:
        return {
            "source_texts": source_texts,
            "anchors": {},
            "support_by_source": {},
            "shared_by_dimension": {},
            "shared_dimension_count": 0,
            "narrow_shared_count": 0,
            "score": 0.0,
        }

    anchors = _structured_anchor_inventory(row, forms)
    support_by_source = {}
    shared_by_dimension = {}

    for source_type, source_text in source_texts.items():
        dimension_hits = {}
        for dimension, tokens in anchors.items():
            matches = []
            for token in tokens:
                aliases = _structured_anchor_aliases(token, dimension)
                if aliases and _text_has_any(source_text, aliases):
                    matches.append(token)
            dimension_hits[dimension] = matches
        support_by_source[source_type] = dimension_hits

    for dimension in ["mechanism", "object", "task", "data"]:
        token_support = defaultdict(set)
        for source_type, hits in support_by_source.items():
            for token in hits.get(dimension, []):
                token_support[token].add(source_type)
        shared_by_dimension[dimension] = {
            token: sorted(sources)
            for token, sources in token_support.items()
            if len(sources) >= 2
        }

    shared_dimension_count = sum(1 for tokens in shared_by_dimension.values() if tokens)
    narrow_shared_count = sum(
        1
        for dimension in ["object", "task", "data"]
        for token in shared_by_dimension.get(dimension, {})
        if token not in GENERIC_SHARED_ANCHORS
    )

    score = 0.0
    if shared_by_dimension.get("mechanism"):
        score += 0.35
    if shared_by_dimension.get("object"):
        score += 0.25
    if shared_by_dimension.get("task"):
        score += 0.20
    if shared_by_dimension.get("data"):
        score += 0.15
    if narrow_shared_count >= 1:
        score += 0.10
    if avg_score >= 0.08:
        score += 0.05

    return {
        "source_texts": source_texts,
        "anchors": anchors,
        "support_by_source": support_by_source,
        "shared_by_dimension": shared_by_dimension,
        "shared_dimension_count": shared_dimension_count,
        "narrow_shared_count": narrow_shared_count,
        "score": round(min(score, 1.0), 3),
    }


def _structured_consistency_reason(features, label):
    parts = []
    for dimension in ["mechanism", "object", "task", "data"]:
        shared = list(features.get("shared_by_dimension", {}).get(dimension, {}).keys())
        if shared:
            parts.append(f"{dimension}={','.join(shared[:2])}")
    anchor_summary = "；".join(parts) if parts else "无稳定共享锚点"
    if label == "high":
        return (
            "high_due_to_structured_shared_anchors",
            f"跨源证据共同支持 {anchor_summary}，结构化锚点已形成较强一致性；Jaccard 仅作为辅助参考。",
        )
    return (
        "medium_due_to_structured_anchor_overlap",
        f"跨源证据共同支持 {anchor_summary}，虽然表面词项仍有差异，但结构化锚点已足够支撑中等一致性。",
    )


def _semantic_consistency_decision(row, forms):
    grouped = _evidence_text_by_source(row)
    constraint_diversity = len(
        {
            str(item.get("constraint_signature", "")).strip()
            for item in forms
            if str(item.get("constraint_signature", "")).strip()
        }
    )
    if len(grouped) < 2:
        if constraint_diversity <= 1:
            return {
                "label": "medium",
                "reason_breakdown": "medium_due_to_single_stable_source",
                "reason_note": "当前仅有单源或近似单源证据，但约束签名稳定，暂保留为中等一致性。",
            }
        return {
            "label": "low",
            "reason_breakdown": "low_due_to_sparse_cross_source_overlap",
            "reason_note": "仅有单源或近似单源证据，当前不足以做稳定跨源一致性判断。",
        }

    scores = _cross_source_scores(grouped)
    avg_score = sum(item[2] for item in scores) / max(len(scores), 1)
    structured = _structured_source_semantic_features(row, forms, avg_score)

    if (
        structured["score"] >= 0.85
        and structured["shared_dimension_count"] >= 4
        and structured["shared_by_dimension"].get("mechanism")
        and structured["narrow_shared_count"] >= 1
        and (avg_score >= 0.02 or structured["narrow_shared_count"] >= 2)
    ):
        reason_breakdown, reason_note = _structured_consistency_reason(structured, "high")
        return {"label": "high", "reason_breakdown": reason_breakdown, "reason_note": reason_note}

    if (
        structured["score"] >= 0.45
        and structured["shared_dimension_count"] >= 2
        and structured["shared_by_dimension"].get("mechanism")
        and (
            structured["shared_by_dimension"].get("object")
            or structured["shared_by_dimension"].get("task")
            or structured["shared_by_dimension"].get("data")
        )
        and (
            structured["shared_dimension_count"] >= 3
            or structured["narrow_shared_count"] >= 1
            or avg_score >= 0.08
        )
    ):
        reason_breakdown, reason_note = _structured_consistency_reason(structured, "medium")
        return {"label": "medium", "reason_breakdown": reason_breakdown, "reason_note": reason_note}

    if avg_score >= 0.28 and constraint_diversity <= 2:
        return {
            "label": "high",
            "reason_breakdown": "high_due_to_surface_overlap",
            "reason_note": "跨源表面词项重叠已较高，且约束签名较集中，可直接判为高一致性。",
        }
    if avg_score >= 0.12 and constraint_diversity <= 3:
        return {
            "label": "medium",
            "reason_breakdown": "medium_due_to_surface_overlap",
            "reason_note": "跨源表面词项已出现可解释重叠，结合约束集中度可判为中等一致性。",
        }
    if _visual_control_shared_anchor_gate(row, forms, avg_score, constraint_diversity):
        return {
            "label": "medium",
            "reason_breakdown": "medium_due_to_visual_control_anchor_gate",
            "reason_note": "视觉控制类对象虽然表面词差异较大，但跨源都保留了 visual/control 窄锚点。",
        }

    return {
        "label": "low",
        "reason_breakdown": "",
        "reason_note": "",
    }


def _consistency_reason_breakdown(row, forms):
    decision = _semantic_consistency_decision(row, forms)
    if decision["reason_breakdown"]:
        return decision["reason_breakdown"], decision["reason_note"]

    grouped = _evidence_text_by_source(row)
    source_count = len(grouped)
    if source_count < 2:
        return "low_due_to_sparse_cross_source_overlap", "仅有单源或近似单源证据，当前不足以做稳定跨源一致性判断。"

    constraint_diversity = len(
        {
            str(item.get("constraint_signature", "")).strip()
            for item in forms
            if str(item.get("constraint_signature", "")).strip()
        }
    )
    scores = _cross_source_scores(grouped)
    avg_score = sum(item[2] for item in scores) / max(len(scores), 1)
    all_text = {key: " ".join(values).lower() for key, values in grouped.items()}
    patent_text = all_text.get("patent", "")
    news_text = all_text.get("news", "")
    paper_text = all_text.get("paper", "")

    shared_task_anchors = {
        "control",
        "控制",
        "manipulation",
        "操控",
        "grasp",
        "抓取",
        "training",
        "训练",
        "policy",
        "策略",
        "planning",
        "规划",
        "trajectory",
        "轨迹",
    }
    paper_tokens = _tokenize(paper_text)
    news_tokens = _tokenize(news_text)
    patent_tokens = _tokenize(patent_text)
    shared_task_anchor_count = len((paper_tokens | news_tokens) & patent_tokens & shared_task_anchors)

    patent_noise_terms = {
        "welding", "焊接", "inspection", "检测", "巡检", "navigation", "导航",
        "warehouse", "仓储", "grid", "电网", "spray", "喷涂", "fault", "缺陷",
    }
    news_upper_terms = {
        "embodied", "具身", "world", "model", "世界模型", "vla",
        "vision", "language", "action", "foundation", "agent", "智能体",
    }
    world_model_surface_terms = {
        "world", "model", "世界模型", "vla", "vision", "language", "action",
        "video", "视频",
    }

    has_patent = bool(patent_text)
    has_news = bool(news_text)
    patent_noise_hits = sum(1 for term in patent_noise_terms if term in patent_text)
    news_upper_hits = sum(1 for term in news_upper_terms if term in news_text)
    world_model_hits = sum(1 for term in world_model_surface_terms if term in (paper_text + " " + news_text))

    if has_patent and patent_noise_hits >= 2 and shared_task_anchor_count == 0:
        return "low_due_to_patent_anchor_drift", "专利侧锚点更像工业焊接/巡检/导航等异质任务，未与 paper/news 共享更窄任务锚点。"
    if has_news and news_upper_hits >= 3 and avg_score < 0.12:
        return "low_due_to_news_upper_level_pull", "新闻侧主要停留在 embodied/VLA/world-model 上层叙事，把对象往上层主题拉走。"
    if world_model_hits >= 4 and avg_score < 0.12:
        return "low_due_to_world_model_surface_pull", "跨源表面共享 world-model/VLA/video 等上层词，但缺少更窄任务对象锚点。"
    if constraint_diversity >= 2 and shared_task_anchor_count == 0:
        return "low_due_to_missing_shared_task_anchor", "不同来源都提到了相关对象，但没有稳定共享的 task/control/manipulation/training 锚点。"
    if avg_score < 0.08:
        return "low_due_to_sparse_cross_source_overlap", "跨源重叠过稀，当前更像局部相关证据拼接，而不是同一对象的稳定跨源支撑。"
    if avg_score < 0.12 and shared_task_anchor_count >= 1:
        return "medium_anchor_shared_surface_far", "已出现共享窄锚点，但表面词项仍偏远，当前更适合作为可解释的中间态。"
    return "", ""


def _anchor_aliases(anchor, anchor_type):
    text = str(anchor or "").strip().lower()
    aliases = set()
    if not text:
        return aliases

    if anchor_type == "subject":
        if "机械臂机器人" in text:
            aliases.update({"manipulator", "manipulators", "mechanical arm", "robotic hand", "anthropomorphic finger", "dexterous", "gripper", "机械臂", "机械手", "灵巧手", "夹爪", "手指", "手"})
        elif "导航机器人" in text:
            aliases.update({"navigation", "navigate", "robot navigation", "path planning", "路径规划", "导航", "定位", "无碰撞路径"})
        elif "感知多模态" in text:
            aliases.update({"multimodal", "multi-modal", "多模态", "perception", "感知", "vision-language", "visual", "audio"})
        elif "感知机器人" in text:
            aliases.update({"perception", "感知", "sensor", "视觉", "camera"})
        else:
            aliases.update(_tokenize(text))
        return aliases

    if anchor_type == "process":
        if "三维" in text:
            aliases.update({"3d", "three-dimensional", "三维", "bim", "laser scanner", "point cloud"})
        elif "传感" in text:
            aliases.update({"sensor", "sensing", "传感", "camera", "image sensor", "monocular", "视觉"})
        elif "多模态" in text:
            aliases.update({"multimodal", "multi-modal", "多模态", "vision-language", "audio", "visual"})
        elif "时序" in text:
            aliases.update({"temporal", "time-series", "时序", "sequence"})
        elif "轨迹" in text:
            aliases.update({"trajectory", "轨迹"})
        else:
            aliases.update(_tokenize(text))
        return aliases

    if anchor_type == "capability":
        if "训练" in text:
            aliases.update({"train", "training", "trained", "训练", "learn", "learning", "policy"})
        elif "规划" in text:
            aliases.update({"plan", "planning", "planner", "规划", "path"})
        elif "控制" in text or "操控" in text:
            aliases.update({"control", "controller", "控制", "manipulation", "操控", "grasp", "抓取"})
        elif "推理" in text:
            aliases.update({"reason", "reasoning", "推理", "inference"})
        elif "仿真" in text:
            aliases.update({"simulation", "simulate", "仿真"})
        else:
            aliases.update(_tokenize(text))
        return aliases

    return _tokenize(text)


def _alignment_profile(row, forms):
    subject_anchor = _most_common_nonempty(form.get("technical_subject_anchor", "") for form in forms)
    process_anchor = _most_common_nonempty(form.get("technical_process_anchor", "") for form in forms)
    capability_anchor = _most_common_nonempty(form.get("capability_slot", "") for form in forms)
    item_pattern = _most_common_nonempty(form.get("technical_item_pattern", "") for form in forms)
    return {
        "subject_anchor": subject_anchor,
        "process_anchor": process_anchor,
        "capability_anchor": capability_anchor,
        "item_pattern": item_pattern,
        "subject_aliases": _anchor_aliases(subject_anchor, "subject"),
        "process_aliases": _anchor_aliases(process_anchor, "process"),
        "capability_aliases": _anchor_aliases(capability_anchor or row.get("mechanism_core", ""), "capability"),
    }


def _text_has_any(text, aliases):
    lowered = str(text or "").lower()
    if not lowered or not aliases:
        return False
    return any(alias in lowered for alias in aliases)


def _evidence_alignment_label(item, profile):
    title = str(item.get("title", "")).strip()
    snippet = str(item.get("snippet", "")).strip()
    text = str(item.get("text", "")).strip()
    source_text = " ".join(part for part in [title, snippet, text] if part).lower()
    title_text = title.lower()

    subject_hit = _text_has_any(source_text, profile["subject_aliases"])
    process_hit = (not profile["process_anchor"]) or _text_has_any(source_text, profile["process_aliases"])
    capability_hit = _text_has_any(source_text, profile["capability_aliases"])
    broad_title = any(term in title_text for term in ALIGNMENT_BROAD_TITLE_HINTS)

    subject_conflicts = ALIGNMENT_CONFLICT_HINTS.get(profile["subject_anchor"], set())
    conflict_hit = _text_has_any(source_text, subject_conflicts)

    if conflict_hit and not subject_hit:
        return "topic_close_object_mismatch", "证据包含明显冲突对象锚点，且未出现目标技术主语。"
    if subject_hit and capability_hit and process_hit and not broad_title:
        return "same_object_strict", "证据同时支持技术主语、能力项与过程/载体锚点。"
    if subject_hit and capability_hit:
        return "upper_lower_related", "证据支持主语与能力项，但过程/载体不足或标题仍偏综述/方法说明。"
    if (capability_hit and process_hit) or subject_hit:
        return "upper_lower_related", "证据与目标对象存在上下位或邻近关系，但不足以判为严格同一对象。"
    if _text_has_any(source_text, profile["process_aliases"] | profile["capability_aliases"]):
        return "topic_close_object_mismatch", "证据只共享过程/能力词，缺少稳定技术主语，更像主题接近。"
    return "evidence_insufficient", "证据中缺少足够的对象级锚点。"


def _object_semantic_alignment(row, forms):
    profile = _alignment_profile(row, forms)
    items = _safe_items(row.get("evidence_items", []))
    if not items:
        return {
            "object_semantic_alignment": "evidence_insufficient",
            "alignment_confidence": "low",
            "alignment_reason": "缺少可用 evidence_items，无法做对象级严格对齐。",
            "release_alignment_pass": False,
            "release_alignment_risk": "high",
        }

    labels = []
    reason_lines = []
    for item in items:
        label, reason = _evidence_alignment_label(item, profile)
        labels.append(label)
        reason_lines.append(f"{item.get('source_type', 'unknown')}:{label}:{reason}")

    strict_count = sum(label == "same_object_strict" for label in labels)
    related_count = sum(label == "upper_lower_related" for label in labels)
    mismatch_count = sum(label == "topic_close_object_mismatch" for label in labels)
    insufficient_count = sum(label == "evidence_insufficient" for label in labels)

    if strict_count >= 2 and mismatch_count == 0:
        return {
            "object_semantic_alignment": "same_object_strict",
            "alignment_confidence": "high",
            "alignment_reason": "至少两条证据同时支持技术主语、能力项与过程/载体锚点。",
            "release_alignment_pass": True,
            "release_alignment_risk": "low",
        }
    if strict_count >= 1 and mismatch_count == 0 and related_count >= 1:
        return {
            "object_semantic_alignment": "upper_lower_related",
            "alignment_confidence": "medium",
            "alignment_reason": "证据之间存在对象相关性，但至少一部分仍停留在上下位或邻近主题，不足以判为严格同一对象。",
            "release_alignment_pass": False,
            "release_alignment_risk": "medium",
        }
    if mismatch_count >= 1:
        return {
            "object_semantic_alignment": "topic_close_object_mismatch",
            "alignment_confidence": "medium" if strict_count else "low",
            "alignment_reason": "部分证据共享主题词，但出现对象锚点漂移或异质对象混入。",
            "release_alignment_pass": False,
            "release_alignment_risk": "high",
        }
    if insufficient_count >= max(1, len(items) - 1):
        return {
            "object_semantic_alignment": "evidence_insufficient",
            "alignment_confidence": "low",
            "alignment_reason": "证据大多缺少技术主语与能力项的共同支撑，当前无法证明是严格同一对象。",
            "release_alignment_pass": False,
            "release_alignment_risk": "high",
        }
    return {
        "object_semantic_alignment": "upper_lower_related",
        "alignment_confidence": "low",
        "alignment_reason": "证据与目标对象存在一定相关性，但严格对象级对齐仍不足。",
        "release_alignment_pass": False,
        "release_alignment_risk": "medium",
    }


def _source_semantic_consistency(row, forms):
    return _semantic_consistency_decision(row, forms)["label"]


def _natural_phrase_in_source(row, forms):
    llm_judgment = str(row.get("llm_small_topic_judgment", "")).strip()
    llm_pattern = str(row.get("llm_small_topic_pattern", "")).strip()
    if llm_judgment == "small_topic" and llm_pattern in {"scene+mechanism", "data+mechanism", "application+mechanism", "problem+mechanism"}:
        return True

    mechanism = str(row.get("mechanism_core", "")).strip().lower()
    for form in forms:
        phrase = str(form.get("raw_phrase", "")).strip()
        canonical = str(form.get("canonical_candidate_name_en", "")).strip()
        relation_summary = str(form.get("relation_summary", "")).strip()
        token_count = len(re.findall(r"[A-Za-z0-9\u4e00-\u9fff]+", phrase))
        task_tokens = [str(item).strip() for item in _safe_list(form.get("task_constraint_tokens", [])) if str(item).strip()]
        object_tokens = [str(item).strip() for item in _safe_list(form.get("object_modifier_tokens", [])) if str(item).strip()]
        data_tokens = [str(item).strip() for item in _safe_list(form.get("data_modifier_tokens", [])) if str(item).strip()]
        scene_tokens = [str(item).strip() for item in _safe_list(form.get("scene_tokens", [])) if str(item).strip()]
        unique_anchor_values = {item for item in task_tokens + object_tokens + data_tokens + scene_tokens if item}
        has_data_or_scene_anchor = bool(data_tokens or scene_tokens)
        has_multi_anchor_values = len(unique_anchor_values) >= 2
        if relation_summary and has_data_or_scene_anchor and has_multi_anchor_values:
            return True
        if phrase and token_count >= 4 and has_data_or_scene_anchor and has_multi_anchor_values:
            lowered = phrase.lower()
            if mechanism and mechanism in lowered:
                return True
        if canonical and canonical != mechanism and has_data_or_scene_anchor and has_multi_anchor_values:
            return True
    return False


def _reverse_validation_status(row, forms):
    llm_judgment = str(row.get("llm_small_topic_judgment", "")).strip()
    consistency = _source_semantic_consistency(row, forms)
    natural_in_source = _natural_phrase_in_source(row, forms)
    if natural_in_source:
        return "natural_topic_in_source"
    if llm_judgment in {"compressed_label", "upper_topic"} and bool(row.get("scope_shell_heavy", False)) and int(row.get("non_scope_constraint_count", 0) or 0) <= 1:
        return "only_upper_topic_in_source"
    if consistency == "low":
        return "mixed_or_unclear"
    if llm_judgment == "small_topic":
        return "natural_topic_in_source"
    if llm_judgment in {"compressed_label", "unclear"}:
        return "only_upper_topic_in_source"
    return "mixed_or_unclear"


def _reverse_validation_note(row, forms, status, consistency):
    llm_judgment = str(row.get("llm_small_topic_judgment", "")).strip()
    patent_count = int(row.get("patent_count", 0) or 0)
    raw_phrase_count = len(_dedupe_preserve(str(item.get("raw_phrase", "")).strip() for item in forms))
    if status == "natural_topic_in_source":
        if patent_count > 0:
            return "原始证据中已能找到较自然的小主题表达，且已出现专利侧支撑；当前更像组织/命名层需要继续收口。"
        return "原始证据中已能找到较自然的小主题表达；当前对象主要问题在命名自然化，而不是原始数据过粗。"
    if status == "only_upper_topic_in_source":
        if llm_judgment == "compressed_label":
            return "原始文本大多只停留在 scope 内机制表达，当前对象更像系统压缩标签，保留在 hotspot/manual_review 更稳妥。"
        return "原始文本仍以上层主题表达为主，当前对象尚不足以直接作为最终弱信号结论。"
    if consistency == "low":
        return f"不同证据之间语义一致性偏低，且聚合中包含 {raw_phrase_count} 类原始短语，存在混题风险，适合进入失败案例分析。"
    return "原始证据中同时存在细粒度线索与上层表达，当前对象可保留为 scope 内细候选，但需继续人工核验。"


def build_reverse_validation_table(scored_df, candidate_forms_df, manual_review_df=None, top_k=15):
    if scored_df is None or scored_df.empty or candidate_forms_df is None or candidate_forms_df.empty:
        return pd.DataFrame(columns=REVERSE_VALIDATION_COLUMNS)

    candidate_forms_df = candidate_forms_df.copy()
    cluster_map = defaultdict(list)
    for _, row in candidate_forms_df.iterrows():
        cluster_key = str(row.get("candidate_cluster_id", "")).strip()
        if cluster_key:
            cluster_map[cluster_key].append(row.to_dict())

    subset = scored_df[scored_df["signal_type"].isin(["weak_signal", "hotspot"])].copy()
    if manual_review_df is not None and not manual_review_df.empty:
        manual_review_subset = pd.DataFrame()
        if _has_nonempty_cluster_ids(manual_review_df) and _has_nonempty_cluster_ids(scored_df):
            review_cluster_ids = {
                str(item).strip()
                for item in manual_review_df.get("candidate_cluster_id", [])
                if str(item).strip()
            }
            if review_cluster_ids:
                manual_review_subset = scored_df[
                    scored_df["candidate_cluster_id"].fillna("").astype(str).isin(review_cluster_ids)
                ].copy()
        if manual_review_subset.empty:
            review_names = {
                str(item).strip()
                for item in manual_review_df.get("display_candidate_name", [])
                if str(item).strip()
            }
            if review_names:
                manual_review_subset = scored_df[
                    scored_df["display_candidate_name"].astype(str).isin(review_names)
                ].copy()
        subset = pd.concat([subset, manual_review_subset], ignore_index=True)
    if subset.empty:
        return pd.DataFrame(columns=REVERSE_VALIDATION_COLUMNS)

    dedupe_key = ["candidate_cluster_id"] if _has_nonempty_cluster_ids(subset) else ["display_candidate_name"]
    subset = subset.sort_values(
        by=["weak_signal_score", "hotspot_score", "source_count", "cluster_evidence_count", "total_mentions"],
        ascending=[False, False, False, False, False],
    ).drop_duplicates(subset=dedupe_key, keep="first").head(top_k).copy()

    rows = []
    for _, row in subset.iterrows():
        cluster_key = str(row.get("candidate_cluster_id", "")).strip()
        forms = cluster_map.get(cluster_key, [])
        raw_candidate_texts = _dedupe_preserve(str(item.get("raw_candidate_text", "")).strip() for item in forms)
        raw_phrases = _dedupe_preserve(str(item.get("raw_phrase", "")).strip() for item in forms)
        relation_phrases = _dedupe_preserve(str(item.get("relation_summary", "")).strip() for item in forms)
        source_extraction_modes = _dedupe_preserve(str(item.get("source_extraction_mode", "")).strip() for item in forms)

        status = _reverse_validation_status(row, forms)
        consistency = _source_semantic_consistency(row, forms)
        alignment = _object_semantic_alignment(row, forms)
        consistency_reason_breakdown, consistency_reason_note = _consistency_reason_breakdown(row, forms)
        needs_surface_rewrite = status != "natural_topic_in_source" or str(row.get("llm_small_topic_judgment", "")).strip() in {"upper_topic", "compressed_label", "unclear"}
        failure_case = bool(row.get("should_go_to_failure_case_section", False)) or status in {"only_upper_topic_in_source", "mixed_or_unclear"}
        note = _reverse_validation_note(row, forms, status, consistency)

        evidence_items = _safe_items(row.get("evidence_items", []))
        evidence_titles = _dedupe_preserve(item.get("title", "") for item in evidence_items)
        orgs = _dedupe_preserve(item.get("org", "") for item in evidence_items)

        rows.append(
            {
                "display_candidate_name": str(row.get("display_candidate_name", "")).strip(),
                "research_surface_name": str(row.get("topic_summary_name", "")).strip() or str(row.get("display_candidate_name", "")).strip(),
                "scope_name": str(row.get("scope_name", "")).strip(),
                "signal_bucket": str(row.get("signal_type", "")).strip(),
                "topic_granularity": str(row.get("topic_granularity", "")).strip(),
                "mechanism_core": str(row.get("mechanism_core", "")).strip(),
                "constraint_signature": str(row.get("constraint_signature", "")).strip(),
                "scope_shell_heavy": bool(row.get("scope_shell_heavy", False)),
                "survives_without_scope": bool(row.get("survives_without_scope", False)),
                "internal_candidate_label": str(row.get("internal_candidate_label", "")).strip(),
                "canonical_candidate_name_en": str(row.get("canonical_candidate_name_en", "")).strip(),
                "candidate_cluster_id": cluster_key,
                "source_count": int(row.get("source_count", 0) or 0),
                "total_mentions": int(row.get("total_mentions", 0) or 0),
                "news_count": int(row.get("news_count", 0) or 0),
                "paper_count": int(row.get("paper_count", 0) or 0),
                "patent_count": int(row.get("patent_count", 0) or 0),
                "raw_candidate_texts": _json_dump(raw_candidate_texts),
                "raw_phrases": _json_dump(raw_phrases),
                "relation_phrases": _json_dump(relation_phrases),
                "source_extraction_modes": _json_dump(source_extraction_modes),
                "evidence_titles": _json_dump(evidence_titles),
                "evidence_items": _json_dump(evidence_items),
                "source_types": _json_dump(_safe_list(row.get("source_types", []))),
                "mention_ids": _json_dump(_safe_list(row.get("mention_ids", []))),
                "mention_dates": _json_dump(_safe_list(row.get("mention_dates", []))),
                "orgs": _json_dump(orgs),
                "reverse_validation_status": status,
                "source_semantic_consistency": consistency,
                "object_semantic_alignment": alignment["object_semantic_alignment"],
                "alignment_confidence": alignment["alignment_confidence"],
                "alignment_reason": alignment["alignment_reason"],
                "release_alignment_pass": bool(alignment["release_alignment_pass"]),
                "release_alignment_risk": alignment["release_alignment_risk"],
                "consistency_reason_breakdown": consistency_reason_breakdown,
                "consistency_reason_note": consistency_reason_note,
                "needs_surface_rewrite": bool(needs_surface_rewrite),
                "should_go_to_failure_case_section": bool(failure_case),
                "reverse_validation_note": note,
            }
        )

    return pd.DataFrame(rows, columns=REVERSE_VALIDATION_COLUMNS)


def merge_reverse_validation_into_manual_review(manual_review_df, reverse_validation_df):
    if manual_review_df is None or manual_review_df.empty:
        return manual_review_df
    if reverse_validation_df is None or reverse_validation_df.empty:
        return manual_review_df
    merged = manual_review_df.copy()
    merge_key = "candidate_cluster_id" if _has_nonempty_cluster_ids(manual_review_df) and _has_nonempty_cluster_ids(reverse_validation_df) else "display_candidate_name"
    rv = reverse_validation_df[
        [
            "candidate_cluster_id",
            "display_candidate_name",
            "research_surface_name",
            "reverse_validation_status",
            "source_semantic_consistency",
            "object_semantic_alignment",
            "alignment_confidence",
            "alignment_reason",
            "release_alignment_pass",
            "release_alignment_risk",
            "consistency_reason_breakdown",
            "consistency_reason_note",
            "needs_surface_rewrite",
            "should_go_to_failure_case_section",
            "reverse_validation_note",
        ]
    ].drop_duplicates(subset=[merge_key], keep="first")
    merged = merged.merge(rv, on=merge_key, how="left", suffixes=("", "_rv"))
    return merged