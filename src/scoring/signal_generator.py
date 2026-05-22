from collections import defaultdict

import pandas as pd

from ..utils.semantic_utils import semantic_similarity
from ..validation.object_family_canonicalizer import get_canonicalizer, match_family


def _canonical_group_token(value):
    token = str(value or "").strip().lower()
    alias_map = {
        "robotics": "robot",
        "robots": "robot",
        "agents": "agent",
        "videos": "video",
        "visuals": "visual",
        "sensors": "sensor",
        "trajectories": "trajectory",
    }
    return alias_map.get(token, token)


def _first_token(values, preferred=None):
    normalized = [_canonical_group_token(value) for value in _normalize_aliases(values)]
    normalized = [value for value in normalized if value]
    if not normalized:
        return ""
    preferred = preferred or []
    for token in preferred:
        if token in normalized:
            return token
    return normalized[0]


def _object_family_candidate_key(concept, canonicalizer):
    """
    对象族感知的聚合键生成。

    优先级：
    1. 若短语可命中 object family registry → 按 family_id 聚合
    2. 特殊处理：simulation (of_006) 按 mechanism_core 细分
    3. 若无法命中 → 退回通用 _coarse_family_key

    Returns:
        (cluster_key, family_info)
    """
    # 尝试匹配对象族
    raw_phrase = str(concept.get("raw_phrase", "")).strip()
    display_name = str(concept.get("display_candidate_name", "")).strip()
    canonical_name_en = str(concept.get("canonical_candidate_name_en", "")).strip()
    mechanism_core = str(concept.get("mechanism_core", "")).strip().lower()

    # 依次尝试匹配，直到找到匹配
    family = None
    for term in [canonical_name_en, raw_phrase, display_name]:
        if term:
            family = match_family(term)
            if family:
                break

    if family:
        family_id = family.get('family_id', '')
        canonical_term = family.get('canonical_term', '')

        # 特殊处理：simulation (of_006) 按 mechanism_core 细分
        # 拆分规则：
        # - mechanism_core == "simulation" → of_027 (embodied simulation)
        # - mechanism_core == "training" → of_028 (robot training)
        # - mechanism_core == "planning" → of_029 (robot planning)
        if family_id == 'of_006':
            if mechanism_core == 'simulation':
                # 具身智能仿真
                return "family:of_027:embodied simulation", {
                    'family_id': 'of_027',
                    'canonical_term': 'embodied simulation',
                    'term_type': 'method',
                    'cross_source_pattern': 'news_paper',
                    'patent_role': 'optional',
                    'priority': 'high',
                }
            elif mechanism_core == 'training':
                # 机器人训练
                return "family:of_028:robot training", {
                    'family_id': 'of_028',
                    'canonical_term': 'robot training',
                    'term_type': 'method',
                    'cross_source_pattern': 'news_paper',
                    'patent_role': 'optional',
                    'priority': 'medium',
                }
            elif mechanism_core == 'planning':
                # 机器人规划
                return "family:of_029:robot planning", {
                    'family_id': 'of_029',
                    'canonical_term': 'robot planning',
                    'term_type': 'task',
                    'cross_source_pattern': 'news_paper',
                    'patent_role': 'optional',
                    'priority': 'medium',
                }
            # 其他 mechanism_core 保留原 family

        return f"family:{family_id}:{canonical_term}", family

    # 退回通用键
    return _coarse_family_key(concept, canonicalizer), None


def _coarse_family_key(concept, canonicalizer=None):
    """
    候选聚合阶段的粗粒度家族键。

    优化策略 (v4 - 对象族感知聚合):
    1. 优先使用对象族感知聚合（由 _object_family_candidate_key 调用）
    2. 当 mechanism_core 存在时：scope + mechanism + anchors
    3. 当 mechanism_core 缺失时：使用 fallback 策略
       - 优先使用归一化后的 raw_phrase_core
       - 使用 task + object + data anchors 组合
       - 避免生成 scope + nan 这类空键
    4. 目标：提升跨源聚合率，减少无效空键
    """
    if canonicalizer is None:
        canonicalizer = get_canonicalizer()

    primary_scope = str(concept.get("primary_scope", "")).strip()
    mechanism = str(concept.get("mechanism_core", "")).strip()

    # 获取 task anchor（优先选择具体的任务类型）
    task_anchor = _first_token(
        concept.get("task_constraint_tokens", []),
        preferred=["driving", "navigation", "manipulation", "grasping", "robot", "agent", "training", "planning", "control"],
    )

    # 获取 object anchor
    object_anchor = _first_token(
        concept.get("object_modifier_tokens", []),
        preferred=["robot", "agent", "environment", "perception", "memory", "policy", "model"],
    )

    # 获取 data anchor
    data_anchor = _first_token(
        concept.get("data_modifier_tokens", []),
        preferred=["video", "sensor", "trajectory", "visual", "3d", "multimodal", "simulation"],
    )

    # 获取 scene anchor (新增)
    scene_anchor = _first_token(
        concept.get("scene_tokens", []),
        preferred=["indoor", "outdoor", "driving", "navigation", "manipulation"],
    )

    # 获取 raw_phrase 的核心词作为 fallback，并使用 canonicalizer 归一化
    raw_phrase = str(concept.get("raw_phrase", "")).strip()
    raw_phrase_core = ""
    canonical_term = ""
    if raw_phrase:
        # 首先尝试从整个短语中提取归一化术语
        canonical_term, norm_type, _ = canonicalizer.extract_canonical_from_phrase(raw_phrase)
        if norm_type != 'none':
            raw_phrase_core = canonical_term.lower()
        else:
            # 如果没有找到，提取 raw_phrase 中的核心词
            raw_tokens = raw_phrase.lower().split()
            # 过滤掉常见停用词和 scope 词
            stop_words = {"the", "a", "an", "for", "with", "based", "using", "via", "through"}
            scope_words = {"embodied", "intelligence", "world", "model"}
            filtered = [t for t in raw_tokens if t not in stop_words and t not in scope_words and len(t) >= 3]
            if filtered:
                raw_phrase_core = filtered[0]

    # 主要区分维度：task 或 object（优先 task）
    primary_anchor = task_anchor or object_anchor
    secondary_anchor = data_anchor or scene_anchor

    # 策略1: mechanism_core 存在时，使用原有逻辑
    if mechanism and mechanism != "nan":
        parts = [primary_scope, mechanism, primary_anchor, secondary_anchor]
        return " || ".join(part for part in parts if part)

    # 策略2: mechanism_core 缺失时，使用 fallback
    # 构建 fallback 键：scope + anchors + raw_phrase_core
    fallback_parts = [primary_scope]

    # 添加 anchors
    if primary_anchor:
        fallback_parts.append(primary_anchor)
    if secondary_anchor and secondary_anchor != primary_anchor:
        fallback_parts.append(secondary_anchor)

    # 如果有归一化后的 raw_phrase_core，添加作为补充
    if raw_phrase_core and raw_phrase_core not in fallback_parts:
        fallback_parts.append(raw_phrase_core)

    # 如果 fallback_parts 只有 scope，尝试从 display_candidate_name 提取并归一化
    if len(fallback_parts) == 1:
        display_name = str(concept.get("display_candidate_name", "")).strip()
        if display_name:
            # 使用 canonicalizer 归一化 display_name
            canonical_display, norm_type, _ = canonicalizer.canonicalize(display_name)
            if norm_type != 'none':
                fallback_parts.append(canonical_display.lower())
            else:
                # 提取 display_name 中的关键词
                name_tokens = display_name.lower().split()
                filtered_name = [t for t in name_tokens if t not in {"the", "a", "an", "for", "with"} and len(t) >= 2]
                if filtered_name:
                    fallback_parts.append(filtered_name[0])

    return " || ".join(fallback_parts) if len(fallback_parts) > 1 else f"{primary_scope} || ungrouped"


def _build_event_summary(event):
    technologies = event.get("technology", [])
    if not isinstance(technologies, list):
        technologies = [technologies]
    technologies = [str(item).strip() for item in technologies if str(item).strip() and str(item).strip() != "未知"]
    reasons = event.get("weak_signal_reasons", [])
    if not isinstance(reasons, list):
        reasons = [reasons] if reasons else []
    segments = [
        str(event.get("subject", "")).strip(),
        str(event.get("action", "")).strip(),
        " / ".join(technologies),
        str(event.get("scene", "")).strip(),
        str(event.get("time", "")).strip(),
        "；".join(str(item).strip() for item in reasons if str(item).strip()),
    ]
    return " | ".join(part for part in segments if part and part != "未知")


def _normalize_scope_names(raw_value):
    if isinstance(raw_value, list):
        return [str(item).strip() for item in raw_value if str(item).strip()]
    if isinstance(raw_value, str) and raw_value.strip():
        return [item.strip() for item in raw_value.split(",") if item.strip()]
    return []


def _normalize_aliases(raw_value):
    if isinstance(raw_value, list):
        return [str(item).strip() for item in raw_value if str(item).strip()]
    if isinstance(raw_value, str) and raw_value.strip():
        return [raw_value.strip()]
    return []


SOURCE_TYPE_NORMALIZATION = {
    "专利": "patent",
    "文献": "paper",
    "研报": "report",
    "资讯": "news",
    "patent": "patent",
    "paper": "paper",
    "literature": "paper",
    "report": "report",
    "news": "news",
}


def _safe_text(value):
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "nat", "none"} else text


def _safe_float(value, default=0.0):
    try:
        if value is None or pd.isna(value):
            return default
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return default


def _safe_bool(value):
    if isinstance(value, bool):
        return value
    text = _safe_text(value).lower()
    return text in {"1", "true", "yes", "y", "是", "通过"}


def _normalize_source_type(value):
    text = _safe_text(value)
    return SOURCE_TYPE_NORMALIZATION.get(text, SOURCE_TYPE_NORMALIZATION.get(text.lower(), text.lower() or "unknown"))


def _build_evidence_item(record, concept):
    title = _safe_text(record.get("title", ""))
    text = _safe_text(record.get("text", ""))
    # 优化：延长 snippet 从 120 到 300 字符，增加语义信息量
    snippet = text[:300].replace("\n", " ")
    if len(text) > 300:
        snippet += "..."
    item = {
        "id": _safe_text(record.get("id", "")),
        "source_type": _normalize_source_type(record.get("source_type", "unknown")),
        "org": _safe_text(record.get("org", "Unknown")) or "Unknown",
        "date": _safe_text(record.get("date", "")),
        "url": _safe_text(record.get("url", "")),
        "title": title,
        "snippet": snippet,
        "text": text[:600],  # 延长 text 从 400 到 600
        "raw_candidate_text": _safe_text(concept.get("raw_candidate_text", "")),
        "display_candidate_name": _safe_text(concept.get("display_candidate_name", "")),
    }
    for key in ["event_quality_score", "event_quality_tier", "event_quality_reason"]:
        if key in record:
            item[key] = record.get(key, "")
    return item


# 技术术语权重增强列表
_TECH_TERMS = {
    "robot", "robotic", "robotics", "agent", "agents",
    "training", "planning", "control", "navigation", "manipulation",
    "video", "visual", "trajectory", "trajectories", "sensor", "sensors",
    "multimodal", "simulation", "simulator", "embodied", "intelligence",
    "world model", "world models", "reinforcement", "learning",
    "perception", "action", "policy", "policies", "environment",
    "3d", "three-dimensional", "autonomous", "autonomy",
    "grasping", "grasp", "motion", "dynamics", "kinematics",
    "benchmark", "benchmarks", "dataset", "datasets", "evaluation",
}


def _tech_term_weighted_similarity(text_a, text_b):
    """
    带技术术语权重的相似度计算。
    在 fast_mode (Jaccard) 基础上，对技术术语给予额外权重。
    """
    tokens_a = set(token for token in text_a.lower().split() if token)
    tokens_b = set(token for token in text_b.lower().split() if token)

    if not tokens_a or not tokens_b:
        return 0.0

    # 基础 Jaccard 相似度
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    base_score = len(intersection) / len(union) if union else 0.0

    # 技术术语加成
    tech_intersection = intersection & _TECH_TERMS
    tech_bonus = len(tech_intersection) * 0.05  # 每个匹配的技术术语加 0.05

    # 组合分数，上限 1.0
    final_score = min(base_score + tech_bonus, 1.0)
    return round(final_score, 3)


def _focused_evidence_text(item, focus_terms):
    base_text = " ".join(
        [
            str(item.get("title", "")).strip(),
            str(item.get("snippet", "")).strip(),
            str(item.get("text", "")).strip(),
        ]
    ).strip()
    lowered = base_text.lower()
    snippets = []
    for term in focus_terms or []:
        alias_lower = str(term).strip().lower()
        if not alias_lower:
            continue
        start = lowered.find(alias_lower)
        while start != -1:
            left = max(0, start - 120)  # 扩大上下文窗口
            right = min(len(base_text), start + len(alias_lower) + 120)
            snippets.append(base_text[left:right].strip())
            start = lowered.find(alias_lower, start + len(alias_lower))
    return " ".join(snippets[:4]) if snippets else base_text


def _cross_source_semantic_validation(evidence_items, focus_terms, threshold=0.15):
    """
    跨源语义验证 (优化版)。

    优化策略：
    1. 降低阈值从 0.28 到 0.15
    2. 使用技术术语加权相似度
    3. 分层阈值：基础阈值 0.15，强验证阈值 0.25
    """
    if not evidence_items or len(evidence_items) < 2:
        return False, 0.0, 0
    best_score = 0.0
    validated_pairs = 0
    for idx, left in enumerate(evidence_items):
        for right in evidence_items[idx + 1:]:
            if left.get("source_type") == right.get("source_type"):
                continue
            # 使用技术术语加权相似度
            score = _tech_term_weighted_similarity(
                _focused_evidence_text(left, focus_terms),
                _focused_evidence_text(right, focus_terms),
            )
            if score >= threshold:
                validated_pairs += 1
            if score > best_score:
                best_score = score
    return best_score >= threshold, round(best_score, 3), validated_pairs


def _family_scoped_semantic_validation(evidence_items, family_info, focus_terms):
    """
    对象族范围的语义验证。

    只在以下对象上做：
    - 已命中 object family registry 的 candidate
    - 或 source_count >= 2 的高价值 candidate

    Args:
        evidence_items: 证据项列表
        family_info: 对象族信息（来自 match_family）
        focus_terms: 关注术语列表

    Returns:
        dict: {
            "family_semantic_consistency": "high" | "medium" | "low",
            "family_semantic_validation_score": 0.0-1.0,
            "family_semantic_validation_method": "term_weighted" | "embedding",
            "family_semantic_validation_note": "解释"
        }
    """
    if not family_info:
        return None

    if not evidence_items or len(evidence_items) < 2:
        return {
            "family_semantic_consistency": "low",
            "family_semantic_validation_score": 0.0,
            "family_semantic_validation_method": "term_weighted",
            "family_semantic_validation_note": "证据项不足，无法验证"
        }

    # 获取对象族的别名作为额外的 focus_terms
    family_aliases = family_info.get('alias_terms', [])
    zh_terms = family_info.get('zh_en_pairs', {}).get('zh', [])
    en_terms = family_info.get('zh_en_pairs', {}).get('en', [])
    canonical = family_info.get('canonical_term', '')

    # 合并所有关注术语
    all_focus_terms = list(set(focus_terms + family_aliases + zh_terms + en_terms + [canonical]))

    # 使用更低的阈值进行对象族范围验证
    threshold = 0.10
    best_score = 0.0
    validated_pairs = 0
    total_pairs = 0

    for idx, left in enumerate(evidence_items):
        for right in evidence_items[idx + 1:]:
            if left.get("source_type") == right.get("source_type"):
                continue
            total_pairs += 1
            # 使用技术术语加权相似度
            score = _tech_term_weighted_similarity(
                _focused_evidence_text(left, all_focus_terms),
                _focused_evidence_text(right, all_focus_terms),
            )
            if score >= threshold:
                validated_pairs += 1
            if score > best_score:
                best_score = score

    # 判断一致性级别
    if best_score >= 0.25:
        consistency = "high"
        note = f"对象族内语义一致性高，最佳分数={best_score:.3f}，验证对数={validated_pairs}/{total_pairs}"
    elif best_score >= 0.15:
        consistency = "medium"
        note = f"对象族内语义一致性中等，最佳分数={best_score:.3f}，验证对数={validated_pairs}/{total_pairs}"
    else:
        consistency = "low"
        note = f"对象族内语义一致性低，最佳分数={best_score:.3f}，验证对数={validated_pairs}/{total_pairs}"

    return {
        "family_semantic_consistency": consistency,
        "family_semantic_validation_score": round(best_score, 3),
        "family_semantic_validation_method": "term_weighted",
        "family_semantic_validation_note": note
    }


def _best_pair_score(left_items, right_items, focus_terms, threshold=0.15):
    best_score = 0.0
    validated_pairs = 0
    for left in left_items:
        for right in right_items:
            score = _tech_term_weighted_similarity(
                _focused_evidence_text(left, focus_terms),
                _focused_evidence_text(right, focus_terms),
            )
            if score >= threshold:
                validated_pairs += 1
            if score > best_score:
                best_score = score
    return round(best_score, 3), validated_pairs


def _event_document_validation(evidence_items, focus_terms, threshold=0.15):
    news_items = [item for item in evidence_items if item.get("source_type") == "news"]
    paper_items = [item for item in evidence_items if item.get("source_type") in {"paper", "report"}]
    patent_items = [item for item in evidence_items if item.get("source_type") == "patent"]
    event_paper_score, paper_pairs = _best_pair_score(news_items, paper_items, focus_terms, threshold=threshold)
    event_patent_score, patent_pairs = _best_pair_score(news_items, patent_items, focus_terms, threshold=threshold)
    best_score = max(event_paper_score, event_patent_score)
    return {
        "event_doc_semantic_validated": best_score >= threshold,
        "event_doc_validation_score": round(best_score, 3),
        "event_doc_semantic_pairs": paper_pairs + patent_pairs,
        "event_paper_semantic_score": event_paper_score,
        "event_patent_semantic_score": event_patent_score,
    }


def _time_validation_metrics(date_values):
    parsed = []
    for raw in date_values:
        dt = pd.to_datetime(str(raw).strip(), errors="coerce", utc=True)
        if not pd.isna(dt):
            parsed.append(dt)
    if len(parsed) < 2:
        return 0, len(parsed), 0.0, False
    parsed = sorted(parsed)
    split_index = max(1, len(parsed) // 2)
    early_count = split_index
    recent_count = len(parsed) - split_index
    validation_ratio = round(recent_count / max(early_count, 1), 2)
    time_validated = recent_count >= early_count and recent_count >= 2
    return early_count, recent_count, validation_ratio, time_validated


def _empty_candidate_columns():
    return [
        "tech_name", "technology", "display_candidate_name", "normalized_candidate_text",
        "canonical_candidate_name_en", "constraint_signature", "internal_candidate_label",
        "relation_target", "relation_task", "relation_data_modinality", "relation_method",
        "relation_summary", "relation_signature", "topic_summary_name", "topic_naturalness_reason",
        "compression_mode",
        "mechanism_core", "news_count", "paper_count", "report_count", "patent_count",
        "total_mentions", "org_count", "orgs", "source_types", "source_count", "mention_ids",
        "mention_dates", "evidence_titles", "evidence_items", "weak_signal_event_count",
        "candidate_evidence_quality", "candidate_core_evidence_quality",
        "low_quality_evidence_ratio", "high_quality_evidence_count",
        "quality_risk_flag", "quality_adjusted_rank_score",
        "candidate_evidence_quality_reason",
        "candidate_id", "tech_chain_node_id", "tech_chain_name",
        "tech_chain_official_name", "mapping_relation", "mapping_confidence",
        "mapping_method", "matched_term", "matched_field", "parent_technology",
        "maturity_level", "technology_status", "bottleneck_level",
        "strategic_importance_level", "mapping_reason",
        "tech_chain_mapping_confidence", "tech_chain_mapping_risk",
        "tech_chain_mapping_coverage_flag",
        "first_seen_date", "last_seen_date", "temporal_validation_tier",
        "temporal_validation_status", "temporal_validation_passed",
        "temporal_momentum_score", "growth_rate", "source_growth_rate",
        "org_growth_rate", "high_quality_growth_rate", "date_coverage_ratio",
        "temporal_validation_reason",
        "key_core_score", "key_core_tier", "weak_signal_component",
        "growth_validation_component", "tech_chain_bottleneck_component",
        "strategic_importance_component", "evidence_confidence_component",
        "asset_support_component", "reverse_validation_component",
        "quality_gate_passed", "mapping_gate_passed", "temporal_gate_passed",
        "object_gate_passed", "key_core_gate_passed", "key_core_reason",
        "key_core_risk", "recommended_action", "top_evidence_ids",
        "weak_signal_event_ratio", "low_attention_ratio", "niche_actor_ratio",
        "non_dominant_ratio", "cross_domain_ratio", "traceable_ratio", "multi_source_validated",
        "multi_source_validation_score", "semantic_validated", "semantic_validation_score",
        "semantic_validation_pairs", "event_doc_semantic_validated", "event_doc_validation_score",
        "event_doc_semantic_pairs", "event_paper_semantic_score", "event_patent_semantic_score",
        "literature_validated", "patent_validated", "time_validation_early_count",
        "time_validation_recent_count", "time_validation_ratio", "time_validated",
        "weak_signal_focus_candidate", "is_observation_scope", "scope_name", "scope_names",
        "is_scope_internal_candidate", "candidate_cluster_id", "display_candidate_aliases",
        "alias_count", "true_alias_count", "cluster_evidence_count", "cluster_item_count",
        "is_scope_echo", "has_mechanism_core", "has_task_constraint", "has_non_scope_constraint",
        "generic_core_only", "template_variant_count", "candidate_stage", "raw_phrase_type",
        "mechanism_core_tokens", "task_constraint_tokens", "object_modifier_tokens",
        "data_modifier_tokens", "method_modifier_tokens", "constraint_diversity",
        "merged_from_long_phrase_count", "generic_core_collision_count", "strong_ready_reason",
        "topic_granularity", "display_tier", "non_scope_constraint_count",
        "survives_without_scope", "scope_shell_heavy", "scope_shell_reason",
        "weak_signal_score", "hotspot_score", "score", "explanation",
        # 新增：内部标签与最终表述分离
        "raw_phrase_cluster", "final_research_object_name",
        # 新增：对象族归一化信息
        "canonical_term", "normalization_type", "parent_term", "cross_source_pattern", "patent_role",
        # 新增：对象族感知聚合信息
        "family_id", "family_matched", "term_type", "family_priority",
        # 新增：跨源模式解释
        "cross_source_pattern_strength", "cross_source_pattern_reason",
        # 新增：专利侧角色解释
        "patent_role_reason", "patent_support_mode",
        # 新增：对象族范围语义验证
        "family_semantic_consistency", "family_semantic_validation_score",
        "family_semantic_validation_method", "family_semantic_validation_note",
    ]


def generate_candidate_outputs(candidate_forms_df, data_df):
    empty_columns = _empty_candidate_columns()
    if candidate_forms_df is None or candidate_forms_df.empty or data_df is None or data_df.empty:
        empty_df = pd.DataFrame(columns=empty_columns)
        return {"candidates_df": empty_df, "near_strong_candidates_df": empty_df.copy()}

    # 获取 canonicalizer 实例
    canonicalizer = get_canonicalizer()

    active_forms = candidate_forms_df[
        candidate_forms_df["candidate_stage"].isin(["scope_overview", "formed_candidate", "formed_candidate_strong"])
    ].copy()
    if active_forms.empty or "id" not in active_forms.columns or "id" not in data_df.columns:
        empty_df = pd.DataFrame(columns=empty_columns)
        return {"candidates_df": empty_df, "near_strong_candidates_df": empty_df.copy()}

    metadata_by_id = data_df.drop_duplicates(subset="id", keep="last").set_index("id").to_dict("index")
    group_maps = defaultdict(list)
    mechanism_collision = defaultdict(set)
    family_maps = {}  # 记录每个 cluster_key 对应的 family_info

    for _, concept in active_forms.iterrows():
        primary_scope = str(concept.get("primary_scope", "")).strip()
        mechanism_core = str(concept.get("mechanism_core", "")).strip()
        constraint_signature = str(concept.get("constraint_signature", "")).strip()
        canonical_name = str(concept.get("canonical_candidate_name_en", "")).strip() or str(concept.get("normalized_candidate_text", "")).strip()

        # 优先使用 candidate_forms 中已有的 candidate_cluster_id，确保整个流程中键的一致性
        existing_cluster_id = str(concept.get("candidate_cluster_id", "")).strip()
        if existing_cluster_id and existing_cluster_id not in {"", "nan", "None"}:
            cluster_key = existing_cluster_id
            # 仍然尝试获取 family_info
            family_info = None
            family = match_family(canonical_name) if canonical_name else None
            if family:
                family_info = family
                family_maps[cluster_key] = family_info
        else:
            # 使用对象族感知聚合生成新的 cluster_key
            cluster_key, family_info = _object_family_candidate_key(concept, canonicalizer)

            # 记录 family_info
            if family_info:
                family_maps[cluster_key] = family_info

        # 如果没有生成有效的 cluster_key，使用 fallback
        if not cluster_key:
            cluster_key = " || ".join([primary_scope, mechanism_core, constraint_signature, canonical_name]) or str(concept.get("raw_candidate_text", "")).strip()

        group_maps[cluster_key].append(concept.to_dict())
        if mechanism_core and constraint_signature:
            mechanism_collision[mechanism_core].add(constraint_signature)

    candidates = []
    for cluster_key, items in group_maps.items():
        first = items[0]
        mention_records = []
        for item in items:
            record = metadata_by_id.get(item["id"])
            if record:
                mention_records.append(record)

        counts = defaultdict(int)
        orgs = set()
        mention_ids = set()
        mention_dates = []
        evidence_titles = []
        evidence_items = []
        for item, record in zip(items, mention_records):
            source_type = _normalize_source_type(record.get("source_type", item.get("source_type", "unknown")))
            counts[source_type] += 1
            org = _safe_text(record.get("org", "Unknown")) or "Unknown"
            orgs.add(org)
            mention_ids.add(item["id"])
            date_text = _safe_text(record.get("date"))
            if date_text:
                mention_dates.append(date_text)
            title = _safe_text(record.get("title", ""))
            if title and title not in evidence_titles:
                evidence_titles.append(title)
                evidence_items.append(_build_evidence_item(record, item))

        source_types = sorted([k for k, v in counts.items() if v > 0])
        total_mentions = sum(counts.values())
        raw_variant_aliases = sorted(
            {
                str(item.get("raw_phrase", "")).strip()
                for item in items
                if str(item.get("raw_phrase", "")).strip()
            }
        )
        alias_terms = sorted(
            {
                alias
                for item in items
                for alias in _normalize_aliases(item.get("display_candidate_aliases", []))
                if alias
            }
        )
        if not alias_terms and first.get("canonical_candidate_name_en"):
            alias_terms = [first["canonical_candidate_name_en"]]
        alias_terms = sorted({*alias_terms, *raw_variant_aliases[:4]})
        early_mentions, recent_mentions, validation_ratio, time_validated = _time_validation_metrics(mention_dates)
        multi_source_validated = len(source_types) >= 2
        multi_source_validation_score = 2.0 if len(source_types) >= 3 else 1.5 if len(source_types) == 2 else 0.0
        semantic_validated, semantic_score, semantic_pair_count = _cross_source_semantic_validation(evidence_items, alias_terms)
        event_doc_validation = _event_document_validation(evidence_items, alias_terms)
        candidate_stage = str(first.get("candidate_stage", "")).strip()
        weak_signal_focus_candidate = (
            candidate_stage == "formed_candidate_strong"
            and bool(first.get("is_scope_internal_candidate", False))
            and not bool(first.get("is_scope_echo", False))
            and bool(first.get("has_mechanism_core", False))
            and bool(first.get("has_non_scope_constraint", False))
            and (multi_source_validated or semantic_validated or event_doc_validation["event_doc_semantic_validated"])
        )
        constraint_signatures = {
            str(item.get("constraint_signature", "")).strip()
            for item in items
            if str(item.get("constraint_signature", "")).strip()
        }

        # 对象族归一化（使用分层归一）
        display_name = str(first.get("display_candidate_name", "")).strip()
        canonical_name_en = str(first.get("canonical_candidate_name_en", "")).strip()
        raw_phrase = str(first.get("raw_phrase", "")).strip()

        # 使用分层归一化（始终执行，用于获取 canonical_term 等信息）
        norm_result = canonicalizer.canonicalize_with_layers(canonical_name_en or raw_phrase or display_name)

        # 如果没有匹配，尝试从 raw_variant_aliases 中提取
        if norm_result.normalization_type == 'none' and raw_variant_aliases:
            for alias in raw_variant_aliases:
                alias_result = canonicalizer.canonicalize_with_layers(alias)
                if alias_result.normalization_type != 'none':
                    norm_result = alias_result
                    break

        # 优先从 family_maps 获取拆分后的 family_info（覆盖 norm_result）
        family_info = family_maps.get(cluster_key)
        if family_info:
            family_matched = True
            family_id = family_info.get('family_id', '')
            term_type = family_info.get('term_type', 'unknown')
            family_priority = family_info.get('priority', 'low')
        else:
            # 使用 norm_result 的信息
            family_matched = norm_result.family_id != ""
            family_id = norm_result.family_id
            term_type = norm_result.term_type
            family_priority = norm_result.priority

        # 获取跨源模式信息
        cross_source_pattern_name, pattern_info = canonicalizer.get_cross_source_pattern(source_types)
        cross_source_pattern_strength = pattern_info.get('strength', 'weak') if pattern_info else 'weak'
        cross_source_pattern_reason = pattern_info.get('reason', '') if pattern_info else ''

        # 获取专利角色信息
        if family_info:
            patent_role = family_info.get('patent_role', 'optional')
        else:
            patent_role = norm_result.patent_role
        patent_role_reason = ""
        patent_support_mode = ""
        if patent_role == 'engineering_signal':
            patent_role_reason = "专利侧提供工程信号补充"
            patent_support_mode = "技术实现细节、工程参数、应用场景"
        elif patent_role == 'application_trace':
            patent_role_reason = "专利侧提供产业化证据"
            patent_support_mode = "产品化、商业化、落地场景"
        elif patent_role == 'validation':
            patent_role_reason = "专利侧提供技术成熟度证据"
            patent_support_mode = "技术路线稳定性、市场认可度"
        else:
            patent_role_reason = "专利侧不强制要求"
            patent_support_mode = "任何补充信息"

        # 获取对象族信息（用于 family-scoped semantic validation）
        # family_info 已在上面设置

        # 执行 family-scoped semantic validation
        family_semantic_result = _family_scoped_semantic_validation(
            evidence_items, family_info, alias_terms
        )
        if family_semantic_result is None:
            family_semantic_result = {
                "family_semantic_consistency": "",
                "family_semantic_validation_score": 0.0,
                "family_semantic_validation_method": "",
                "family_semantic_validation_note": ""
            }

        candidate_quality_values = [
            _safe_float(item.get("candidate_evidence_quality"), default=-1.0)
            for item in items
        ]
        candidate_quality_values = [value for value in candidate_quality_values if value >= 0]
        candidate_evidence_quality = (
            round(sum(candidate_quality_values) / len(candidate_quality_values), 2)
            if candidate_quality_values
            else 0.0
        )
        candidate_core_quality_values = [
            _safe_float(item.get("candidate_core_evidence_quality"), default=-1.0)
            for item in items
        ]
        candidate_core_quality_values = [value for value in candidate_core_quality_values if value >= 0]
        candidate_core_evidence_quality = (
            round(sum(candidate_core_quality_values) / len(candidate_core_quality_values), 2)
            if candidate_core_quality_values
            else candidate_evidence_quality
        )
        low_quality_evidence_ratio = _safe_float(first.get("low_quality_evidence_ratio"), 0.0)
        high_quality_evidence_count = int(_safe_float(first.get("high_quality_evidence_count"), 0.0))
        quality_risk_flag = str(first.get("quality_risk_flag", "")).strip()
        quality_adjusted_rank_score = _safe_float(
            first.get("quality_adjusted_rank_score"),
            _safe_float(first.get("weak_signal_score", first.get("score", 0.0))),
        )
        candidate_evidence_quality_reason = str(first.get("candidate_evidence_quality_reason", "")).strip()
        tech_chain_source = sorted(
            items,
            key=lambda item: (
                str(item.get("mapping_relation", "")).strip() == "no_match",
                -_safe_float(item.get("mapping_confidence"), 0.0),
            ),
        )[0]
        mapping_confidence = _safe_float(tech_chain_source.get("mapping_confidence"), 0.0)
        tech_chain_mapping_confidence = _safe_float(
            tech_chain_source.get("tech_chain_mapping_confidence"),
            mapping_confidence,
        )

        candidates.append(
            {
                "tech_name": str(first.get("display_candidate_name", "")).strip() or str(first.get("canonical_candidate_name_en", "")).strip() or str(first.get("raw_candidate_text", "")).strip(),
                "technology": str(first.get("display_candidate_name", "")).strip() or str(first.get("canonical_candidate_name_en", "")).strip(),
                "display_candidate_name": str(first.get("display_candidate_name", "")).strip() or str(first.get("canonical_candidate_name_en", "")).strip(),
                "normalized_candidate_text": str(first.get("normalized_candidate_text", "")).strip(),
                "canonical_candidate_name_en": str(first.get("canonical_candidate_name_en", "")).strip(),
                "constraint_signature": str(first.get("constraint_signature", "")).strip(),
                "internal_candidate_label": str(first.get("internal_candidate_label", "")).strip(),
                "relation_target": str(first.get("relation_target", "")).strip(),
                "relation_task": str(first.get("relation_task", "")).strip(),
                "relation_data_modality": str(first.get("relation_data_modality", "")).strip(),
                "relation_method": str(first.get("relation_method", "")).strip(),
                "relation_summary": str(first.get("relation_summary", "")).strip(),
                "relation_signature": str(first.get("relation_signature", "")).strip(),
                "topic_summary_name": str(first.get("topic_summary_name", "")).strip(),
                "topic_naturalness_reason": str(first.get("topic_naturalness_reason", "")).strip(),
                "compression_mode": str(first.get("compression_mode", "")).strip(),
                "mechanism_core": str(first.get("mechanism_core", "")).strip(),
                "news_count": counts.get("news", 0),
                "paper_count": counts.get("paper", 0) + counts.get("report", 0),
                "report_count": counts.get("paper", 0) + counts.get("report", 0),
                "patent_count": counts.get("patent", 0),
                "total_mentions": total_mentions,
                "org_count": len(orgs),
                "orgs": sorted(orgs),
                "source_types": source_types,
                "source_count": len(source_types),
                "mention_ids": sorted(mention_ids),
                "mention_dates": mention_dates,
                "evidence_titles": evidence_titles[:5],
                "evidence_items": evidence_items[:5],
                "weak_signal_event_count": len(mention_ids),
                "candidate_evidence_quality": candidate_evidence_quality,
                "candidate_core_evidence_quality": candidate_core_evidence_quality,
                "low_quality_evidence_ratio": low_quality_evidence_ratio,
                "high_quality_evidence_count": high_quality_evidence_count,
                "quality_risk_flag": quality_risk_flag,
                "quality_adjusted_rank_score": quality_adjusted_rank_score,
                "candidate_evidence_quality_reason": candidate_evidence_quality_reason,
                "candidate_id": str(tech_chain_source.get("candidate_id", first.get("candidate_id", ""))).strip(),
                "tech_chain_node_id": str(tech_chain_source.get("tech_chain_node_id", "")).strip(),
                "tech_chain_name": str(tech_chain_source.get("tech_chain_name", "")).strip(),
                "tech_chain_official_name": str(tech_chain_source.get("tech_chain_official_name", "")).strip(),
                "mapping_relation": str(tech_chain_source.get("mapping_relation", "")).strip(),
                "mapping_confidence": mapping_confidence,
                "mapping_method": str(tech_chain_source.get("mapping_method", "")).strip(),
                "matched_term": str(tech_chain_source.get("matched_term", "")).strip(),
                "matched_field": str(tech_chain_source.get("matched_field", "")).strip(),
                "parent_technology": str(tech_chain_source.get("parent_technology", "")).strip(),
                "maturity_level": str(tech_chain_source.get("maturity_level", "")).strip(),
                "technology_status": str(tech_chain_source.get("technology_status", "")).strip(),
                "bottleneck_level": str(tech_chain_source.get("bottleneck_level", "")).strip(),
                "strategic_importance_level": str(tech_chain_source.get("strategic_importance_level", "")).strip(),
                "mapping_reason": str(tech_chain_source.get("mapping_reason", "")).strip(),
                "tech_chain_mapping_confidence": tech_chain_mapping_confidence,
                "tech_chain_mapping_risk": str(tech_chain_source.get("tech_chain_mapping_risk", "")).strip(),
                "tech_chain_mapping_coverage_flag": str(tech_chain_source.get("tech_chain_mapping_coverage_flag", "")).strip(),
                "first_seen_date": str(first.get("first_seen_date", "")).strip(),
                "last_seen_date": str(first.get("last_seen_date", "")).strip(),
                "temporal_validation_tier": str(first.get("temporal_validation_tier", "")).strip(),
                "temporal_validation_status": str(first.get("temporal_validation_status", "")).strip(),
                "temporal_validation_passed": _safe_bool(first.get("temporal_validation_passed", False)),
                "temporal_momentum_score": _safe_float(first.get("temporal_momentum_score"), 0.0),
                "growth_rate": _safe_float(first.get("growth_rate"), 0.0),
                "source_growth_rate": _safe_float(first.get("source_growth_rate"), 0.0),
                "org_growth_rate": _safe_float(first.get("org_growth_rate"), 0.0),
                "high_quality_growth_rate": _safe_float(first.get("high_quality_growth_rate"), 0.0),
                "date_coverage_ratio": _safe_float(first.get("date_coverage_ratio"), 0.0),
                "temporal_validation_reason": str(first.get("temporal_validation_reason", "")).strip(),
                "key_core_score": _safe_float(first.get("key_core_score"), 0.0),
                "key_core_tier": str(first.get("key_core_tier", "")).strip(),
                "weak_signal_component": _safe_float(first.get("weak_signal_component"), 0.0),
                "growth_validation_component": _safe_float(first.get("growth_validation_component"), 0.0),
                "tech_chain_bottleneck_component": _safe_float(first.get("tech_chain_bottleneck_component"), 0.0),
                "strategic_importance_component": _safe_float(first.get("strategic_importance_component"), 0.0),
                "evidence_confidence_component": _safe_float(first.get("evidence_confidence_component"), 0.0),
                "asset_support_component": _safe_float(first.get("asset_support_component"), 0.0),
                "reverse_validation_component": _safe_float(first.get("reverse_validation_component"), 0.0),
                "quality_gate_passed": _safe_bool(first.get("quality_gate_passed", False)),
                "mapping_gate_passed": _safe_bool(first.get("mapping_gate_passed", False)),
                "temporal_gate_passed": _safe_bool(first.get("temporal_gate_passed", False)),
                "object_gate_passed": _safe_bool(first.get("object_gate_passed", False)),
                "key_core_gate_passed": _safe_bool(first.get("key_core_gate_passed", False)),
                "key_core_reason": str(first.get("key_core_reason", "")).strip(),
                "key_core_risk": str(first.get("key_core_risk", "")).strip(),
                "recommended_action": str(first.get("recommended_action", "")).strip(),
                "top_evidence_ids": str(first.get("top_evidence_ids", "")).strip(),
                "weak_signal_event_ratio": round(len(mention_ids) / max(total_mentions, 1), 2),
                "low_attention_ratio": round(sum(1 for record in mention_records if record.get("source_type") in {"paper", "patent"}) / max(total_mentions, 1), 2),
                "niche_actor_ratio": round(sum(1 for org in orgs if org not in {"Unknown", "arXiv"}) / max(len(orgs), 1), 2),
                "non_dominant_ratio": round(sum(1 for item in items if item.get("has_mechanism_core")) / max(len(items), 1), 2),
                "cross_domain_ratio": round(sum(1 for item in items if item.get("has_non_scope_constraint")) / max(len(items), 1), 2),
                "traceable_ratio": round(sum(1 for record in mention_records if bool(record.get("url") or record.get("date"))) / max(total_mentions, 1), 2),
                "multi_source_validated": multi_source_validated,
                "multi_source_validation_score": multi_source_validation_score,
                "semantic_validated": semantic_validated,
                "semantic_validation_score": semantic_score,
                "semantic_validation_pairs": semantic_pair_count,
                "event_doc_semantic_validated": event_doc_validation["event_doc_semantic_validated"],
                "event_doc_validation_score": event_doc_validation["event_doc_validation_score"],
                "event_doc_semantic_pairs": event_doc_validation["event_doc_semantic_pairs"],
                "event_paper_semantic_score": event_doc_validation["event_paper_semantic_score"],
                "event_patent_semantic_score": event_doc_validation["event_patent_semantic_score"],
                "literature_validated": event_doc_validation["event_paper_semantic_score"] > 0,
                "patent_validated": event_doc_validation["event_patent_semantic_score"] > 0,
                "time_validation_early_count": early_mentions,
                "time_validation_recent_count": recent_mentions,
                "time_validation_ratio": validation_ratio,
                "time_validated": time_validated,
                "weak_signal_focus_candidate": weak_signal_focus_candidate,
                "is_observation_scope": bool(first.get("is_observation_scope", False)),
                "scope_name": str(first.get("scope_name", "")).strip(),
                "scope_names": _normalize_scope_names(first.get("scope_names", [])),
                "primary_scope": str(first.get("primary_scope", "")).strip(),
                "is_scope_internal_candidate": bool(first.get("is_scope_internal_candidate", False)),
                "candidate_cluster_id": cluster_key,
                "display_candidate_aliases": alias_terms,
                "alias_count": len(alias_terms),
                "true_alias_count": len(alias_terms),
                "cluster_evidence_count": len(mention_ids),
                "cluster_item_count": len(items),
                "is_scope_echo": bool(first.get("is_scope_echo", False)),
                "has_mechanism_core": bool(first.get("has_mechanism_core", False)),
                "has_task_constraint": bool(first.get("has_task_constraint", False)),
                "has_non_scope_constraint": bool(first.get("has_non_scope_constraint", False)),
                "generic_core_only": bool(first.get("generic_core_only", False)),
                "template_variant_count": len({str(item.get('raw_phrase', '')).strip() for item in items if str(item.get('raw_phrase', '')).strip()}) - len(alias_terms),
                "candidate_stage": candidate_stage,
                "raw_phrase_type": str(first.get("raw_phrase_type", "")).strip(),
                "mechanism_core_tokens": _normalize_aliases(first.get("mechanism_core_tokens", [])),
                "task_constraint_tokens": _normalize_aliases(first.get("task_constraint_tokens", [])),
                "object_modifier_tokens": _normalize_aliases(first.get("object_modifier_tokens", [])),
                "data_modifier_tokens": _normalize_aliases(first.get("data_modifier_tokens", [])),
                "method_modifier_tokens": _normalize_aliases(first.get("method_modifier_tokens", [])),
                "constraint_diversity": len(constraint_signatures),
                "merged_from_long_phrase_count": sum(1 for item in items if len(str(item.get("raw_phrase", "")).split()) >= 4),
                "generic_core_collision_count": max(len(mechanism_collision.get(str(first.get("mechanism_core", "")).strip(), set())) - 1, 0),
                "strong_ready_reason": str(first.get("strong_ready_reason", "")).strip(),
                "topic_granularity": str(first.get("topic_granularity", "")).strip(),
                "display_tier": str(first.get("display_tier", "")).strip(),
                "non_scope_constraint_count": int(first.get("non_scope_constraint_count", 0) or 0),
                "survives_without_scope": bool(first.get("survives_without_scope", False)),
                "scope_shell_heavy": bool(first.get("scope_shell_heavy", False)),
                "scope_shell_reason": str(first.get("scope_shell_reason", "")).strip(),
                "weak_signal_score": float(first.get("weak_signal_score", 0.0) or 0.0),
                "hotspot_score": float(first.get("hotspot_score", 0.0) or 0.0),
                "score": float(first.get("score", first.get("weak_signal_score", first.get("hotspot_score", 0.0))) or 0.0),
                "explanation": str(first.get("explanation", "")).strip(),
                # 新增：内部标签与最终表述分离
                "raw_phrase_cluster": "；".join(raw_variant_aliases[:5]) if raw_variant_aliases else "",
                "final_research_object_name": str(first.get("topic_summary_name", "")).strip() or str(first.get("display_candidate_name", "")).strip(),
                # 新增：对象族归一化信息
                "canonical_term": norm_result.canonical_term,
                "normalization_type": norm_result.normalization_type,
                "parent_term": norm_result.parent_term,
                "cross_source_pattern": cross_source_pattern_name,
                "patent_role": patent_role,
                # 新增：对象族感知聚合信息
                "family_id": family_id,
                "family_matched": family_matched,
                "term_type": term_type,
                "family_priority": family_priority,
                # 新增：跨源模式解释
                "cross_source_pattern_strength": cross_source_pattern_strength,
                "cross_source_pattern_reason": cross_source_pattern_reason,
                # 新增：专利侧角色解释
                "patent_role_reason": patent_role_reason,
                "patent_support_mode": patent_support_mode,
                # 新增：对象族范围语义验证
                "family_semantic_consistency": family_semantic_result.get("family_semantic_consistency", ""),
                "family_semantic_validation_score": family_semantic_result.get("family_semantic_validation_score", 0.0),
                "family_semantic_validation_method": family_semantic_result.get("family_semantic_validation_method", ""),
                "family_semantic_validation_note": family_semantic_result.get("family_semantic_validation_note", ""),
            }
        )

    candidates_df = pd.DataFrame(candidates, columns=_empty_candidate_columns())
    if candidates_df.empty:
        return {"candidates_df": candidates_df, "near_strong_candidates_df": candidates_df.copy()}
    
    # 设置信号类型
    def _set_signal_type(row):
        if row.get("candidate_stage") == "scope_overview":
            return "scope_overview"
        elif row.get("candidate_stage") == "formed_candidate_strong":
            return "weak_signal"
        elif row.get("candidate_stage") == "formed_candidate":
            if row.get("is_scope_internal_candidate") and row.get("has_mechanism_core") and row.get("has_non_scope_constraint"):
                return "near_strong"
            else:
                return "other"
        else:
            return "other"
    
    candidates_df["signal_type"] = candidates_df.apply(_set_signal_type, axis=1)
    
    candidates_df = candidates_df.sort_values(
        by=["total_mentions", "source_count", "org_count", "tech_name"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    near_strong_candidates_df = candidates_df[
        (candidates_df["signal_type"] == "near_strong")
    ].copy()
    return {
        "candidates_df": candidates_df,
        "near_strong_candidates_df": near_strong_candidates_df.reset_index(drop=True),
    }


def generate_candidates(candidate_forms_df, data_df):
    return generate_candidate_outputs(candidate_forms_df, data_df)["candidates_df"]
