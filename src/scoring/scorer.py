from __future__ import annotations

from datetime import timedelta
import json
from pathlib import Path
import re

import pandas as pd

from ..extraction.tech_lexicon import BARE_MECHANISM_CORES, DOMINANT_TECH_TERMS, OBSERVATION_SCOPE_SET


def _safe_list(value):
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"nan", "none", "null"}:
            return []
        if text[0] in "[{":
            try:
                parsed = json.loads(text)
            except Exception:
                return [text]
            return _safe_list(parsed)
        return [text]
    try:
        if pd.isna(value):
            return []
    except Exception:
        pass
    return [value]


def _safe_text(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _safe_float(value, default=0.0):
    if value is None:
        return default
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"nan", "none", "null"}:
            return default
        value = text
    else:
        try:
            if pd.isna(value):
                return default
        except Exception:
            pass
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value, default=0):
    return int(_safe_float(value, float(default)))


def _safe_bool(value):
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "0", "false", "no", "n", "nan", "none", "null"}:
            return False
        if text in {"1", "true", "yes", "y"}:
            return True
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass
    return bool(value)


def _safe_evidence_items(value):
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def _joined_evidence_text(row):
    parts = []
    for item in _safe_evidence_items(row.get("evidence_items", [])):
        parts.extend(
            [
                _safe_text(item.get("title", "")),
                _safe_text(item.get("snippet", "")),
                _safe_text(item.get("text", "")),
                _safe_text(item.get("raw_candidate_text", "")),
            ]
        )
    parts.extend(_safe_list(row.get("evidence_titles", [])))
    parts.append(_safe_text(row.get("raw_phrase_example", "")))
    return " ".join(part for part in parts if part).lower()


def _parse_datetime(raw_value):
    if raw_value is None or (isinstance(raw_value, float) and pd.isna(raw_value)):
        return None
    text = str(raw_value).strip()
    if not text:
        return None
    parsed = pd.to_datetime(text, errors="coerce", utc=True)
    if pd.isna(parsed):
        return None
    return parsed.to_pydatetime()


def _calculate_growth_score(mention_dates):
    parsed_dates = [dt for dt in (_parse_datetime(item) for item in _safe_list(mention_dates)) if dt is not None]
    if not parsed_dates:
        return 0.0
    latest_date = max(parsed_dates)
    recent_threshold = latest_date - timedelta(days=30)
    recent_mentions = sum(1 for dt in parsed_dates if dt >= recent_threshold)
    return round(min(4.0, (recent_mentions / len(parsed_dates)) * 4.0), 2)


def _calculate_novelty_score(row):
    tech_name = str(row.get("display_candidate_name", row.get("tech_name", ""))).strip()
    total_mentions = row.get("total_mentions", 0)
    source_count = row.get("source_count", len(_safe_list(row.get("source_types", []))))
    novelty_score = 0.0
    token_count = len([token for token in tech_name.replace("-", " ").split() if token])
    if token_count >= 2:
        novelty_score += 1.0
    if any(char.isdigit() for char in tech_name):
        novelty_score += 1.0
    if "-" in tech_name or "/" in tech_name:
        novelty_score += 0.5
    if source_count >= 2 and total_mentions <= 6:
        novelty_score += 1.5
    if len(tech_name) >= 14:
        novelty_score += 0.5
    return round(min(4.0, novelty_score), 2)


def _calculate_mainstream_penalty(total_mentions, source_count):
    penalty = 0.0
    if total_mentions >= 12:
        penalty += min(6.0, (total_mentions - 11) * 0.8)
    elif total_mentions >= 8:
        penalty += 1.5
    if total_mentions >= 10 and source_count >= 3:
        penalty += 1.5
    return round(penalty, 2)


def _calculate_validation_bonus(row):
    bonus = 0.0
    bonus += float(row.get("multi_source_validation_score", 0.0))
    if bool(row.get("semantic_validated", False)):
        bonus += min(float(row.get("semantic_validation_score", 0.0)) * 2.0, 2.0)
    if bool(row.get("event_doc_semantic_validated", False)):
        bonus += min(float(row.get("event_doc_validation_score", 0.0)) * 3.0, 2.5)
    if bool(row.get("time_validated", False)):
        bonus += 1.5
    bonus += min(float(row.get("weak_signal_event_ratio", 0.0)) * 2.0, 2.0)
    return round(bonus, 2)


def _calculate_evidence_strength_score(row):
    """
    计算证据强度分数，替代简单的 source_count 例外逻辑。

    证据强度分数综合考虑：
    1. 跨源验证强度（multi_source_validated, semantic_validated）
    2. 文献/专利验证强度（literature_validated, patent_validated）
    3. 时间验证强度（time_validated, growth_score）
    4. 证据簇大小（cluster_evidence_count）
    5. 机构分散度（org_count）

    分数范围：0.0 - 10.0
    """
    score = 0.0

    # 1. 跨源验证强度（最高 3.0 分）
    source_count = int(row.get("source_count", 0) or 0)
    if source_count >= 3:
        score += 3.0
    elif source_count == 2:
        score += 2.0
    elif source_count == 1:
        # 单源但有强语义验证也可得分
        if bool(row.get("semantic_validated", False)):
            score += 1.0

    # 2. 语义验证强度（最高 2.0 分）
    if bool(row.get("multi_source_validated", False)):
        semantic_score = float(row.get("semantic_validation_score", 0.0) or 0.0)
        score += min(semantic_score * 2.0, 2.0)

    # 3. 文献/专利验证强度（最高 2.0 分）
    literature_validated = bool(row.get("literature_validated", False))
    patent_validated = bool(row.get("patent_validated", False))
    if literature_validated and patent_validated:
        score += 2.0
    elif literature_validated:
        score += 1.2
    elif patent_validated:
        score += 1.0

    # 4. 时间验证强度（最高 1.5 分）
    if bool(row.get("time_validated", False)):
        time_ratio = float(row.get("time_validation_ratio", 0.0) or 0.0)
        score += min(time_ratio * 0.5, 1.5)

    # 5. 证据簇大小（最高 1.0 分）
    cluster_evidence_count = int(row.get("cluster_evidence_count", 0) or 0)
    if cluster_evidence_count >= 5:
        score += 1.0
    elif cluster_evidence_count >= 3:
        score += 0.7
    elif cluster_evidence_count >= 2:
        score += 0.5

    # 6. 机构分散度（最高 0.5 分）
    org_count = int(row.get("org_count", 0) or 0)
    if org_count >= 3:
        score += 0.5
    elif org_count >= 2:
        score += 0.3

    return round(min(score, 10.0), 2)


def _infer_failure_case_type(row):
    """
    失败案例类型化：把失败案例分成不同类型。
    这是 plan.md 第6条要求的核心实现。

    类型包括：
    - scope_shell_only: 仅停留在 scope 壳层
    - generic_core_only: 只有泛化机制核，缺少约束
    - cross_source_mixed: 跨源但语义混杂
    - surface_name_not_natural: 命名不够自然
    - promising_but_stage_unclear: 有希望但阶段不明
    - not_failure_case: 不是失败案例
    """
    # 先检查是否是失败案例
    object_hierarchy_tier = str(row.get("object_hierarchy_tier", "")).strip()
    if object_hierarchy_tier in {"weak_signal", "natural_small_topic", "scope_internal_candidate", "scope_overview"}:
        return "not_failure_case", "已进入有效候选层"

    # 检查失败案例类型
    is_scope_echo = bool(row.get("is_scope_echo", False))
    generic_core_only = bool(row.get("generic_core_only", False))
    has_mechanism_core = bool(row.get("has_mechanism_core", False))
    has_non_scope_constraint = bool(row.get("has_non_scope_constraint", False))
    scope_shell_heavy = bool(row.get("scope_shell_heavy", False))
    source_semantic_consistency = str(row.get("source_semantic_consistency", "")).strip()
    reverse_validation_status = str(row.get("reverse_validation_status", "")).strip()
    llm_judgment = str(row.get("llm_small_topic_judgment", "")).strip()
    stage_hypothesis = str(row.get("stage_hypothesis", "")).strip()
    source_count = int(row.get("source_count", 0) or 0)
    cluster_evidence_count = int(row.get("cluster_evidence_count", 0) or 0)

    # 类型1：scope_shell_only - 仅停留在 scope 壳层
    if is_scope_echo or (has_mechanism_core and scope_shell_heavy and not has_non_scope_constraint):
        return "scope_shell_only", "原文太粗：仅停留在 scope 壳层，缺少具体约束"

    # 类型2：generic_core_only - 只有泛化机制核
    if generic_core_only or (not has_non_scope_constraint):
        return "generic_core_only", "聚合过头：只有泛化机制核，缺少非 scope 约束"

    # 类型3：cross_source_mixed - 跨源但语义混杂
    if source_count >= 2 and source_semantic_consistency == "low":
        return "cross_source_mixed", "跨源但语义混杂：不同来源指向的主题不一致"

    # 类型4：surface_name_not_natural - 命名不够自然
    if llm_judgment in {"compressed_label", "unclear"} and reverse_validation_status in {"mixed_or_unclear", "only_upper_topic_in_source"}:
        return "surface_name_not_natural", "命名没收好：当前命名仍偏系统压缩标签"

    # 类型5：promising_but_stage_unclear - 有希望但阶段不明
    if has_mechanism_core and has_non_scope_constraint and cluster_evidence_count >= 2:
        if stage_hypothesis == "unclear" or not stage_hypothesis:
            return "promising_but_stage_unclear", "已经有希望但阶段还不稳：需继续观察证据积累"

    # 默认类型
    return "generic_failure", "未能进入有效候选层，原因待进一步分析"


def _infer_weak_signal_tier(row):
    """
    弱信号二次分层：把 weak_signal 对象细分为不同层级。

    分层标准：
    - core_weak_signal: natural_topic_in_source + high/medium semantic + source_count >= 2
    - candidate_weak_signal: natural_topic_in_source + source_count >= 1
    - borderline_hotspot: 其他

    这是任务2的核心实现：对 weak_signal 进行分级，而不是继续增加数量。
    """
    # 先检查是否是 weak_signal
    signal_type = str(row.get("signal_type", "")).strip()
    if signal_type != "weak_signal":
        return "", "不是 weak_signal 对象"

    reverse_validation_status = str(row.get("reverse_validation_status", "")).strip()
    source_semantic_consistency = str(row.get("source_semantic_consistency", "")).strip()
    family_semantic_consistency = str(row.get("family_semantic_consistency", "")).strip()
    source_count = int(row.get("source_count", 0) or 0)
    weak_signal_score = float(row.get("weak_signal_score", 0.0) or 0.0)
    cluster_evidence_count = int(row.get("cluster_evidence_count", 0) or 0)

    # 综合语义一致性：优先使用 family_semantic_consistency，否则使用 source_semantic_consistency
    semantic_consistency = family_semantic_consistency or source_semantic_consistency

    # 第一层：core_weak_signal - 核心弱信号
    # 条件：natural_topic_in_source + high/medium semantic + source_count >= 2
    if reverse_validation_status == "natural_topic_in_source":
        if semantic_consistency in {"high", "medium"} and source_count >= 2:
            return "core_weak_signal", "反向验证通过，语义一致性高/中，跨源数>=2，为核心弱信号"

        # 第二层：candidate_weak_signal - 候选弱信号
        # 条件：natural_topic_in_source + source_count >= 1
        if source_count >= 1:
            if semantic_consistency == "low":
                return "candidate_weak_signal", "反向验证通过，但语义一致性低，为候选弱信号"
            return "candidate_weak_signal", "反向验证通过，为候选弱信号"

    # 检查其他反向验证状态
    if reverse_validation_status == "only_upper_topic_in_source":
        return "borderline_hotspot", "反向验证显示原文只包含上层主题表达，为边界热点"

    if reverse_validation_status == "mixed_or_unclear":
        return "borderline_hotspot", "反向验证显示语义混杂或不明确，为边界热点"

    # 没有反向验证信息的情况
    if not reverse_validation_status:
        # 使用其他指标判断
        if source_count >= 2 and semantic_consistency in {"high", "medium"}:
            return "candidate_weak_signal", "无反向验证，但跨源且语义一致性高/中，为候选弱信号"
        if source_count >= 2 and weak_signal_score >= 8.0:
            return "candidate_weak_signal", f"无反向验证，但跨源且弱信号分数高({weak_signal_score:.1f})，为候选弱信号"

    # 默认：borderline_hotspot
    return "borderline_hotspot", "当前证据不足以确认为核心或候选弱信号，为边界热点"


def _infer_candidate_weak_signal_stability(row):
    """
    candidate_weak_signal 内部排序：评估候选弱信号的稳定性。

    稳定性分级：
    - promotion_ready: 更可能升为 core 的（跨源+证据充足+语义一致性中等）
    - observation_pool: 适合留在观察池（单源或证据不足）
    - manual_review: 需要人工复核（语义一致性低或边界情况）

    判断依据：
    1. source_count >= 2 且 cluster_evidence_count >= 5 → promotion_ready
    2. source_count >= 2 且 semantic_consistency == "medium" → promotion_ready
    3. source_count == 1 或 cluster_evidence_count < 3 → observation_pool
    4. semantic_consistency == "low" → manual_review
    """
    # 先检查是否是 candidate_weak_signal
    weak_signal_tier = str(row.get("weak_signal_tier", "")).strip()
    if weak_signal_tier != "candidate_weak_signal":
        return "", "不是 candidate_weak_signal 对象"

    source_semantic_consistency = str(row.get("source_semantic_consistency", "")).strip()
    family_semantic_consistency = str(row.get("family_semantic_consistency", "")).strip()
    source_count = int(row.get("source_count", 0) or 0)
    cluster_evidence_count = int(row.get("cluster_evidence_count", 0) or 0)
    weak_signal_score = float(row.get("weak_signal_score", 0.0) or 0.0)

    # 综合语义一致性
    semantic_consistency = family_semantic_consistency or source_semantic_consistency

    # 判断稳定性
    reasons = []

    # 条件1：跨源 + 证据充足
    if source_count >= 2 and cluster_evidence_count >= 5:
        return "promotion_ready", f"跨源数={source_count}，证据数={cluster_evidence_count}，具备升级潜力"

    # 条件2：跨源 + 语义一致性中等
    if source_count >= 2 and semantic_consistency == "medium":
        return "promotion_ready", f"跨源数={source_count}，语义一致性中等，具备升级潜力"

    # 条件3：语义一致性低 → 需要人工复核
    if semantic_consistency == "low":
        return "manual_review", "语义一致性低，需要人工复核确认"

    # 条件4：单源或证据不足
    if source_count == 1:
        return "observation_pool", f"单源（source_count=1），建议继续观察证据积累"

    if cluster_evidence_count < 3:
        return "observation_pool", f"证据不足（evidence_count={cluster_evidence_count}），建议继续观察"

    # 默认：观察池
    return "observation_pool", "当前指标适中，建议继续观察"


def _infer_hotspot_subtype(row):
    """
    Hotspot 子类型判断：把 hotspot 对象细分为不同类型。
    这是 plan.md 第3条要求的核心实现。

    子类型包括：
    - promising_natural_topic: 较像自然小主题，但暂不具备弱信号属性
    - compressed_label: 仍偏系统压缩标签，需继续收口
    - cross_source_mixed: 跨源但语义混杂
    - upper_topic_only: 仅停留在上层主题层面

    判断依据：
    - LLM 小主题判断
    - 反向验证状态
    - 语义一致性
    - scope 壳依赖程度
    """
    # 先检查是否是 hotspot
    signal_type = str(row.get("signal_type", "")).strip()
    if signal_type != "hotspot":
        return "", "不是 hotspot 对象"

    llm_judgment = str(row.get("llm_small_topic_judgment", "")).strip()
    llm_pattern = str(row.get("llm_small_topic_pattern", "")).strip()
    reverse_validation_status = str(row.get("reverse_validation_status", "")).strip()
    source_semantic_consistency = str(row.get("source_semantic_consistency", "")).strip()
    scope_shell_heavy = bool(row.get("scope_shell_heavy", False))
    survives_without_scope = bool(row.get("survives_without_scope", False))
    source_count = int(row.get("source_count", 0) or 0)

    # 子类型1：promising_natural_topic - 较像自然小主题
    if llm_judgment == "small_topic" and llm_pattern in {"scene+mechanism", "data+mechanism", "application+mechanism", "problem+mechanism"}:
        if reverse_validation_status == "natural_topic_in_source" or survives_without_scope:
            return "promising_natural_topic", "LLM 判断为自然小主题，且反向验证或去 scope 成立，暂不具备弱信号属性"

    # 子类型2：cross_source_mixed - 跨源但语义混杂
    if source_count >= 2 and source_semantic_consistency == "low":
        return "cross_source_mixed", "跨源但语义混杂：不同来源指向的主题不一致，需人工复核"

    # 子类型3：upper_topic_only - 仅停留在上层主题层面
    if reverse_validation_status == "only_upper_topic_in_source":
        return "upper_topic_only", "反向验证显示原文只包含上层主题表达，缺少细粒度线索"

    # 子类型4：compressed_label - 仍偏系统压缩标签
    if llm_judgment in {"compressed_label", "unclear"} or scope_shell_heavy:
        return "compressed_label", "当前命名仍偏系统压缩标签，需继续收口或人工复核"

    # 默认子类型
    if llm_judgment == "upper_topic":
        return "upper_topic_only", "LLM 判断为上层主题，需继续收口"

    return "promising_natural_topic", "已进入 hotspot 层，需进一步判断是否具备弱信号属性"


def _infer_object_hierarchy_tier(row):
    """
    显式化对象层级体系：scope_overview / scope_internal_candidate / natural_small_topic / weak_signal / failure_case
    这是 plan.md 第5条要求的核心实现。
    """
    # 初始化层级
    tier = "failure_case"
    tier_reason = ""

    # 第一层：scope_overview - 观察范围母主题
    if bool(row.get("is_observation_scope", False)):
        return "scope_overview", "观察范围母主题，不直接作为弱信号对象"

    # 检查是否被过滤
    if bool(row.get("is_scope_echo", False)):
        return "failure_case", "scope echo，仅停留在观察范围壳层"
    if bool(row.get("generic_core_only", False)):
        return "failure_case", "generic core only，缺少非 scope 约束"
    if not bool(row.get("has_mechanism_core", False)):
        return "failure_case", "缺少机制核"
    if not bool(row.get("has_non_scope_constraint", False)):
        return "failure_case", "缺少非 scope 约束"

    # 检查候选阶段
    candidate_stage = str(row.get("candidate_stage", "")).strip()
    if candidate_stage not in {"formed_candidate", "formed_candidate_strong"}:
        return "failure_case", f"候选阶段未成形: {candidate_stage}"

    # 第二层：scope_internal_candidate - scope 内细候选
    topic_granularity = str(row.get("topic_granularity", "")).strip()
    scope_shell_heavy = bool(row.get("scope_shell_heavy", False))
    survives_without_scope = bool(row.get("survives_without_scope", False))
    llm_judgment = str(row.get("llm_small_topic_judgment", "")).strip()

    if topic_granularity == "scope_internal_candidate" or scope_shell_heavy:
        if llm_judgment in {"upper_topic", "compressed_label", "unclear"}:
            return "scope_internal_candidate", "scope 内细候选，待继续收口"
        return "scope_internal_candidate", "scope 内细候选，命名和语义组织仍偏系统压缩标签"

    # 第三层：natural_small_topic - 自然小主题
    is_natural_small_topic = bool(row.get("is_natural_small_topic", False))
    if is_natural_small_topic:
        # 第四层：weak_signal - 弱信号
        weak_signal_readiness = str(row.get("weak_signal_readiness", "")).strip()
        if weak_signal_readiness == "weak_signal_ready":
            return "weak_signal", "具备弱信号属性：早期性、非主导性、验证支撑"
        return "natural_small_topic", "自然小主题，但暂不具备弱信号属性"

    # 默认返回 failure_case
    return "failure_case", "未能进入有效候选层"


def _infer_final_research_object_name(row):
    """
    统一确定最终研究对象名称，确保每个对象只有一个最终名字。

    优先级：
    1. topic_summary_name（LLM 收口后的自然名称）
    2. research_surface_name（反向验证后的自然名称）
    3. display_candidate_name（系统生成的展示名）

    同时返回名称来源，便于追溯。
    """
    topic_summary = _safe_text(row.get("topic_summary_name", ""))
    research_surface = _safe_text(row.get("research_surface_name", ""))
    display_name = _safe_text(row.get("display_candidate_name", ""))

    if topic_summary and topic_summary not in {"", "nan", "None"}:
        return topic_summary, "topic_summary_name"

    if research_surface and research_surface not in {"", "nan", "None"}:
        return research_surface, "research_surface_name"

    return display_name, "display_candidate_name"


def _infer_final_research_bucket(row):
    """
    最终研究层级分层：core_result / candidate_result / family_backbone / observation_pool / failure_case
    这是统一的结果口径，所有报告和输出以该字段为准。

    分层规则：
    1. core_result: 名字自然 + 粒度够小(total_mentions<20, evidence<20) + 验证够稳(source>=2, reverse=natural_topic)
    2. candidate_result: 指标符合但命名待收口，或单源但证据充足
    3. family_backbone: 过宽(mentions>=100 或 evidence>=50)，适合作为 family 主干
    4. observation_pool: 证据不足，需继续观察
    5. failure_case: 存在不确定性或异常
    """
    total_mentions = int(row.get("total_mentions", 0) or 0)
    evidence_count = int(row.get("cluster_evidence_count", 0) or 0)
    source_count = int(row.get("source_count", 0) or 0)
    reverse_status = str(row.get("reverse_validation_status", "")).strip()
    is_natural = bool(row.get("is_natural_small_topic", False))
    weak_readiness = str(row.get("weak_signal_readiness", "")).strip()
    signal_type = str(row.get("signal_type", "")).strip()
    llm_type = str(row.get("llm_small_topic_type", "")).strip()

    # 排除条件：不是 weak_signal
    if signal_type != "weak_signal":
        # 检查是否是 hotspot 或 scope_overview
        if signal_type == "hotspot":
            # hotspot 中证据过多的降级为 family_backbone
            if total_mentions >= 100 or evidence_count >= 50:
                return "family_backbone", "证据过多，更适合作为对象族主干"
            return "observation_pool", "热点对象，需继续观察"
        if signal_type == "scope_overview":
            return "family_backbone", "观察范围母主题，作为对象族主干"
        return "failure_case", "未进入弱信号候选层"

    # 条件1：family_backbone - 过宽
    if total_mentions >= 100 or evidence_count >= 50:
        return "family_backbone", f"证据过多(mentions={total_mentions}, evidence={evidence_count})，更适合作为对象族主干"

    # 条件2：排除 not_weak_signal
    if weak_readiness == "not_weak_signal":
        # 检查是否适合作为 family_backbone
        if total_mentions >= 50 or evidence_count >= 30:
            return "family_backbone", "不具备弱信号属性但证据充足，作为对象族主干"
        return "observation_pool", "不具备弱信号属性"

    # 条件3：排除 small_direction（方向级别）
    if llm_type == "small_direction":
        return "candidate_result", "方向级别对象，命名待收口"

    # 条件4：core_result - 三项条件均满足
    name_natural = is_natural or reverse_status == "natural_topic_in_source"
    granularity_ok = total_mentions < 20 and evidence_count < 20
    validation_ok = source_count >= 2 and reverse_status == "natural_topic_in_source"

    if name_natural and granularity_ok and validation_ok:
        return "core_result", "名字自然、粒度够小、验证够稳"

    # 条件5：candidate_result - 指标符合但某项不足
    if is_natural or reverse_status == "natural_topic_in_source":
        if not granularity_ok:
            return "candidate_result", "粒度偏大，待收口"
        if not validation_ok:
            return "candidate_result", "验证不够稳定，待继续观察"
        return "candidate_result", "指标符合，待命名收口"

    # 默认：observation_pool
    return "observation_pool", "需继续观察"


def _infer_foresight_significance(row):
    """
    预见意义解释层：说明对象为什么值得持续关注。
    不参与主评分，只作为解释层补充。
    """
    mechanism = str(row.get("mechanism_core", "")).strip().lower()
    scope_name = str(row.get("scope_name", "")).strip().lower()
    llm_type = str(row.get("llm_small_topic_type", "")).strip()
    llm_pattern = str(row.get("llm_small_topic_pattern", "")).strip()
    stage_hypothesis = str(row.get("stage_hypothesis", "")).strip()
    source_count = int(row.get("source_count", 0) or 0)
    patent_count = int(row.get("patent_count", 0) or 0)
    paper_count = int(row.get("paper_count", 0) or 0)
    news_count = int(row.get("news_count", 0) or 0)

    # 基于机制核判断可能的能力增强方向
    capability_enhancement = {
        "training": "可能提升模型学习效率或泛化能力",
        "planning": "可能增强复杂任务下的决策与路径规划能力",
        "control": "可能提升系统在动态环境中的稳定性和响应精度",
        "simulation": "可能降低真实场景测试成本，加速迭代验证",
        "reasoning": "可能增强系统对复杂因果关系的理解与推断能力",
        "memory": "可能提升长期任务中的信息保持与检索效率",
        "grounding": "可能增强系统与物理世界的交互落地能力",
        "alignment": "可能提升系统行为与人类意图的一致性",
        "retrieval": "可能提升知识获取与利用效率",
    }

    # 基于主题模板判断可能的应用落地方向
    application_direction = {
        "scene+mechanism": "更可能在特定场景（如导航、操控）中落地应用",
        "data+mechanism": "更可能在特定数据模态（如视频、轨迹）驱动下形成技术路线",
        "application+mechanism": "更可能直接对应具体应用需求",
        "problem+mechanism": "更可能针对特定技术痛点提供解决方案",
    }

    # 基于发展阶段判断技术路线演进
    stage_evolution = {
        "concept_proposal": "尚处于概念提出阶段，需关注后续实验验证进展",
        "experimental_validation": "已有实验验证支撑，需关注向原型或早期应用转化的信号",
        "early_application": "已出现早期应用或工程化线索，需关注规模化扩散与竞争格局",
        "scaled_application": "已进入规模化应用阶段，弱信号属性减弱，更适合作为热点追踪",
        "unclear": "发展阶段尚不明确，需持续观察证据积累",
    }

    significance_parts = []

    # 能力增强维度
    if mechanism in capability_enhancement:
        significance_parts.append(f"能力增强：{capability_enhancement[mechanism]}")

    # 应用落地维度
    if llm_pattern in application_direction:
        significance_parts.append(f"应用方向：{application_direction[llm_pattern]}")
    elif llm_type in {"small_application", "small_capability"}:
        significance_parts.append("应用方向：已具备较明确的应用或能力边界")

    # 技术路线演进维度
    if stage_hypothesis in stage_evolution:
        significance_parts.append(f"技术演进：{stage_evolution[stage_hypothesis]}")

    # 多源验证维度
    if source_count >= 2:
        if patent_count > 0 and paper_count > 0:
            significance_parts.append("验证支撑：论文与专利双重验证，技术路线较稳定")
        elif patent_count > 0:
            significance_parts.append("验证支撑：已有专利侧支撑，工程化潜力值得关注")
        elif paper_count > 0:
            significance_parts.append("验证支撑：已有论文侧验证，学术关注度较高")

    # 跨源扩散维度
    if news_count > 0 and (paper_count > 0 or patent_count > 0):
        significance_parts.append("扩散信号：学术/专利与资讯侧同时出现，可能正在从研究走向应用")

    if not significance_parts:
        return "当前证据不足以判断明确的前瞻意义，建议持续观察"

    return "；".join(significance_parts)


def _infer_weak_signal_readiness(row):
    """
    判断对象是否具备弱信号属性（早期性、非主导性、扩散不足、但已有增长和验证支撑）。
    这是第二层判断，在判断对象是否是自然小主题之后。
    """
    # 第一层：先判断是否是自然小主题
    llm_judgment = str(row.get("llm_small_topic_judgment", "")).strip()
    llm_pattern = str(row.get("llm_small_topic_pattern", "")).strip()
    llm_type = str(row.get("llm_small_topic_type", "")).strip()
    topic_granularity = str(row.get("topic_granularity", "")).strip()
    scope_shell_heavy = bool(row.get("scope_shell_heavy", False))
    survives_without_scope = bool(row.get("survives_without_scope", False))

    # 判断是否是自然小主题
    is_natural_small_topic = False
    natural_topic_reason = ""

    if llm_judgment == "small_topic" and llm_pattern in {"scene+mechanism", "data+mechanism", "application+mechanism", "problem+mechanism"}:
        is_natural_small_topic = True
        natural_topic_reason = "LLM判断为自然小主题，且主题模板明确"
    elif topic_granularity == "fine_grained_topic" and survives_without_scope and not scope_shell_heavy:
        is_natural_small_topic = True
        natural_topic_reason = "主题粒度细粒度且去scope可成立"
    elif llm_judgment == "small_topic":
        is_natural_small_topic = True
        natural_topic_reason = "LLM判断为自然小主题"

    if not is_natural_small_topic:
        return False, "not_natural_small_topic", natural_topic_reason or "当前对象尚未形成自然小主题表达"

    # 第二层：判断是否具备弱信号属性
    # 弱信号属性：早期性、非主导性、扩散不足、但已有增长和验证支撑
    total_mentions = int(row.get("total_mentions", 0) or 0)
    source_count = int(row.get("source_count", 0) or 0)
    growth_score = float(row.get("growth_score", 0.0) or 0.0)
    novelty_score = float(row.get("novelty_score", 0.0) or 0.0)
    time_validated = bool(row.get("time_validated", False))
    literature_validated = bool(row.get("literature_validated", False))
    patent_validated = bool(row.get("patent_validated", False))
    multi_source_validated = bool(row.get("multi_source_validated", False))
    cluster_evidence_count = int(row.get("cluster_evidence_count", 0) or 0)
    weak_signal_score = float(row.get("weak_signal_score", 0.0) or 0.0)

    # 早期性判断：总提及数不应过高
    is_early_stage = total_mentions <= 12
    # 非主导性判断：跨源数不应过多
    is_non_dominant = source_count <= 4
    # 扩散不足但有增长：增长分存在但不过高
    has_growth_signal = growth_score >= 0.5 and growth_score <= 3.0
    # 验证支撑：至少有一些验证
    has_validation_support = time_validated or literature_validated or patent_validated or multi_source_validated
    # 新颖性支撑
    has_novelty = novelty_score >= 1.0

    readiness_reasons = []
    readiness_blockers = []

    if not is_early_stage:
        readiness_blockers.append(f"提及数过高({total_mentions})，已进入主流视野")
    else:
        readiness_reasons.append("早期性：提及数适中")

    if not is_non_dominant:
        readiness_blockers.append(f"跨源数过高({source_count})，已形成主导地位")
    else:
        readiness_reasons.append("非主导性：跨源数适中")

    if has_growth_signal:
        readiness_reasons.append(f"增长信号：增长分={growth_score:.1f}")

    if has_validation_support:
        readiness_reasons.append("验证支撑：存在时间/文献/专利验证")

    if has_novelty:
        readiness_reasons.append(f"新颖性：新颖性分={novelty_score:.1f}")

    # 综合判断
    # 需要满足：早期性 + 非主导性 + (增长信号 或 验证支撑)
    weak_signal_ready = (
        is_early_stage
        and is_non_dominant
        and (has_growth_signal or has_validation_support)
        and weak_signal_score >= 6.0
        and cluster_evidence_count >= 2
    )

    if weak_signal_ready:
        reason = "具备弱信号属性：" + "；".join(readiness_reasons)
    else:
        if readiness_blockers:
            reason = "不具备弱信号属性：" + "；".join(readiness_blockers)
        else:
            reason = "暂不具备弱信号属性：" + "；".join(readiness_reasons) if readiness_reasons else "证据不足以判断弱信号属性"

    return weak_signal_ready, "weak_signal_ready" if weak_signal_ready else "not_weak_signal", reason


def _infer_stage_hypothesis(row):
    """
    推断技术发展阶段假设。

    阶段类型：
    - concept_proposal: 概念提出阶段
    - experimental_validation: 实验验证阶段
    - early_application: 早期应用阶段
    - scaled_application: 规模化应用阶段
    - unclear: 阶段不明确

    判断依据：
    - 证据文本中的关键词标记
    - 来源类型组合（news/paper/patent）
    - 证据簇大小和候选阶段
    """
    text = _joined_evidence_text(row)
    source_types = {str(item).strip().lower() for item in _safe_list(row.get("source_types", []))}
    stage = str(row.get("candidate_stage", "")).strip()
    total_mentions = int(row.get("total_mentions", 0) or 0)
    cluster_evidence_count = int(row.get("cluster_evidence_count", 0) or 0)

    # 规模化应用标记
    scaled_markers = [
        "deployment", "deployed", "field test", "pilot", "commercial",
        "customer", "production", "launch", "released", "release",
        "mass production", "industrial application", "commercialization",
    ]
    # 产品/平台标记
    product_markers = [
        "product", "platform", "industrial", "device", "system",
        "solution", "service", "offering",
    ]
    # 原型/框架标记
    prototype_markers = [
        "prototype", "open-source", "open source", "demo", "framework",
        "library", "toolkit", "demonstration", "proof of concept", "poc",
        "github", "codebase", "implementation",
    ]
    # 实验验证标记
    experiment_markers = [
        "benchmark", "dataset", "evaluation", "simulat", "real-world",
        "real world", "experiment", "ablation", "testbed", "sim2real",
        "sim-to-real", "empirical", "evaluation results", "experimental setup",
    ]
    # 概念提出标记
    concept_markers = [
        "toward", "towards", "survey", "frontier", "challenge",
        "perspective", "what do", "understanding", "world models and",
        "future work", "roadmap", "vision", "outlook",
    ]
    # 学术发表标记
    academic_markers = [
        "arxiv", "conference", "journal", "proceedings", "preprint",
        "paper", "publication", "cite", "citation",
    ]

    # 判断逻辑：从规模化应用到概念提出，逐层回退

    # 1. 规模化应用阶段
    if any(marker in text for marker in scaled_markers):
        if "patent" in source_types or "news" in source_types:
            return "scaled_application", "证据包含部署、平台或产品化线索，更像已开始规模化应用阶段。"
        return "early_application", "证据包含原型、试点或真实场景落地线索，更像早期应用阶段。"

    # 2. 产品/平台阶段
    if any(marker in text for marker in product_markers):
        if "patent" in source_types:
            return "early_application", "证据包含专利与产品/平台线索，更像已进入早期应用或工程化阶段。"
        return "early_application", "证据包含系统、平台或产品化实现线索，更像早期应用阶段。"

    # 3. 跨源组合判断（专利+论文/新闻）
    if "patent" in source_types and ("paper" in source_types or "news" in source_types) and total_mentions >= 2:
        return "early_application", "证据同时出现论文/新闻与专利，说明主题已从研究验证开始走向早期应用。"

    # 4. 专利+系统/装置
    if "patent" in source_types and any(marker in text for marker in ["platform", "system", "device", "apparatus", "method"]):
        return "early_application", "证据包含专利与系统/装置线索，更像已进入早期应用或工程化阶段。"

    # 5. 新闻+论文/专利+真实场景
    if "news" in source_types and ("paper" in source_types or "patent" in source_types) and any(
        marker in text for marker in ["real-world", "real world", "field", "pilot", "platform", "release", "deployment"]
    ):
        return "early_application", "证据包含研究与新闻扩散并伴随真实场景线索，更像早期应用阶段。"

    # 6. 原型/框架/开源
    if any(marker in text for marker in prototype_markers):
        return "early_application", "证据包含原型、框架或开源实现线索，更像早期应用阶段。"

    # 7. 实验验证阶段
    if any(marker in text for marker in experiment_markers):
        return "experimental_validation", "证据主要体现为 benchmark、dataset、simulation 或 evaluation，更像实验验证阶段。"

    # 8. 论文验证+实验线索
    if (
        "paper" in source_types
        and cluster_evidence_count >= 2
        and stage in {"formed_candidate", "formed_candidate_strong"}
        and any(marker in text for marker in ["simulation", "simulator", "benchmark", "dataset", "evaluation", "experiment"])
    ):
        return "experimental_validation", "证据以论文验证为主，且出现 simulation/benchmark/evaluation 线索，更像实验验证阶段。"

    # 9. 概念提出阶段
    if "paper" in source_types and any(marker in text for marker in concept_markers):
        return "concept_proposal", "证据更像概念提出、综述或方法讨论，尚未进入明确应用验证。"

    # 10. 纯论文+低提及数
    if source_types == {"paper"} and total_mentions <= 2 and not any(
        marker in text for marker in product_markers + prototype_markers + experiment_markers
    ):
        return "concept_proposal", "当前证据几乎只停留在论文提出层，缺少实验或落地线索，更像概念提出阶段。"

    # 11. 学术发表标记
    if any(marker in text for marker in academic_markers) and "paper" in source_types:
        if cluster_evidence_count >= 3:
            return "experimental_validation", "证据包含学术发表线索且证据簇较大，可能已进入实验验证阶段。"
        return "concept_proposal", "证据包含学术发表线索，更像概念提出或早期研究阶段。"

    return "unclear", "现有证据不足以稳定判断技术发展阶段。"


def _hotspot_score(row):
    return round((2.5 * row["source_count"]) + (1.2 * row["org_count"]) + row["total_mentions"] + row["growth_score"], 2)


def _weak_signal_score(row):
    return round(
        (2 * row["source_count"])
        + row["org_count"]
        + row["growth_score"]
        + row["novelty_score"]
        + row["validation_bonus"]
        - row["mainstream_penalty"],
        2,
    )


def _quality_adjusted_rank_score(row):
    base_score = _safe_float(
        row.get("weak_signal_score"),
        _safe_float(row.get("score"), _safe_float(row.get("hotspot_score"), 0.0)),
    )
    quality = _safe_float(row.get("candidate_evidence_quality"), 0.0)
    if quality <= 0:
        return round(base_score, 2)
    quality_factor = 0.7 + min(quality, 10.0) / 10.0 * 0.3
    return round(base_score * quality_factor, 2)


def _has_display_ready_name(row):
    display_name = str(row.get("display_candidate_name", row.get("tech_name", ""))).strip().lower()
    mechanism_core = str(row.get("mechanism_core", "")).strip().lower()
    if not display_name:
        return False
    if display_name == mechanism_core and mechanism_core in BARE_MECHANISM_CORES:
        return False
    return display_name not in DOMINANT_TECH_TERMS


def _contains_ascii_letters(text):
    return bool(re.search(r"[A-Za-z]", str(text or "")))


SCOPE_SHELL_TASK_TOKENS = {"agent", "embodied", "interactive", "control"}
SCOPE_SHELL_OBJECT_TOKENS = {"agent", "environment", "control", "memory", "policy"}
SCOPE_SHELL_DATA_TOKENS = {"simulation", "video", "visual"}
SCOPE_SHELL_METHOD_TOKENS = {"reinforcement", "policy-guided", "retrieval-based", "causal", "dynamic", "sparse"}
SCOPE_SURFACE_TERMS = {
    "world model",
    "embodied intelligence",
    "世界模型",
    "具身智能",
}


def _is_scope_shell_constraint(value, key):
    token = str(value or "").strip()
    return (
        (key == "task_constraint_tokens" and token in SCOPE_SHELL_TASK_TOKENS)
        or (key == "object_modifier_tokens" and token in SCOPE_SHELL_OBJECT_TOKENS)
        or (key == "data_modifier_tokens" and token in SCOPE_SHELL_DATA_TOKENS)
        or (key == "method_modifier_tokens" and token in SCOPE_SHELL_METHOD_TOKENS)
    )


def _scope_shell_profile(row):
    raw_count = int(row.get("non_scope_constraint_count", 0) or 0)
    raw_survive = bool(row.get("survives_without_scope", False))
    raw_heavy = bool(row.get("scope_shell_heavy", False))
    raw_reason = str(row.get("scope_shell_reason", "")).strip()
    if raw_count > 0 or raw_survive or raw_heavy or raw_reason:
        return {
            "non_scope_constraint_count": raw_count,
            "survives_without_scope": raw_survive,
            "scope_shell_heavy": raw_heavy,
            "scope_shell_reason": raw_reason,
        }

    pairs = []
    for key in ["task_constraint_tokens", "object_modifier_tokens", "data_modifier_tokens", "method_modifier_tokens"]:
        for value in _safe_list(row.get(key, [])):
            token = str(value).strip()
            if token:
                pairs.append((token, key))

    filtered = [(value, key) for value, key in pairs if value not in {"", str(row.get("mechanism_core", "")).strip()}]
    non_shell_pairs = [(value, key) for value, key in filtered if not _is_scope_shell_constraint(value, key)]
    non_scope_constraint_count = len({value for value, _ in non_shell_pairs})
    has_anchor = any(key in {"task_constraint_tokens", "object_modifier_tokens", "data_modifier_tokens"} for _, key in non_shell_pairs)
    survives_without_scope = bool(non_scope_constraint_count >= 2 and has_anchor)
    scope_shell_heavy = False
    scope_shell_reason = ""
    if bool(row.get("has_mechanism_core", False)) and bool(row.get("has_non_scope_constraint", False)):
        if not survives_without_scope:
            scope_shell_heavy = True
            if non_scope_constraint_count < 2:
                scope_shell_reason = "non_scope_constraints_insufficient"
            elif not has_anchor:
                scope_shell_reason = "constraints_too_generic_without_scope"
            else:
                scope_shell_reason = "scope_shell_dependency"
    return {
        "non_scope_constraint_count": non_scope_constraint_count,
        "survives_without_scope": survives_without_scope,
        "scope_shell_heavy": scope_shell_heavy,
        "scope_shell_reason": scope_shell_reason,
    }


def _resolved_topic_granularity(row):
    raw = str(row.get("topic_granularity", "")).strip()
    llm_judgment = str(row.get("llm_small_topic_judgment", "")).strip()
    llm_pattern = str(row.get("llm_small_topic_pattern", "")).strip()
    if llm_judgment == "small_topic":
        if llm_pattern in {"scene+mechanism", "data+mechanism", "application+mechanism", "problem+mechanism"}:
            return "fine_grained_topic"
        if bool(row.get("scope_shell_heavy", False)):
            return "scope_internal_candidate"
    if raw in {"fine_grained_topic", "scope_internal_candidate", "generic_or_failed"}:
        if bool(row.get("scope_shell_heavy", False)) and raw == "fine_grained_topic":
            return "scope_internal_candidate"
        return raw
    if bool(row.get("is_observation_scope", False)):
        return "generic_or_failed"
    if bool(row.get("is_scope_echo", False)) or bool(row.get("generic_core_only", False)):
        return "generic_or_failed"
    if bool(row.get("has_non_scope_constraint", False)) and bool(row.get("has_mechanism_core", False)):
        profile = _scope_shell_profile(row)
        if bool(profile["survives_without_scope"]) or int(profile["non_scope_constraint_count"]) >= 3:
            return "fine_grained_topic"
        return "scope_internal_candidate"
    return "generic_or_failed"


def _resolved_display_tier(row):
    raw = str(row.get("display_tier", "")).strip()
    llm_judgment = str(row.get("llm_small_topic_judgment", "")).strip()
    llm_pattern = str(row.get("llm_small_topic_pattern", "")).strip()
    if llm_judgment == "small_topic" and llm_pattern in {"scene+mechanism", "data+mechanism", "application+mechanism", "problem+mechanism"}:
        return "weak_signal"
    if llm_judgment in {"upper_topic", "compressed_label"}:
        return "hotspot"
    if raw in {"weak_signal", "hotspot", "demo", "manual_review"}:
        if raw == "weak_signal" and bool(row.get("scope_shell_heavy", False)):
            return "hotspot"
        return raw
    granularity = _resolved_topic_granularity(row)
    if bool(row.get("is_observation_scope", False)):
        return "demo"
    if granularity == "fine_grained_topic":
        return "weak_signal"
    if granularity == "scope_internal_candidate":
        return "hotspot"
    return "manual_review"


def _load_demo_rules(config_path=None):
    default_rules = {
        "whitelist_mechanism_cores": ["memory", "planning", "control", "simulation", "retrieval", "compression", "alignment"],
        "whitelist_constraints": ["robot", "manipulation", "navigation", "video", "multimodal", "causal", "dynamic", "retrieval-based", "embodied", "agent"],
        "blacklist_generic_names": ["planning", "training", "memory", "control", "simulation"],
        "display_name_overrides": {},
    }
    if not config_path:
        return default_rules
    path = Path(config_path)
    if not path.exists():
        return default_rules
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return default_rules
    try:
        loaded = json.loads(raw)
    except Exception:
        try:
            import yaml  # type: ignore
        except Exception:
            return default_rules
        loaded = yaml.safe_load(raw) or {}
    if isinstance(loaded, dict):
        default_rules.update(loaded)
    return default_rules


def _apply_demo_display_override(display_name, canonical_name, aliases, overrides):
    candidates = [str(display_name or "").strip(), str(canonical_name or "").strip()]
    candidates.extend(str(item).strip() for item in _safe_list(aliases))
    for key in candidates:
        if key and key in overrides:
            return overrides[key]
    return str(display_name or "").strip()


def _prepare_scored_candidates(candidates_df):
    if candidates_df.empty:
        return candidates_df.copy()
    scored_df = candidates_df.copy()
    # 安全地处理 source_types 列
    if "source_types" in scored_df.columns:
        scored_df["source_types"] = scored_df["source_types"].apply(_safe_list)
    else:
        # 从 source_type 列创建 source_types
        if "source_type" in scored_df.columns:
            scored_df["source_types"] = scored_df["source_type"].apply(
                lambda x: [_safe_text(x)] if _safe_text(x) else []
            )
        else:
            scored_df["source_types"] = [[] for _ in range(len(scored_df))]
    
    # 安全地处理 evidence_items 列
    if "evidence_items" in scored_df.columns:
        scored_df["evidence_items"] = scored_df["evidence_items"].apply(_safe_evidence_items)
    else:
        scored_df["evidence_items"] = [[] for _ in range(len(scored_df))]
    if "mention_dates" in scored_df.columns:
        scored_df["mention_dates"] = scored_df["mention_dates"].apply(_safe_list)
    else:
        scored_df["mention_dates"] = [[] for _ in range(len(scored_df))]
    defaults = {
        "total_mentions": 0,
        "source_count": 0,
        "org_count": 0,
        "demo_score": 0.0,
        "tech_name": "",
        "weak_signal_event_ratio": 0.0,
        "candidate_evidence_quality": 0.0,
        "candidate_core_evidence_quality": 0.0,
        "low_quality_evidence_ratio": 0.0,
        "high_quality_evidence_count": 0,
        "quality_risk_flag": "",
        "quality_adjusted_rank_score": 0.0,
        "candidate_evidence_quality_reason": "",
        "multi_source_validated": False,
        "multi_source_validation_score": 0.0,
        "semantic_validated": False,
        "semantic_validation_score": 0.0,
        "event_doc_semantic_validated": False,
        "event_doc_validation_score": 0.0,
        "event_paper_semantic_score": 0.0,
        "event_patent_semantic_score": 0.0,
        "time_validated": False,
        "weak_signal_focus_candidate": False,
        "is_observation_scope": False,
        "is_scope_internal_candidate": False,
        "is_scope_echo": False,
        "has_mechanism_core": False,
        "has_task_constraint": False,
        "has_non_scope_constraint": False,
        "candidate_stage": "",
        "generic_core_only": False,
        "alias_count": 0,
        "cluster_evidence_count": 0,
        "template_variant_count": 0,
        "topic_granularity": "",
        "display_tier": "",
        "non_scope_constraint_count": 0,
        "survives_without_scope": False,
        "scope_shell_heavy": False,
        "scope_shell_reason": "",
        "internal_candidate_label": "",
        "topic_summary_name": "",
        "topic_naturalness_reason": "",
        "raw_phrase_example": "",
        "llm_refined_topic_name": "",
        "llm_small_topic_judgment": "",
        "llm_small_topic_type": "",
        "llm_small_topic_pattern": "",
        "llm_small_topic_reason": "",
        "llm_refined": False,
        "topic_refiner_active": False,
        "rule_display_candidate_name": "",
        "rule_topic_summary_name": "",
        "reverse_validation_status": "",
        "source_semantic_consistency": "",
        "needs_surface_rewrite": False,
        "reverse_validation_note": "",
        "raw_candidate_texts": "",
        "raw_phrases": "",
        "relation_phrases": "",
        "suggested_label": "",
        "human_judgment": "",
        "review_notes": "",
        "baseline_rank": "",
        "appears_in_frequency_baseline": False,
        "comparison_note": "",
        "stage_hypothesis": "",
        "stage_hypothesis_reason": "",
        "is_natural_small_topic": False,
        "natural_small_topic_reason": "",
        "weak_signal_readiness": "",
        "weak_signal_readiness_reason": "",
        "foresight_significance": "",
        "object_hierarchy_tier": "",
        "object_hierarchy_reason": "",
        "failure_case_type": "",
        "failure_case_reason": "",
        # 新增：内部标签与最终表述分离
        "raw_phrase_cluster": "",
        "final_research_object_name": "",
        "final_research_object_name_source": "",  # 追踪最终名称来源
        # 新增：hotspot 子类型
        "hotspot_subtype": "",
        "hotspot_subtype_reason": "",
        # 新增：证据强度分数
        "evidence_strength_score": 0.0,
        # 新增：弱信号二次分层
        "weak_signal_tier": "",
        "weak_signal_tier_reason": "",
        # 新增：candidate_weak_signal 内部排序
        "candidate_stability": "",
        "candidate_stability_reason": "",
    }
    missing_defaults = {
        column: default
        for column, default in defaults.items()
        if column not in scored_df.columns
    }
    if missing_defaults:
        scored_df = pd.concat(
            [
                scored_df,
                pd.DataFrame(
                    {column: [default] * len(scored_df) for column, default in missing_defaults.items()},
                    index=scored_df.index,
                ),
            ],
            axis=1,
        )

    numeric_defaults = {
        column: default
        for column, default in defaults.items()
        if isinstance(default, (int, float)) and not isinstance(default, bool)
    }
    numeric_defaults.update(
        {
            "news_count": 0,
            "paper_count": 0,
            "report_count": 0,
            "patent_count": 0,
            "time_validation_ratio": 0.0,
            "traceable_ratio": 0.0,
            "growth_score": 0.0,
            "novelty_score": 0.0,
            "mainstream_penalty": 0.0,
            "validation_bonus": 0.0,
            "weak_signal_score": 0.0,
            "hotspot_score": 0.0,
            "evidence_count": 0,
            "rank": 0,
        }
    )
    bool_defaults = {
        column: default
        for column, default in defaults.items()
        if isinstance(default, bool)
    }
    bool_defaults.update(
        {
            "literature_validated": False,
            "patent_validated": False,
            "is_natural_small_topic": False,
        }
    )
    missing_numeric_defaults = {
        column: default
        for column, default in numeric_defaults.items()
        if column not in scored_df.columns
    }
    if missing_numeric_defaults:
        scored_df = pd.concat(
            [
                scored_df,
                pd.DataFrame(
                    {column: [default] * len(scored_df) for column, default in missing_numeric_defaults.items()},
                    index=scored_df.index,
                ),
            ],
            axis=1,
        )

    missing_bool_defaults = {
        column: default
        for column, default in bool_defaults.items()
        if column not in scored_df.columns
    }
    if missing_bool_defaults:
        scored_df = pd.concat(
            [
                scored_df,
                pd.DataFrame(
                    {column: [default] * len(scored_df) for column, default in missing_bool_defaults.items()},
                    index=scored_df.index,
                ),
            ],
            axis=1,
        )

    scored_df = scored_df.copy()
    for column, default in numeric_defaults.items():
        if isinstance(default, int):
            scored_df[column] = scored_df[column].apply(lambda value: _safe_int(value, default))
        else:
            scored_df[column] = scored_df[column].apply(lambda value: _safe_float(value, default))
    for column, default in bool_defaults.items():
        scored_df[column] = scored_df[column].apply(_safe_bool)

    scored_df["growth_score"] = scored_df["mention_dates"].apply(_calculate_growth_score)
    scored_df["novelty_score"] = scored_df.apply(_calculate_novelty_score, axis=1)
    scored_df["mainstream_penalty"] = scored_df.apply(
        lambda row: _calculate_mainstream_penalty(row["total_mentions"], row["source_count"]),
        axis=1,
    )
    scored_df["validation_bonus"] = scored_df.apply(_calculate_validation_bonus, axis=1)
    scored_df["weak_signal_score"] = scored_df.apply(_weak_signal_score, axis=1)
    scored_df["hotspot_score"] = scored_df.apply(_hotspot_score, axis=1)
    scored_df["quality_adjusted_rank_score"] = scored_df.apply(_quality_adjusted_rank_score, axis=1)
    scored_df["source_spread"] = scored_df["source_types"].apply(lambda items: "/".join(items))
    scored_df["evidence_count"] = scored_df["evidence_items"].apply(len)
    scored_df["raw_phrase_example"] = scored_df["evidence_items"].apply(
        lambda items: items[0].get("raw_candidate_text", "") if items else ""
    )
    scored_df["topic_granularity"] = scored_df.apply(_resolved_topic_granularity, axis=1)
    scored_df["display_tier"] = scored_df.apply(_resolved_display_tier, axis=1)
    stage_profiles = scored_df.apply(_infer_stage_hypothesis, axis=1)
    scored_df["stage_hypothesis"] = [item[0] for item in stage_profiles]
    scored_df["stage_hypothesis_reason"] = [item[1] for item in stage_profiles]
    profiles = scored_df.apply(_scope_shell_profile, axis=1)
    scored_df["non_scope_constraint_count"] = [int(item["non_scope_constraint_count"]) for item in profiles]
    scored_df["survives_without_scope"] = [bool(item["survives_without_scope"]) for item in profiles]
    scored_df["scope_shell_heavy"] = [bool(item["scope_shell_heavy"]) for item in profiles]
    scored_df["scope_shell_reason"] = [str(item["scope_shell_reason"]) for item in profiles]
    # 新增：自然小主题与弱信号两层判断
    weak_signal_readiness_profiles = scored_df.apply(_infer_weak_signal_readiness, axis=1)
    scored_df["is_natural_small_topic"] = [item[0] for item in weak_signal_readiness_profiles]
    scored_df["weak_signal_readiness"] = [item[1] for item in weak_signal_readiness_profiles]
    scored_df["weak_signal_readiness_reason"] = [item[2] for item in weak_signal_readiness_profiles]
    scored_df["natural_small_topic_reason"] = scored_df.apply(
        lambda row: row.get("weak_signal_readiness_reason", "").split("。")[0] if row.get("is_natural_small_topic", False) else "",
        axis=1,
    )
    # 新增：预见意义解释层
    scored_df["foresight_significance"] = scored_df.apply(_infer_foresight_significance, axis=1)
    # 新增：对象层级显式化
    hierarchy_profiles = scored_df.apply(_infer_object_hierarchy_tier, axis=1)
    scored_df["object_hierarchy_tier"] = [item[0] for item in hierarchy_profiles]
    scored_df["object_hierarchy_reason"] = [item[1] for item in hierarchy_profiles]
    # 新增：失败案例类型化
    failure_profiles = scored_df.apply(_infer_failure_case_type, axis=1)
    scored_df["failure_case_type"] = [item[0] for item in failure_profiles]
    scored_df["failure_case_reason"] = [item[1] for item in failure_profiles]
    # 新增：hotspot 子类型判断
    hotspot_subtype_profiles = scored_df.apply(_infer_hotspot_subtype, axis=1)
    scored_df["hotspot_subtype"] = [item[0] for item in hotspot_subtype_profiles]
    scored_df["hotspot_subtype_reason"] = [item[1] for item in hotspot_subtype_profiles]
    # 新增：证据强度分数
    scored_df["evidence_strength_score"] = scored_df.apply(_calculate_evidence_strength_score, axis=1)
    # 新增：统一最终研究对象名称
    final_name_profiles = scored_df.apply(_infer_final_research_object_name, axis=1)
    scored_df["final_research_object_name"] = [item[0] for item in final_name_profiles]
    scored_df["final_research_object_name_source"] = [item[1] for item in final_name_profiles]
    # 新增：弱信号二次分层（需要在 signal_type 确定之后，所以这里先不调用）
    # 将在 score_signals 和 score_all_candidates 中调用
    return scored_df


def _research_signal_type(row):
    tech_name = str(row.get("display_candidate_name", row.get("tech_name", ""))).strip().lower()
    mechanism_core = str(row.get("mechanism_core", "")).strip()
    candidate_stage = str(row.get("candidate_stage", "")).strip()
    is_strong = candidate_stage == "formed_candidate_strong"
    topic_granularity = _resolved_topic_granularity(row)
    display_tier = _resolved_display_tier(row)
    scope_shell_profile = _scope_shell_profile(row)
    survives_without_scope = bool(scope_shell_profile["survives_without_scope"])
    non_scope_constraint_count = int(scope_shell_profile["non_scope_constraint_count"])
    scope_shell_heavy = bool(scope_shell_profile["scope_shell_heavy"])
    has_anchor_constraint = any(
        str(token).strip() and not _is_scope_shell_constraint(str(token).strip(), key)
        for key in ["task_constraint_tokens", "object_modifier_tokens", "data_modifier_tokens"]
        for token in _safe_list(row.get(key, []))
    )
    strict_object_gate = survives_without_scope or (
        non_scope_constraint_count >= 2
        and has_anchor_constraint
    )
    llm_judgment = str(row.get("llm_small_topic_judgment", "")).strip()
    llm_pattern = str(row.get("llm_small_topic_pattern", "")).strip()
    llm_type = str(row.get("llm_small_topic_type", "")).strip()
    topic_refiner_active = bool(row.get("topic_refiner_active", False))
    llm_supports_small_topic = (
        llm_judgment == "small_topic"
        and llm_pattern in {"scene+mechanism", "data+mechanism", "application+mechanism", "problem+mechanism"}
    )
    llm_supports_final_weak_signal = llm_supports_small_topic and llm_type in {
        "small_application",
        "small_capability",
        "small_problem",
    }
    # 使用 evidence_strength_score 替代 source_count 例外逻辑
    evidence_strength_score = float(row.get("evidence_strength_score", 0.0) or 0.0)
    llm_supports_directional_weak_signal = (
        llm_supports_small_topic
        and llm_type == "small_direction"
        and row["source_count"] >= 2  # 保持原有逻辑
        and survives_without_scope
        and not scope_shell_heavy
        and row["cluster_evidence_count"] >= 2
    )
    display_name = str(row.get("display_candidate_name", "")).strip()
    has_scope_surface_in_name = any(term in display_name for term in SCOPE_SURFACE_TERMS)
    # 保持原有的单源例外逻辑，不使用 evidence_strength_score
    llm_supports_single_source_exception = (
        llm_supports_small_topic
        and llm_type in {"small_application", "small_capability", "small_problem"}
        and row["source_count"] == 1
        and row["cluster_evidence_count"] >= 3
        and row["total_mentions"] >= 3
        and bool(row.get("time_validated", False))
        and row["weak_signal_score"] >= 10
        and survives_without_scope
        and not scope_shell_heavy
        and not has_scope_surface_in_name
    )
    llm_final_gate = (
        llm_supports_final_weak_signal
        or llm_supports_directional_weak_signal
        or llm_supports_single_source_exception
    )
    reverse_validation_status = str(row.get("reverse_validation_status", "")).strip()
    source_semantic_consistency = str(row.get("source_semantic_consistency", "")).strip()
    reverse_validation_backpressure = (
        reverse_validation_status in {"mixed_or_unclear", "only_upper_topic_in_source"}
        and source_semantic_consistency == "low"
    )
    if bool(row.get("is_observation_scope", False)) or tech_name in OBSERVATION_SCOPE_SET:
        return "scope_overview"
    if not bool(row.get("is_scope_internal_candidate", False)):
        return "other"
    if bool(row.get("is_scope_echo", False)) or not bool(row.get("has_mechanism_core", False)) or not bool(row.get("has_non_scope_constraint", False)):
        return "other"
    if bool(row.get("generic_core_only", False)) or mechanism_core in BARE_MECHANISM_CORES and str(row.get("display_candidate_name", "")).strip().lower() == mechanism_core:
        return "other"
    if tech_name in DOMINANT_TECH_TERMS or not _has_display_ready_name(row):
        return "other"
    if topic_granularity == "generic_or_failed":
        return "other"
    if topic_refiner_active and row["source_count"] >= 2 and not llm_supports_small_topic and topic_granularity == "fine_grained_topic":
        return "hotspot"
    if (
        topic_granularity == "fine_grained_topic"
        and display_tier == "weak_signal"
        and (
            llm_final_gate
            if topic_refiner_active
            else (is_strong or llm_final_gate)
        )
        and (strict_object_gate or llm_final_gate)
        and (not scope_shell_heavy or llm_final_gate)
        and (
            row["source_count"] >= 2
            or llm_supports_single_source_exception
        )
        and row["weak_signal_score"] >= 6
        and row["cluster_evidence_count"] >= 2
        and (row["novelty_score"] >= 1.5 or row["growth_score"] >= 1.5 or bool(row.get("time_validated", False)))
        and not reverse_validation_backpressure
    ):
        return "weak_signal"
    if (
        topic_granularity in {"fine_grained_topic", "scope_internal_candidate"}
        and display_tier in {"hotspot", "weak_signal"}
        and candidate_stage in {"formed_candidate", "formed_candidate_strong"}
        and (scope_shell_heavy or topic_granularity == "scope_internal_candidate" or not strict_object_gate)
        and row["hotspot_score"] >= 8
        and row["cluster_evidence_count"] >= 2
        and row["source_count"] >= 2
        and (bool(row.get("has_task_constraint", False)) or len(_safe_list(row.get("object_modifier_tokens", []))) > 0 or len(_safe_list(row.get("data_modifier_tokens", []))) > 0)
    ):
        return "hotspot"
    return "other"


def score_demo_signals(candidates_df, demo_config=None):
    rules = _load_demo_rules(demo_config)
    scored_df = _prepare_scored_candidates(candidates_df)
    if scored_df.empty:
        return scored_df
    blacklist = {item.lower() for item in rules.get("blacklist_generic_names", [])}
    whitelist_cores = set(rules.get("whitelist_mechanism_cores", []))
    whitelist_constraints = set(rules.get("whitelist_constraints", []))
    overrides = {str(k).strip(): str(v).strip() for k, v in rules.get("display_name_overrides", {}).items()}
    scores = []
    for _, row in scored_df.iterrows():
        display_name = str(row.get("display_candidate_name", row.get("tech_name", ""))).strip()
        canonical_name = str(row.get("canonical_candidate_name_en", "")).strip()
        aliases = _safe_list(row.get("display_candidate_aliases", []))
        mechanism_core = str(row.get("mechanism_core", "")).strip()
        constraint_tokens = set(
            _safe_list(row.get("task_constraint_tokens", []))
            + _safe_list(row.get("object_modifier_tokens", []))
            + _safe_list(row.get("data_modifier_tokens", []))
            + _safe_list(row.get("method_modifier_tokens", []))
        )
        display_name = _apply_demo_display_override(display_name, canonical_name, aliases, overrides)
        object_score = 1 if mechanism_core else 0
        object_score += 2 if bool(row.get("has_non_scope_constraint", False)) else 0
        object_score += 2 if display_name and display_name.lower() not in blacklist and display_name.lower() != mechanism_core else 0
        early_score = int(row.get("total_mentions", 0) <= 6) + int(bool(row.get("time_validated", False))) + int(row.get("mainstream_penalty", 0) <= 1.5)
        multi_source_score = 0 if row["source_count"] <= 1 else 1 if row["source_count"] == 2 else 2
        explainable_score = 2 if display_name and any(token in whitelist_constraints for token in constraint_tokens) else 0
        penalty = 0
        if bool(row.get("is_scope_echo", False)):
            penalty -= 999
        if display_name.lower() in blacklist or bool(row.get("generic_core_only", False)):
            penalty -= 3
        if any(token in display_name.lower() for token in ["policy", "benchmark", "evaluation"]):
            penalty -= 4
        if len(canonical_name.split()) >= 6:
            penalty -= 2
        if int(row.get("source_count", 0)) <= 1:
            penalty -= 1
        if int(row.get("cluster_evidence_count", 0)) <= 1:
            penalty -= 1
        whitelist_bonus = 1 if mechanism_core in whitelist_cores else 0
        demo_score = object_score + early_score + multi_source_score + explainable_score + whitelist_bonus + penalty
        scores.append((display_name, demo_score))
    scored_df["display_candidate_name"] = [
        _apply_demo_display_override(
            name,
            canonical,
            aliases,
            overrides,
        )
        for name, canonical, aliases in zip(
            scored_df["display_candidate_name"].tolist(),
            scored_df.get("canonical_candidate_name_en", pd.Series([""] * len(scored_df))).tolist(),
            scored_df.get("display_candidate_aliases", pd.Series([[] for _ in range(len(scored_df))])).tolist(),
        )
    ]
    scored_df["demo_score"] = [score for _, score in scores]
    scored_df["signal_type"] = scored_df.apply(
        lambda row: "scope_overview"
        if bool(row.get("is_observation_scope", False))
        else "demo_signal"
        if row["demo_score"] >= 4 and not bool(row.get("is_scope_echo", False)) and not bool(row.get("generic_core_only", False))
        else "other",
        axis=1,
    )
    scored_df["display_tier"] = "demo"
    scored_df["score"] = scored_df["demo_score"]
    scored_df["explanation"] = scored_df.apply(build_signal_explanation, axis=1)
    ranked = scored_df[scored_df["signal_type"] == "demo_signal"].sort_values(
        by=["demo_score", "cluster_evidence_count", "source_count", "total_mentions"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    if "display_candidate_name" in ranked.columns:
        ranked = ranked.drop_duplicates(subset=["display_candidate_name"], keep="first").reset_index(drop=True)
    diversified_rows = []
    mechanism_limits = {}
    for _, row in ranked.iterrows():
        mechanism_core = str(row.get("mechanism_core", "")).strip().lower()
        if mechanism_core:
            current = mechanism_limits.get(mechanism_core, 0)
            if current >= 2:
                continue
            mechanism_limits[mechanism_core] = current + 1
        diversified_rows.append(row.to_dict())
        if len(diversified_rows) >= 5:
            break
    ranked = pd.DataFrame(diversified_rows, columns=ranked.columns) if diversified_rows else ranked.head(0).copy()
    ranked = ranked[ranked["demo_score"] >= 2].reset_index(drop=True)
    if len(ranked) > 5:
        ranked = ranked.head(5).copy()
    ranked["rank"] = range(1, len(ranked) + 1)
    return ranked


def build_manual_review_table(hotspot_df, weak_signal_df, top_k=10):
    review_columns = [
        "review_rank", "signal_bucket", "display_candidate_name", "scope_name",
        "mechanism_core", "constraint_signature", "source_count", "total_mentions",
        "literature_validated", "patent_validated", "time_validated",
        "stage_hypothesis", "stage_hypothesis_reason",
        "raw_phrase_example", "internal_candidate_label", "rule_display_candidate_name", "final_research_object_name", "topic_naturalness_reason",
        "llm_small_topic_judgment", "llm_small_topic_type", "llm_small_topic_pattern", "llm_small_topic_reason",
        "topic_granularity", "naming_quality", "is_patent_backed",
        "non_scope_constraint_count", "survives_without_scope", "scope_shell_heavy", "scope_shell_reason",
        "should_go_to_failure_case_section",
        "suggested_label", "why_flagged", "human_judgment", "review_notes",
    ]
    frames = []
    for bucket_name, frame in [("weak_signal", weak_signal_df), ("hotspot", hotspot_df)]:
        if frame is None or frame.empty:
            continue
        temp = frame.copy()
        temp["signal_bucket"] = bucket_name
        frames.append(temp)
    if not frames:
        return pd.DataFrame(columns=review_columns)
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sort_values(
        by=["signal_bucket", "weak_signal_score", "hotspot_score", "source_count", "total_mentions"],
        ascending=[True, False, False, False, False],
    ).drop_duplicates(subset=["display_candidate_name"], keep="first").head(top_k).copy()

    merged["topic_granularity"] = merged.apply(_resolved_topic_granularity, axis=1)
    shell_profiles = merged.apply(_scope_shell_profile, axis=1)
    merged["non_scope_constraint_count"] = [int(item["non_scope_constraint_count"]) for item in shell_profiles]
    merged["survives_without_scope"] = [bool(item["survives_without_scope"]) for item in shell_profiles]
    merged["scope_shell_heavy"] = [bool(item["scope_shell_heavy"]) for item in shell_profiles]
    merged["scope_shell_reason"] = [str(item["scope_shell_reason"]) for item in shell_profiles]

    def _naming_quality(row):
        display_name = str(row.get("display_candidate_name", "")).strip()
        if not display_name:
            return "missing"
        if _contains_ascii_letters(display_name):
            return "mixed_language"
        if any(shell in display_name for shell in ["驱动", "机制", "范式", "体系"]):
            return "upper_topic_style"
        if len(display_name) > 14:
            return "too_long"
        return "natural"

    merged["naming_quality"] = merged.apply(_naming_quality, axis=1)
    merged["is_patent_backed"] = merged.apply(
        lambda row: bool(row.get("patent_validated", False)) or int(row.get("patent_count", 0) or 0) > 0,
        axis=1,
    )
    merged["raw_phrase_example"] = merged.apply(
        lambda row: (
            _safe_evidence_items(row.get("evidence_items", []))[0].get("raw_candidate_text", "")
            if _safe_evidence_items(row.get("evidence_items", []))
            else ""
        ),
        axis=1,
    )
    merged["internal_candidate_label"] = merged.apply(
        lambda row: str(row.get("internal_candidate_label", "")).strip()
        or f"canonical={str(row.get('canonical_candidate_name_en', '')).strip()} || constraints={str(row.get('constraint_signature', '')).strip()}",
        axis=1,
    )
    merged["rule_display_candidate_name"] = merged.apply(
        lambda row: str(row.get("rule_display_candidate_name", "")).strip() or str(row.get("display_candidate_name", "")).strip(),
        axis=1,
    )
    merged["final_research_object_name"] = merged.apply(
        lambda row: str(row.get("topic_summary_name", "")).strip() or str(row.get("display_candidate_name", "")).strip(),
        axis=1,
    )
    merged["topic_naturalness_reason"] = merged.apply(
        lambda row: str(row.get("topic_naturalness_reason", "")).strip()
        or ("已形成 scope 内细粒度候选，但当前命名和语义组织仍偏系统压缩标签，建议继续收口" if bool(row.get("scope_shell_heavy", False)) else "主题表达可用"),
        axis=1,
    )
    merged["llm_small_topic_judgment"] = merged.apply(
        lambda row: str(row.get("llm_small_topic_judgment", "")).strip(),
        axis=1,
    )
    merged["llm_small_topic_reason"] = merged.apply(
        lambda row: str(row.get("llm_small_topic_reason", "")).strip() or str(row.get("topic_naturalness_reason", "")).strip(),
        axis=1,
    )
    merged["llm_small_topic_type"] = merged.apply(
        lambda row: str(row.get("llm_small_topic_type", "")).strip(),
        axis=1,
    )
    merged["llm_small_topic_pattern"] = merged.apply(
        lambda row: str(row.get("llm_small_topic_pattern", "")).strip(),
        axis=1,
    )

    def _label_row(row):
        display_name = str(row.get("display_candidate_name", "")).strip()
        topic_granularity = str(row.get("topic_granularity", "")).strip()
        naming_quality = str(row.get("naming_quality", "")).strip()
        is_patent_backed = bool(row.get("is_patent_backed", False))
        scope_shell_heavy = bool(row.get("scope_shell_heavy", False))
        survives_without_scope = bool(row.get("survives_without_scope", False))
        llm_judgment = str(row.get("llm_small_topic_judgment", "")).strip()
        if bool(row.get("generic_core_only", False)) or not bool(row.get("has_non_scope_constraint", False)):
            return "偏方法标签", "缺少稳定的非 scope 约束", True
        if llm_judgment == "small_topic":
            return "较像弱信号", "LLM 判断该对象已经较接近自然小主题表达，可优先进入研究关注列表", False
        if llm_judgment == "compressed_label":
            return "待进一步收口", "LLM 判断当前对象仍主要是系统压缩标签，建议继续收口", True
        if llm_judgment == "unclear":
            return "待进一步收口", "LLM 判断当前命名仍偏泛化或系统压缩标签，建议继续收口", True
        if llm_judgment == "upper_topic":
            return "scope内细候选", "LLM 判断该对象已进入 scope 内细候选层，但当前仍更适合作为待人工确认对象", True
        if scope_shell_heavy and not survives_without_scope:
            return "待进一步收口", "该对象已经形成 scope 内细粒度候选，但当前仍偏系统压缩标签，建议继续收口或人工复核", True
        if topic_granularity == "scope_internal_candidate":
            return "scope内细候选", "该对象已进入 scope 内细粒度候选层，但当前更适合作为待继续收口的研究对象", True
        if naming_quality in {"mixed_language", "too_long"}:
            return "待人工确认", "命名自然性不足，建议作为待人工确认对象复核", True
        if str(row.get("candidate_stage", "")).strip() != "formed_candidate_strong":
            return "待人工确认", "已成形但尚未进入 strong", False
        if topic_granularity == "fine_grained_topic" and (is_patent_backed or bool(row.get("literature_validated", False))):
            return "较像弱信号", "已形成较自然的小主题表达，且存在跨源或文献/专利证据支撑", False
        return "待人工确认", "当前仍主要依赖文本聚合，验证支撑较弱", True

    labels = merged.apply(_label_row, axis=1)
    merged["suggested_label"] = [item[0] for item in labels]
    merged["why_flagged"] = [item[1] for item in labels]
    merged["should_go_to_failure_case_section"] = [item[2] for item in labels]
    merged["human_judgment"] = ""
    merged["review_notes"] = ""
    merged["review_rank"] = range(1, len(merged) + 1)
    return merged[review_columns]


def build_llm_small_topic_comparison_table(scored_df, top_k=15):
    columns = [
        "comparison_rank",
        "signal_type",
        "scope_name",
        "raw_phrase_example",
        "rule_display_candidate_name",
        "llm_refined_topic_name",
        "final_research_object_name",
        "llm_small_topic_judgment",
        "llm_small_topic_type",
        "llm_small_topic_pattern",
        "llm_small_topic_reason",
        "human_judgment",
        "review_notes",
    ]
    if scored_df is None or scored_df.empty:
        return pd.DataFrame(columns=columns)
    subset = scored_df[scored_df["is_scope_internal_candidate"] == True].copy()
    if subset.empty:
        return pd.DataFrame(columns=columns)
    subset = subset.sort_values(
        by=["weak_signal_score", "hotspot_score", "source_count", "cluster_evidence_count", "total_mentions"],
        ascending=[False, False, False, False, False],
    ).drop_duplicates(subset=["display_candidate_name"], keep="first").head(top_k).copy()
    subset["raw_phrase_example"] = subset["evidence_items"].apply(
        lambda items: _safe_evidence_items(items)[0].get("raw_candidate_text", "") if _safe_evidence_items(items) else ""
    )
    subset["rule_display_candidate_name"] = subset.apply(
        lambda row: str(row.get("rule_display_candidate_name", "")).strip() or str(row.get("display_candidate_name", "")).strip(),
        axis=1,
    )
    subset["llm_refined_topic_name"] = subset.apply(
        lambda row: str(row.get("llm_refined_topic_name", "")).strip() or str(row.get("display_candidate_name", "")).strip(),
        axis=1,
    )
    subset["final_research_object_name"] = subset.apply(
        lambda row: str(row.get("topic_summary_name", "")).strip() or str(row.get("display_candidate_name", "")).strip(),
        axis=1,
    )
    subset["human_judgment"] = ""
    subset["review_notes"] = ""
    subset["comparison_rank"] = range(1, len(subset) + 1)
    subset["signal_type"] = subset["signal_type"].fillna("")
    return subset[columns]


def score_signals(candidates_df, data_df=None):
    scored_df = _prepare_scored_candidates(candidates_df)
    if scored_df.empty:
        empty = scored_df.copy()
        return empty, empty, empty
    scored_df = refresh_research_layers(scored_df)
    scored_df["score"] = scored_df["weak_signal_score"]
    scored_df["evidence_preview"] = scored_df["evidence_items"].apply(_build_evidence_preview)

    def _rank(df, score_column):
        if df.empty:
            return df.copy()
        ranked = df.sort_values(
            by=[score_column, "source_count", "org_count", "total_mentions", "tech_name"],
            ascending=[False, False, False, False, True],
        ).reset_index(drop=True)
        if "display_candidate_name" in ranked.columns:
            ranked = ranked.drop_duplicates(subset=["display_candidate_name"], keep="first").reset_index(drop=True)
        ranked["rank"] = range(1, len(ranked) + 1)
        return ranked

    hotspot_df = _rank(scored_df[scored_df["signal_type"] == "hotspot"].copy(), "hotspot_score")
    weak_signal_df = _rank(scored_df[scored_df["signal_type"] == "weak_signal"].copy(), "weak_signal_score")
    scope_overview_df = _rank(scored_df[scored_df["signal_type"] == "scope_overview"].copy(), "hotspot_score")
    return hotspot_df, weak_signal_df, scope_overview_df


def refresh_research_layers(scored_df):
    """
    Recompute research layers after downstream validation columns are merged in.

    The first scoring pass happens before reverse validation, so fields such as
    reverse_validation_status and source_semantic_consistency are still empty.
    Calling this after validation keeps final_research_bucket and weak_signal_tier
    aligned with the final evidence state.
    """
    if scored_df is None or scored_df.empty:
        return scored_df

    refreshed = scored_df.copy()
    refreshed["signal_type"] = refreshed.apply(_research_signal_type, axis=1)
    score_source = "weak_signal_score" if "weak_signal_score" in refreshed.columns else "hotspot_score"
    refreshed["score"] = refreshed.get(score_source, 0)
    refreshed["explanation"] = refreshed.apply(build_signal_explanation, axis=1)

    weak_signal_tier_profiles = refreshed.apply(_infer_weak_signal_tier, axis=1)
    refreshed["weak_signal_tier"] = [item[0] for item in weak_signal_tier_profiles]
    refreshed["weak_signal_tier_reason"] = [item[1] for item in weak_signal_tier_profiles]

    candidate_stability_profiles = refreshed.apply(_infer_candidate_weak_signal_stability, axis=1)
    refreshed["candidate_stability"] = [item[0] for item in candidate_stability_profiles]
    refreshed["candidate_stability_reason"] = [item[1] for item in candidate_stability_profiles]

    final_bucket_profiles = refreshed.apply(_infer_final_research_bucket, axis=1)
    refreshed["final_research_bucket"] = [item[0] for item in final_bucket_profiles]
    refreshed["final_research_bucket_reason"] = [item[1] for item in final_bucket_profiles]
    refreshed["signal_bucket"] = refreshed["final_research_bucket"].where(
        refreshed["final_research_bucket"].astype(str).str.strip() != "",
        refreshed["signal_type"],
    )
    return refreshed


def score_all_candidates(candidates_df):
    scored_df = _prepare_scored_candidates(candidates_df)
    if scored_df.empty:
        return scored_df
    scored_df = refresh_research_layers(scored_df)
    scored_df["score"] = scored_df["hotspot_score"]
    return scored_df.sort_values(
        by=["hotspot_score", "quality_adjusted_rank_score", "source_count", "org_count", "total_mentions", "tech_name"],
        ascending=[False, False, False, False, False, True],
    ).reset_index(drop=True)


def _build_evidence_preview(evidence_items):
    if not evidence_items:
        return "暂无代表性证据"
    previews = []
    for item in evidence_items[:3]:
        previews.append(
            f"[{item.get('source_type', 'unknown')}] {item.get('org', 'Unknown')}: {item.get('title', '').strip() or item.get('snippet', '').strip()}"
        )
    return " | ".join(previews)


def build_signal_explanation(row):
    # 优先使用统一的最终研究对象名称
    final_name = _safe_text(row.get("final_research_object_name", ""))
    if not final_name or final_name in {"", "nan", "None"}:
        final_name = _safe_text(row.get("topic_summary_name", "")) or _safe_text(row.get("display_candidate_name", ""))
    base = (
        f"跨源性={row.get('source_count', 0)}({ '/'.join(_safe_list(row.get('source_types', []))) })，"
        f"机构分散度={row.get('org_count', 0)}，总提及数={row.get('total_mentions', 0)}，"
        f"机制核={row.get('mechanism_core', '')}，"
        f"约束签名={row.get('constraint_signature', '')}，"
        f"主题粒度={row.get('topic_granularity', '')}，分层={row.get('display_tier', '')}，"
        f"原始短语={row.get('raw_phrase_example', '') or row.get('raw_candidate_text', '')}，"
        f"内部标签={row.get('internal_candidate_label', '')}，"
        f"最终对象={final_name}，"
        f"自然性解释={row.get('topic_naturalness_reason', '')}，"
        f"LLM小主题判断={_safe_text(row.get('llm_small_topic_judgment', ''))}，"
        f"LLM小主题类型={_safe_text(row.get('llm_small_topic_type', ''))}，"
        f"LLM主题模板={_safe_text(row.get('llm_small_topic_pattern', ''))}，"
        f"反向验证={_safe_text(row.get('reverse_validation_status', ''))}，"
        f"语义一致性={_safe_text(row.get('source_semantic_consistency', ''))}，"
        f"阶段判断={_safe_text(row.get('stage_hypothesis', ''))}，"
        f"需继续改写={bool(row.get('needs_surface_rewrite', False))}，"
        f"人工建议={_safe_text(row.get('suggested_label', ''))}，"
        f"非scope约束数={int(row.get('non_scope_constraint_count', 0) or 0)}，"
        f"去scope可成立={bool(row.get('survives_without_scope', False))}，"
        f"scope壳依赖={bool(row.get('scope_shell_heavy', False))}，"
        f"stage={row.get('candidate_stage', '')}，"
        f"scope_echo={bool(row.get('is_scope_echo', False))}，"
        f"generic_core_only={bool(row.get('generic_core_only', False))}，"
        f"别名数={int(row.get('alias_count', 0) or 0)}，"
        f"证据簇大小={int(row.get('cluster_evidence_count', 0) or 0)}，"
        f"增长分={row.get('growth_score', 0.0):.1f}，新颖性={row.get('novelty_score', 0.0):.1f}，"
        f"文献验证={row.get('literature_validated', False)}，专利验证={row.get('patent_validated', False)}，"
        f"时间验证={row.get('time_validated', False)}"
    )
    raw_phrases = _safe_text(row.get("raw_phrases", ""))
    if raw_phrases:
        base += f"，原始短语簇={raw_phrases}"
    reverse_note = _safe_text(row.get("reverse_validation_note", ""))
    if reverse_note:
        base += f"，反向验证说明={reverse_note}"
    stage_reason = _safe_text(row.get("stage_hypothesis_reason", ""))
    if stage_reason:
        base += f"，阶段说明={stage_reason}"
    comparison_note = _safe_text(row.get("comparison_note", ""))
    if comparison_note:
        base += f"，基线对照={comparison_note}"
    if bool(row.get("is_observation_scope", False)):
        return f"观察范围概览项（不直接作为弱信号对象）。{base}"
    scope_name = str(row.get("scope_name", "")).strip()
    if scope_name:
        return f"观察范围={scope_name}。{base}"
    return base


def format_signal_output(signal_df, title, score_column="weak_signal_score"):
    if signal_df.empty:
        return f"\n{title}：\n暂无符合条件的信号"
    output = [f"\n{title}："]
    for _, row in signal_df.head(10).iterrows():
        # 优先使用统一的最终研究对象名称
        label = str(row.get("final_research_object_name", "")).strip()
        if not label or label in {"", "nan", "None"}:
            label = str(row.get("display_candidate_name", row.get("tech_name", "未知"))).strip() or "未知"
        scope_segment = f"，观察范围={row.get('scope_name', '')}" if str(row.get("scope_name", "")).strip() else ""
        output.append(
            f"{int(row.get('rank', 0) or 0)}. {label}：得分{row.get(score_column, row.get('score', 0)):.1f}，"
            f"跨源{row.get('source_count', 0)}，机构{row.get('org_count', 0)}，提及{row.get('total_mentions', 0)}{scope_segment}"
        )
        aliases = row.get("display_candidate_aliases", [])
        if isinstance(aliases, list) and aliases:
            output.append(f"   别名：{'; '.join(aliases[:5])}")
        output.append(f"   解释：{row.get('explanation', '')}")
    return "\n".join(output)
