"""
对象族级别评估模块

用于评估对象族的稳定性、可解释性和可承接性。
"""

from __future__ import annotations

import pandas as pd
from typing import Dict, List, Optional

FAMILY_EVALUATION_COLUMNS = [
    "family_id",
    "canonical_term",
    "term_type",
    "cross_source_pattern",
    "patent_role",
    "priority",
    "family_coverage",
    "family_cross_source_count",
    "family_evidence_count",
    "family_source_pattern",
    "family_reverse_validation_status",
    "family_stage_hypothesis_distribution",
    "family_semantic_consistency",
    "family_semantic_validation_score",
]


def _empty_family_metrics_df() -> pd.DataFrame:
    return pd.DataFrame(columns=FAMILY_EVALUATION_COLUMNS)


def _normalize_family_metrics_df(family_metrics_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if family_metrics_df is None or family_metrics_df.empty:
        return _empty_family_metrics_df()

    normalized = family_metrics_df.copy()
    column_defaults = {
        "family_id": "",
        "canonical_term": "",
        "term_type": "unknown",
        "cross_source_pattern": "unknown",
        "patent_role": "optional",
        "priority": "low",
        "family_coverage": 0,
        "family_cross_source_count": 0,
        "family_evidence_count": 0,
        "family_source_pattern": "unknown",
        "family_reverse_validation_status": "unknown",
        "family_stage_hypothesis_distribution": dict,
        "family_semantic_consistency": "unknown",
        "family_semantic_validation_score": 0.0,
    }

    for column, default in column_defaults.items():
        if column in normalized.columns:
            continue
        if callable(default):
            normalized[column] = [default() for _ in range(len(normalized))]
        else:
            normalized[column] = default

    return normalized[FAMILY_EVALUATION_COLUMNS]


def evaluate_families(candidates_df: pd.DataFrame, registry) -> pd.DataFrame:
    """
    对象族级别评估。

    Args:
        candidates_df: 候选数据框
        registry: 对象族注册表（ObjectFamilyCanonicalizer 实例）

    Returns:
        包含对象族评估指标的 DataFrame
    """
    if candidates_df is None:
        candidates_df = pd.DataFrame()

    metrics = []
    family_definitions = registry.get_all_families() if registry is not None and hasattr(registry, "get_all_families") else []
    if not family_definitions:
        return _empty_family_metrics_df()

    if 'family_id' not in candidates_df.columns:
        return _empty_family_metrics_df()

    for family in family_definitions:
        family_id = family.get('family_id', '')
        canonical_term = family.get('canonical_term', '')

        # 筛选该对象族的候选
        family_candidates = candidates_df[candidates_df['family_id'] == family_id]

        if family_candidates.empty:
            # 没有匹配的候选，记录基础信息
            metrics.append({
                "family_id": family_id,
                "canonical_term": canonical_term,
                "term_type": family.get('term_type', 'unknown'),
                "cross_source_pattern": family.get('cross_source_pattern', 'unknown'),
                "patent_role": family.get('patent_role', 'optional'),
                "priority": family.get('priority', 'low'),
                "family_coverage": 0,
                "family_cross_source_count": 0,
                "family_evidence_count": 0,
                "family_source_pattern": "none",
                "family_reverse_validation_status": "none",
                "family_stage_hypothesis_distribution": {},
                "family_semantic_consistency": "none",
                "family_semantic_validation_score": 0.0,
            })
            continue

        # 计算指标
        family_coverage = len(family_candidates)
        family_cross_source_count = (family_candidates['source_count'] >= 2).sum()
        family_evidence_count = family_candidates['cluster_evidence_count'].sum()

        # 跨源模式分布
        if 'cross_source_pattern' in family_candidates.columns:
            pattern_mode = family_candidates['cross_source_pattern'].mode()
            family_source_pattern = pattern_mode.iloc[0] if len(pattern_mode) > 0 else "unknown"
        else:
            family_source_pattern = "unknown"

        # 反向验证状态分布
        if 'reverse_validation_status' in family_candidates.columns:
            rvs_mode = family_candidates['reverse_validation_status'].mode()
            family_reverse_validation_status = rvs_mode.iloc[0] if len(rvs_mode) > 0 else "unknown"
        else:
            family_reverse_validation_status = "unknown"

        # 阶段假设分布
        if 'stage_hypothesis' in family_candidates.columns:
            stage_distribution = family_candidates['stage_hypothesis'].value_counts().to_dict()
        else:
            stage_distribution = {}

        # 语义一致性
        if 'family_semantic_consistency' in family_candidates.columns:
            consistency_mode = family_candidates['family_semantic_consistency'].mode()
            family_semantic_consistency = consistency_mode.iloc[0] if len(consistency_mode) > 0 else "unknown"
        else:
            family_semantic_consistency = "unknown"

        # 语义验证分数
        if 'family_semantic_validation_score' in family_candidates.columns:
            family_semantic_validation_score = family_candidates['family_semantic_validation_score'].mean()
        else:
            family_semantic_validation_score = 0.0

        metrics.append({
            "family_id": family_id,
            "canonical_term": canonical_term,
            "term_type": family.get('term_type', 'unknown'),
            "cross_source_pattern": family.get('cross_source_pattern', 'unknown'),
            "patent_role": family.get('patent_role', 'optional'),
            "priority": family.get('priority', 'low'),
            "family_coverage": family_coverage,
            "family_cross_source_count": family_cross_source_count,
            "family_evidence_count": family_evidence_count,
            "family_source_pattern": family_source_pattern,
            "family_reverse_validation_status": family_reverse_validation_status,
            "family_stage_hypothesis_distribution": stage_distribution,
            "family_semantic_consistency": family_semantic_consistency,
            "family_semantic_validation_score": round(family_semantic_validation_score, 3),
        })

    return _normalize_family_metrics_df(pd.DataFrame(metrics))


def generate_family_report(family_metrics_df: pd.DataFrame) -> str:
    """
    生成对象族评估报告。

    Args:
        family_metrics_df: 对象族评估指标 DataFrame

    Returns:
        Markdown 格式的报告
    """
    family_metrics_df = _normalize_family_metrics_df(family_metrics_df)

    report_lines = [
        "# 对象族级别评估报告",
        "",
        "## 概览",
        "",
        f"- 总对象族数: {len(family_metrics_df)}",
        f"- 有候选覆盖的对象族数: {(family_metrics_df['family_coverage'] > 0).sum()}",
        f"- 有跨源候选的对象族数: {(family_metrics_df['family_cross_source_count'] > 0).sum()}",
        "",
        "## 按优先级统计",
        "",
    ]

    # 按优先级统计
    if family_metrics_df.empty:
        priority_stats = pd.DataFrame(columns=['family_coverage', 'family_cross_source_count', 'family_count'])
    else:
        priority_stats = family_metrics_df.groupby('priority').agg({
            'family_coverage': 'sum',
            'family_cross_source_count': 'sum',
            'family_id': 'count'
        }).rename(columns={'family_id': 'family_count'})

    report_lines.append("| 优先级 | 对象族数 | 覆盖候选数 | 跨源候选数 |")
    report_lines.append("|--------|----------|------------|------------|")
    for priority, row in priority_stats.iterrows():
        report_lines.append(f"| {priority} | {int(row['family_count'])} | {int(row['family_coverage'])} | {int(row['family_cross_source_count'])} |")

    report_lines.extend([
        "",
        "## 按跨源模式统计",
        "",
    ])

    # 按跨源模式统计
    if family_metrics_df.empty:
        pattern_stats = pd.DataFrame(columns=['family_coverage', 'family_cross_source_count', 'family_count'])
    else:
        pattern_stats = family_metrics_df.groupby('cross_source_pattern').agg({
            'family_coverage': 'sum',
            'family_cross_source_count': 'sum',
            'family_id': 'count'
        }).rename(columns={'family_id': 'family_count'})

    report_lines.append("| 跨源模式 | 对象族数 | 覆盖候选数 | 跨源候选数 |")
    report_lines.append("|----------|----------|------------|------------|")
    for pattern, row in pattern_stats.iterrows():
        report_lines.append(f"| {pattern} | {int(row['family_count'])} | {int(row['family_coverage'])} | {int(row['family_cross_source_count'])} |")

    report_lines.extend([
        "",
        "## 高优先级对象族详情",
        "",
    ])

    # 高优先级对象族详情
    high_priority = family_metrics_df[family_metrics_df['priority'] == 'high']
    if not high_priority.empty:
        report_lines.append("| 对象族 | 类型 | 跨源模式 | 覆盖数 | 跨源数 | 语义一致性 |")
        report_lines.append("|--------|------|----------|--------|--------|------------|")
        for _, row in high_priority.iterrows():
            report_lines.append(
                f"| {row['canonical_term']} | {row['term_type']} | {row['cross_source_pattern']} | "
                f"{int(row['family_coverage'])} | {int(row['family_cross_source_count'])} | {row['family_semantic_consistency']} |"
            )

    report_lines.extend([
        "",
        "## 有覆盖的对象族列表",
        "",
    ])

    # 有覆盖的对象族
    covered = family_metrics_df[family_metrics_df['family_coverage'] > 0].sort_values(
        by=['family_coverage', 'family_cross_source_count'],
        ascending=[False, False]
    )

    if not covered.empty:
        report_lines.append("| 对象族 | 类型 | 优先级 | 覆盖数 | 跨源数 | 语义分数 |")
        report_lines.append("|--------|------|--------|--------|--------|----------|")
        for _, row in covered.head(20).iterrows():
            report_lines.append(
                f"| {row['canonical_term']} | {row['term_type']} | {row['priority']} | "
                f"{int(row['family_coverage'])} | {int(row['family_cross_source_count'])} | {row['family_semantic_validation_score']} |"
            )

    return "\n".join(report_lines)
