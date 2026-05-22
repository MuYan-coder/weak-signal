from __future__ import annotations

import pandas as pd


FREQUENCY_BASELINE_COLUMNS = [
    "baseline_rank",
    "display_candidate_name",
    "final_research_object_name",
    "scope_name",
    "signal_type",
    "topic_granularity",
    "mechanism_core",
    "constraint_signature",
    "candidate_stage",
    "total_mentions",
    "source_count",
    "news_count",
    "paper_count",
    "patent_count",
    "cluster_evidence_count",
    "llm_small_topic_judgment",
    "reverse_validation_status",
    "topic_naturalness_reason",
]


BASELINE_COMPARISON_COLUMNS = [
    "comparison_rank",
    "display_candidate_name",
    "scope_name",
    "research_signal_type",
    "research_rank",
    "baseline_rank",
    "appears_in_research_frontier",
    "appears_in_frequency_baseline",
    "total_mentions",
    "source_count",
    "news_count",
    "paper_count",
    "patent_count",
    "mechanism_core",
    "constraint_signature",
    "topic_granularity",
    "llm_small_topic_judgment",
    "reverse_validation_status",
    "comparison_note",
]


def _empty_frequency_baseline():
    return pd.DataFrame(columns=FREQUENCY_BASELINE_COLUMNS)


def _empty_comparison():
    return pd.DataFrame(columns=BASELINE_COMPARISON_COLUMNS)


def _research_frontier(scored_df: pd.DataFrame, top_k: int = 15) -> pd.DataFrame:
    if scored_df is None or scored_df.empty:
        return pd.DataFrame()
    subset = scored_df[scored_df["signal_type"].isin(["weak_signal", "hotspot"])].copy()
    if subset.empty:
        return pd.DataFrame()
    subset["research_priority"] = subset["signal_type"].map({"weak_signal": 0, "hotspot": 1}).fillna(9)
    subset = subset.sort_values(
        by=[
            "research_priority",
            "weak_signal_score",
            "hotspot_score",
            "source_count",
            "cluster_evidence_count",
            "total_mentions",
        ],
        ascending=[True, False, False, False, False, False],
    ).drop_duplicates(subset=["display_candidate_name"], keep="first").head(top_k).copy()
    subset["research_rank"] = range(1, len(subset) + 1)
    return subset


def build_frequency_baseline_table(
    scored_df: pd.DataFrame,
    reverse_validation_df: pd.DataFrame | None = None,
    top_k: int = 15,
) -> pd.DataFrame:
    if scored_df is None or scored_df.empty:
        return _empty_frequency_baseline()
    subset = scored_df[
        (scored_df.get("is_scope_internal_candidate", False) == True)
        & (scored_df.get("is_observation_scope", False) != True)
        & (scored_df.get("candidate_stage", "").isin(["formed_candidate", "formed_candidate_strong"]))
    ].copy()
    if subset.empty:
        return _empty_frequency_baseline()
    subset = subset.sort_values(
        by=["total_mentions", "source_count", "cluster_evidence_count", "org_count", "display_candidate_name"],
        ascending=[False, False, False, False, True],
    ).drop_duplicates(subset=["display_candidate_name"], keep="first").head(top_k).copy()
    subset["baseline_rank"] = range(1, len(subset) + 1)
    subset["final_research_object_name"] = subset.apply(
        lambda row: str(row.get("topic_summary_name", "")).strip() or str(row.get("display_candidate_name", "")).strip(),
        axis=1,
    )
    if reverse_validation_df is not None and not reverse_validation_df.empty:
        rv = reverse_validation_df[["display_candidate_name", "reverse_validation_status"]].drop_duplicates(
            subset=["display_candidate_name"],
            keep="first",
        )
        subset = subset.merge(rv, on="display_candidate_name", how="left")
    else:
        subset["reverse_validation_status"] = ""
    for column in FREQUENCY_BASELINE_COLUMNS:
        if column not in subset.columns:
            subset[column] = ""
    return subset[FREQUENCY_BASELINE_COLUMNS]


def build_baseline_comparison_table(
    scored_df: pd.DataFrame,
    frequency_baseline_df: pd.DataFrame,
    reverse_validation_df: pd.DataFrame | None = None,
    top_k: int = 15,
) -> pd.DataFrame:
    if scored_df is None or scored_df.empty:
        return _empty_comparison()
    frontier = _research_frontier(scored_df, top_k=top_k)
    baseline = frequency_baseline_df.copy() if frequency_baseline_df is not None else _empty_frequency_baseline()
    names = []
    names.extend(frontier.get("display_candidate_name", pd.Series(dtype=str)).tolist())
    names.extend(baseline.get("display_candidate_name", pd.Series(dtype=str)).tolist())
    ordered_names = []
    seen = set()
    for name in names:
        label = str(name).strip()
        if not label or label in seen:
            continue
        seen.add(label)
        ordered_names.append(label)
    if not ordered_names:
        return _empty_comparison()

    scored_index = scored_df.sort_values(
        by=["weak_signal_score", "hotspot_score", "total_mentions"],
        ascending=[False, False, False],
    ).drop_duplicates(subset=["display_candidate_name"], keep="first").set_index("display_candidate_name")
    frontier_index = frontier.set_index("display_candidate_name") if not frontier.empty else pd.DataFrame()
    baseline_index = baseline.set_index("display_candidate_name") if not baseline.empty else pd.DataFrame()
    reverse_index = (
        reverse_validation_df.sort_values(by=["display_candidate_name"]).drop_duplicates(subset=["display_candidate_name"], keep="first").set_index("display_candidate_name")
        if reverse_validation_df is not None and not reverse_validation_df.empty
        else pd.DataFrame()
    )

    rows = []
    for rank, name in enumerate(ordered_names, start=1):
        scored_row = scored_index.loc[name] if name in scored_index.index else pd.Series(dtype=object)
        frontier_row = frontier_index.loc[name] if not frontier_index.empty and name in frontier_index.index else pd.Series(dtype=object)
        baseline_row = baseline_index.loc[name] if not baseline_index.empty and name in baseline_index.index else pd.Series(dtype=object)
        reverse_row = reverse_index.loc[name] if not reverse_index.empty and name in reverse_index.index else pd.Series(dtype=object)
        research_signal_type = str(frontier_row.get("signal_type", "")).strip()
        research_rank = frontier_row.get("research_rank", "")
        baseline_rank = baseline_row.get("baseline_rank", "")
        appears_in_research = bool(research_signal_type)
        appears_in_baseline = pd.notna(baseline_rank) and str(baseline_rank).strip() != ""

        if appears_in_research and appears_in_baseline:
            comparison_note = "研究前排与简单频次基线均命中，可进一步比较其是否更像自然小主题。"
        elif appears_in_research:
            comparison_note = "研究链命中但简单频次基线未命中，说明对象更依赖结构化约束与判定层。"
        else:
            comparison_note = "简单频次基线命中但研究前排未命中，说明该对象可能更偏高频上层主题或需人工复核。"

        rows.append(
            {
                "comparison_rank": rank,
                "display_candidate_name": name,
                "scope_name": scored_row.get("scope_name", ""),
                "research_signal_type": research_signal_type,
                "research_rank": research_rank,
                "baseline_rank": baseline_rank,
                "appears_in_research_frontier": appears_in_research,
                "appears_in_frequency_baseline": appears_in_baseline,
                "total_mentions": scored_row.get("total_mentions", 0),
                "source_count": scored_row.get("source_count", 0),
                "news_count": scored_row.get("news_count", 0),
                "paper_count": scored_row.get("paper_count", 0),
                "patent_count": scored_row.get("patent_count", 0),
                "mechanism_core": scored_row.get("mechanism_core", ""),
                "constraint_signature": scored_row.get("constraint_signature", ""),
                "topic_granularity": scored_row.get("topic_granularity", ""),
                "llm_small_topic_judgment": scored_row.get("llm_small_topic_judgment", ""),
                "reverse_validation_status": reverse_row.get("reverse_validation_status", ""),
                "comparison_note": comparison_note,
            }
        )
    result = pd.DataFrame(rows)
    for column in BASELINE_COMPARISON_COLUMNS:
        if column not in result.columns:
            result[column] = ""
    return result[BASELINE_COMPARISON_COLUMNS]
