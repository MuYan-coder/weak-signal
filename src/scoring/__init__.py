"""评分与信号生成模块"""

from .scorer import (
    score_signals,
    score_all_candidates,
    build_manual_review_table,
)
from .signal_generator import (
    generate_candidate_outputs,
    generate_candidates,
)
from .topic_refiner import (
    refine_research_scored_candidates,
)
from .key_core_scorer import (
    score_key_core_candidates,
    build_key_core_candidate_table,
)

__all__ = [
    "score_signals",
    "score_all_candidates",
    "build_manual_review_table",
    "generate_candidate_outputs",
    "generate_candidates",
    "refine_research_scored_candidates",
    "score_key_core_candidates",
    "build_key_core_candidate_table",
]
