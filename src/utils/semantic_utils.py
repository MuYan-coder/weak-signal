from __future__ import annotations

from functools import lru_cache
import os

import numpy as np


_FAST_SEMANTIC_MODE = False


def _lexical_similarity(text_a: str, text_b: str) -> float:
    tokens_a = {token for token in text_a.lower().split() if token}
    tokens_b = {token for token in text_b.lower().split() if token}
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def set_semantic_fast_mode(enabled: bool) -> None:
    global _FAST_SEMANTIC_MODE
    enabled = bool(enabled)
    if _FAST_SEMANTIC_MODE != enabled:
        _FAST_SEMANTIC_MODE = enabled
        _get_model.cache_clear()
        _encode_text.cache_clear()


def is_semantic_fast_mode() -> bool:
    env_flag = os.getenv("WEAK_SIGNAL_FAST_SEMANTIC", "").strip().lower()
    if env_flag in {"1", "true", "yes", "on"}:
        return True
    return _FAST_SEMANTIC_MODE


@lru_cache(maxsize=1)
def _get_model():
    if is_semantic_fast_mode():
        return None
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        return None


@lru_cache(maxsize=2048)
def _encode_text(text: str):
    model = _get_model()
    if model is None:
        return None
    embedding = model.encode([text])[0]
    return tuple(float(value) for value in embedding)


def semantic_similarity(text_a: str, text_b: str) -> float:
    text_a = (text_a or "").strip()
    text_b = (text_b or "").strip()
    if not text_a or not text_b:
        return 0.0

    model = _get_model()
    if model is None:
        return round(_lexical_similarity(text_a, text_b), 3)

    vec_a = np.array(_encode_text(text_a))
    vec_b = np.array(_encode_text(text_b))
    denominator = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if denominator == 0:
        return 0.0
    score = float(np.dot(vec_a, vec_b) / denominator)
    return round(score, 3)
