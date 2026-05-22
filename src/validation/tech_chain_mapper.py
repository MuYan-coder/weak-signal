"""Lightweight technology-chain mapping for v2.6.

The mapper is intentionally conservative: it uses local CSV priors and simple
text matching, records why a candidate was mapped, and keeps no-match rows
instead of dropping candidates.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from ..utils.config import Config


TECH_CHAIN_MAPPING_COLUMNS = [
    "candidate_id",
    "candidate_name",
    "tech_chain_node_id",
    "tech_chain_name",
    "tech_chain_official_name",
    "mapping_relation",
    "mapping_confidence",
    "mapping_method",
    "matched_term",
    "matched_field",
    "parent_technology",
    "maturity_level",
    "technology_status",
    "bottleneck_level",
    "strategic_importance_level",
    "mapping_reason",
]

TECH_CHAIN_CANDIDATE_COLUMNS = [
    "tech_chain_node_id",
    "tech_chain_name",
    "tech_chain_official_name",
    "mapping_relation",
    "mapping_confidence",
    "mapping_method",
    "matched_term",
    "matched_field",
    "parent_technology",
    "maturity_level",
    "technology_status",
    "bottleneck_level",
    "strategic_importance_level",
    "mapping_reason",
    "tech_chain_mapping_confidence",
    "tech_chain_mapping_risk",
    "tech_chain_mapping_coverage_flag",
]

PRIMARY_CANDIDATE_NAME_FIELDS = [
    "final_research_object_name",
    "topic_summary_name",
    "research_surface_name",
    "display_candidate_name",
    "tech_name",
    "technology",
    "canonical_candidate_name_en",
    "normalized_candidate_text",
    "raw_phrase",
    "raw_candidate_text",
]

DISPLAY_CANDIDATE_NAME_FIELDS = [
    "final_research_object_name",
    "topic_summary_name",
    "research_surface_name",
    "display_candidate_name",
    "tech_name",
    "technology",
]

CONTEXT_CANDIDATE_FIELDS = [
    "mechanism_core",
    "relation_target",
    "relation_task",
    "relation_data_modality",
    "relation_method",
    "relation_summary",
    "constraint_signature",
    "internal_candidate_label",
    "stable_object_label",
]

LIST_CANDIDATE_FIELDS = [
    "display_candidate_aliases",
    "mechanism_core_tokens",
    "task_constraint_tokens",
    "object_modifier_tokens",
    "data_modifier_tokens",
    "method_modifier_tokens",
    "scene_tokens",
]

NODE_PRIMARY_FIELDS = ["name", "official_name", "short_name", "name_en"]
NODE_CONTEXT_FIELDS = [
    "description",
    "classification_name",
    "application_scenario",
    "key_function",
    "technical_route",
]
TERM_PRIMARY_FIELDS = ["term_name", "official_name", "short_name", "name_en"]
MULTI_VALUE_SPLIT_RE = re.compile(r"[|;；,，、/\n\r]+")
UNKNOWN_VALUES = {"", "nan", "none", "null", "nat", "unknown", "未知", "无"}
SHORT_ALLOWED_TERMS = {"ai", "rl", "ros", "imu", "llm", "slam", "3d"}
GENERIC_CANDIDATE_KEYS = {
    "机器人",
    "robot",
    "robots",
    "具身智能",
    "embodiedintelligence",
    "worldmodel",
    "世界模型",
}


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    return "" if text.lower() in UNKNOWN_VALUES else text


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return default


def _safe_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple) or isinstance(value, set):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text[0] in "[{":
            try:
                parsed = json.loads(text)
                return _safe_list(parsed)
            except Exception:
                pass
        return [text]
    try:
        if pd.isna(value):
            return []
    except Exception:
        pass
    return [value]


def _split_multi(value: Any) -> List[str]:
    values: List[str] = []
    for item in _safe_list(value):
        text = _safe_text(item)
        if not text:
            continue
        for part in MULTI_VALUE_SPLIT_RE.split(text):
            cleaned = _safe_text(part)
            if cleaned:
                values.append(cleaned)
    return _dedupe(values)


def _dedupe(values: Iterable[str]) -> List[str]:
    seen = set()
    result = []
    for value in values:
        text = _safe_text(value)
        if not text:
            continue
        key = _normalize_key(text)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def _normalize_key(value: Any) -> str:
    text = unicodedata.normalize("NFKC", _safe_text(value)).lower()
    if not text:
        return ""
    text = re.sub(r"[\s_\-]+", "", text)
    text = re.sub(r"[^\w\u4e00-\u9fff]+", "", text)
    return text


def _is_usable_term_key(key: str) -> bool:
    if not key:
        return False
    if key in SHORT_ALLOWED_TERMS:
        return True
    if re.search(r"[\u4e00-\u9fff]", key):
        return len(key) >= 2
    return len(key) >= 3


def _tokenize(value: Any) -> set[str]:
    text = unicodedata.normalize("NFKC", _safe_text(value)).lower()
    if not text:
        return set()

    tokens = set(re.findall(r"[a-z0-9]+", text))
    for seq in re.findall(r"[\u4e00-\u9fff]{2,}", text):
        tokens.add(seq)
        if len(seq) > 2:
            tokens.update(seq[idx : idx + 2] for idx in range(len(seq) - 1))
            tokens.update(seq[idx : idx + 3] for idx in range(len(seq) - 2))
    return {token for token in tokens if token and token not in UNKNOWN_VALUES}


def _text_similarity(left: str, right: str) -> float:
    left_text = _safe_text(left)
    right_text = _safe_text(right)
    if not left_text or not right_text:
        return 0.0

    left_key = _normalize_key(left_text)
    right_key = _normalize_key(right_text)
    substring_bonus = 0.0
    if left_key and right_key:
        if left_key in right_key or right_key in left_key:
            shorter = min(len(left_key), len(right_key))
            longer = max(len(left_key), len(right_key))
            substring_bonus = min(0.42, shorter / max(longer, 1) * 0.55)

    left_tokens = _tokenize(left_text)
    right_tokens = _tokenize(right_text)
    if not left_tokens or not right_tokens:
        return round(substring_bonus, 3)
    intersection = left_tokens & right_tokens
    union = left_tokens | right_tokens
    jaccard = len(intersection) / len(union) if union else 0.0
    coverage = len(intersection) / max(min(len(left_tokens), len(right_tokens)), 1)
    score = max(jaccard, coverage * 0.72, substring_bonus)
    if intersection:
        score += min(0.16, len(intersection) * 0.025)
    return round(min(score, 1.0), 3)


def _resolve_base_dir(base_dir: Optional[Path | str] = None) -> Path:
    if base_dir is None:
        return Config.DATA_DIR / "tech_chain"
    path = Path(base_dir)
    if (path / "technology_nodes.csv").exists():
        return path
    if (path / "tech_chain" / "technology_nodes.csv").exists():
        return path / "tech_chain"
    return path


def _read_csv(path: Path, columns: List[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="utf-8")
    for column in columns:
        if column not in df.columns:
            df[column] = ""
    return df


def load_tech_chain_data(base_dir: Optional[Path | str] = None) -> Dict[str, pd.DataFrame]:
    """Load v2.6 technology-chain CSV priors.

    Missing files return empty tables with expected columns so all runtime paths
    remain backward-compatible.
    """
    resolved = _resolve_base_dir(base_dir)
    node_columns = [
        "tech_node_id",
        "name",
        "official_name",
        "short_name",
        "alias",
        "name_en",
        "description",
        "classification_code",
        "classification_name",
        "maturity_level",
        "technology_status",
        "application_scenario",
        "key_function",
        "technical_route",
        "parent_technology",
        "depends_on",
        "supersedes",
        "improves",
        "cooperates_with",
        "related_policy",
        "owned_by_enterprise",
        "strategic_importance_level",
        "bottleneck_level",
        "source_reference",
        "maintainer",
        "updated_at",
    ]
    term_columns = [
        "term_id",
        "term_name",
        "official_name",
        "short_name",
        "alias",
        "name_en",
        "semantic_tag",
        "source_text",
        "ipc_code",
        "ipc_name",
        "belongs_to_tech_node_id",
        "mapping_confidence",
        "source_reference",
        "maintainer",
        "updated_at",
    ]
    asset_columns = [
        "asset_id",
        "asset_name",
        "asset_category",
        "asset_no",
        "legal_status",
        "application_date",
        "publication_date",
        "grant_date",
        "right_holder_name",
        "inventor_or_author",
        "patent_no",
        "patent_type",
        "patent_country",
        "belongs_to_tech_node_id",
        "related_product",
        "applied_in",
        "related_policy",
        "source_reference",
    ]
    nodes_df = _read_csv(resolved / "technology_nodes.csv", node_columns)
    terms_df = _read_csv(resolved / "technology_terms.csv", term_columns)
    assets_df = _read_csv(resolved / "technology_assets.csv", asset_columns)

    diagnostics = {
        "base_dir": str(resolved),
        "node_count": int(len(nodes_df)),
        "term_count": int(len(terms_df)),
        "asset_count": int(len(assets_df)),
        "duplicate_node_ids": [],
        "orphan_term_node_ids": [],
    }
    if not nodes_df.empty and "tech_node_id" in nodes_df.columns:
        diagnostics["duplicate_node_ids"] = sorted(
            nodes_df.loc[nodes_df["tech_node_id"].astype(str).duplicated(), "tech_node_id"].astype(str).unique().tolist()
        )
        node_ids = set(nodes_df["tech_node_id"].astype(str))
        if not terms_df.empty:
            term_ids = set(terms_df["belongs_to_tech_node_id"].dropna().astype(str))
            diagnostics["orphan_term_node_ids"] = sorted(term_ids - node_ids)

    return {
        "nodes": nodes_df,
        "terms": terms_df,
        "assets": assets_df,
        "diagnostics": diagnostics,
    }


def _candidate_id(row: pd.Series, index: int) -> str:
    for field in ["candidate_id", "candidate_cluster_id", "id"]:
        value = _safe_text(row.get(field))
        if value:
            return value
    name = _candidate_name(row)
    if name:
        return f"name::{_normalize_key(name)}"
    return f"candidate_row_{index + 1}"


def _candidate_name(row: pd.Series) -> str:
    for field in PRIMARY_CANDIDATE_NAME_FIELDS:
        value = _safe_text(row.get(field))
        if value:
            return value
    return ""


def _candidate_name_variants(row: pd.Series) -> List[str]:
    values = []
    for field in PRIMARY_CANDIDATE_NAME_FIELDS:
        value = _safe_text(row.get(field))
        if value:
            values.append(value)
    values.extend(_safe_text(item) for item in _safe_list(row.get("display_candidate_aliases")) if _safe_text(item))
    return _dedupe(values)


def _candidate_context_text(row: pd.Series) -> str:
    parts = _candidate_name_variants(row)
    for field in CONTEXT_CANDIDATE_FIELDS:
        value = _safe_text(row.get(field))
        if value:
            parts.append(value)
    for field in LIST_CANDIDATE_FIELDS:
            parts.extend(_safe_text(item) for item in _safe_list(row.get(field)) if _safe_text(item))
    return " ".join(_dedupe(parts))


def _mapping_input_key(row: pd.Series, index: int) -> str:
    candidate_id = _candidate_id(row, index)
    candidate_name = _candidate_name(row)
    context = _candidate_context_text(row)
    key_parts = [
        _normalize_key(candidate_id),
        _normalize_key(candidate_name),
        _normalize_key(context),
    ]
    key = "::".join(part for part in key_parts if part)
    return key or f"candidate_row_{index + 1}"


def _node_id(row: pd.Series) -> str:
    return _safe_text(row.get("tech_node_id"))


def _node_text(row: pd.Series) -> str:
    parts = []
    for field in NODE_PRIMARY_FIELDS + NODE_CONTEXT_FIELDS:
        value = _safe_text(row.get(field))
        if value:
            parts.append(value)
    parts.extend(_split_multi(row.get("alias")))
    return " ".join(_dedupe(parts))


def _mechanism_keywords(value: Any) -> List[str]:
    mechanism = _safe_text(value).lower()
    mapping = {
        "training": ["训练", "learning", "training", "强化学习", "reinforcement"],
        "planning": ["规划", "planning", "路径规划", "pathplanning"],
        "control": ["控制", "control", "motioncontrol"],
        "simulation": ["仿真", "simulation", "simulator"],
        "tactile": ["触觉", "tactile", "感知"],
        "perception": ["感知", "perception", "视觉", "vision"],
        "grounding": ["交互", "grounding", "具身交互"],
        "navigation": ["导航", "navigation"],
        "manipulation": ["操作", "manipulation", "抓取", "grasping"],
    }
    return mapping.get(mechanism, [])


def _node_matches_mechanism(node: pd.Series, mechanism: Any) -> bool:
    keywords = _mechanism_keywords(mechanism)
    if not keywords:
        return False
    node_key = _normalize_key(_node_text(node))
    return any(_normalize_key(keyword) in node_key for keyword in keywords if _normalize_key(keyword))


def _term_confidence(term: pd.Series) -> float:
    confidence = _safe_float(term.get("mapping_confidence"), 0.78)
    if confidence <= 0:
        return 0.78
    return max(0.0, min(confidence, 1.0))


def _node_payload(node: pd.Series) -> Dict[str, Any]:
    return {
        "tech_chain_node_id": _safe_text(node.get("tech_node_id")),
        "tech_chain_name": _safe_text(node.get("name")),
        "tech_chain_official_name": _safe_text(node.get("official_name")) or _safe_text(node.get("name")),
        "parent_technology": _safe_text(node.get("parent_technology")),
        "maturity_level": _safe_text(node.get("maturity_level")),
        "technology_status": _safe_text(node.get("technology_status")),
        "bottleneck_level": _safe_text(node.get("bottleneck_level")) or "unknown",
        "strategic_importance_level": _safe_text(node.get("strategic_importance_level")) or "unknown",
    }


def _mapping_row(
    candidate_id: str,
    candidate_name: str,
    node: Optional[pd.Series],
    relation: str,
    confidence: float,
    method: str,
    matched_term: str,
    matched_field: str,
    reason: str,
) -> Dict[str, Any]:
    row = {
        "candidate_id": candidate_id,
        "candidate_name": candidate_name,
        "tech_chain_node_id": "",
        "tech_chain_name": "",
        "tech_chain_official_name": "",
        "mapping_relation": relation,
        "mapping_confidence": round(max(0.0, min(float(confidence or 0.0), 1.0)), 3),
        "mapping_method": method,
        "matched_term": matched_term,
        "matched_field": matched_field,
        "parent_technology": "",
        "maturity_level": "",
        "technology_status": "",
        "bottleneck_level": "",
        "strategic_importance_level": "",
        "mapping_reason": reason,
    }
    if node is not None and not node.empty:
        row.update(_node_payload(node))
    return row


def _build_match_indexes(nodes_df: pd.DataFrame, terms_df: pd.DataFrame) -> Dict[str, Any]:
    nodes_by_id = {}
    exact_entries: Dict[str, List[Dict[str, Any]]] = {}
    alias_entries: Dict[str, List[Dict[str, Any]]] = {}
    term_entries: List[Dict[str, Any]] = []
    semantic_nodes = []

    if nodes_df is None:
        nodes_df = pd.DataFrame()
    if terms_df is None:
        terms_df = pd.DataFrame()

    for _, node in nodes_df.iterrows():
        node_id = _node_id(node)
        if not node_id:
            continue
        nodes_by_id[node_id] = node
        semantic_nodes.append({"node": node, "text": _node_text(node)})

        for field in NODE_PRIMARY_FIELDS:
            value = _safe_text(node.get(field))
            key = _normalize_key(value)
            if _is_usable_term_key(key):
                exact_entries.setdefault(key, []).append(
                    {"node": node, "matched_term": value, "matched_field": f"node.{field}", "confidence": 0.98}
                )
                term_entries.append(
                    {
                        "key": key,
                        "text": value,
                        "node": node,
                        "field": f"node.{field}",
                        "confidence": 0.86,
                    }
                )

        for value in _split_multi(node.get("alias")):
            key = _normalize_key(value)
            if _is_usable_term_key(key):
                alias_entries.setdefault(key, []).append(
                    {"node": node, "matched_term": value, "matched_field": "node.alias", "confidence": 0.9}
                )
                term_entries.append(
                    {
                        "key": key,
                        "text": value,
                        "node": node,
                        "field": "node.alias",
                        "confidence": 0.78,
                    }
                )

    for _, term in terms_df.iterrows():
        node = nodes_by_id.get(_safe_text(term.get("belongs_to_tech_node_id")))
        if node is None:
            continue
        confidence = _term_confidence(term)
        for field in TERM_PRIMARY_FIELDS:
            value = _safe_text(term.get(field))
            key = _normalize_key(value)
            if _is_usable_term_key(key):
                exact_entries.setdefault(key, []).append(
                    {
                        "node": node,
                        "matched_term": value,
                        "matched_field": f"term.{field}",
                        "confidence": max(confidence, 0.82),
                    }
                )
                term_entries.append(
                    {
                        "key": key,
                        "text": value,
                        "node": node,
                        "field": f"term.{field}",
                        "confidence": confidence,
                    }
                )
        for value in _split_multi(term.get("alias")):
            key = _normalize_key(value)
            if _is_usable_term_key(key):
                alias_entries.setdefault(key, []).append(
                    {
                        "node": node,
                        "matched_term": value,
                        "matched_field": "term.alias",
                        "confidence": max(confidence * 0.95, 0.7),
                    }
                )
                term_entries.append(
                    {
                        "key": key,
                        "text": value,
                        "node": node,
                        "field": "term.alias",
                        "confidence": max(confidence * 0.95, 0.7),
                    }
                )

    term_entries.sort(key=lambda item: (len(item["key"]), item["confidence"]), reverse=True)
    return {
        "nodes_by_id": nodes_by_id,
        "exact_entries": exact_entries,
        "alias_entries": alias_entries,
        "term_entries": term_entries,
        "semantic_nodes": semantic_nodes,
    }


def _choose_best(entries: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not entries:
        return None
    return sorted(
        entries,
        key=lambda item: (
            float(item.get("confidence", 0.0) or 0.0),
            len(_normalize_key(item.get("matched_term") or item.get("text"))),
        ),
        reverse=True,
    )[0]


def _manual_mapping(
    row: pd.Series,
    index: int,
    manual_index: Dict[str, Dict[str, Any]],
    nodes_by_id: Dict[str, pd.Series],
) -> Optional[Dict[str, Any]]:
    candidate_id = _candidate_id(row, index)
    candidate_name = _candidate_name(row)
    lookup_keys = [candidate_id, _normalize_key(candidate_name)]
    for key in lookup_keys:
        override = manual_index.get(key)
        if not override:
            continue
        node_id = _safe_text(
            override.get("tech_chain_node_id")
            or override.get("manual_tech_chain_node_id")
            or override.get("belongs_to_tech_node_id")
        )
        node = nodes_by_id.get(node_id)
        if node is None:
            continue
        confidence = _safe_float(override.get("mapping_confidence"), 1.0)
        relation = _safe_text(override.get("mapping_relation")) or "manual_override"
        reason = _safe_text(override.get("mapping_reason")) or "manual_review_df 指定映射"
        return _mapping_row(
            candidate_id,
            candidate_name,
            node,
            relation,
            confidence,
            "manual_override",
            _safe_text(override.get("matched_term")) or candidate_name,
            _safe_text(override.get("matched_field")) or "manual_review_df",
            reason,
        )
    return None


def _manual_index(manual_review_df: Optional[pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
    if manual_review_df is None or manual_review_df.empty:
        return {}
    result: Dict[str, Dict[str, Any]] = {}
    for _, row in manual_review_df.iterrows():
        record = row.to_dict()
        for field in ["candidate_id", "candidate_cluster_id", "display_candidate_name", "candidate_name", "final_research_object_name"]:
            value = _safe_text(record.get(field))
            if not value:
                continue
            result[value] = record
            result[_normalize_key(value)] = record
    return result


def _exact_or_alias_match(
    row: pd.Series,
    index: int,
    indexes: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    candidate_id = _candidate_id(row, index)
    candidate_name = _candidate_name(row)
    variants = _candidate_name_variants(row)
    for value in variants:
        key = _normalize_key(value)
        if not _is_usable_term_key(key):
            continue
        best = _choose_best(indexes["exact_entries"].get(key, []))
        if best:
            return _mapping_row(
                candidate_id,
                candidate_name,
                best["node"],
                "exact_match",
                best.get("confidence", 0.98),
                "exact_match",
                best.get("matched_term", value),
                best.get("matched_field", ""),
                f"候选名称与技术链字段精确匹配：{best.get('matched_term', value)}",
            )
    for value in variants:
        key = _normalize_key(value)
        if not _is_usable_term_key(key):
            continue
        best = _choose_best(indexes["alias_entries"].get(key, []))
        if best:
            return _mapping_row(
                candidate_id,
                candidate_name,
                best["node"],
                "close_match",
                best.get("confidence", 0.9),
                "alias_match",
                best.get("matched_term", value),
                best.get("matched_field", ""),
                f"候选名称命中技术链别名：{best.get('matched_term', value)}",
            )
    return None


def _term_belongs_match(
    row: pd.Series,
    index: int,
    indexes: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    candidate_id = _candidate_id(row, index)
    candidate_name = _candidate_name(row)
    context = _candidate_context_text(row)
    context_key = _normalize_key(context)
    if not context_key:
        return None
    context_token_keys = {_normalize_key(token) for token in _tokenize(context)}
    display_context_key = _normalize_key(
        " ".join(_safe_text(row.get(field)) for field in DISPLAY_CANDIDATE_NAME_FIELDS if _safe_text(row.get(field)))
    )
    primary_context_key = _normalize_key(
        " ".join(_safe_text(row.get(field)) for field in PRIMARY_CANDIDATE_NAME_FIELDS if _safe_text(row.get(field)))
    )
    generic_display = display_context_key in GENERIC_CANDIDATE_KEYS

    candidates = []
    for term in indexes["term_entries"]:
        key = term["key"]
        if not _is_usable_term_key(key):
            continue
        if key in SHORT_ALLOWED_TERMS and key not in context_token_keys:
            continue
        if key not in context_key:
            continue
        display_hit = bool(display_context_key and key in display_context_key)
        primary_hit = bool(primary_context_key and key in primary_context_key)
        mechanism_hit = _node_matches_mechanism(term["node"], row.get("mechanism_core"))
        if generic_display and not display_hit:
            continue
        length_bonus = min(len(key) / max(len(context_key), 1), 0.25)
        context_penalty = 0.08 if not (display_hit or primary_hit or mechanism_hit) else 0.0
        confidence = min(
            float(term.get("confidence", 0.78) or 0.78)
            + length_bonus
            + (0.05 if display_hit else 0.0)
            + (0.03 if primary_hit else 0.0)
            + (0.04 if mechanism_hit else 0.0)
            - context_penalty,
            0.94,
        )
        relation = "close_match" if confidence >= 0.78 else "broader_match"
        candidates.append(
            {
                **term,
                "confidence": confidence,
                "relation": relation,
                "display_hit": display_hit,
                "primary_hit": primary_hit,
                "mechanism_hit": mechanism_hit,
            }
        )

    best = sorted(
        candidates,
        key=lambda item: (
            bool(item.get("display_hit", False)),
            bool(item.get("primary_hit", False)),
            bool(item.get("mechanism_hit", False)),
            float(item.get("confidence", 0.0) or 0.0),
            len(item.get("key", "")),
        ),
        reverse=True,
    )[0] if candidates else None
    if not best:
        return None
    return _mapping_row(
        candidate_id,
        candidate_name,
        best["node"],
        best.get("relation", "close_match"),
        best.get("confidence", 0.78),
        "term_belongs_to",
        best.get("text", ""),
        best.get("field", ""),
        f"候选上下文包含术语表条目：{best.get('text', '')}",
    )


def _semantic_match(
    row: pd.Series,
    index: int,
    indexes: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    candidate_id = _candidate_id(row, index)
    candidate_name = _candidate_name(row)
    context = _candidate_context_text(row)
    if not context:
        return None

    best_item = None
    best_score = 0.0
    for item in indexes["semantic_nodes"]:
        score = _text_similarity(context, item["text"])
        if score > best_score:
            best_item = item
            best_score = score
    if best_item is None or best_score < 0.34:
        return None

    relation = "broader_match" if best_score < 0.52 else "related_match"
    confidence = min(0.68, 0.42 + best_score * 0.42)
    node = best_item["node"]
    return _mapping_row(
        candidate_id,
        candidate_name,
        node,
        relation,
        confidence,
        "semantic_match",
        _safe_text(node.get("name")),
        "node.semantic_context",
        f"轻量文本相似度匹配，score={best_score:.3f}；需人工复核",
    )


def build_tech_chain_mapping_table(
    candidates_df: pd.DataFrame,
    tech_chain_data: Dict[str, pd.DataFrame],
    manual_review_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Map candidates to technology-chain nodes using local priors."""
    if candidates_df is None or candidates_df.empty:
        return pd.DataFrame(columns=TECH_CHAIN_MAPPING_COLUMNS)

    nodes_df = tech_chain_data.get("nodes", pd.DataFrame()) if tech_chain_data else pd.DataFrame()
    terms_df = tech_chain_data.get("terms", pd.DataFrame()) if tech_chain_data else pd.DataFrame()
    indexes = _build_match_indexes(nodes_df, terms_df)
    manual = _manual_index(manual_review_df)

    rows = []
    seen_inputs = set()
    for index, candidate in candidates_df.reset_index(drop=True).iterrows():
        input_key = _mapping_input_key(candidate, index)
        if input_key in seen_inputs:
            continue
        seen_inputs.add(input_key)

        candidate_id = _candidate_id(candidate, index)
        candidate_name = _candidate_name(candidate)
        if not candidate_name:
            rows.append(
                _mapping_row(
                    candidate_id,
                    "",
                    None,
                    "no_match",
                    0.0,
                    "no_match",
                    "",
                    "",
                    "候选缺少可用于映射的名称字段",
                )
            )
            continue

        matched = (
            _manual_mapping(candidate, index, manual, indexes["nodes_by_id"])
            or _exact_or_alias_match(candidate, index, indexes)
            or _term_belongs_match(candidate, index, indexes)
            or _semantic_match(candidate, index, indexes)
        )
        if matched is None:
            matched = _mapping_row(
                candidate_id,
                candidate_name,
                None,
                "no_match",
                0.0,
                "no_match",
                "",
                "",
                "未命中技术链节点、别名或术语；建议人工补充 technology_terms.csv",
            )
        rows.append(matched)

    return pd.DataFrame(rows, columns=TECH_CHAIN_MAPPING_COLUMNS)


def _mapping_risk(row: pd.Series) -> Tuple[str, str]:
    relation = _safe_text(row.get("mapping_relation"))
    method = _safe_text(row.get("mapping_method"))
    confidence = _safe_float(row.get("mapping_confidence"), 0.0)
    if relation == "no_match" or method == "no_match":
        return "no_chain_mapping", "unmapped"
    if confidence < 0.55:
        return "low_confidence_mapping", "mapped_review"
    if relation == "broader_match":
        return "broad_chain_mapping", "mapped_broad"
    if method == "semantic_match":
        return "semantic_mapping_review", "mapped_review"
    return "low_mapping_risk", "mapped"


def _mapping_risk_values(
    relation: Any,
    method: Any,
    confidence: Any,
) -> Tuple[str, str]:
    relation_text = _safe_text(relation)
    method_text = _safe_text(method)
    confidence_value = _safe_float(confidence, 0.0)
    if relation_text == "no_match" or method_text == "no_match":
        return "no_chain_mapping", "unmapped"
    if confidence_value < 0.55:
        return "low_confidence_mapping", "mapped_review"
    if relation_text == "broader_match":
        return "broad_chain_mapping", "mapped_broad"
    if method_text == "semantic_match":
        return "semantic_mapping_review", "mapped_review"
    return "low_mapping_risk", "mapped"


def merge_tech_chain_mapping_into_candidates(
    candidates_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
) -> pd.DataFrame:
    """Attach technology-chain mapping fields to candidate rows."""
    if candidates_df is None or candidates_df.empty:
        return candidates_df.copy() if isinstance(candidates_df, pd.DataFrame) else pd.DataFrame()

    merged = candidates_df.reset_index(drop=True).copy()
    merged["candidate_id"] = [
        _safe_text(row.get("candidate_id")) or _candidate_id(row, index)
        for index, row in merged.iterrows()
    ]

    if mapping_df is None or mapping_df.empty:
        for column in TECH_CHAIN_CANDIDATE_COLUMNS:
            if column not in merged.columns:
                merged[column] = 0.0 if column in {"mapping_confidence", "tech_chain_mapping_confidence"} else ""
        merged["tech_chain_mapping_risk"] = merged.get("tech_chain_mapping_risk", "no_chain_mapping")
        merged["tech_chain_mapping_coverage_flag"] = merged.get("tech_chain_mapping_coverage_flag", "unmapped")
        return merged

    mapping = mapping_df.reset_index(drop=True).copy()
    mapping_columns = [
        column for column in TECH_CHAIN_MAPPING_COLUMNS if column not in {"candidate_id", "candidate_name"}
    ]

    if len(mapping) == len(merged):
        joined = merged.drop(columns=[column for column in TECH_CHAIN_CANDIDATE_COLUMNS if column in merged.columns])
        for column in mapping_columns:
            if column in mapping.columns:
                joined[column] = mapping[column].values
    else:
        mapping["_tech_chain_join_id"] = mapping["candidate_id"].astype(str)
        mapping = mapping.drop_duplicates(subset=["_tech_chain_join_id"], keep="first")
        working = merged.drop(columns=[column for column in TECH_CHAIN_CANDIDATE_COLUMNS if column in merged.columns])
        working["_tech_chain_join_id"] = working["candidate_id"].astype(str)

        columns = ["_tech_chain_join_id"] + [
            column
            for column in mapping_columns
            if column in mapping.columns
        ]
        joined = working.merge(mapping[columns], on="_tech_chain_join_id", how="left")
        joined = joined.drop(columns=["_tech_chain_join_id"])

    for column in ["mapping_confidence"]:
        joined[column] = pd.to_numeric(joined.get(column, 0.0), errors="coerce").fillna(0.0)
    for column in TECH_CHAIN_CANDIDATE_COLUMNS:
        if column not in joined.columns:
            joined[column] = 0.0 if column in {"mapping_confidence", "tech_chain_mapping_confidence"} else ""
    joined["tech_chain_mapping_confidence"] = joined["mapping_confidence"]

    risks = [
        _mapping_risk_values(relation, method, confidence)
        for relation, method, confidence in zip(
            joined.get("mapping_relation", pd.Series([""] * len(joined), index=joined.index)),
            joined.get("mapping_method", pd.Series([""] * len(joined), index=joined.index)),
            joined.get("mapping_confidence", pd.Series([0.0] * len(joined), index=joined.index)),
        )
    ]
    joined["tech_chain_mapping_risk"] = [item[0] for item in risks]
    joined["tech_chain_mapping_coverage_flag"] = [item[1] for item in risks]
    return joined


__all__ = [
    "TECH_CHAIN_MAPPING_COLUMNS",
    "TECH_CHAIN_CANDIDATE_COLUMNS",
    "load_tech_chain_data",
    "build_tech_chain_mapping_table",
    "merge_tech_chain_mapping_into_candidates",
]
