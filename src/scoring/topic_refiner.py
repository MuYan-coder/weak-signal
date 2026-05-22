from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pandas as pd

from ..utils.api_stats import record_call
from ..utils.env_config import ensure_env_loaded
from ..utils.llm_client import chat_text, get_provider_and_client


ensure_env_loaded()
PROMPT_VERSION = "small_topic_v8_multisource_grouping_and_naturalization"
GENERIC_OBJECT_TOKENS = {"robot", "agent", "embodied", "control"}
SPECIFIC_TASK_TOKENS = {"driving", "navigation", "manipulation", "grasping", "safety", "decision", "planning"}
SPECIFIC_DATA_TOKENS = {"video", "3d", "visual", "sensor", "multimodal", "temporal", "trajectory", "memory"}
SPECIFIC_METHOD_TOKENS = {"retrieval", "causal", "dynamic", "sparse", "alignment"}
PROMOTABLE_PATTERNS = {"scene+mechanism", "data+mechanism", "application+mechanism", "problem+mechanism"}
SCENE_OR_APPLICATION_TOKENS = {"driving", "navigation", "manipulation", "grasping", "safety", "path", "space", "spatial", "physical", "memory", "reasoning"}
STRONG_DATA_TOKENS = {"video", "sensor", "trajectory", "temporal"}
WEAK_DATA_TOKENS = {"visual", "3d", "multimodal"}

def _normalize_cache_path(cache_path):
    if not cache_path:
        return None
    path = Path(cache_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    return path


def load_topic_cache_bundle(cache_path):
    cache_file = _normalize_cache_path(cache_path)
    if cache_file is None or not cache_file.exists():
        return {}, {}
    try:
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
    except Exception:
        return {}, {}
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    items = payload.get("items", {})
    return (metadata if isinstance(metadata, dict) else {}), (items if isinstance(items, dict) else {})


def save_topic_cache(cache_path, items, metadata=None):
    cache_file = _normalize_cache_path(cache_path)
    if cache_file is None:
        return
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": metadata or {},
        "items": items or {},
    }
    cache_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _parse_json_from_response(raw: str):
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\[\s*\{.*\}\s*\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None
    return None


def _safe_list(value):
    return value if isinstance(value, list) else []


def _candidate_key(row):
    cluster_id = str(row.get("candidate_cluster_id", "")).strip()
    if cluster_id:
        return cluster_id
    return str(row.get("display_candidate_name", row.get("tech_name", ""))).strip()


def _evidence_preview(row, limit=3):
    previews = []
    for item in _safe_list(row.get("evidence_items", []))[:limit]:
        previews.append(
            {
                "source_type": str(item.get("source_type", "")).strip(),
                "date": str(item.get("date", "")).strip(),
                "title": str(item.get("title", "")).strip(),
                "raw_candidate_text": str(item.get("raw_candidate_text", "")).strip(),
                "snippet": str(item.get("snippet", "")).strip()[:160],
            }
        )
    return previews


def _build_prompt(batch_rows):
    payload = []
    for row in batch_rows:
        payload.append(
            {
                "candidate_key": _candidate_key(row),
                "current_name": str(row.get("display_candidate_name", "")).strip(),
                "current_bucket": str(row.get("signal_type", "")).strip(),
                "scope_name": str(row.get("scope_name", "")).strip(),
                "mechanism_core": str(row.get("mechanism_core", "")).strip(),
                "constraint_signature": str(row.get("constraint_signature", "")).strip(),
                "source_count": int(row.get("source_count", 0) or 0),
                "source_types": _safe_list(row.get("source_types", [])),
                "news_count": int(row.get("news_count", 0) or 0),
                "paper_count": int(row.get("paper_count", 0) or 0),
                "patent_count": int(row.get("patent_count", 0) or 0),
                "non_scope_constraint_count": int(row.get("non_scope_constraint_count", 0) or 0),
                "raw_phrase_example": str(row.get("raw_phrase_example", "")).strip(),
                "topic_naturalness_reason": str(row.get("topic_naturalness_reason", "")).strip(),
                "evidence_preview": _evidence_preview(row),
            }
        )
    return f"""
你在帮助做“弱信号小主题收口”。

背景要求：
1. observation scope 是母主题，例如 world model、embodied intelligence。
2. 真正希望得到的对象，不是“母主题 + 机制核”的系统压缩标签，而是母主题内部更小、更具体、更自然的子技术主题表达。
3. 参考弱信号经典表达方式，例如：
   - army's solar tents
   - portable solar chargers for military applications
   - health and safety concerns of photovoltaic solar panels
   - recycling the solar panels waste
它们不是大主题本体，而是大主题内部的小方向、小应用、小问题、小能力。

你的任务：
对下面每个候选，判断它目前更像：
- small_topic：已经较像自然的小主题表达
- upper_topic：已经是 scope 内细粒度候选，但当前仍更像上层候选或待人工确认对象
- compressed_label：当前主要是系统压缩标签，还没收口成自然小主题
- unclear：仍过于泛化或信息不足，不适合作为自然小主题表达

并给出一个更自然、简洁的中文对象表述。

同时请判断它更偏向哪类小主题：
- small_application：小应用/小场景
- small_capability：小能力/小功能
- small_problem：小问题/小痛点
- small_direction：小方向/小模块
- unclear：当前还不够清楚

并给出一个简短的主题表达模板，可选值：
- object+mechanism
- scene+mechanism
- data+mechanism
- problem+mechanism
- application+mechanism
- unclear

命名要求：
1. 优先表达“小方向 / 小应用 / 小问题 / 小能力”
2. 尽量避免只输出“世界模型训练 / 具身智能规划”这类过泛表达
3. 若需要保留 scope，可放在中间作为限定，例如：
   - 机器人世界模型训练
   - 视频世界模型训练
   - 驾驶世界模型仿真
4. method（如 reinforcement）一般不要直接放进主名，除非它真的构成小主题核心
5. 输出中文，尽量自然，不要像系统标签串
6. 不要凭空发明证据里没有出现的强修饰词，例如“实时 / 平台 / 构建 / 生成 / 闭环 / 长期”等；只有当原始短语或证据标题明确支持时才能使用
7. 如果当前对象只有“对象 + 机制核”，但没有更具体的问题、应用、数据、场景或能力限定，优先判为 `upper_topic` 或 `compressed_label`，不要轻易判成 `small_topic`
8. `small_topic` 更接近类似：
   - 基于视频的机器人训练
   - 自动驾驶极端场景仿真
   - 三维导航中的空间记忆
   - 基于轨迹的机器人控制
   - 基于视觉的机器人控制
   而不是：
   - 机器人训练平台
   - 世界模型驱动控制
   - 具身智能规划
9. 请注意多源性质：如果多个来源都提到该对象，但大多数来源只停留在上层主题表达，只有单条证据出现了更细的说法，优先判为 `upper_topic` 或 `compressed_label`，不要仅因“有一条更具体证据”就判成 `small_topic`
10. `small_topic` 应优先保留给这种情况：
   - 原始文本里已经出现较自然的小方向/小应用/小问题表达
   - 或多源证据能较一致地指向同一个具体场景/数据/问题
11. 像下面这类通常仍应留在 `upper_topic / compressed_label`：
   - 机器人训练
   - 智能体世界模型训练
   - 世界模型驱动控制
   - 具身智能任务规划

请只输出 JSON 数组，不要加额外解释。格式：
[
  {{
    "candidate_key": "...",
    "refined_topic_name": "...",
    "small_topic_judgment": "small_topic|upper_topic|compressed_label|unclear",
    "small_topic_type": "small_application|small_capability|small_problem|small_direction|unclear",
    "small_topic_pattern": "object+mechanism|scene+mechanism|data+mechanism|problem+mechanism|application+mechanism|unclear",
    "reason": "一句简短理由"
  }}
]

待处理候选：
{json.dumps(payload, ensure_ascii=False, indent=2)}
"""


def _select_rows_for_refinement(scored_df, top_k=20):
    if scored_df is None or scored_df.empty:
        return []
    subset = scored_df[
        (
            scored_df["signal_type"].isin(["weak_signal", "hotspot"])
        )
        | (
            scored_df.get("is_scope_internal_candidate", False) == True
        )
        | (
            scored_df.get("topic_granularity", "").isin(["fine_grained_topic", "scope_internal_candidate"])
        )
        | (
            scored_df.get("source_count", 0) >= 2
        )
    ].copy()
    if subset.empty:
        return []
    subset = subset.sort_values(
        by=["source_count", "weak_signal_score", "hotspot_score", "cluster_evidence_count", "total_mentions"],
        ascending=[False, False, False, False, False],
    ).drop_duplicates(subset=["display_candidate_name"], keep="first")
    return subset.head(top_k).to_dict(orient="records")


def _row_token_pool(row):
    values = []
    for key in [
        "constraint_signature",
        "raw_phrase_example",
        "display_candidate_name",
        "canonical_candidate_name_en",
    ]:
        values.append(str(row.get(key, "")).strip().lower())
    joined = " ".join(values)
    return set(re.findall(r"[a-z0-9]+", joined))


def _has_specific_anchor(row):
    token_pool = _row_token_pool(row)
    if token_pool & SPECIFIC_TASK_TOKENS:
        return True
    if token_pool & SPECIFIC_DATA_TOKENS:
        return True
    if token_pool & SPECIFIC_METHOD_TOKENS:
        return True
    object_tokens = {token for token in token_pool if token not in GENERIC_OBJECT_TOKENS}
    return bool(object_tokens & {"driving", "navigation", "manipulation", "sensor", "video", "3d", "memory", "visual"})


def _has_natural_source_support(row):
    evidence_text = " ".join(
        [
            str(row.get("raw_phrase_example", "")).strip().lower(),
            str(row.get("topic_naturalness_reason", "")).strip().lower(),
            str(row.get("constraint_signature", "")).strip().lower(),
        ]
    )
    signals = [
        "video",
        "visual",
        "sensor",
        "3d",
        "memory",
        "trajectory",
        "driving",
        "navigation",
        "manipulation",
        "safety",
        "problem",
        "scene",
        "application",
    ]
    return any(token in evidence_text for token in signals)


def _has_explicit_small_topic_phrase(evidence_text):
    cues = [
        "based on",
        "for ",
        "in ",
        "using ",
        "social navigation",
        "indoor",
        "indoor navigation",
        "collaborative",
        "coordination",
        "scene",
        "scenario",
        "path",
        "spatial",
        "physical",
        "memory",
        "reasoning",
        "用于",
        "基于",
        "面向",
        "场景",
        "路径",
        "空间",
        "记忆",
        "推理",
        "感知",
        "控制",
    ]
    return any(cue in evidence_text for cue in cues)


def _has_strong_small_topic_support(row, evidence_text):
    task_tokens = {str(token).strip().lower() for token in _safe_list(row.get("task_constraint_tokens", []))}
    object_tokens = {str(token).strip().lower() for token in _safe_list(row.get("object_modifier_tokens", []))}
    data_tokens = {str(token).strip().lower() for token in _safe_list(row.get("data_modifier_tokens", []))}
    mechanism = str(row.get("mechanism_core", "")).strip().lower()

    has_scene_or_application = bool((task_tokens | object_tokens) & SCENE_OR_APPLICATION_TOKENS)
    has_strong_data = bool(data_tokens & STRONG_DATA_TOKENS)
    has_weak_data = bool(data_tokens & WEAK_DATA_TOKENS)
    has_robotic_object = bool((task_tokens | object_tokens) & {"robot", "robotics", "agent"})
    has_explicit_phrase = _has_explicit_small_topic_phrase(evidence_text)

    if has_scene_or_application and has_explicit_phrase:
        return True
    if has_strong_data and (has_robotic_object or has_scene_or_application or has_explicit_phrase):
        return True
    if mechanism in {"reasoning", "memory"} and (has_scene_or_application or has_strong_data or "physical" in evidence_text):
        return True
    if mechanism in {"control", "planning"} and (has_scene_or_application or (has_strong_data and has_robotic_object)):
        return True
    if mechanism in {"training", "simulation"} and (
        has_scene_or_application
        or (has_strong_data and has_robotic_object)
        or ("driving" in task_tokens and has_explicit_phrase)
    ):
        return True
    if has_weak_data and not (has_scene_or_application or has_strong_data or has_robotic_object):
        return False
    return has_explicit_phrase and (has_scene_or_application or has_strong_data or has_robotic_object)


def _evidence_corpus(row):
    pieces = [
        str(row.get("display_candidate_name", "")).strip().lower(),
        str(row.get("raw_phrase_example", "")).strip().lower(),
        str(row.get("constraint_signature", "")).strip().lower(),
        str(row.get("topic_naturalness_reason", "")).strip().lower(),
    ]
    for item in _safe_list(row.get("evidence_items", []))[:6]:
        if isinstance(item, dict):
            pieces.extend(
                [
                    str(item.get("title", "")).strip().lower(),
                    str(item.get("snippet", "")).strip().lower(),
                    str(item.get("raw_candidate_text", "")).strip().lower(),
                ]
            )
    return " ".join(piece for piece in pieces if piece)


def _surface_name_from_evidence(row, judgment, pattern, refined_name):
    scope_name = str(row.get("scope_name", "")).strip().lower()
    mechanism = str(row.get("mechanism_core", "")).strip().lower()
    evidence_text = _evidence_corpus(row)
    display_name = str(refined_name or row.get("display_candidate_name", "")).strip()
    task_tokens = {str(token).strip().lower() for token in _safe_list(row.get("task_constraint_tokens", []))}
    object_tokens = {str(token).strip().lower() for token in _safe_list(row.get("object_modifier_tokens", []))}
    data_tokens = {str(token).strip().lower() for token in _safe_list(row.get("data_modifier_tokens", []))}

    if judgment == "small_topic" and scope_name == "world model":
        if mechanism == "planning" and "robot" in (task_tokens | object_tokens):
            if "social navigation" in evidence_text:
                return "机器人社交导航规划"
            if "indoor" in evidence_text and any(term in evidence_text for term in ["coordination", "collaborative", "cooperative"]):
                return "室内机器人协同规划"
        if mechanism == "simulation" and ("driving" in task_tokens or "autonomous driving" in evidence_text or "self-driving" in evidence_text):
            if any(term in evidence_text for term in ["scenario", "situations", "extreme", "closed-loop"]):
                return "自动驾驶场景仿真"
            return "自动驾驶世界模型仿真"
        if mechanism == "control" and "trajectory" in data_tokens and "robot" in (task_tokens | object_tokens):
            return "基于轨迹的机器人控制"
        if mechanism == "control" and "visual" in data_tokens and "robot" in (task_tokens | object_tokens):
            return "基于视觉的机器人控制"
        if mechanism == "training" and "visual" in data_tokens:
            if any(term in evidence_text for term in ["visual feature", "visual features", "pixels", "joint-embedding", "pre-trained visual"]):
                return "视觉表征世界模型训练"
            return "基于视觉的世界模型训练"
        if mechanism == "training" and "video" in data_tokens and "robot" in (task_tokens | object_tokens):
            return "基于视频的机器人训练"

    if scope_name == "world model" and judgment in {"upper_topic", "compressed_label"}:
        if mechanism == "planning" and "robot" in (task_tokens | object_tokens):
            if "social navigation" in evidence_text:
                return "机器人社交导航规划"
            if "indoor" in evidence_text and any(term in evidence_text for term in ["coordination", "collaborative", "cooperative"]):
                return "室内机器人协同规划"
            if any(term in evidence_text for term in ["real-time", "near-real-time"]):
                return "机器人实时规划"
            if "task planning" in evidence_text:
                return "机器人任务规划"
            return "机器人世界模型规划"
        if mechanism == "training" and "robot" in (task_tokens | object_tokens):
            if any(term in evidence_text for term in ["for robot training", "robot training"]):
                return "机器人训练用世界模型"
            return "机器人世界模型训练"

    return display_name


def _post_adjust_refinement(row, item):
    adjusted = dict(item)
    judgment = str(adjusted.get("small_topic_judgment", "")).strip()
    pattern = str(adjusted.get("small_topic_pattern", "")).strip()
    reason = str(adjusted.get("reason", "")).strip()
    evidence_text = _evidence_corpus(row)
    source_count = int(row.get("source_count", 0) or 0)
    non_scope_constraint_count = int(row.get("non_scope_constraint_count", 0) or 0)
    has_specific_anchor = _has_specific_anchor(row)
    natural_source_support = _has_natural_source_support(row)
    strong_small_topic_support = _has_strong_small_topic_support(row, evidence_text)
    if judgment == "small_topic":
        if pattern == "object+mechanism" and (non_scope_constraint_count <= 1 or not has_specific_anchor):
            adjusted["small_topic_judgment"] = "upper_topic"
            adjusted["reason"] = reason or "当前对象已进入 scope 内细候选层，但仍更像对象+机制核的能力标签，需继续收口。"
        elif pattern in {"unclear", ""}:
            adjusted["small_topic_judgment"] = "compressed_label"
            adjusted["reason"] = reason or "当前对象仍偏系统压缩标签，尚不足以直接判为自然小主题。"
        elif source_count < 2 and pattern not in {"scene+mechanism", "data+mechanism", "application+mechanism", "problem+mechanism"}:
            adjusted["small_topic_judgment"] = "upper_topic"
            adjusted["reason"] = reason or "当前对象缺少更稳定的多源支撑，宜先保留为 scope 内细候选。"
        elif not strong_small_topic_support:
            adjusted["small_topic_judgment"] = "upper_topic"
            adjusted["reason"] = "当前对象虽已较像 scope 内细候选，但原始证据对具体场景/应用/问题的支撑仍不够稳定，宜继续保留为待收口对象。"
    final_judgment = str(adjusted.get("small_topic_judgment", "")).strip()
    final_pattern = str(adjusted.get("small_topic_pattern", "")).strip()
    if final_judgment == "compressed_label" and has_specific_anchor and final_pattern in PROMOTABLE_PATTERNS:
        adjusted["small_topic_judgment"] = "upper_topic"
        adjusted["reason"] = reason or "当前对象已有较明确的场景/数据边界，但仍需继续收口为更自然的小主题表达。"
        final_judgment = "upper_topic"
    if (
        final_judgment == "upper_topic"
        and final_pattern in PROMOTABLE_PATTERNS
        and source_count >= 2
        and has_specific_anchor
        and natural_source_support
        and strong_small_topic_support
    ):
        adjusted["small_topic_judgment"] = "small_topic"
        adjusted["reason"] = reason or "当前对象已具备较明确的场景/数据/应用边界，且多源证据支持，适合作为更自然的小主题表达。"
        final_judgment = "small_topic"
    if (
        final_judgment == "upper_topic"
        and source_count >= 2
        and str(row.get("mechanism_core", "")).strip().lower() == "planning"
        and "robot" in evidence_text
        and any(term in evidence_text for term in ["social navigation", "indoor navigation", "collaborative", "coordination"])
    ):
        adjusted["small_topic_judgment"] = "small_topic"
        adjusted["small_topic_pattern"] = "scene+mechanism"
        adjusted["reason"] = "原始证据已经出现较明确的机器人导航/协同场景表达，适合作为更自然的小主题对象。"
        final_judgment = "small_topic"

    if final_judgment == "compressed_label":
        if "robot" in evidence_text and str(row.get("mechanism_core", "")).strip().lower() in {"training", "planning"} and source_count >= 3:
            adjusted["small_topic_judgment"] = "upper_topic"
            adjusted["reason"] = "当前对象已进入 scope 内细候选层，原始证据能支持更自然的对象表述，但仍不足以直接判为自然小主题。"
            final_judgment = "upper_topic"
    if (
        final_judgment == "upper_topic"
        and source_count >= 2
        and str(row.get("mechanism_core", "")).strip().lower() in {"control", "simulation", "training"}
        and str(adjusted.get("small_topic_pattern", "")).strip() in {"data+mechanism", "scene+mechanism", "application+mechanism"}
        and any(token in evidence_text for token in ["video", "visual", "trajectory", "driving", "navigation", "social navigation"])
    ):
        adjusted["small_topic_judgment"] = "small_topic"
        adjusted["reason"] = "当前对象具备较明确的数据或场景边界，并且多源证据对该边界支持较稳定，可上推为自然小主题。"
        final_judgment = "small_topic"

    refined_name = _surface_name_from_evidence(
        row,
        final_judgment,
        str(adjusted.get("small_topic_pattern", "")).strip(),
        str(adjusted.get("refined_topic_name", "")).strip(),
    )
    if refined_name:
        adjusted["refined_topic_name"] = refined_name
    return adjusted


def refine_research_scored_candidates(scored_df, cache_path=None, refresh_cache=False, top_k=20, batch_size=5):
    if scored_df is None or scored_df.empty:
        print("[topic_refiner] 跳过：输入 scored_df 为空。")
        return scored_df

    defaults = {
        "rule_display_candidate_name": "",
        "rule_topic_summary_name": "",
        "llm_refined_topic_name": "",
        "llm_small_topic_judgment": "",
        "llm_small_topic_type": "",
        "llm_small_topic_pattern": "",
        "llm_small_topic_reason": "",
        "llm_refined": False,
    }
    refined_df = scored_df.copy()
    for column, default in defaults.items():
        if column not in refined_df.columns:
            refined_df[column] = default
    if "rule_display_candidate_name" in refined_df.columns:
        refined_df["rule_display_candidate_name"] = refined_df["rule_display_candidate_name"].where(
            refined_df["rule_display_candidate_name"].astype(str).str.strip() != "",
            refined_df.get("display_candidate_name", ""),
        )
    if "rule_topic_summary_name" in refined_df.columns:
        refined_df["rule_topic_summary_name"] = refined_df["rule_topic_summary_name"].where(
            refined_df["rule_topic_summary_name"].astype(str).str.strip() != "",
            refined_df.get("topic_summary_name", ""),
        )

    provider, client = get_provider_and_client()
    if client is None or provider is None:
        print("[topic_refiner] 跳过：未检测到可用 LLM API 配置。")
        return refined_df

    selected_rows = _select_rows_for_refinement(refined_df, top_k=top_k)
    if not selected_rows:
        print("[topic_refiner] 跳过：当前没有候选进入收口队列。")
        return refined_df

    cache_metadata, cached_items = ({}, {}) if refresh_cache else load_topic_cache_bundle(cache_path)
    if cache_metadata.get("prompt_version") != PROMPT_VERSION:
        if cache_metadata:
            print(
                "[topic_refiner] 缓存 prompt_version 不匹配，忽略旧缓存："
                f"{cache_metadata.get('prompt_version')} -> {PROMPT_VERSION}"
            )
        cached_items = {}
    pending = [row for row in selected_rows if _candidate_key(row) not in cached_items]
    cache_hits = len(selected_rows) - len(pending)
    print(
        "[topic_refiner] 启动："
        f" cache_path={cache_path or 'None'} | "
        f"selected={len(selected_rows)} | cache_hits={cache_hits} | pending={len(pending)} | "
        f"refresh_cache={refresh_cache}"
    )

    api_written_count = 0

    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        if not batch:
            continue
        prompt = _build_prompt(batch)
        try:
            # 使用配置的Agent模型进行主题细化
            agent_model = os.getenv("AGENT_MODEL", os.getenv("OPENAI_MODEL", "Qwen/Qwen2.5-14B-Instruct"))
            content, usage_info, _ = chat_text(
                prompt,
                model=agent_model,
                max_tokens=1200,
                temperature=0.2,
                timeout=90,
            )
            parsed = _parse_json_from_response(content)
            if isinstance(parsed, dict):
                parsed = parsed.get("items") or parsed.get("results") or parsed.get("data")
            if not isinstance(parsed, list):
                continue
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                candidate_key = str(item.get("candidate_key", "")).strip()
                if not candidate_key:
                    continue
                normalized_item = _post_adjust_refinement(next((row for row in batch if _candidate_key(row) == candidate_key), {}), item)
                cached_items[candidate_key] = {
                    "refined_topic_name": str(normalized_item.get("refined_topic_name", "")).strip(),
                    "small_topic_judgment": str(normalized_item.get("small_topic_judgment", "")).strip(),
                    "small_topic_type": str(normalized_item.get("small_topic_type", "")).strip(),
                    "small_topic_pattern": str(normalized_item.get("small_topic_pattern", "")).strip(),
                    "reason": str(normalized_item.get("reason", "")).strip(),
                }
                api_written_count += 1
            if usage_info["prompt_tokens"] or usage_info["completion_tokens"]:
                record_call(
                    call_type="小主题收口",
                    prompt_tokens=usage_info["prompt_tokens"],
                    completion_tokens=usage_info["completion_tokens"],
                    success=True,
                )
        except Exception as exc:
            print(f"[topic_refiner] 批次调用失败，batch_start={start}，error={exc}")
            continue

    if cache_path:
        save_topic_cache(cache_path, cached_items, metadata={"top_k": top_k, "batch_size": batch_size, "prompt_version": PROMPT_VERSION})

    refined_df["topic_refiner_active"] = True
    applied_count = 0
    for idx, row in refined_df.iterrows():
        candidate_key = _candidate_key(row)
        item = cached_items.get(candidate_key)
        if not item:
            continue
        refined_name = str(item.get("refined_topic_name", "")).strip()
        judgment = str(item.get("small_topic_judgment", "")).strip()
        topic_type = str(item.get("small_topic_type", "")).strip()
        topic_pattern = str(item.get("small_topic_pattern", "")).strip()
        reason = str(item.get("reason", "")).strip()
        if refined_name:
            refined_df.at[idx, "llm_refined_topic_name"] = refined_name
            refined_df.at[idx, "display_candidate_name"] = refined_name
            refined_df.at[idx, "topic_summary_name"] = refined_name
        if judgment:
            refined_df.at[idx, "llm_small_topic_judgment"] = judgment
        if topic_type:
            refined_df.at[idx, "llm_small_topic_type"] = topic_type
        if topic_pattern:
            refined_df.at[idx, "llm_small_topic_pattern"] = topic_pattern
        if reason:
            refined_df.at[idx, "llm_small_topic_reason"] = reason
            refined_df.at[idx, "topic_naturalness_reason"] = reason
        refined_df.at[idx, "llm_refined"] = True
        applied_count += 1

    print(
        "[topic_refiner] 完成："
        f" cache_items={len(cached_items)} | api_written={api_written_count} | applied={applied_count}"
    )

    return refined_df
