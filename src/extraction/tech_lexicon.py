from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List
from urllib.parse import quote_plus


TECH_ALIASES: Dict[str, List[str]] = {
    "humanoid robot": [
        "humanoid robot",
        "humanoid robots",
        "humanoid",
        "humanoids",
        "bipedal robot",
        "bipedal humanoid",
        "dexterous hand",
        "dexterous manipulation",
        "robotic manipulator",
        "robot manipulator",
        "robot arm",
        "robotic arm",
        "end effector",
        "end-effector",
        "人形机器人",
        "类人机器人",
        "双足机器人",
        "仿人机器人",
        "灵巧手",
        "机械灵巧手",
        "机械臂",
        "机械手",
        "机械手指",
        "机器人末端执行器",
        "双臂机器人",
        "双臂",
        "仿生手",
        "夹爪",
    ],
    "embodied intelligence": [
        "embodied intelligence",
        "embodied ai",
        "embodied agent",
        "embodied agents",
        "embodied robotics",
        "embodied planning",
        "physical ai",
        "具身智能",
        "具身智能机器人",
        "具身世界模型",
    ],
    "world model": [
        "world model",
        "world models",
        "world modeling",
        "video world model",
        "robot world model",
        "embodied world model",
        "physical world model",
        "世界模型",
        "具身世界模型",
        "4d world modeling",
    ],
    "edge intelligence": ["edge intelligence", "edge ai"],
    "neuromorphic": ["neuromorphic", "spiking neural network", "snn", "event-based vision"],
    "small model": ["small model", "small language model", "slm", "compact model"],
    "agentic planning": ["agentic planning", "agent planning", "planning agent", "agentic workflow"],
    "robot learning": ["robot learning", "robotic learning", "robot manipulation learning"],
    "synthetic data": ["synthetic data", "synthetic dataset", "simulation data"],
    "multimodal": ["multimodal", "multi-modal", "多模态"],
    "llm": ["llm", "large language model", "foundation model", "大语言模型"],
    "ai": ["artificial intelligence", "ai", "人工智能"],
    "robot": ["robot", "robotics", "机器人", "humanoid"],
    "automation": ["automation", "autonomous system", "自动化"],
    "generative ai": ["generative ai", "generative model", "生成式ai", "生成式人工智能"],
    "machine learning": ["machine learning", "机器学习"],
    "neural network": ["neural network", "deep learning", "神经网络"],
}


DOMINANT_TECH_TERMS = {"ai", "machine learning", "llm", "generative ai", "neural network"}
# 人形机器人提升为主 scope（第一位），具身智能 / 世界模型保留为背景 / family_backbone scope。
# 因为 detect_observation_scopes 按 OBSERVATION_SCOPE_TERMS 顺序收集匹配，
# 当文本同时命中 humanoid robot 和 embodied intelligence 时，primary_scope 为 humanoid robot。
OBSERVATION_SCOPE_TERMS = ("humanoid robot", "embodied intelligence", "world model")
OBSERVATION_SCOPE_SET = set(OBSERVATION_SCOPE_TERMS)
BROAD_TECH_TERMS = DOMINANT_TECH_TERMS | OBSERVATION_SCOPE_SET | {
    "robot", "automation", "multimodal", "robot learning",
}

# 机器人领域锚点：用于 raw/domain filter。candidate_unit 级别如果 source_title + source_text
# 完全不命中这些锚点，视为非机器人领域，直接丢弃 unit（防止挖掘机 / 儿童防坠床 / 建材等
# 专利因 raw_phrase 英文 token 相似而渗漏到机器人操控 cluster）。
ROBOT_DOMAIN_ANCHORS = {
    # 直接机器人实体词
    "robot", "robots", "robotic", "robotics",
    "humanoid", "humanoids", "bipedal", "quadruped",
    "manipulator", "manipulators", "gripper", "grippers",
    "dexterous", "end-effector", "end effector",
    # 中文机器人实体词
    "机器人", "人形机器人", "类人机器人", "双足机器人", "仿人机器人",
    "机械臂", "机械手", "机械手指", "灵巧手", "夹爪", "末端执行器",
    "双臂", "单臂", "仿生手", "机械灵巧手",
    # 具身 / 机器人强相关上下文（本项目主线内基本都属于机器人）
    "具身", "具身智能",
    "tactile", "haptic", "触觉",
    "grasp", "grasping", "抓取",
}


def has_robot_domain_anchor(text: str) -> bool:
    """判断一段文本是否命中至少一个机器人领域锚点。
    用于 raw signal 过滤：挖掘机、儿童床等与人形机器人无关的专利应被此过滤门挡掉。"""
    if not text:
        return False
    haystack = str(text or "").lower()
    for anchor in ROBOT_DOMAIN_ANCHORS:
        if anchor in haystack:
            return True
    return False

SCOPE_CONTEXT_TERMS = {
    "humanoid robot": {
        "humanoid", "robot", "robotics", "bipedal", "dexterous",
        "manipulation", "manipulator", "gripper", "grasp", "grasping",
        "arm", "hand", "finger", "tactile", "haptic", "perception",
        "locomotion", "walking", "whole-body", "whole body",
        "control", "embodied", "end-effector", "end effector",
    },
    "embodied intelligence": {
        "embodied", "robot", "robotics", "decision", "decision-making", "planning",
        "control", "manipulation", "interactive", "simulation", "grounding",
        "policy", "agent", "reasoning", "training", "framework",
    },
    "world model": {
        "world", "model", "models", "planning", "simulation", "interactive",
        "robot", "robotics", "embodied", "decision", "decision-making",
        "training", "framework", "generative", "environment", "modeling",
        "control", "reasoning", "policy", "video",
    },
}

DISCOVERY_HEADWORDS = {
    "model", "models", "learning", "planning", "reasoning", "robot", "robotics",
    "agent", "agents", "network", "networks", "intelligence",
    "simulation", "simulator", "benchmark", "benchmarks", "grounding",
    "perception", "world", "multimodal", "neuromorphic",
}

DISCOVERY_CORE_TERMS = {
    "model", "models", "learning", "robot", "robotics", "agent", "agents",
    "network", "networks", "planning", "reasoning",
    "multimodal", "neuromorphic", "grounding", "perception", "simulation",
}

DISCOVERY_STOPWORDS = {
    "the", "a", "an", "and", "or", "for", "with", "from", "into", "over", "under",
    "based", "using", "via", "toward", "towards", "through", "on", "in", "of", "to",
    "by", "at", "across", "beyond", "new", "real", "world", "large", "small",
    "calculating", "impact", "effects", "study", "analysis", "systematic",
}

CONNECTOR_TERMS = {"for", "in", "with", "via", "using", "toward", "towards", "based", "on", "of"}
SCOPE_ECHO_SHELL_TOKENS = {
    "model", "models", "modeling", "framework", "frameworks", "system", "systems",
    "method", "methods", "approach", "approaches", "technology", "technologies",
    "technique", "techniques", "application", "applications", "video", "robot",
    "robots", "robotics", "interactive", "generative",
}
BROAD_OBJECT_SHELL_TOKENS = {
    "robot", "robots", "robotics", "video", "videos", "interactive", "generative",
    "embodied", "agent", "agents",
}
BARE_MECHANISM_CORES = {
    "training", "planning", "control", "simulation", "reasoning", "memory", "grounding",
}

MECHANISM_CORE_ALIASES: Dict[str, List[str]] = {
    "training": ["training", "train", "trained", "fine tuning", "fine-tuning", "optimization", "optimize", "训练", "skill learning", "技能学习", "技能习得", "模仿学习"],
    "planning": ["planning", "plan", "planner", "规划", "path planning", "route planning", "motion planning", "路径规划", "地图构建", "建图", "局部地图构建"],
    "simulation": ["simulation", "simulator", "simulating", "仿真"],
    "calibration": ["calibration", "calibrate", "calibrated"],
    "compression": ["compression", "compress", "compressed", "quantization", "quantized"],
    "memory": ["memory", "spatial memory", "long term memory", "long-term memory", "记忆"],
    "tactile": ["tactile", "haptic", "触觉"],
    "reasoning": ["reasoning", "reasoner"],
    "control": ["control", "controller", "controlling", "控制", "操控", "操纵", "抓取", "码垛", "位姿调节"],
    "grounding": ["grounding", "grounded", "落地"],
    "alignment": ["alignment", "aligned", "preference optimization"],
    "policy": ["policy", "policies", "策略", "策略模型"],
    "retrieval": ["retrieval", "retrieve", "retriever", "检索"],
    "distillation": ["distillation", "distill", "distilled"],
}

TASK_CONSTRAINT_ALIASES: Dict[str, List[str]] = {
    "robot": ["robot", "robots", "机器人", "人形机器人"],
    "robotics": ["robotics"],
    "manipulation": ["manipulation", "manipulator", "grasping", "操控", "抓取", "码垛", "搬运"],
    "assembly": ["assembly", "assemble", "assembling", "装配", "装联", "装调", "插接"],
    "navigation": ["navigation", "nav", "导航", "路径规划", "定位", "建图", "地图构建"],
    "driving": ["driving", "autonomous driving", "驾驶"],
    "video": ["video", "videos", "视频"],
    "interactive": ["interactive", "interaction", "interacting", "交互"],
    "embodied": ["embodied", "embodiment", "具身"],
    "multimodal": ["multimodal", "multi modal", "multi-modal", "多模态"],
    "agent": ["agent", "agents", "agentic", "智能体"],
    "decision-making": ["decision making", "decision-making"],
    "control": ["control", "controller", "controlling", "控制"],
    "benchmark": ["benchmark", "benchmarks"],
    "evaluation": ["evaluation", "evaluate", "evaluation-driven"],
}
OBJECT_MODIFIER_ALIASES: Dict[str, List[str]] = {
    "robot": ["robot", "robots", "robotic", "机器人", "人形机器人"],
    "agent": ["agent", "agents", "agentic", "智能体"],
    "policy": ["policy", "policies", "策略", "策略模型"],
    "environment": ["environment", "environmental"],
    "manipulator": ["manipulator", "gripper", "grasping", "操控", "robot arm", "robotic arm", "机械臂", "双臂", "单臂", "灵巧手", "夹爪"],
    "navigation": ["navigation", "navigator", "导航"],
    "decision-making": ["decision making", "decision-making"],
    "control": ["control", "controller", "controlling", "控制"],
    "memory": ["memory", "spatial memory", "long term memory", "long-term memory", "记忆"],
    "tactile": ["tactile", "haptic", "触觉"],
    "perception": ["perception", "perceptual", "感知"],
}
DATA_MODIFIER_ALIASES: Dict[str, List[str]] = {
    "video": ["video", "videos", "视频"],
    "multimodal": ["multimodal", "multi modal", "multi-modal", "多模态"],
    "temporal": ["temporal", "long-horizon", "streaming"],
    "3d": ["3d", "three-dimensional", "spatial", "三维", "3d", "point cloud", "point-cloud", "3d point cloud", "3d point-cloud", "三维点云", "空间点云", "点云"],
    "egocentric": ["egocentric", "ego-centric"],
    "simulation": ["simulation", "simulator", "simulated", "仿真"],
    "trajectory": ["trajectory", "trajectories", "轨迹"],
    "sensor": ["sensor", "sensors", "传感", "imu", "激光雷达", "雷达", "深度相机"],
    "visual": ["visual", "vision", "视觉", "图像", "彩色图像", "camera", "cameras"],
}
METHOD_MODIFIER_ALIASES: Dict[str, List[str]] = {
    "causal": ["causal", "causality-aware", "因果"],
    "dynamic": ["dynamic", "dynamical", "dynamics-aware", "动态"],
    "sparse": ["sparse", "sparsity-aware", "稀疏"],
    "retrieval-based": ["retrieval-based", "retrieval based", "memory retrieval", "dynamic retrieval"],
    "reinforcement": ["reinforcement", "reinforcement learning", "rl", "强化学习", "强化"],
    "policy-guided": ["policy-guided", "policy guided", "guided policy", "policy-aware", "策略引导"],
}
WORLD_MODEL_CONTEXT_TERMS = {
    "ai", "robot", "robotics", "embodied", "agent", "agents", "video",
    "planning", "simulation", "control", "policy", "decision", "decision-making",
    "environment", "navigation", "manipulation", "multimodal",
}

PATENT_PROXY_SCOPE_TERMS = {
    # humanoid robot proxy：要求较强的机器人/操控/触觉类 token 命中
    "humanoid robot": {"robot", "manipulator", "tactile", "dexterous", "humanoid", "grasping", "manipulation"},
    # world model proxy 维持原状
    "world model": {"robot", "agent", "video", "visual", "trajectory", "simulation", "driving", "planning", "3d"},
    # embodied intelligence proxy：收紧，去掉 control/3d/sensor/perception 这类过泛 token
    "embodied intelligence": {"robot", "agent", "tactile", "manipulation", "navigation", "grounding"},
}

PROXY_TOKEN_ZH_MAP: Dict[str, str] = {
    "trajectory": "轨迹",
    "trajectories": "轨迹",
    "perception": "感知",
    "perceptual": "感知",
    "simulation": "仿真",
    "simulator": "仿真",
    "control": "控制",
    "controller": "控制",
    "controlling": "控制",
    "reasoning": "推理",
    "planning": "规划",
    "training": "训练",
    "visual": "视觉",
    "vision": "视觉",
    "sensor": "传感",
    "sensors": "传感",
    "robot": "机器人",
    "robotics": "机器人",
    "driving": "驾驶",
    "video": "视频",
    "agent": "智能体",
    "agents": "智能体",
}

SCENE_TOKEN_SET = {
    "robot", "robotics", "manipulation", "assembly", "navigation",
    "video", "interactive", "embodied", "multimodal",
}

NON_SHELL_TASK_TOKENS = {
    "manipulation", "assembly", "navigation", "decision-making", "control", "benchmark", "evaluation",
}


@dataclass(frozen=True)
class TopicBundle:
    topic: str
    canonical_topic: str
    research_mode: str
    rss_feeds: List[str]
    rss_keywords: List[str]
    filter_keywords: List[str]
    arxiv_query: str
    patent_query: str
    note: str


TOPIC_PRESETS: Dict[str, Dict[str, List[str] | str]] = {
    "ai": {
        "aliases": ["ai", "artificial intelligence", "machine learning", "robotics", "large language model", "multimodal", "foundation model"],
        "validation_aliases": ["world model", "embodied intelligence"],
        "rss_queries": ["artificial intelligence", "machine learning robotics"],
        "note": "AI 主题当前最完整，RSS、论文和专利链路都围绕 AI 领域设计。",
    },
    "new_energy": {
        "aliases": ["new energy", "renewable energy", "energy storage", "battery", "photovoltaic", "wind power", "hydrogen energy"],
        "validation_aliases": ["solid-state battery", "sodium-ion battery", "perovskite solar", "hydrogen storage", "grid storage"],
        "rss_queries": ["new energy", "renewable energy battery"],
        "note": "新能源主题已接入主题化 RSS 检索，但论文与专利侧仍是更稳定的主通道。",
    },
}


DEFAULT_RSS_FEEDS = [
    "https://www.technologyreview.com/feed/",
    "https://www.wired.com/feed/category/science/latest/rss",
]


def aliases_for(term: str) -> List[str]:
    canonical = str(term).strip().lower()
    aliases = TECH_ALIASES.get(canonical, [])
    values = [canonical, *aliases]
    seen = set()
    deduped = []
    for value in values:
        if value not in seen:
            deduped.append(value)
            seen.add(value)
    return deduped


def normalize_signal_phrase(text: str) -> str:
    return _normalize_text_key(text)


def tokenize_signal_phrase(text: str) -> List[str]:
    normalized = normalize_signal_phrase(text)
    if not normalized:
        return []
    return [token for token in normalized.replace("-", " ").split() if token]


def _extract_canonical_terms(texts: Iterable[str], alias_map: Dict[str, List[str]]) -> List[str]:
    haystack = " ".join(normalize_signal_phrase(text) for text in texts if str(text).strip())
    if not haystack:
        return []

    matched = []
    for canonical, aliases in alias_map.items():
        terms = [canonical, *aliases]
        if any(normalize_signal_phrase(alias) in haystack for alias in terms):
            matched.append(canonical)
    return matched


def extract_mechanism_core_tokens(*texts: str) -> List[str]:
    return _extract_canonical_terms(texts, MECHANISM_CORE_ALIASES)


def extract_task_constraint_tokens(*texts: str) -> List[str]:
    return _extract_canonical_terms(texts, TASK_CONSTRAINT_ALIASES)


def extract_scene_tokens(*texts: str) -> List[str]:
    return [token for token in extract_task_constraint_tokens(*texts) if token in SCENE_TOKEN_SET]


def extract_object_modifier_tokens(*texts: str) -> List[str]:
    return _extract_canonical_terms(texts, OBJECT_MODIFIER_ALIASES)


def extract_data_modifier_tokens(*texts: str) -> List[str]:
    return _extract_canonical_terms(texts, DATA_MODIFIER_ALIASES)


def extract_method_modifier_tokens(*texts: str) -> List[str]:
    return _extract_canonical_terms(texts, METHOD_MODIFIER_ALIASES)


def _scope_alias_token_set(scopes: Iterable[str]) -> set[str]:
    tokens: set[str] = set()
    for scope in scopes:
        for alias in aliases_for(scope):
            tokens.update(tokenize_signal_phrase(alias))
    return tokens


def strip_scope_and_shell_tokens(text: str, scopes: Iterable[str]) -> List[str]:
    scope_tokens = _scope_alias_token_set(scopes)
    tokens = tokenize_signal_phrase(text)
    filtered = []
    for token in tokens:
        if token in scope_tokens:
            continue
        if token in SCOPE_ECHO_SHELL_TOKENS:
            continue
        if token in CONNECTOR_TERMS or token in DISCOVERY_STOPWORDS:
            continue
        filtered.append(token)
    return filtered


def is_scope_echo_candidate(
    text: str,
    scopes: Iterable[str],
    mechanism_tokens: Iterable[str] | None = None,
    task_tokens: Iterable[str] | None = None,
) -> bool:
    mechanism_tokens = [str(token).strip() for token in (mechanism_tokens or []) if str(token).strip()]
    task_tokens = [str(token).strip() for token in (task_tokens or []) if str(token).strip()]
    qualifying_task_tokens = [token for token in task_tokens if token in NON_SHELL_TASK_TOKENS]
    if mechanism_tokens or qualifying_task_tokens:
        return False
    return True


def supports_world_model_context(text: str) -> bool:
    haystack = normalize_signal_phrase(text)
    if "world model" not in haystack and "world models" not in haystack and "world modeling" not in haystack:
        return False
    tokens = set(tokenize_signal_phrase(haystack))
    return bool(tokens & WORLD_MODEL_CONTEXT_TERMS)


def detect_supported_observation_scopes(text: str) -> List[str]:
    scopes = detect_observation_scopes(text)
    if "world model" in scopes and not supports_world_model_context(text):
        scopes = [scope for scope in scopes if scope != "world model"]
    return scopes


def detect_patent_proxy_scopes(
    mechanism_tokens: Iterable[str] | None = None,
    task_tokens: Iterable[str] | None = None,
    object_tokens: Iterable[str] | None = None,
    data_tokens: Iterable[str] | None = None,
    scene_tokens: Iterable[str] | None = None,
) -> List[str]:
    signal_tokens = {
        str(token).strip()
        for token in [
            *(mechanism_tokens or []),
            *(task_tokens or []),
            *(object_tokens or []),
            *(data_tokens or []),
            *(scene_tokens or []),
        ]
        if str(token).strip()
    }
    if not signal_tokens:
        return []

    scopes = []
    # humanoid robot 放在最前，与主 scope 顺序保持一致
    if signal_tokens & PATENT_PROXY_SCOPE_TERMS["humanoid robot"]:
        scopes.append("humanoid robot")
    if signal_tokens & PATENT_PROXY_SCOPE_TERMS["world model"]:
        scopes.append("world model")
    if signal_tokens & PATENT_PROXY_SCOPE_TERMS["embodied intelligence"]:
        scopes.append("embodied intelligence")
    return scopes


def detect_context_proxy_scopes(
    mechanism_tokens: Iterable[str] | None = None,
    task_tokens: Iterable[str] | None = None,
    object_tokens: Iterable[str] | None = None,
    data_tokens: Iterable[str] | None = None,
    scene_tokens: Iterable[str] | None = None,
    method_tokens: Iterable[str] | None = None,
) -> List[str]:
    signal_tokens = {
        str(token).strip()
        for token in [
            *(mechanism_tokens or []),
            *(task_tokens or []),
            *(object_tokens or []),
            *(data_tokens or []),
            *(scene_tokens or []),
            *(method_tokens or []),
        ]
        if str(token).strip()
    }
    if not signal_tokens:
        return []

    scopes = []
    humanoid_hits = signal_tokens & PATENT_PROXY_SCOPE_TERMS["humanoid robot"]
    embodied_hits = signal_tokens & PATENT_PROXY_SCOPE_TERMS["embodied intelligence"]
    world_hits = signal_tokens & PATENT_PROXY_SCOPE_TERMS["world model"]
    if humanoid_hits and (mechanism_tokens or task_tokens or object_tokens or data_tokens):
        scopes.append("humanoid robot")
    if embodied_hits and (mechanism_tokens or task_tokens or object_tokens or data_tokens):
        scopes.append("embodied intelligence")
    world_model_gate = {"planning", "simulation", "video", "3d", "trajectory", "temporal", "memory", "navigation"}
    if world_hits and signal_tokens & world_model_gate:
        scopes.append("world model")
    return scopes


def diagnose_observation_scope_detection(
    text: str,
    source_type: str = "",
    mechanism_tokens: Iterable[str] | None = None,
    task_tokens: Iterable[str] | None = None,
    object_tokens: Iterable[str] | None = None,
    data_tokens: Iterable[str] | None = None,
    scene_tokens: Iterable[str] | None = None,
    method_tokens: Iterable[str] | None = None,
) -> Dict[str, object]:
    haystack = normalize_signal_phrase(text)
    direct_matches: List[str] = []
    alias_matches: Dict[str, List[str]] = {}
    for scope in OBSERVATION_SCOPE_TERMS:
        canonical_norm = normalize_signal_phrase(scope)
        if canonical_norm and canonical_norm in haystack:
            direct_matches.append(scope)
        matched_aliases = []
        for alias in aliases_for(scope):
            alias_norm = normalize_signal_phrase(alias)
            if alias_norm and alias_norm in haystack and alias_norm != canonical_norm:
                matched_aliases.append(alias)
        if matched_aliases:
            alias_matches[scope] = matched_aliases[:3]

    supported_scopes = detect_supported_observation_scopes(text)
    rejected_scopes = [
        scope
        for scope in set(direct_matches) | set(alias_matches.keys())
        if scope not in supported_scopes
    ]

    source_type = str(source_type or "").strip().lower()
    proxy_scopes: List[str] = []
    proxy_mode = ""
    if not supported_scopes:
        if source_type == "patent":
            proxy_scopes = detect_patent_proxy_scopes(
                mechanism_tokens=mechanism_tokens,
                task_tokens=task_tokens,
                object_tokens=object_tokens,
                data_tokens=data_tokens,
                scene_tokens=scene_tokens,
            )
            proxy_mode = "proxy_patent" if proxy_scopes else ""
        elif source_type in {"paper", "report", "news"}:
            proxy_scopes = detect_context_proxy_scopes(
                mechanism_tokens=mechanism_tokens,
                task_tokens=task_tokens,
                object_tokens=object_tokens,
                data_tokens=data_tokens,
                scene_tokens=scene_tokens,
                method_tokens=method_tokens,
            )
            proxy_mode = "proxy_context" if proxy_scopes else ""

    final_scopes = supported_scopes or proxy_scopes
    if supported_scopes:
        reason = "命中显式 scope alias"
    elif proxy_scopes:
        reason = f"未命中显式 scope，转为 {proxy_mode}"
    elif rejected_scopes:
        reason = "命中 scope alias，但上下文支持不足"
    else:
        reason = "未命中直接匹配、alias 匹配或 proxy 推断"

    return {
        "supported_scopes": final_scopes,
        "scope_match_mode": "explicit" if supported_scopes else proxy_mode,
        "direct_matches": direct_matches,
        "alias_matches": alias_matches,
        "rejected_scopes": rejected_scopes,
        "proxy_scopes": proxy_scopes,
        "reason": reason,
    }


def has_non_scope_constraint(
    task_tokens: Iterable[str] | None = None,
    object_tokens: Iterable[str] | None = None,
    data_tokens: Iterable[str] | None = None,
    scene_tokens: Iterable[str] | None = None,
    mechanism_tokens: Iterable[str] | None = None,
    method_tokens: Iterable[str] | None = None,
) -> bool:
    mechanism_tokens = {str(token).strip() for token in (mechanism_tokens or []) if str(token).strip()}
    if any(
        str(token).strip() in NON_SHELL_TASK_TOKENS and str(token).strip() not in mechanism_tokens
        for token in (task_tokens or [])
    ):
        return True
    if any(
        str(token).strip() not in BROAD_OBJECT_SHELL_TOKENS and str(token).strip() not in mechanism_tokens
        for token in (object_tokens or [])
    ):
        return True
    if any(
        str(token).strip() not in {"simulation", "video"} and str(token).strip() not in mechanism_tokens
        for token in (data_tokens or [])
    ):
        return True
    if any(str(token).strip() for token in (method_tokens or [])):
        return True
    if any(str(token).strip() in {"manipulation", "navigation", "multimodal"} for token in (scene_tokens or [])):
        return True
    return False


def is_bare_mechanism_candidate(
    mechanism_tokens: Iterable[str] | None = None,
    task_tokens: Iterable[str] | None = None,
    object_tokens: Iterable[str] | None = None,
    data_tokens: Iterable[str] | None = None,
    scene_tokens: Iterable[str] | None = None,
    method_tokens: Iterable[str] | None = None,
) -> bool:
    mechanism_tokens = [str(token).strip() for token in (mechanism_tokens or []) if str(token).strip()]
    if not mechanism_tokens:
        return False
    if not set(mechanism_tokens).issubset(BARE_MECHANISM_CORES):
        return False
    return not has_non_scope_constraint(
        task_tokens,
        object_tokens,
        data_tokens,
        scene_tokens,
        mechanism_tokens,
        method_tokens,
    )


def _normalize_text_key(text: str) -> str:
    lowered = str(text or "").strip().lower()
    lowered = re.sub(r"[^a-z0-9\u4e00-\u9fff\s\-]+", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def normalize_proxy_token(token: str) -> str:
    normalized = _normalize_text_key(str(token or ""))
    if not normalized:
        return ""
    return PROXY_TOKEN_ZH_MAP.get(normalized, normalized)


def canonicalize_term(term: str) -> str:
    raw = str(term).strip()
    if not raw or raw == "未知":
        return "未知"

    lowered = _normalize_text_key(raw)
    for tech_name, aliases in TECH_ALIASES.items():
        alias_values = [tech_name, *aliases]
        if any(_normalize_text_key(alias) == lowered for alias in alias_values):
            return tech_name
    return lowered


def canonicalize_topic(topic: str) -> str:
    raw = str(topic).strip().lower()
    if not raw:
        return "ai"
    if raw in {"ai", "artificial intelligence", "人工智能"}:
        return "ai"
    if raw in {"新能源", "new energy", "renewable energy", "energy"}:
        return "new_energy"
    return raw


def extract_technologies(text: str) -> List[str]:
    haystack = str(text or "").lower()
    matched = []
    for canonical, aliases in TECH_ALIASES.items():
        if any(alias.lower() in haystack for alias in aliases):
            matched.append(canonical)
    return matched or ["未知"]


def detect_observation_scopes(text: str) -> List[str]:
    haystack = str(text or "").lower()
    scopes = []
    for scope in OBSERVATION_SCOPE_TERMS:
        if any(alias.lower() in haystack for alias in aliases_for(scope)):
            scopes.append(scope)
    return scopes


def normalize_technologies(items: Iterable[str], fallback_text: str = "") -> List[str]:
    normalized = []
    seen = set()

    for item in items or []:
        raw = str(item).strip()
        if not raw or raw == "未知":
            continue

        canonical = canonicalize_term(raw)
        if canonical not in seen:
            normalized.append(canonical)
            seen.add(canonical)

    if not normalized:
        for item in discover_candidate_terms(fallback_text) + extract_technologies(fallback_text):
            if item != "未知" and item not in seen:
                normalized.append(item)
                seen.add(item)

    return normalized or ["未知"]


def is_broad_term(term: str) -> bool:
    canonical = canonicalize_term(term)
    return canonical in BROAD_TECH_TERMS


def is_observation_scope(term: str) -> bool:
    canonical = canonicalize_term(term)
    return canonical in OBSERVATION_SCOPE_SET


def _phrase_tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z0-9\-]+", str(text or "").lower())


def _contains_alias_window(tokens: List[str], alias_tokens: List[str], start: int, end: int) -> bool:
    window = tokens[start:end]
    alias_len = len(alias_tokens)
    if alias_len == 0 or len(window) < alias_len:
        return False
    for idx in range(len(window) - alias_len + 1):
        if window[idx:idx + alias_len] == alias_tokens:
            return True
    return False


def _is_specific_scope_candidate_phrase(phrase: str, scope: str) -> bool:
    normalized = _normalize_text_key(phrase)
    if not normalized or normalized == scope or normalized in BROAD_TECH_TERMS:
        return False

    tokens = [token for token in normalized.replace("-", " ").split() if token]
    if len(tokens) < 2:
        return False

    if tokens[0] in DISCOVERY_STOPWORDS | {"this", "that", "these", "those"}:
        return False
    if tokens[-1] in DISCOVERY_STOPWORDS | CONNECTOR_TERMS:
        return False

    alias_tokens = set(scope.replace("-", " ").split())
    extra_tokens = [token for token in tokens if token not in alias_tokens]
    context_terms = SCOPE_CONTEXT_TERMS.get(scope, set())
    if not any(token in context_terms for token in extra_tokens):
        return False
    if any(token not in context_terms and token not in CONNECTOR_TERMS for token in extra_tokens):
        return False

    if len(tokens) >= 3:
        return True

    if any(token in context_terms for token in tokens) and any(token in CONNECTOR_TERMS for token in tokens):
        return True

    if "-" in normalized or len(normalized) >= 24:
        return True

    return False


def discover_scope_candidates(text: str, scopes: Iterable[str], limit: int = 6) -> List[str]:
    scopes = [scope for scope in scopes if scope in OBSERVATION_SCOPE_SET]
    if not scopes:
        return []

    tokens = _phrase_tokenize(text)
    if not tokens:
        return []

    discovered = []
    seen = set()

    for scope in scopes:
        for alias in aliases_for(scope):
            alias_tokens = _phrase_tokenize(alias)
            if not alias_tokens:
                continue

            alias_len = len(alias_tokens)
            for idx in range(len(tokens) - alias_len + 1):
                if tokens[idx:idx + alias_len] != alias_tokens:
                    continue

                left_bound = max(0, idx - 2)
                right_bound = min(len(tokens), idx + alias_len + 3)
                for left in range(left_bound, idx + 1):
                    for right in range(idx + alias_len, right_bound + 1):
                        if not _contains_alias_window(tokens, alias_tokens, left, right):
                            continue
                        phrase_tokens = tokens[left:right]
                        if len(phrase_tokens) <= alias_len:
                            continue
                        phrase = " ".join(phrase_tokens)
                        normalized = _normalize_text_key(phrase)
                        if normalized in seen:
                            continue
                        if not _is_specific_scope_candidate_phrase(normalized, scope):
                            continue
                        seen.add(normalized)
                        discovered.append(normalized)
                        if len(discovered) >= limit:
                            return discovered

    return discovered


def discover_candidate_terms(text: str, limit: int = 6) -> List[str]:
    haystack = str(text or "").lower()
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9\\-]+", haystack)
    candidates = []
    seen = set()

    for size in (3, 2):
        for idx in range(len(tokens) - size + 1):
            gram_tokens = tokens[idx:idx + size]
            if gram_tokens[0] in DISCOVERY_STOPWORDS or gram_tokens[-1] in DISCOVERY_STOPWORDS:
                continue
            if gram_tokens[-1] not in DISCOVERY_HEADWORDS and canonicalize_term(" ".join(gram_tokens)) == " ".join(gram_tokens):
                continue
            if not any(token in DISCOVERY_HEADWORDS for token in gram_tokens):
                continue
            if not any(token in DISCOVERY_CORE_TERMS for token in gram_tokens):
                continue
            if all(token in DISCOVERY_STOPWORDS for token in gram_tokens[:-1]):
                continue

            phrase = " ".join(gram_tokens)
            canonical = canonicalize_term(phrase)
            if canonical == "未知":
                continue
            if canonical not in seen:
                candidates.append(canonical)
                seen.add(canonical)
            if len(candidates) >= limit:
                return candidates

    return candidates


def keyword_match(*texts: str, keywords: Iterable[str]) -> bool:
    keywords = [str(keyword).strip().lower() for keyword in keywords if str(keyword).strip()]
    if not keywords:
        return True

    haystack = " ".join(str(text or "").lower() for text in texts)
    for keyword in keywords:
        if keyword in haystack:
            return True
        for alias in aliases_for(keyword):
            if alias in haystack:
                return True
    return False


def build_or_query(keywords: Iterable[str]) -> str:
    items = [str(keyword).strip() for keyword in keywords if str(keyword).strip()]
    filtered_items = [item for item in items if len(item) > 2 or " " in item]
    items = filtered_items or items
    return " OR ".join(f'"{item}"' for item in items)


def build_google_news_rss(query: str) -> str:
    encoded = quote_plus(query)
    return f"https://news.google.com/rss/search?q={encoded}&hl=en-US&gl=US&ceid=US:en"


def build_topic_bundle(topic: str, research_mode: str) -> TopicBundle:
    canonical_topic = canonicalize_topic(topic)
    preset = TOPIC_PRESETS.get(canonical_topic)
    clean_topic = str(topic).strip() or "AI"

    if preset:
        discovery_terms = [str(item) for item in preset["aliases"]]
        validation_terms = [str(item) for item in preset["validation_aliases"]]
        rss_queries = [str(item) for item in preset.get("rss_queries", [])]
        note = str(preset["note"])
    else:
        discovery_terms = [clean_topic]
        validation_terms = [clean_topic]
        rss_queries = [clean_topic]
        note = "当前主题使用通用构造规则，建议优先从论文与专利侧扩展。"

    if research_mode == "discovery":
        filter_keywords: List[str] = []
        rss_keywords: List[str] = []
        query_terms = discovery_terms
    else:
        filter_keywords = validation_terms
        rss_keywords = validation_terms
        query_terms = validation_terms

    rss_feeds = [build_google_news_rss(query) for query in rss_queries[:2]]
    rss_feeds.extend(DEFAULT_RSS_FEEDS)

    return TopicBundle(
        topic=clean_topic,
        canonical_topic=canonical_topic,
        research_mode=research_mode,
        rss_feeds=rss_feeds,
        rss_keywords=rss_keywords,
        filter_keywords=filter_keywords,
        arxiv_query=build_or_query(query_terms),
        patent_query=build_or_query(query_terms),
        note=note,
    )