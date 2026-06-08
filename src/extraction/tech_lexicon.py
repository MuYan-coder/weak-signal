from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List


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


def is_observation_scope(term: str) -> bool:
    canonical = canonicalize_term(term)
    return canonical in OBSERVATION_SCOPE_SET


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


def _list_values(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        values: List[str] = []
        for item in value:
            values.extend(_list_values(item))
        return values
    if isinstance(value, dict):
        for key in ("term", "name", "value", "canonical", "canonical_term", "pattern_id"):
            if value.get(key):
                return [str(value.get(key)).strip()]
        return []
    text = str(value).strip()
    return [text] if text else []


def _dedupe_text(values: Iterable[Any]) -> List[str]:
    seen = set()
    deduped: List[str] = []
    for value in values or []:
        text = str(value or "").strip()
        if not text:
            continue
        key = normalize_signal_phrase(text)
        if not key or key in seen:
            continue
        deduped.append(text)
        seen.add(key)
    return deduped


def _alias_map_from_terms(values: Iterable[Any]) -> Dict[str, List[str]]:
    alias_map: Dict[str, List[str]] = {}
    for value in _list_values(list(values or [])):
        text = str(value or "").strip()
        if not text:
            continue
        alias_map[text] = _dedupe_text([text])
    return alias_map


def _pack_section(pack: Any, name: str) -> Dict[str, Any]:
    if pack is None:
        return {}
    if isinstance(pack, dict):
        value = pack.get(name, {})
    else:
        value = getattr(pack, name, {})
    return value if isinstance(value, dict) else {}


def _pack_value(pack: Any, name: str, default: Any = "") -> Any:
    if pack is None:
        return default
    if isinstance(pack, dict):
        return pack.get(name, default)
    return getattr(pack, name, default)


def _coerce_pack(domain_source: Any) -> Any:
    if domain_source is None:
        return None
    if hasattr(domain_source, "domain_pack"):
        return getattr(domain_source, "domain_pack")
    return domain_source


def _matches_alias_map(texts: Iterable[Any], alias_map: Dict[str, List[str]]) -> List[str]:
    haystack = " ".join(normalize_signal_phrase(text) for text in texts or [] if str(text or "").strip())
    if not haystack:
        return []
    matched: List[str] = []
    for canonical, aliases in alias_map.items():
        terms = _dedupe_text([canonical, *(aliases or [])])
        if any((alias_norm := normalize_signal_phrase(alias)) and alias_norm in haystack for alias in terms):
            matched.append(canonical)
    return matched


def is_sentence_like_mechanism(text: str, max_term_chars: int = 15) -> bool:
    """判断一段文本是否像一个完整句子而非术语/短语。
    用于阻止 "硅是从沙子中提炼出来的" 这类内容作为 mechanism_core。
    如果文本包含常见句子标志词或中文字符数超过 max_term_chars，视为句子。"""
    if not text:
        return False
    text = str(text).strip()
    # 中文字符计数
    zh_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    if zh_chars > max_term_chars:
        return True
    # 句子标志：包含动词性助词、判断词等
    sentence_markers = ["是", "的", "了", "在", "从", "被", "把", "将", "让", "给", "着", "过"]
    marker_count = sum(1 for marker in sentence_markers if marker in text)
    # 多个句子标志词 + 较长文本 = 很可能是句子
    if marker_count >= 3 and zh_chars > 8:
        return True
    return False


@dataclass
class DomainLexicon:
    pack_id: str = "neutral"
    pack_name: str = ""
    use_legacy_robot_rules: bool = False
    observation_scope_aliases: Dict[str, List[str]] = field(default_factory=dict)
    mechanism_aliases: Dict[str, List[str]] = field(default_factory=dict)
    task_aliases: Dict[str, List[str]] = field(default_factory=dict)
    object_aliases: Dict[str, List[str]] = field(default_factory=dict)
    data_aliases: Dict[str, List[str]] = field(default_factory=dict)
    method_aliases: Dict[str, List[str]] = field(default_factory=dict)
    scene_aliases: Dict[str, List[str]] = field(default_factory=dict)
    generic_terms: List[str] = field(default_factory=list)
    shell_terms: List[str] = field(default_factory=list)
    off_domain_terms: List[str] = field(default_factory=list)
    domain_anchor_terms: List[str] = field(default_factory=list)
    valid_candidate_patterns: List[Dict[str, Any]] = field(default_factory=list)
    invalid_candidate_patterns: List[Dict[str, Any]] = field(default_factory=list)
    minimum_specificity_rule: Dict[str, Any] = field(default_factory=dict)

    @property
    def observation_scopes(self) -> List[str]:
        return list(self.observation_scope_aliases.keys())

    @property
    def domain_specific_terms(self) -> List[str]:
        values: List[str] = []
        for alias_map in [
            self.object_aliases,
            self.mechanism_aliases,
            self.task_aliases,
            self.data_aliases,
            self.method_aliases,
            self.scene_aliases,
        ]:
            values.extend(alias_map.keys())
        return _dedupe_text(values)

    def aliases_for(self, term: str) -> List[str]:
        if self.use_legacy_robot_rules:
            return aliases_for(term)
        canonical = str(term or "").strip()
        if not canonical:
            return []
        for alias_map in [
            self.observation_scope_aliases,
            self.object_aliases,
            self.mechanism_aliases,
            self.task_aliases,
            self.data_aliases,
            self.method_aliases,
            self.scene_aliases,
        ]:
            if canonical in alias_map:
                return _dedupe_text([canonical, *alias_map.get(canonical, [])])
        return [canonical]

    def canonicalize_term(self, term: str) -> str:
        if self.use_legacy_robot_rules:
            return canonicalize_term(term)
        raw = str(term or "").strip()
        if not raw:
            return ""
        normalized = normalize_signal_phrase(raw)
        for alias_map in [
            self.observation_scope_aliases,
            self.object_aliases,
            self.mechanism_aliases,
            self.task_aliases,
            self.data_aliases,
            self.method_aliases,
            self.scene_aliases,
        ]:
            for canonical, aliases in alias_map.items():
                terms = _dedupe_text([canonical, *(aliases or [])])
                if any(normalize_signal_phrase(alias) == normalized for alias in terms):
                    return canonical
        return raw

    def _matches_off_domain(self, text: str) -> bool:
        if not self.off_domain_terms:
            return False
        haystack = normalize_signal_phrase(text)
        return any((term_norm := normalize_signal_phrase(term)) and term_norm in haystack for term in self.off_domain_terms)

    def is_off_domain_term(self, text: str) -> bool:
        normalized = normalize_signal_phrase(text)
        if not normalized:
            return False
        return any(normalize_signal_phrase(term) == normalized for term in self.off_domain_terms)

    def detect_supported_observation_scopes(self, text: str) -> List[str]:
        if self.use_legacy_robot_rules:
            return detect_supported_observation_scopes(text)
        if self._matches_off_domain(text):
            return []
        return _matches_alias_map([text], self.observation_scope_aliases)

    def diagnose_observation_scope_detection(
        self,
        text: str,
        source_type: str = "",
        mechanism_tokens: Iterable[str] | None = None,
        task_tokens: Iterable[str] | None = None,
        object_tokens: Iterable[str] | None = None,
        data_tokens: Iterable[str] | None = None,
        scene_tokens: Iterable[str] | None = None,
        method_tokens: Iterable[str] | None = None,
    ) -> Dict[str, object]:
        if self.use_legacy_robot_rules:
            return diagnose_observation_scope_detection(
                text,
                source_type=source_type,
                mechanism_tokens=mechanism_tokens,
                task_tokens=task_tokens,
                object_tokens=object_tokens,
                data_tokens=data_tokens,
                scene_tokens=scene_tokens,
                method_tokens=method_tokens,
            )

        haystack = normalize_signal_phrase(text)
        direct_matches: List[str] = []
        alias_matches: Dict[str, List[str]] = {}
        for scope, aliases in self.observation_scope_aliases.items():
            scope_norm = normalize_signal_phrase(scope)
            if scope_norm and scope_norm in haystack:
                direct_matches.append(scope)
            matched_aliases = [
                alias
                for alias in aliases or []
                if (alias_norm := normalize_signal_phrase(alias))
                and alias_norm in haystack
                and alias_norm != scope_norm
            ]
            if matched_aliases:
                alias_matches[scope] = matched_aliases[:3]

        off_domain_hit = self._matches_off_domain(text)
        supported_scopes = [] if off_domain_hit else self.detect_supported_observation_scopes(text)
        rejected_scopes = [
            scope
            for scope in set(direct_matches) | set(alias_matches.keys())
            if scope not in supported_scopes
        ]
        if supported_scopes:
            reason = "domain_pack_scope_alias"
        elif off_domain_hit:
            reason = "domain_pack_off_domain_term"
        elif rejected_scopes:
            reason = "domain_pack_scope_alias_rejected"
        else:
            reason = "domain_pack_no_scope_match"
        return {
            "supported_scopes": supported_scopes,
            "scope_match_mode": "domain_pack_explicit" if supported_scopes else "",
            "direct_matches": direct_matches,
            "alias_matches": alias_matches,
            "rejected_scopes": rejected_scopes,
            "proxy_scopes": [],
            "reason": reason,
        }

    def extract_mechanism_core_tokens(self, *texts: str) -> List[str]:
        if self.use_legacy_robot_rules:
            return extract_mechanism_core_tokens(*texts)
        return _matches_alias_map(texts, self.mechanism_aliases)

    def extract_task_constraint_tokens(self, *texts: str) -> List[str]:
        if self.use_legacy_robot_rules:
            return extract_task_constraint_tokens(*texts)
        return _matches_alias_map(texts, self.task_aliases)

    def extract_scene_tokens(self, *texts: str) -> List[str]:
        if self.use_legacy_robot_rules:
            return extract_scene_tokens(*texts)
        return _matches_alias_map(texts, self.scene_aliases)

    def extract_object_modifier_tokens(self, *texts: str) -> List[str]:
        if self.use_legacy_robot_rules:
            return extract_object_modifier_tokens(*texts)
        return _matches_alias_map(texts, self.object_aliases)

    def extract_data_modifier_tokens(self, *texts: str) -> List[str]:
        if self.use_legacy_robot_rules:
            return extract_data_modifier_tokens(*texts)
        return _matches_alias_map(texts, self.data_aliases)

    def extract_method_modifier_tokens(self, *texts: str) -> List[str]:
        if self.use_legacy_robot_rules:
            return extract_method_modifier_tokens(*texts)
        return _matches_alias_map(texts, self.method_aliases)

    def extract_technologies(self, text: str) -> List[str]:
        if self.use_legacy_robot_rules:
            return extract_technologies(text)
        values: List[str] = []
        for alias_map in [
            self.object_aliases,
            self.mechanism_aliases,
            self.task_aliases,
            self.scene_aliases,
            self.observation_scope_aliases,
        ]:
            values.extend(_matches_alias_map([text], alias_map))
        values = [value for value in _dedupe_text(values) if not self.is_generic_or_shell(value)]
        return values or ["unknown"]

    def normalize_technologies(self, items: Iterable[str], fallback_text: str = "") -> List[str]:
        if self.use_legacy_robot_rules:
            return normalize_technologies(items, fallback_text=fallback_text)
        normalized: List[str] = []
        seen = set()
        for item in items or []:
            canonical = self.canonicalize_term(str(item or "").strip())
            if not canonical or canonical.lower() in {"unknown", "none", "n/a"}:
                continue
            if self.is_off_domain_term(canonical):
                continue
            if self.is_generic_or_shell(canonical):
                continue
            key = normalize_signal_phrase(canonical)
            if key not in seen:
                normalized.append(canonical)
                seen.add(key)
        for item in self.extract_technologies(fallback_text):
            if item.lower() == "unknown" or self.is_off_domain_term(item) or self.is_generic_or_shell(item):
                continue
            key = normalize_signal_phrase(item)
            if key not in seen:
                normalized.append(item)
                seen.add(key)
        return normalized or ["unknown"]

    def is_observation_scope(self, term: str) -> bool:
        if self.use_legacy_robot_rules:
            return is_observation_scope(term)
        canonical = self.canonicalize_term(term)
        return canonical in self.observation_scope_aliases

    def has_robot_domain_anchor(self, text: str) -> bool:
        if self.use_legacy_robot_rules:
            return has_robot_domain_anchor(text)
        return bool(self.detect_supported_observation_scopes(text) or self.extract_technologies(text) != ["unknown"])

    def has_domain_anchor(self, text: str) -> bool:
        """判断文本是否包含至少一个领域锚点词。
        领域锚点从 Domain Pack 的 core_keywords + technical_object_types +
        observation_scopes.technical_object 自动构建。
        用于在 analysis_field scope 模式下仍确保文档与领域相关。"""
        if self.use_legacy_robot_rules:
            return has_robot_domain_anchor(text)
        if not self.domain_anchor_terms:
            # 没有配置锚点时默认通过（向后兼容）
            return True
        haystack = normalize_signal_phrase(text)
        if not haystack:
            return False
        for anchor in self.domain_anchor_terms:
            anchor_norm = normalize_signal_phrase(anchor)
            if anchor_norm and anchor_norm in haystack:
                return True
        return False

    def is_valid_mechanism_core(self, text: str) -> bool:
        """判断文本是否是一个合理的 mechanism_core。
        拒绝像完整句子一样的内容（如'硅是从沙子中提炼出来的'）。"""
        if not text:
            return False
        # 如果在 mechanism_aliases 中有精确匹配，直接接受
        if _matches_alias_map([text], self.mechanism_aliases):
            return True
        # 句子样的文本不应作为 mechanism
        if is_sentence_like_mechanism(text):
            return False
        return True

    def has_non_scope_constraint(
        self,
        task_tokens: Iterable[str] | None = None,
        object_tokens: Iterable[str] | None = None,
        data_tokens: Iterable[str] | None = None,
        scene_tokens: Iterable[str] | None = None,
        mechanism_tokens: Iterable[str] | None = None,
        method_tokens: Iterable[str] | None = None,
    ) -> bool:
        if self.use_legacy_robot_rules:
            return has_non_scope_constraint(task_tokens, object_tokens, data_tokens, scene_tokens, mechanism_tokens, method_tokens)
        for value in [
            *(task_tokens or []),
            *(object_tokens or []),
            *(data_tokens or []),
            *(scene_tokens or []),
            *(mechanism_tokens or []),
            *(method_tokens or []),
        ]:
            text = str(value or "").strip()
            if text and not self.is_generic_or_shell(text) and not self.is_observation_scope(text):
                return True
        return False

    def is_scope_echo_candidate(
        self,
        text: str,
        scopes: Iterable[str],
        mechanism_tokens: Iterable[str] | None = None,
        task_tokens: Iterable[str] | None = None,
    ) -> bool:
        if self.use_legacy_robot_rules:
            return is_scope_echo_candidate(text, scopes, mechanism_tokens=mechanism_tokens, task_tokens=task_tokens)
        if mechanism_tokens:
            return False
        if any(token and not self.is_generic_or_shell(token) for token in task_tokens or []):
            return False
        text_norm = normalize_signal_phrase(text)
        if not text_norm:
            return True
        scope_or_shell = _dedupe_text([*(scopes or []), *self.generic_terms, *self.shell_terms])
        stripped = text_norm
        for term in scope_or_shell:
            term_norm = normalize_signal_phrase(term)
            if term_norm:
                stripped = stripped.replace(term_norm, " ")
        return not stripped.strip()

    def is_generic_or_shell(self, term: str) -> bool:
        normalized = normalize_signal_phrase(term)
        if not normalized:
            return True
        for value in [*self.generic_terms, *self.shell_terms]:
            value_norm = normalize_signal_phrase(value)
            if value_norm and normalized == value_norm:
                return True
        return False

    def match_terms(self, alias_map: Dict[str, List[str]], *texts: str) -> List[str]:
        return _matches_alias_map(texts, alias_map)

    def candidate_slot_hits(
        self,
        task_tokens: Iterable[str] | None = None,
        object_tokens: Iterable[str] | None = None,
        data_tokens: Iterable[str] | None = None,
        scene_tokens: Iterable[str] | None = None,
        mechanism_tokens: Iterable[str] | None = None,
        method_tokens: Iterable[str] | None = None,
    ) -> Dict[str, bool]:
        return {
            "technical_object": any(not self.is_generic_or_shell(value) for value in object_tokens or []),
            "mechanism": any(not self.is_generic_or_shell(value) for value in mechanism_tokens or []),
            "performance": any(not self.is_generic_or_shell(value) for value in task_tokens or []),
            "task": any(not self.is_generic_or_shell(value) for value in task_tokens or []),
            "data_modality": any(not self.is_generic_or_shell(value) for value in data_tokens or []),
            "data": any(not self.is_generic_or_shell(value) for value in data_tokens or []),
            "method": any(not self.is_generic_or_shell(value) for value in method_tokens or []),
            "scene": any(not self.is_generic_or_shell(value) for value in scene_tokens or []),
        }

    def valid_candidate_pattern_matches(
        self,
        task_tokens: Iterable[str] | None = None,
        object_tokens: Iterable[str] | None = None,
        data_tokens: Iterable[str] | None = None,
        scene_tokens: Iterable[str] | None = None,
        mechanism_tokens: Iterable[str] | None = None,
        method_tokens: Iterable[str] | None = None,
        *,
        evidence_present: bool = True,
    ) -> bool:
        if self.use_legacy_robot_rules:
            return False
        slot_hits = self.candidate_slot_hits(
            task_tokens=task_tokens,
            object_tokens=object_tokens,
            data_tokens=data_tokens,
            scene_tokens=scene_tokens,
            mechanism_tokens=mechanism_tokens,
            method_tokens=method_tokens,
        )
        for pattern in self.valid_candidate_patterns:
            if not isinstance(pattern, dict):
                continue
            if pattern.get("evidence_required") and not evidence_present:
                continue
            required_slots = [str(slot or "").strip() for slot in pattern.get("required_slots", []) if str(slot or "").strip()]
            min_required = int(pattern.get("min_required_slot_count", len(required_slots)) or len(required_slots))
            hit_count = sum(1 for slot in required_slots if slot_hits.get(slot, False))
            if hit_count >= min_required:
                return True
        return False


def build_domain_lexicon(domain_source: Any = None) -> DomainLexicon:
    pack = _coerce_pack(domain_source)
    pack_id = str(_pack_value(pack, "pack_id", "neutral") or "neutral").strip() or "neutral"
    pack_name = str(_pack_value(pack, "pack_name", pack_id) or pack_id).strip()
    use_legacy = pack_id == "humanoid_robot"

    if use_legacy:
        return DomainLexicon(
            pack_id=pack_id,
            pack_name=pack_name,
            use_legacy_robot_rules=True,
            observation_scope_aliases={scope: aliases_for(scope) for scope in OBSERVATION_SCOPE_TERMS},
            mechanism_aliases=MECHANISM_CORE_ALIASES,
            task_aliases=TASK_CONSTRAINT_ALIASES,
            object_aliases=OBJECT_MODIFIER_ALIASES,
            data_aliases=DATA_MODIFIER_ALIASES,
            method_aliases=METHOD_MODIFIER_ALIASES,
            scene_aliases={term: [term] for term in SCENE_TOKEN_SET},
        )

    search_strategy = _pack_section(pack, "search_strategy")
    observation_scopes = _pack_section(pack, "observation_scopes")
    candidate_formation = _pack_section(pack, "candidate_formation")
    domain_identity = _pack_section(pack, "domain_identity")
    source = _pack_section(pack, "source")
    source_query = source.get("based_on_user_input", {}) if isinstance(source.get("based_on_user_input", {}), dict) else {}
    evidence_rules = _pack_section(pack, "evidence_rules")

    main_scope = str(
        observation_scopes.get("main_scope")
        or domain_identity.get("field_name")
        or source_query.get("field_name")
        or ""
    ).strip()
    scope_alias_values = _dedupe_text(
        [
            main_scope,
            *_list_values(observation_scopes.get("sub_scopes")),
            *_list_values(observation_scopes.get("scope_aliases")),
            *_list_values(observation_scopes.get("scope_echo_terms")),
            *_list_values(search_strategy.get("core_keywords")),
            *_list_values(search_strategy.get("synonyms")),
            *_list_values(search_strategy.get("english_terms")),
            domain_identity.get("field_name", ""),
            source_query.get("field_name", ""),
        ]
    )
    observation_scope_aliases: Dict[str, List[str]] = {}
    if main_scope:
        observation_scope_aliases[main_scope] = scope_alias_values or [main_scope]
    for sub_scope in _list_values(observation_scopes.get("sub_scopes")):
        sub_scope = str(sub_scope or "").strip()
        if sub_scope and sub_scope not in observation_scope_aliases:
            observation_scope_aliases[sub_scope] = _dedupe_text([sub_scope])

    data_method_terms = _list_values(candidate_formation.get("data_or_method_types"))
    generic_terms = _dedupe_text(candidate_formation.get("generic_terms", []))
    default_shells = [
        "技术", "方法", "系统", "应用", "数据", "性能", "效果", "场景",
        "technology", "method", "system", "application", "data", "performance", "effect", "scene"
    ]
    if len(generic_terms) < 2:
        seen = {normalize_signal_phrase(t) for t in generic_terms}
        for default_word in default_shells:
            if (default_norm := normalize_signal_phrase(default_word)) and default_norm not in seen:
                seen.add(default_norm)
                generic_terms.append(default_word)

    shell_terms = _dedupe_text(candidate_formation.get("shell_terms", []))
    if len(shell_terms) < 2:
        seen = {normalize_signal_phrase(t) for t in shell_terms}
        for default_word in default_shells:
            if (default_norm := normalize_signal_phrase(default_word)) and default_norm not in seen:
                seen.add(default_norm)
                shell_terms.append(default_word)

    off_domain_terms = _dedupe_text(
        [
            *_list_values(observation_scopes.get("off_domain_anchor_terms")),
            *_list_values(search_strategy.get("exclude_terms")),
            *_list_values(domain_identity.get("out_of_scope_domains")),
            *_list_values(evidence_rules.get("evidence_rejection_patterns")),
        ]
    )

    valid_patterns = candidate_formation.get("valid_candidate_patterns", [])
    invalid_patterns = candidate_formation.get("invalid_candidate_patterns", [])
    # 构建领域锚点集合：从 core_keywords + technical_object_types + observation_scopes.technical_object 合并
    domain_anchor_sources = [
        *_list_values(search_strategy.get("core_keywords")),
        *_list_values(search_strategy.get("synonyms")),
        *_list_values(candidate_formation.get("technical_object_types")),
        *_list_values(observation_scopes.get("technical_object")),
        *_list_values(observation_scopes.get("mechanism")),
    ]
    domain_anchor_terms = _dedupe_text(domain_anchor_sources)

    mechanism_aliases = _alias_map_from_terms(candidate_formation.get("mechanism_types", []))
    task_aliases = _alias_map_from_terms(candidate_formation.get("task_or_performance_types", []))
    object_aliases = _alias_map_from_terms(candidate_formation.get("technical_object_types", []))
    data_aliases = _alias_map_from_terms(data_method_terms)
    method_aliases = _alias_map_from_terms(data_method_terms)
    scene_aliases = _alias_map_from_terms(candidate_formation.get("scene_or_application_types", []))

    # Support enrichment of aliases from canonicalization
    canonicalization = _pack_section(pack, "canonicalization")
    object_families = canonicalization.get("object_families", []) or []
    alias_groups = canonicalization.get("alias_groups", []) or []

    def _merge_aliases(target_dict: Dict[str, List[str]], key: str, aliases: List[str]):
        normalized_key = str(key).strip()
        if not normalized_key:
            return False
        matched_key = None
        for canonical, existing_aliases in target_dict.items():
            if canonical.lower() == normalized_key.lower() or any(a.lower() == normalized_key.lower() for a in existing_aliases):
                matched_key = canonical
                break
        if matched_key:
            merged = _dedupe_text([*target_dict[matched_key], *aliases])
            target_dict[matched_key] = merged
            return True
        return False

    # 1. Process object_families (which have term_type to help map directly)
    for family in object_families:
        if not isinstance(family, dict):
            continue
        canonical = str(family.get("canonical_term", "")).strip()
        if not canonical:
            continue
        aliases = _dedupe_text(family.get("aliases", []))
        term_type = str(family.get("term_type", "unknown")).strip().lower()

        merged = False
        if term_type in {"task", "capability"}:
            merged = _merge_aliases(task_aliases, canonical, aliases)
        elif term_type in {"model_name", "component"}:
            merged = _merge_aliases(object_aliases, canonical, aliases)
        elif term_type == "method":
            merged = _merge_aliases(method_aliases, canonical, aliases) or _merge_aliases(data_aliases, canonical, aliases)
        elif term_type == "scenario":
            merged = _merge_aliases(scene_aliases, canonical, aliases)

        if not merged:
            for target in [object_aliases, mechanism_aliases, task_aliases, method_aliases, data_aliases, scene_aliases]:
                if _merge_aliases(target, canonical, aliases):
                    break

    # 2. Process alias_groups
    for group in alias_groups:
        if not isinstance(group, dict):
            continue
        canonical = str(group.get("canonical", "")).strip()
        if not canonical:
            continue
        aliases = _dedupe_text(group.get("aliases", []))

        merged = False
        for target in [object_aliases, mechanism_aliases, task_aliases, method_aliases, data_aliases, scene_aliases]:
            if _merge_aliases(target, canonical, aliases):
                merged = True
                break
        if not merged:
            object_aliases[canonical] = _dedupe_text([canonical, *aliases])

    return DomainLexicon(
        pack_id=pack_id,
        pack_name=pack_name,
        use_legacy_robot_rules=False,
        observation_scope_aliases=observation_scope_aliases,
        mechanism_aliases=mechanism_aliases,
        task_aliases=task_aliases,
        object_aliases=object_aliases,
        data_aliases=data_aliases,
        method_aliases=method_aliases,
        scene_aliases=scene_aliases,
        generic_terms=generic_terms,
        shell_terms=shell_terms,
        off_domain_terms=off_domain_terms,
        domain_anchor_terms=domain_anchor_terms,
        valid_candidate_patterns=valid_patterns if isinstance(valid_patterns, list) else [],
        invalid_candidate_patterns=invalid_patterns if isinstance(invalid_patterns, list) else [],
        minimum_specificity_rule=(
            candidate_formation.get("minimum_specificity_rule", {})
            if isinstance(candidate_formation.get("minimum_specificity_rule", {}), dict)
            else {}
        ),
    )
