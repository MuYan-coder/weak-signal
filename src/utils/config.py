"""配置管理模块"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


class Config:
    """配置类"""
    
    # 项目根目录
    ROOT_DIR = Path(__file__).resolve().parents[2]
    
    # API配置
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.siliconflow.cn/v1")
    
    # 模型配置
    EXTRACTION_MODEL = os.getenv("EXTRACTION_MODEL", "Qwen/Qwen2.5-14B-Instruct")
    REPORT_MODEL = os.getenv("REPORT_MODEL", "deepseek-ai/DeepSeek-V3")
    AGENT_MODEL = os.getenv("AGENT_MODEL", "Qwen/Qwen2.5-32B-Instruct")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "deepseek-ai/DeepSeek-V3")
    
    # 语义相似度配置
    FAST_SEMANTIC_MODE = os.getenv("WEAK_SIGNAL_FAST_SEMANTIC", "").lower() in ("1", "true", "yes", "on")
    
    # 数据目录
    DATA_DIR = ROOT_DIR / "data"
    RESULT_DIR = ROOT_DIR / "result"
    MEMORY_DIR = ROOT_DIR / "memory"
    
    # 弱信号判定阈值
    WEAK_SIGNAL_THRESHOLDS = {
        "visibility_low": 10,
        "diffusion_low": 2,
        "influence_low": 3,
        "min_source_count": 2,
        "min_org_count": 2,
        "time_window_days": 30,
        "duplicate_window_days": 7,
    }
    
    # 评分权重
    SCORING_WEIGHTS = {
        "novelty": 1.0,
        "growth": 1.0,
        "cross_source": 1.5,
        "semantic": 2.0,
        "time": 1.5,
        "evidence": 2.0,
    }
    
    @classmethod
    def ensure_dirs(cls):
        """确保必要目录存在"""
        for dir_path in [cls.DATA_DIR, cls.RESULT_DIR, cls.MEMORY_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def is_api_available(cls) -> bool:
        """检查API是否可用"""
        return bool(cls.OPENAI_API_KEY and cls.OPENAI_BASE_URL)
