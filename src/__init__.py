"""
产业技术预见智能体 - 弱信号识别系统

基于大模型驱动的产业技术弱信号识别系统，实现从多源文本抽取到弱信号判定的完整流程。
"""

__version__ = "0.04"
__author__ = "Tech Foresight Agent"

from .core.agent import TechForesightAgent, analyze_data, analyze_text
from .core.pipeline import AnalysisPipeline

__all__ = [
    "TechForesightAgent",
    "analyze_data",
    "analyze_text",
    "AnalysisPipeline",
]
