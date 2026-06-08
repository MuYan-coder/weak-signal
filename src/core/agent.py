"""产业技术预见智能体主类"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from .pipeline import AnalysisPipeline
from ..utils.config import Config
from ..utils.env_config import ensure_env_loaded
from ..utils.llm_client import get_provider_and_client, chat_text
from ..utils.api_stats import record_call

ensure_env_loaded()


class TechForesightAgent:
    """
    产业技术预见智能体
    
    基于大模型驱动的产业技术弱信号识别系统
    """
    
    def __init__(self):
        """初始化智能体"""
        Config.ensure_dirs()
        self.pipeline = AnalysisPipeline()
        self._version = "0.04"
        self._llm_available = self._check_llm_availability()
    
    def _check_llm_availability(self) -> bool:
        """检查LLM是否可用"""
        provider, client = get_provider_and_client()
        return client is not None and provider is not None
    
    def _llm_decide_analysis_strategy(self, data_info: Dict[str, Any]) -> Dict[str, Any]:
        """使用LLM决定分析策略"""
        if not self._llm_available:
            return {
                "strategy": "default",
                "sample_size": 100,
                "focus_areas": [],
                "reasoning": "LLM不可用，使用默认策略"
            }
        
        prompt = f"""你是一个技术趋势分析专家。请根据以下数据信息，决定最佳的分析策略。

数据信息：
- 数据来源类型: {data_info.get('source_types', [])}
- 数据总量: {data_info.get('total_count', 0)}
- 时间范围: {data_info.get('time_range', '未知')}

请返回JSON格式的分析策略建议：
{{
  "strategy": "comprehensive|focused|exploratory",
  "sample_size": 建议的采样数量(整数),
  "focus_areas": ["建议关注的技术领域1", "领域2"],
  "reasoning": "策略选择的理由"
}}
"""
        
        try:
            print("[智能体] 正在调用LLM进行策略决策...")
            agent_model = os.getenv("AGENT_MODEL", os.getenv("OPENAI_MODEL", "Qwen/Qwen2.5-14B-Instruct"))
            result, usage_info, _ = chat_text(
                prompt,
                model=agent_model,
                temperature=0.3,
                max_tokens=500,
                timeout=30,
            )
            
            if usage_info["prompt_tokens"] or usage_info["completion_tokens"]:
                record_call(
                    call_type="智能体决策",
                    prompt_tokens=usage_info["prompt_tokens"],
                    completion_tokens=usage_info["completion_tokens"],
                    success=True,
                )
            
            import json
            # 尝试解析JSON
            result = result.strip()
            # 移除可能的markdown代码块标记
            if result.startswith("```"):
                result = result.split("\n", 1)[1] if "\n" in result else result
                result = result.rsplit("```", 1)[0] if "```" in result else result
            
            strategy = json.loads(result.strip())
            print(f"[智能体] LLM策略决策完成")
            return strategy
        except Exception as e:
            print(f"[智能体] LLM决策失败: {e}，使用默认策略")
            return {
                "strategy": "default",
                "sample_size": 100,
                "focus_areas": [],
                "reasoning": f"LLM决策失败: {e}"
            }
    
    def _llm_summarize_results(self, results: Dict[str, Any]) -> str:
        """使用LLM总结分析结果"""
        if not self._llm_available:
            return "LLM不可用，无法生成智能总结"
        
        # 提取关键信息
        events_count = len(results.get('events_df', []))
        candidates_count = len(results.get('candidate_forms_df', []))
        signals_count = len(results.get('signals_df', []))
        
        prompt = f"""你是一个技术趋势分析专家。请根据以下分析结果，生成一份简洁的洞察总结。

分析结果概览：
- 抽取事件数: {events_count}
- 候选对象数: {candidates_count}
- 识别信号数: {signals_count}

请用2-3句话总结主要发现和洞察。
"""
        
        try:
            agent_model = os.getenv("AGENT_MODEL", os.getenv("OPENAI_MODEL", "Qwen/Qwen2.5-14B-Instruct"))
            result, usage_info, _ = chat_text(
                prompt,
                model=agent_model,
                temperature=0.3,
                max_tokens=300,
                timeout=60,
            )
            
            if usage_info["prompt_tokens"] or usage_info["completion_tokens"]:
                record_call(
                    call_type="智能体总结",
                    prompt_tokens=usage_info["prompt_tokens"],
                    completion_tokens=usage_info["completion_tokens"],
                    success=True,
                )
            
            return result.strip()
        except Exception as e:
            return f"智能总结生成失败: {e}"
    
    def analyze(
        self,
        data_path: Optional[str] = None,
        sample_size: int = 100,
        use_cache: bool = True,
        use_llm_strategy: bool = True,
        source_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        执行分析
        
        Args:
            data_path: 数据文件路径（可选，默认加载data目录下所有数据）
            sample_size: 采样数量
            use_cache: 是否使用缓存
            use_llm_strategy: 是否使用LLM决定分析策略
            source_config: 数据源配置，包含:
                - sources: 数据源类型列表 ['patent', 'literature', 'report', 'news']
                - counts: 各数据源数量 {'patent': 50, 'literature': 30}
            
        Returns:
            分析结果字典
        """
        print(f"=" * 60)
        print(f"产业技术预见智能体 v{self._version}")
        print(f"=" * 60)
        
        # 显示LLM状态
        print(f"\n[智能体] LLM状态: {'可用' if self._llm_available else '不可用'}")
        if self._llm_available:
            provider, _ = get_provider_and_client()
            agent_model = os.getenv("AGENT_MODEL", "未配置")
            extraction_model = os.getenv("EXTRACTION_MODEL", "未配置")
            report_model = os.getenv("REPORT_MODEL", "未配置")
            print(f"[智能体] Provider: {provider}")
            print(f"[智能体] 事件抽取模型: {extraction_model}")
            print(f"[智能体] 主题细化模型: {agent_model}")
            print(f"[智能体] 报告生成模型: {report_model}")
        
        # 显示数据源配置
        if source_config:
            self._print_source_config(source_config, sample_size)
        
        # 转换路径
        if data_path:
            data_path = Path(data_path)
        
        # 使用LLM决定分析策略（如果启用）
        if use_llm_strategy and self._llm_available:
            print("\n[智能体] 正在使用LLM决定分析策略...")
            data_info = {
                "source_types": ["专利", "文献", "研报", "资讯"],
                "total_count": sample_size,
            }
            strategy = self._llm_decide_analysis_strategy(data_info)
            print(f"[智能体] 分析策略: {strategy.get('strategy', 'default')}")
            print(f"[智能体] 策略理由: {strategy.get('reasoning', 'N/A')}")
            if strategy.get('focus_areas'):
                print(f"[智能体] 关注领域: {', '.join(strategy['focus_areas'])}")
        
        # 运行流水线
        print("\n[智能体] 开始执行分析流水线...")
        results = self.pipeline.run_full_pipeline(
            data_path=data_path,
            sample_size=sample_size,
            use_cache=use_cache,
            cache_dir=Config.MEMORY_DIR / "cache",
            source_config=source_config,
        )
        
        # 使用LLM总结结果
        if self._llm_available and results:
            print("\n[智能体] 正在使用LLM生成洞察总结...")
            summary = self._llm_summarize_results(results)
            print(f"\n[智能体] 洞察总结:\n{summary}")
            results['llm_summary'] = summary
        
        return results
    
    def _print_source_config(self, source_config: Dict[str, Any], sample_size: int):
        """打印数据源配置信息"""
        print(f"\n[智能体] 数据源配置:")
        
        sources = source_config.get('sources')
        counts = source_config.get('counts', {})
        
        # 数据源名称映射
        source_names = {
            'patent': '专利',
            'literature': '文献',
            'report': '研报',
            'news': '资讯'
        }
        
        if counts:
            # 使用指定的数量
            total = sum(counts.values())
            for src, cnt in counts.items():
                src_name = source_names.get(src, src)
                print(f"  - {src_name}: {cnt} 条")
            print(f"  总计: {total} 条")
        elif sources:
            # 指定了数据源类型，平均分配
            per_source = sample_size // len(sources)
            for src in sources:
                src_name = source_names.get(src, src)
                print(f"  - {src_name}: {per_source} 条")
            print(f"  总计: {sample_size} 条")
        else:
            # 默认从所有数据源平均分配
            per_source = sample_size // 4
            print(f"  - 专利: {per_source} 条")
            print(f"  - 文献: {per_source} 条")
            print(f"  - 研报: {per_source} 条")
            print(f"  - 资讯: {per_source} 条")
            print(f"  总计: {sample_size} 条")
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        分析单条文本
        
        Args:
            text: 输入文本
            
        Returns:
            分析结果
        """
        from ..extraction.event_extractor import extract_events_from_text
        from ..extraction.candidate_former import build_candidate_forms
        
        # 创建临时DataFrame
        df = pd.DataFrame([{"text": text, "id": "single", "source_type": "single"}])
        
        # 抽取事件
        events_df = extract_events_from_text(df)
        
        # 候选成形
        candidate_forms = build_candidate_forms(events_df)
        
        return {
            "events": events_df.to_dict('records'),
            "candidates": candidate_forms,
        }
    
    def get_version(self) -> str:
        """获取版本号"""
        return self._version
    
    def get_config(self) -> Dict[str, Any]:
        """获取配置信息"""
        return {
            "version": self._version,
            "data_dir": str(Config.DATA_DIR),
            "result_dir": str(Config.RESULT_DIR),
            "memory_dir": str(Config.MEMORY_DIR),
            "extraction_model": Config.EXTRACTION_MODEL,
            "report_model": Config.REPORT_MODEL,
            "agent_model": Config.AGENT_MODEL,
        }


# 便捷函数
def create_agent() -> TechForesightAgent:
    """创建智能体实例"""
    return TechForesightAgent()


def analyze_data(
    data_path: Optional[str] = None,
    sample_size: int = 100,
) -> Dict[str, Any]:
    """便捷分析函数"""
    agent = create_agent()
    return agent.analyze(data_path=data_path, sample_size=sample_size)


def analyze_text(text: str) -> Dict[str, Any]:
    """便捷单条文本分析函数"""
    agent = create_agent()
    return agent.analyze_text(text)
