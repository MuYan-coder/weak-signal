"""产业技术预见智能体 - 主程序入口"""

import argparse
import sys
from pathlib import Path

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.agent import TechForesightAgent, analyze_data, analyze_text
from src.core.pipeline import AnalysisPipeline


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="产业技术预见智能体 - 弱信号识别",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 默认模式：从所有数据源各取25条，共100条
  python main.py
  
  # 指定总采样数量（从各数据源平均分配）
  python main.py --sample 200
  
  # 只分析专利数据
  python main.py --sources patent --sample 100
  
  # 分析专利和文献，各50条
  python main.py --sources patent,literature --sample 50
  
  # 自定义各数据源数量
  python main.py --patent 50 --literature 30 --news 20
  
  # 分析指定文件
  python main.py --data my_data.xlsx --sample 50
        """
    )
    
    # 数据源选择参数
    parser.add_argument("--data", "-d", type=str, help="指定数据文件路径（xlsx/csv/json）")
    parser.add_argument("--sources", type=str, default=None, 
                        help="选择数据源类型，用逗号分隔: patent,literature,report,news")
    
    # 采样数量参数
    parser.add_argument("--sample", "-s", type=int, default=100, help="总采样数量（默认100）")
    parser.add_argument("--patent", type=int, default=None, help="专利数据数量")
    parser.add_argument("--literature", type=int, default=None, help="文献数据数量")
    parser.add_argument("--report", type=int, default=None, help="研报数据数量")
    parser.add_argument("--news", type=int, default=None, help="资讯数据数量")
    
    # 其他参数
    parser.add_argument("--no-cache", action="store_true", help="不使用缓存")
    parser.add_argument("--no-llm-strategy", action="store_true", help="不使用LLM策略决策")
    parser.add_argument("--text", "-t", type=str, help="分析单条文本")
    parser.add_argument("--list-sources", action="store_true", help="列出可用的数据源文件")
    parser.add_argument("--from-events", type=str, help="从已抽取的 events.json/csv 继续分析并生成报告")
    parser.add_argument("--from-result", type=str, help="从历史 result 目录恢复；默认仅重新生成报告")
    parser.add_argument("--resume-stage", choices=["events", "report"], default="report", help="恢复阶段：events=从事件继续跑，report=仅重生成报告")
    
    args = parser.parse_args()
    
    # 如果只是列出数据源
    if args.list_sources:
        list_available_sources()
        return
    
    # 如果指定了单条文本
    if args.text:
        result = analyze_text(args.text)
        print("\n" + "=" * 60)
        print("分析结果:")
        print("=" * 60)
        print(f"事件: {result.get('events', [])}")
        print(f"候选: {result.get('candidates', [])}")
        return

    if args.from_events:
        pipeline = AnalysisPipeline()
        results = pipeline.run_from_events(events_path=Path(args.from_events))
        if results:
            print("\n" + "=" * 60)
            print("从已抽取事件继续分析完成!")
            print(f"结果目录: {results.get('result_dir', 'N/A')}")
            print("=" * 60)
            return
        print("\n[ERROR] 从已抽取事件继续分析失败")
        sys.exit(1)

    if args.from_result:
        pipeline = AnalysisPipeline()
        result_path = Path(args.from_result)
        if args.resume_stage == "events":
            events_path = result_path / "events.json" if result_path.is_dir() else result_path
            results = pipeline.run_from_events(events_path=events_path)
        else:
            results = pipeline.regenerate_report_from_result(result_path)
        if results:
            print("\n" + "=" * 60)
            print("历史结果恢复完成!")
            print(f"结果目录: {results.get('result_dir', 'N/A')}")
            print("=" * 60)
            return
        print("\n[ERROR] 历史结果恢复失败")
        sys.exit(1)
    
    # 构建数据源配置
    source_config = build_source_config(args)
    
    # 创建智能体并运行分析
    agent = TechForesightAgent()
    
    results = agent.analyze(
        data_path=args.data,
        sample_size=args.sample,
        use_cache=not args.no_cache,
        use_llm_strategy=not args.no_llm_strategy,
        source_config=source_config,
    )
    
    if results:
        print("\n" + "=" * 60)
        print("分析完成!")
        print(f"结果目录: {results.get('result_dir', 'N/A')}")
        print("=" * 60)
    else:
        print("\n[ERROR] 分析失败")
        sys.exit(1)


def list_available_sources():
    """列出可用的数据源文件"""
    from src.utils.config import Config
    
    print("=" * 60)
    print("可用的数据源文件")
    print("=" * 60)
    
    data_files = list(Config.DATA_DIR.glob("*.csv")) + \
                 list(Config.DATA_DIR.glob("*.xlsx")) + \
                 list(Config.DATA_DIR.glob("*.json"))
    
    if not data_files:
        print("\ndata目录中没有找到数据文件")
        return
    
    # 按类型分组
    sources = {'专利': [], '文献': [], '研报': [], '资讯': [], '其他': []}
    
    for f in data_files:
        fname = f.name.lower()
        if '专利' in fname or 'patent' in fname:
            sources['专利'].append(f)
        elif '文献' in fname or 'literature' in fname or 'paper' in fname:
            sources['文献'].append(f)
        elif '研报' in fname or 'report' in fname:
            sources['研报'].append(f)
        elif '资讯' in fname or 'news' in fname:
            sources['资讯'].append(f)
        else:
            sources['其他'].append(f)
    
    for source_type, files in sources.items():
        if files:
            print(f"\n[{source_type}]")
            for f in files:
                try:
                    import pandas as pd
                    if f.suffix == '.csv':
                        df = pd.read_csv(f)
                    elif f.suffix in ['.xlsx', '.xls']:
                        df = pd.read_excel(f)
                    else:
                        df = pd.read_json(f)
                    print(f"  - {f.name}: {len(df)} 条")
                except:
                    print(f"  - {f.name}")


def build_source_config(args):
    """构建数据源配置"""
    config = {
        'sources': None,
        'counts': {}
    }
    
    # 解析数据源类型
    if args.sources:
        config['sources'] = [s.strip().lower() for s in args.sources.split(',')]
    
    # 解析各数据源数量
    if args.patent is not None:
        config['counts']['patent'] = args.patent
    if args.literature is not None:
        config['counts']['literature'] = args.literature
    if args.report is not None:
        config['counts']['report'] = args.report
    if args.news is not None:
        config['counts']['news'] = args.news
    
    return config


if __name__ == "__main__":
    main()
