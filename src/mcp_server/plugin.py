from src.mcp_server.settings import WeakSignalMCPSettings
from src.mcp_server.service import (
    get_service_info, 
    list_data_sources,
    validate_domain_pack_tool as _validate_domain_pack,
    get_recommended_source_counts_tool as _get_recommended_source_counts,
    prepare_domain_pack_tool as _prepare_domain_pack,
    run_analysis_tool as _run_analysis,
    start_analysis_job_tool as _start_analysis_job,
    read_result_resource_tool as _read_result_resource,
    list_runs_tool as _list_runs,
    run_end_to_end_analysis_tool as _run_end_to_end_analysis
)
from mcp.server.fastmcp import FastMCP

def register_weak_signal_mcp(mcp: FastMCP, settings: WeakSignalMCPSettings | None = None) -> None:
    if settings is None:
        from pathlib import Path
        settings = WeakSignalMCPSettings(project_root=Path.cwd())

    tool_prefix = settings.tool_prefix

    # Dynamically define the tool function to have the correct name in the schema
    # FastMCP derives the tool name from the function name.
    
    # We can use a wrapper to rename the tool, or register it.
    # FastMCP's @mcp.tool() uses the function name. We can pass name parameter: @mcp.tool(name=...)
    
    @mcp.tool(name=f"{tool_prefix}get_service_info", description="获取弱信号服务的状态、版本和目录信息")
    def get_service_info_tool() -> dict:
        return get_service_info(settings)

    @mcp.tool(name=f"{tool_prefix}list_data_sources", description="列出 data/ 目录下可用于分析的本地数据文件")
    def list_data_sources_tool(include_nested: bool = False) -> dict:
        return list_data_sources(settings, include_nested)

    @mcp.tool(name=f"{tool_prefix}analyze_weak_signals", description="一键执行完整的弱信号分析流程。输入技术领域和关键词，自动完成领域包准备、数据抽取、分析计算并生成最终报告摘要。")
    def analyze_weak_signals_tool(field_name: str, keywords: list[str], exclude_terms: list[str] = None, sample_size: int = 20, source_types: list[str] = None, data_path: str = "") -> dict:
        return _run_end_to_end_analysis(field_name, keywords, exclude_terms, sample_size, source_types, data_path, settings)

    # All other fine-grained tools and resources have been removed to expose only a single end-to-end tool.

    # In the future, we will register more tools, resources, and prompts here.
    # For now, it only registers the health check tool.
