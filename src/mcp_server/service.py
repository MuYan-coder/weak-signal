from typing import Any, Dict, List
from src.mcp_server.settings import WeakSignalMCPSettings

def get_service_info(settings: WeakSignalMCPSettings) -> Dict[str, Any]:
    return {
        "name": "weak-signal-mcp",
        "project": "产业技术预见智能体",
        "version": "0.1.0",
        "capabilities": ["tools", "resources", "prompts"],
        "mode": "plugin_registered",
        "tool_prefix": settings.tool_prefix,
        "data_dir": str(settings.data_dir),
        "result_dir": str(settings.result_dir),
        "memory_dir": str(settings.memory_dir)
    }

def _guess_source_type(filename: str) -> str:
    filename = filename.lower()
    if "patent" in filename or "专利" in filename:
        return "patent"
    if "literature" in filename or "paper" in filename or "文献" in filename or "论文" in filename:
        return "literature"
    if "report" in filename or "研报" in filename or "报告" in filename:
        return "report"
    if "news" in filename or "资讯" in filename or "新闻" in filename:
        return "news"
    return "unknown"

def list_data_sources(settings: WeakSignalMCPSettings, include_nested: bool = False) -> Dict[str, Any]:
    """列出 data_dir 下可用于分析的本地数据文件"""
    data_dir = settings.data_dir
    if not data_dir or not data_dir.exists():
        return {
            "data_dir": str(data_dir).replace("\\", "/") if data_dir else "",
            "files": []
        }
    
    files = []
    pattern = "**/*" if include_nested else "*"
    
    for file_path in data_dir.glob(pattern):
        if file_path.is_file() and file_path.suffix.lower() in [".xlsx", ".csv", ".json", ".jsonl"]:
            try:
                rel_path = file_path.relative_to(settings.project_root)
            except ValueError:
                rel_path = file_path
                
            files.append({
                "name": file_path.name,
                "relative_path": str(rel_path).replace("\\", "/"),
                "suffix": file_path.suffix.lower(),
                "source_type_hint": _guess_source_type(file_path.name),
                "size_bytes": file_path.stat().st_size
            })
            
    return {
        "data_dir": str(data_dir).replace("\\", "/"),
        "files": files
    }

def validate_domain_pack_tool(
    pack_ref: str,
    require_dry_run: bool = False,
    dry_run_report_path: str = "",
    settings: WeakSignalMCPSettings | None = None
) -> Dict[str, Any]:
    """校验指定 Domain Pack 是否符合 schema 和质量门槛"""
    from src.domain.domain_pack_loader import load_domain_pack
    from src.domain.domain_pack_validator import validate_domain_pack
    from src.domain.domain_pack_dry_run import load_domain_pack_dry_run_report
    from src.mcp_server.path_guard import resolve_safe_path
    from src.mcp_server.serialization import to_jsonable
    
    try:
        pack = load_domain_pack(pack_ref)
    except Exception as e:
        return {"error": f"Failed to load pack_ref {pack_ref}: {str(e)}"}
        
    dry_run_report = None
    if dry_run_report_path and settings:
        try:
            safe_path = resolve_safe_path(
                dry_run_report_path, 
                settings, 
                allowed_roots=[settings.result_dir, settings.memory_dir], 
                must_exist=True
            )
            dry_run_report = load_domain_pack_dry_run_report(safe_path)
        except Exception as e:
            return {"error": f"Failed to load dry_run_report: {str(e)}"}
            
    try:
        report = validate_domain_pack(
            pack,
            dry_run_report=dry_run_report,
            require_dry_run=require_dry_run
        )
        return to_jsonable(report.to_dict(), settings)
    except Exception as e:
        return {"error": f"Validation failed: {str(e)}"}

def get_recommended_source_counts_tool(
    pack_ref: str,
    total_sample_size: int,
    source_types: List[str],
    backend: str = "mock",
    settings: WeakSignalMCPSettings | None = None
) -> Dict[str, Any]:
    """自动查询数据库当前可用数量，并根据 Domain Pack 配置推荐各数据源采样量"""
    from src.domain.domain_pack_loader import load_domain_pack
    from src.domain.domain_pack_source_counts import recommend_source_counts
    from src.data_access.repository import DataRepository
    from src.data_access.models import SourceQuery
    from src.mcp_server.serialization import to_jsonable
    
    try:
        pack = load_domain_pack(pack_ref)
    except Exception as e:
        return {"error": f"Failed to load pack_ref {pack_ref}: {str(e)}"}
        
    try:
        import os
        old_backend = os.getenv("WEAK_SIGNAL_DATA_BACKEND")
        os.environ["WEAK_SIGNAL_DATA_BACKEND"] = backend
        try:
            repo = DataRepository.from_env()
            core_keywords = pack.search_strategy.get("core_keywords", []) if isinstance(pack.search_strategy, dict) else []
            query = SourceQuery(
                tech_field_id=pack.pack_id,
                tech_field_name=pack.pack_name,
                source_types=source_types, 
                keywords=core_keywords
            )
            available_counts = repo.get_source_counts(query)
        finally:
            if old_backend is not None:
                os.environ["WEAK_SIGNAL_DATA_BACKEND"] = old_backend
            else:
                del os.environ["WEAK_SIGNAL_DATA_BACKEND"]
                
        rec = recommend_source_counts(
            pack,
            available_counts=available_counts,
            total_sample_size=total_sample_size,
            source_types=source_types
        )
        
        return {
            "total_count": rec.total_count,
            "backend": backend,
            "final_counts": rec.final_counts,
            "rows": to_jsonable([row.__dict__ for row in rec.rows], settings)
        }
    except Exception as e:
        return {"error": f"Failed to recommend source counts: {str(e)}"}

def prepare_domain_pack_tool(
    field_id: str,
    field_name: str,
    keywords: List[str],
    synonyms: List[str] = None,
    exclude_terms: List[str] = None,
    source_types: List[str] = None,
    settings: WeakSignalMCPSettings | None = None
) -> Dict[str, Any]:
    """生成并准备 Domain Pack"""
    from src.domain.domain_pack_workflow import DomainPackWorkflow
    from src.domain.domain_pack_generator import DomainPackGenerationRequest
    from src.mcp_server.serialization import to_jsonable
    
    if not settings:
        return {"error": "Missing MCP settings"}
        
    try:
        req = DomainPackGenerationRequest(
            field_id=field_id,
            field_name=field_name,
            keywords=keywords or [],
            synonyms=synonyms or [],
            exclude_terms=exclude_terms or [],
            source_types=source_types or ["paper", "news", "policy", "report", "patent"]
        )
        
        workflow = DomainPackWorkflow(memory_dir=settings.memory_dir)
        result = workflow.prepare_domain_pack(req)
        
        return {
            "status": result.status,
            "manual_review_required": result.manual_review_required,
            "domain_pack_id": result.domain_pack.pack_id,
            "domain_pack_hash": result.domain_pack.domain_pack_hash,
            "domain_pack_version": result.domain_pack.domain_pack_version,
            "validation": to_jsonable(result.validation_report.to_dict(), settings),
            "review_hints": result.review_hints
        }
    except Exception as e:
        return {"error": f"Failed to prepare domain pack: {str(e)}"}

def run_analysis_tool(
    sample_size: int = 20,
    use_cache: bool = True,
    domain_pack_ref: str = "neutral",
    data_path: str = "",
    source_config: Dict[str, Any] = None,
    settings: WeakSignalMCPSettings | None = None
) -> Dict[str, Any]:
    """同步运行弱信号分析流程。该工具主要用于小样本调试、验收和单用户本地场景"""
    from src.core.pipeline import AnalysisPipeline
    from src.mcp_server.path_guard import resolve_safe_path
    from src.mcp_server.serialization import report_preview, result_resources
    
    if not settings:
        return {"error": "Missing MCP settings"}
        
    if sample_size > 100:
        return {"error": f"sample_size {sample_size} exceeds synchronous limit 100. Use start_analysis_job instead."}
        
    if data_path:
        try:
            safe_data_path = resolve_safe_path(data_path, settings, [settings.data_dir], must_exist=True)
            if safe_data_path.suffix.lower() not in [".xlsx", ".xls", ".csv", ".json", ".jsonl"]:
                return {"error": "data_path must be a valid data file (.xlsx, .csv, .json)"}
            data_path = str(safe_data_path)
        except Exception as e:
            return {"error": f"Invalid data_path: {str(e)}"}
            
    try:
        pipeline = AnalysisPipeline(domain_pack_ref=domain_pack_ref)
        
        # Override RESULT_DIR temporarily if needed or pass it to pipeline?
        # AnalysisPipeline uses Config.RESULT_DIR, so we might need to rely on that or monkeypatch.
        # But for MCP we assume Config is already configured correctly with project root.
        
        result = pipeline.run_full_pipeline(
            data_path=data_path,
            sample_size=sample_size,
            use_cache=use_cache,
            source_config=source_config,
            domain_context=domain_pack_ref
        )
        
        if not result:
            return {"error": "Pipeline returned empty result"}
            
        result_dir_path = result.get("result_dir")
        run_id = result_dir_path.name if result_dir_path else "unknown"
        
        domain_context = result.get("domain_context")
        events_df = result.get("events_df")
        candidate_forms_df = result.get("candidate_forms_df")
        signals_df = result.get("signals_df")
        
        weak_signal_count = 0
        if signals_df is not None and not signals_df.empty and "signal_type" in signals_df.columns:
            weak_signal_count = len(signals_df[signals_df["signal_type"] == "weak_signal"])
            
        return {
            "result_dir": str(result_dir_path).replace("\\", "/") if result_dir_path else "",
            "domain_pack_id": domain_context.domain_pack_id if domain_context else "neutral",
            "domain_pack_hash": domain_context.domain_pack_hash if domain_context else "",
            "event_count": len(events_df) if events_df is not None else 0,
            "candidate_count": len(candidate_forms_df) if candidate_forms_df is not None else 0,
            "signal_count": len(signals_df) if signals_df is not None else 0,
            "weak_signal_count": weak_signal_count,
            "report_preview": report_preview(result.get("report", {})),
            "resources": {
                res.split("/")[-1]: res for res in result_resources(run_id, settings)
            }
        }
    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}"}

def start_analysis_job_tool(
    sample_size: int = 100,
    use_cache: bool = True,
    domain_pack_ref: str = "neutral",
    data_path: str = "",
    source_config: Dict[str, Any] = None,
    settings: WeakSignalMCPSettings | None = None
) -> Dict[str, Any]:
    """以 job 模式启动完整弱信号分析"""
    # For now, simulate background job
    return {
        "job_id": "job_12345",
        "status": "queued",
        "message": "Analysis job started in background (mock implementation for MCP proxy)",
        "sample_size": sample_size
    }

def read_result_resource_tool(
    resource_uri: str,
    settings: WeakSignalMCPSettings | None = None
) -> Dict[str, Any]:
    """读取分析产出的指定资源"""
    from src.mcp_server.path_guard import resolve_result_dir
    import json
    import pandas as pd
    from src.mcp_server.serialization import to_jsonable
    
    if not settings:
        return {"error": "Missing MCP settings"}
        
    if not resource_uri.startswith("weak-signal://results/"):
        return {"error": f"Invalid resource URI: {resource_uri}"}
        
    parts = resource_uri.replace("weak-signal://results/", "").split("/")
    if len(parts) != 2:
        return {"error": "Malformed resource URI. Expected format: weak-signal://results/{run_id}/{resource_type}"}
        
    run_id, resource_type = parts
    
    try:
        run_dir = resolve_result_dir(run_id, settings)
    except Exception as e:
        return {"error": str(e)}
        
    file_map = {
        "report": "report.txt",
        "signals": "signals.json",
        "weak-signals": "weak_signals.json",
        "events": "events.json",
        "domain-pack": "domain_pack.yaml",
        "evidence-links": "signal_evidence_links.json",
        "reliability": "signal_reliability.json"
    }
    
    if resource_type not in file_map:
        return {"error": f"Unsupported resource type: {resource_type}"}
        
    target_file = run_dir / file_map[resource_type]
    if not target_file.exists():
        if file_map[resource_type].endswith(".json"):
            csv_file = run_dir / file_map[resource_type].replace(".json", ".csv")
            if csv_file.exists():
                target_file = csv_file
                
    if not target_file.exists():
        return {"error": f"Resource file not found: {target_file.name}"}
        
    try:
        if target_file.suffix in [".txt", ".yaml"]:
            with open(target_file, "r", encoding="utf-8") as f:
                data = f.read()
        elif target_file.suffix == ".json":
            with open(target_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        elif target_file.suffix == ".csv":
            df = pd.read_csv(target_file, encoding="utf-8-sig")
            data = df.to_dict(orient="records")
        else:
            return {"error": f"Unsupported file type for reading: {target_file.suffix}"}
            
        return {
            "run_id": run_id,
            "resource_type": resource_type,
            "data": to_jsonable(data, settings)
        }
    except Exception as e:
        return {"error": f"Failed to read resource: {str(e)}"}

def list_runs_tool(
    limit: int = 10,
    settings: WeakSignalMCPSettings | None = None
) -> Dict[str, Any]:
    """列出最近的分析运行记录"""
    from src.mcp_server.serialization import result_resources
    import os
    
    if not settings:
        return {"error": "Missing MCP settings"}
        
    result_dir = settings.result_dir
    if not result_dir or not result_dir.exists():
        return {"runs": []}
        
    runs = []
    for item in sorted(result_dir.iterdir(), key=os.path.getmtime, reverse=True):
        if item.is_dir():
            run_id = item.name
            runs.append({
                "run_id": run_id,
                "timestamp": str(item.stat().st_mtime),
                "resources": {
                    res.split("/")[-1]: res for res in result_resources(run_id, settings)
                }
            })
            if len(runs) >= limit:
                break
                
    return {"runs": runs}

def run_end_to_end_analysis_tool(
    field_name: str,
    keywords: List[str],
    exclude_terms: List[str] = None,
    sample_size: int = 20,
    source_types: List[str] = None,
    data_path: str = "",
    settings: WeakSignalMCPSettings | None = None
) -> Dict[str, Any]:
    """一键执行完整的弱信号分析流程"""
    import uuid
    import time
    import os

    if not settings:
        return {"error": "Missing MCP settings"}

    start_time = time.time()

    # 生成 field_id (简单转换，或者直接用 uuid)
    import re
    field_id = re.sub(r'[^a-zA-Z0-9_]', '', field_name.lower())
    if not field_id:
        field_id = f"field_{uuid.uuid4().hex[:8]}"

    if not source_types:
        source_types = ["paper", "patent", "news", "report", "policy"]
    exclude_terms = exclude_terms or []

    # 步骤 1: 准备 Domain Pack
    pack_result = prepare_domain_pack_tool(
        field_id=field_id,
        field_name=field_name,
        keywords=keywords,
        synonyms=[],
        exclude_terms=exclude_terms,
        source_types=source_types,
        settings=settings
    )

    if "error" in pack_result:
        return {"error": f"Failed to prepare domain pack: {pack_result['error']}"}

    # 优先尝试从 memory/domain_packs/ 目录用完整文件路径加载生成的 domain pack
    domain_pack_ref = _resolve_generated_pack_ref(pack_result, settings)

    # 步骤 2: 构造数据源配置
    source_config = None
    if not data_path:
        source_config = _build_source_config(
            pack_ref=domain_pack_ref,
            pack_result=pack_result,
            field_name=field_name,
            keywords=keywords,
            exclude_terms=exclude_terms,
            source_types=source_types,
            sample_size=sample_size,
            settings=settings,
        )

    # 步骤 3: 执行全管线分析
    analysis_result = run_analysis_tool(
        sample_size=sample_size,
        use_cache=True,
        domain_pack_ref=domain_pack_ref,
        data_path=data_path,
        source_config=source_config,
        settings=settings
    )

    if "error" in analysis_result:
        return {"error": f"Analysis failed: {analysis_result['error']}"}

    elapsed_time = round(time.time() - start_time, 2)

    # 组装最终结果
    return {
        "status": "success",
        "domain_pack_id": pack_result.get("domain_pack_id", domain_pack_ref),
        "elapsed_seconds": elapsed_time,
        "summary": {
            "event_count": analysis_result.get("event_count", 0),
            "candidate_count": analysis_result.get("candidate_count", 0),
            "signal_count": analysis_result.get("signal_count", 0),
            "weak_signal_count": analysis_result.get("weak_signal_count", 0)
        },
        "report_preview": analysis_result.get("report_preview", ""),
        "resources": analysis_result.get("resources", {})
    }


def _resolve_generated_pack_ref(
    pack_result: Dict[str, Any],
    settings: WeakSignalMCPSettings,
) -> str:
    """Resolve the best domain_pack_ref for a generated Domain Pack.

    Tries these strategies in order:
    1. Full path to memory/domain_packs/{hash}.yaml (most reliable)
    2. Full path to result/*/domain_pack.yaml snapshot
    3. pack_id string (may be found by memory-dir search in load_domain_pack)
    4. "neutral" fallback
    """
    pack_hash = pack_result.get("domain_pack_hash", "")
    pack_id = pack_result.get("domain_pack_id", "neutral")

    # Strategy 1: memory directory via hash
    if pack_hash:
        memory_pack = settings.memory_dir / "domain_packs" / f"{pack_hash}.yaml"
        if memory_pack.exists():
            return str(memory_pack)

    # Strategy 2: also check config domain_pack_dir for pack_id
    config_pack = settings.config_domain_pack_dir / f"{pack_id}.yaml"
    if config_pack and config_pack.exists():
        return str(config_pack)

    # Strategy 3: return pack_id — the updated load_domain_pack will
    # also search memory/domain_packs/ by matching pack_id in YAML metadata
    return pack_id if pack_id else "neutral"


def _build_source_config(
    pack_ref: str,
    pack_result: Dict[str, Any],
    field_name: str,
    keywords: List[str],
    exclude_terms: List[str],
    source_types: List[str],
    sample_size: int,
    settings: WeakSignalMCPSettings,
) -> Dict[str, Any]:
    """Build source_config for data loading, preferring db backend when available.

    When WEAK_SIGNAL_DATA_BACKEND=db and a valid Domain Pack is loaded, the
    function constructs a SourceQuery carrying the pack's search strategy so
    that the pipeline can dynamically retrieve domain-relevant documents from
    MySQL / Elasticsearch instead of falling back to static local test files.
    """
    import os

    backend = os.getenv("WEAK_SIGNAL_DATA_BACKEND", "mock").strip().lower()

    if backend == "db":
        try:
            from src.domain.domain_pack_loader import load_domain_pack
            from src.data_access.models import SourceQuery

            domain_pack = load_domain_pack(pack_ref)
            search_strategy = (
                domain_pack.search_strategy
                if isinstance(domain_pack.search_strategy, dict)
                else {}
            )
            core_keywords = search_strategy.get("core_keywords", []) or keywords
            english_terms = search_strategy.get("english_terms", []) or []
            synonyms = search_strategy.get("synonyms", []) or []
            strategy_exclude = search_strategy.get("exclude_terms", []) or exclude_terms

            all_keywords = list(core_keywords) + list(english_terms)

            query = SourceQuery(
                tech_field_id=domain_pack.pack_id,
                tech_field_name=field_name,
                source_types=list(source_types),
                keywords=all_keywords,
                synonyms=list(synonyms),
                exclude_terms=list(strategy_exclude),
                counts={
                    st: max(1, sample_size // len(source_types))
                    for st in source_types
                },
            )
            return {
                "backend": "db",
                "query": query,
                "sources": source_types,
                "counts": dict(query.counts),
            }
        except Exception as exc:
            print(f"  [Domain Pack] db backend init failed ({exc}), falling back to local files")

    # Fallback: even distribution + mock/local-file backend
    per_source = max(1, sample_size // len(source_types))
    counts = {st: per_source for st in source_types}
    return {
        "backend": backend,
        "sources": source_types,
        "counts": counts,
    }
