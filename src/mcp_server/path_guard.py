from pathlib import Path
from src.mcp_server.settings import WeakSignalMCPSettings

ALLOWED_ARTIFACTS = {
    "report.txt",
    "report_metadata.json",
    "domain_pack.yaml",
    "events.json",
    "event_quality.json",
    "candidate_forms.json",
    "scored.json",
    "refined.json",
    "validated.json",
    "signals.json",
    "weak_signals.json",
    "weak_signals_comparison.json",
    "temporal_validation.json",
    "reverse_validation.json",
    "signal_evidence_links.json",
    "signal_reliability.json",
    "source_documents.json",
    "report_evidence_packets.json",
    "report_grounded_facts.json",
    "report_grounding_check.json",
    "family_evaluation.json",
    "family_evaluation_report.md",
    "signal_generation_diagnostics.json",
    "migration_validation.md"
}

def assert_allowed_artifact(artifact: str) -> None:
    """验证目标产物是否在白名单中"""
    if artifact not in ALLOWED_ARTIFACTS:
        raise ValueError(f"Artifact '{artifact}' is not in the allowed reading whitelist.")

def resolve_safe_path(value: str | Path, settings: WeakSignalMCPSettings, allowed_roots: list[Path], must_exist: bool = True) -> Path:
    """解析并验证路径是否在允许的根目录列表中"""
    # 转换为绝对路径
    path_obj = Path(value)
    
    # 防止空路径
    if str(path_obj) == ".":
        raise ValueError("Invalid path: '.' is not allowed as a direct input file/dir.")
        
    # 处理相对路径和绝对路径
    if not path_obj.is_absolute():
        path_obj = (settings.project_root / path_obj).resolve()
    else:
        path_obj = path_obj.resolve()
        
    # 验证是否在 allowed_roots 之内
    is_allowed = False
    for root in allowed_roots:
        resolved_root = root.resolve()
        try:
            # relative_to will raise ValueError if path_obj is not under resolved_root
            path_obj.relative_to(resolved_root)
            is_allowed = True
            break
        except ValueError:
            continue
            
    if not is_allowed:
        raise PermissionError(f"Access denied. Path '{value}' is outside the allowed directories.")
        
    # 如果要求必须存在
    if must_exist and not path_obj.exists():
        raise FileNotFoundError(f"File or directory not found: {path_obj}")
        
    return path_obj

def resolve_result_dir(run_id_or_path: str, settings: WeakSignalMCPSettings) -> Path:
    """将 run_id 或相对路径解析为安全的 result_dir"""
    # 如果输入只是一个没有分隔符的 run_id（通常是时间戳）
    if "/" not in run_id_or_path and "\\" not in run_id_or_path:
        target_path = settings.result_dir / run_id_or_path
    else:
        target_path = Path(run_id_or_path)
        
    return resolve_safe_path(
        value=target_path,
        settings=settings,
        allowed_roots=[settings.result_dir],
        must_exist=True
    )
