import json
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
import numpy as np
from src.mcp_server.settings import WeakSignalMCPSettings

def dataframe_summary(df: pd.DataFrame, preview_rows: int = 5) -> Dict[str, Any]:
    """返回 DataFrame 的摘要和预览，避免大表阻塞传输"""
    if df is None or df.empty:
        return {"row_count": 0, "columns": [], "preview": []}
    
    # 将 numpy types 转换为原生 types 方便 JSON 序列化
    preview = df.head(preview_rows).replace({np.nan: None}).to_dict(orient="records")
    return {
        "row_count": len(df),
        "columns": list(df.columns),
        "preview": preview
    }

def to_jsonable(value: Any, settings: WeakSignalMCPSettings | None = None) -> Any:
    """递归将不可 JSON 序列化的对象转换为可序列化对象"""
    if isinstance(value, pd.DataFrame):
        return dataframe_summary(value)
    elif isinstance(value, Path):
        # 尝试返回相对路径
        if settings and settings.project_root:
            try:
                # 转换出可读的相对路径
                return str(value.relative_to(settings.project_root)).replace("\\", "/")
            except ValueError:
                return str(value).replace("\\", "/")
        return str(value).replace("\\", "/")
    elif isinstance(value, (np.integer, np.int64, np.int32)):
        return int(value)
    elif isinstance(value, (np.floating, np.float64, np.float32, float)):
        import math
        if math.isnan(value) or math.isinf(value):
            return None
        return float(value)
    elif isinstance(value, (np.bool_, bool)):
        return bool(value)
    elif isinstance(value, np.ndarray):
        return [to_jsonable(v, settings) for v in value.tolist()]
    elif isinstance(value, dict):
        return {k: to_jsonable(v, settings) for k, v in value.items()}
    elif isinstance(value, list):
        return [to_jsonable(v, settings) for v in value]
    elif isinstance(value, tuple):
        return [to_jsonable(v, settings) for v in value]
    elif value is None or isinstance(value, (int, float, str)):
        return value
    else:
        return str(value)

def result_resources(run_id: str, settings: WeakSignalMCPSettings | None = None) -> List[str]:
    """生成运行结果的 resource URIs"""
    scheme = settings.resource_scheme if settings else "weak-signal"
    return [
        f"{scheme}://results/{run_id}/report",
        f"{scheme}://results/{run_id}/signals",
        f"{scheme}://results/{run_id}/weak-signals",
        f"{scheme}://results/{run_id}/events",
        f"{scheme}://results/{run_id}/domain-pack"
    ]

def report_preview(text: str, limit: int = 300) -> str:
    """截断报告文本用于预览"""
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + f"... [Truncated. Use resource URI to read full text. length={len(text)}]"
