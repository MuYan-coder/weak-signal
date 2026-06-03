"""API 调用统计模块，支持跨 Streamlit rerun 持久化。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATS_FILE = PROJECT_ROOT / "data" / "api_stats.json"


@dataclass
class APIStats:
    """记录和统计 API 调用情况"""

    calls: List[Dict] = field(default_factory=list)
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_calls: int = 0

    @classmethod
    def from_dict(cls, payload: Dict) -> "APIStats":
        return cls(
            calls=payload.get("calls", []),
            total_prompt_tokens=payload.get("total_prompt_tokens", 0),
            total_completion_tokens=payload.get("total_completion_tokens", 0),
            total_calls=payload.get("total_calls", 0),
        )

    def to_dict(self) -> Dict:
        return {
            "calls": self.calls,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_calls": self.total_calls,
        }

    def record(
        self,
        call_type: str,
        prompt_tokens: int,
        completion_tokens: int,
        success: bool = True,
        metadata: Dict | None = None,
    ):
        """记录一次 API 调用"""
        metadata = metadata or {}
        self.calls.append(
            {
                "type": call_type,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "success": success,
                "metadata": metadata,
            }
        )
        if success:
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self.total_calls += 1

    def reset(self):
        self.calls = []
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_calls = 0

    def get_summary(self) -> str:
        """返回统计摘要"""
        if not self.calls:
            return "未进行任何 API 调用。"

        successful_calls = sum(1 for c in self.calls if c["success"])
        failed_calls = len(self.calls) - successful_calls

        summary = f"""
════════════════════════════════════════════════════════════════
【API 调用统计】
════════════════════════════════════════════════════════════════
总调用次数: {len(self.calls)} 次 (成功: {successful_calls}, 失败: {failed_calls})
总 Token 消耗: {self.total_prompt_tokens + self.total_completion_tokens:,} 个
  - Prompt Token: {self.total_prompt_tokens:,}
  - Completion Token: {self.total_completion_tokens:,}

【按类型统计】
"""

        by_type = {}
        for call in self.calls:
            call_type = call["type"]
            if call_type not in by_type:
                by_type[call_type] = {"count": 0, "tokens": 0, "success": 0, "fail": 0}
            by_type[call_type]["count"] += 1
            by_type[call_type]["tokens"] += call["total_tokens"]
            if call["success"]:
                by_type[call_type]["success"] += 1
            else:
                by_type[call_type]["fail"] += 1

        for call_type, stats in by_type.items():
            summary += f"\n  {call_type}:"
            summary += f"\n    - 调用次数: {stats['count']} (成功: {stats['success']}, 失败: {stats['fail']})"
            summary += f"\n    - Token 消耗: {stats['tokens']:,}"

        summary += "\n════════════════════════════════════════════════════════════════\n"
        return summary

    def print_summary(self):
        print(self.get_summary())


def _load_stats() -> APIStats:
    if not STATS_FILE.exists():
        return APIStats()

    try:
        payload = json.loads(STATS_FILE.read_text(encoding="utf-8"))
        return APIStats.from_dict(payload)
    except Exception:
        return APIStats()


def _save_stats(stats: APIStats):
    STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATS_FILE.write_text(
        json.dumps(stats.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


_stats = _load_stats()


def record_call(
    call_type: str,
    prompt_tokens: int,
    completion_tokens: int,
    success: bool = True,
    metadata: Dict | None = None,
):
    """记录一次调用并持久化。"""
    global _stats
    _stats = _load_stats()
    _stats.record(call_type, prompt_tokens, completion_tokens, success, metadata=metadata)
    _save_stats(_stats)
