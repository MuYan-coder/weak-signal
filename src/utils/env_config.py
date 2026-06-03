from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_KEYS = {
    "DEEPSEEK_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "PATENTSVIEW_API_KEY",
    "LENS_API_TOKEN",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_MODEL",
    "EXTRACTION_MODEL",
    "REPORT_MODEL",
    "AGENT_MODEL",
}


def _normalize_env_value(raw_value: str) -> str:
    value = str(raw_value or "").strip()
    if "&&" in value:
        value = value.split("&&", 1)[0].strip()
    value = value.rstrip("\\").strip()
    return value.strip('"').strip("'")


def ensure_env_loaded() -> None:
    """从项目根目录 `.env` 补充环境变量。

    只在系统环境中缺失时写入，避免覆盖用户当前 shell 中已设置的值。
    """
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = _normalize_env_value(value)
        # 加载所有变量，不仅仅是 ENV_KEYS 中的
        if not os.getenv(key):
            os.environ[key] = value


def has_env_key(name: str) -> bool:
    ensure_env_loaded()
    return bool(os.getenv(name))


def get_openai_compatible_base_url(raw_base_url: str | None = None) -> str:
    """Return a base URL suitable for the OpenAI-compatible client.

    讯飞 MaaS 当前对 Anthropic 原生接口使用 `/anthropic`，但对 OpenAI 兼容接口
    需要 `/anthropic/v1`。这里统一做一次兼容修正，避免调用层各自拼接。
    """
    ensure_env_loaded()
    base_url = (raw_base_url or os.getenv("ANTHROPIC_BASE_URL", "")).strip()
    if not base_url:
        return ""
    if "xf-yun.com" in base_url and not base_url.rstrip("/").endswith("/v1"):
        return base_url.rstrip("/") + "/v1"
    return base_url


def get_preferred_chat_model() -> str:
    """Return the default chat-completions model for the active provider."""
    ensure_env_loaded()
    # 优先从环境变量读取配置的模型
    if os.getenv("OPENAI_MODEL"):
        return os.getenv("OPENAI_MODEL")
    if os.getenv("EXTRACTION_MODEL"):
        return os.getenv("EXTRACTION_MODEL")
    if os.getenv("ANTHROPIC_AUTH_TOKEN") and os.getenv("ANTHROPIC_BASE_URL"):
        return "astron-code-latest"
    return "deepseek-chat"
