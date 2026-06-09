"""Weak Signal MCP Server — entry point.

Launch via:
    python -m src.mcp_server.server
or via the console script registered in pyproject.toml:
    weak-signal-mcp
"""

from mcp.server.fastmcp import FastMCP
from src.mcp_server.settings import WeakSignalMCPSettings
from src.mcp_server.plugin import register_weak_signal_mcp


def create_server() -> FastMCP:
    settings = WeakSignalMCPSettings.from_env()
    mcp = FastMCP("weak-signal-mcp")
    register_weak_signal_mcp(mcp, settings)
    return mcp


def main() -> None:
    """Console entry point — validates environment before starting."""
    import sys

    print("[weak-signal-mcp] Starting server …", file=sys.stderr)

    # --- pre-flight checks ---
    _check_python_version()
    _check_env_configured()

    mcp = create_server()
    mcp.run(transport="stdio")


def _check_python_version() -> None:
    import sys

    if sys.version_info < (3, 11):
        print(
            f"[weak-signal-mcp] WARNING: Python {sys.version_info.major}.{sys.version_info.minor} "
            "detected. This server is tested on Python 3.11+. "
            "Some features may not work correctly.",
            file=sys.stderr,
        )


def _check_env_configured() -> None:
    import os

    from src.utils.env_config import ensure_env_loaded
    ensure_env_loaded()

    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_anthropic = bool(os.getenv("ANTHROPIC_AUTH_TOKEN"))
    has_deepseek = bool(os.getenv("DEEPSEEK_API_KEY"))

    if not (has_openai or has_anthropic or has_deepseek):
        print(
            "[weak-signal-mcp] WARNING: No LLM API key detected.\n"
            "  The server will start, but Domain Pack generation and event extraction\n"
            "  require an LLM backend. Set one of:\n"
            "    - OPENAI_API_KEY + OPENAI_BASE_URL\n"
            "    - ANTHROPIC_AUTH_TOKEN + ANTHROPIC_BASE_URL\n"
            "    - DEEPSEEK_API_KEY\n"
            "  Copy .env.example to .env and fill in your credentials.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
