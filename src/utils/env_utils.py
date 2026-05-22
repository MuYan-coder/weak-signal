import os
from pathlib import Path

ENV_KEYS = {
    "DEEPSEEK_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
}


def _normalize_env_value(raw_value: str) -> str:
    value = str(raw_value or "").strip()
    if "&&" in value:
        value = value.split("&&", 1)[0].strip()
    value = value.rstrip("\\").strip()
    return value.strip('"').strip("'")


def _candidate_env_paths():
    project_root = Path(__file__).resolve().parent.parent
    return [
        project_root / '.env',
        project_root.parent.parent / 'weak_signal_demo' / '.env',
    ]


def load_project_env():
    for env_path in _candidate_env_paths():
        if not env_path.exists():
            continue
        try:
            for raw_line in env_path.read_text(encoding='utf-8').splitlines():
                line = raw_line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                key, value = line.split('=', 1)
                key = key.strip()
                value = _normalize_env_value(value)
                if key in ENV_KEYS and value and key not in os.environ:
                    os.environ[key] = value
        except Exception:
            continue


def has_deepseek_key():
    load_project_env()
    return bool(os.getenv('DEEPSEEK_API_KEY'))


def has_anthropic_key():
    load_project_env()
    return bool(os.getenv('ANTHROPIC_AUTH_TOKEN') and os.getenv('ANTHROPIC_BASE_URL'))
