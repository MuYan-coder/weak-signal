from __future__ import annotations

import os
from typing import Any

import httpx
from anthropic import Anthropic
from openai import OpenAI

from .env_config import ensure_env_loaded, get_preferred_chat_model


ensure_env_loaded()
_provider: str | None = None
_client: Any = None

DEFAULT_TIMEOUT = 120.0


def get_provider_and_client():
    global _provider, _client
    if _client is not None and _provider is not None:
        return _provider, _client

    # 优先检查 OPENAI_API_KEY（支持硅基流动等兼容API）
    openai_key = os.getenv("OPENAI_API_KEY")
    openai_base_url = os.getenv("OPENAI_BASE_URL")
    if openai_key and openai_base_url:
        _provider = "openai"
        _client = OpenAI(
            api_key=openai_key,
            base_url=openai_base_url,
            timeout=httpx.Timeout(DEFAULT_TIMEOUT, connect=30.0),
        )
        return _provider, _client

    # 兼容旧的 ANTHROPIC 配置
    anthropic_token = os.getenv("ANTHROPIC_AUTH_TOKEN")
    anthropic_base_url = os.getenv("ANTHROPIC_BASE_URL")
    if anthropic_token and anthropic_base_url:
        _provider = "anthropic"
        _client = Anthropic(
            api_key=anthropic_token,
            base_url=anthropic_base_url,
            timeout=DEFAULT_TIMEOUT,
        )
        return _provider, _client

    # 兼容旧的 DEEPSEEK 配置
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    if deepseek_key:
        _provider = "openai"
        _client = OpenAI(
            api_key=deepseek_key,
            base_url="https://api.deepseek.com",
            timeout=httpx.Timeout(DEFAULT_TIMEOUT, connect=30.0),
        )
        return _provider, _client

    _provider = None
    _client = None
    return _provider, _client


def chat_text(
    prompt: str,
    *,
    system: str | None = None,
    model: str | None = None,
    max_tokens: int = 512,
    temperature: float = 0.1,
    timeout: int | None = None,
):
    provider, client = get_provider_and_client()
    if client is None or provider is None:
        return "", {"prompt_tokens": 0, "completion_tokens": 0}, None

    model_name = model or get_preferred_chat_model()

    if provider == "anthropic":
        kwargs = {
            "model": model_name,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        response = client.messages.create(**kwargs)
        text = "".join(getattr(block, "text", "") for block in getattr(response, "content", []))
        usage = getattr(response, "usage", None)
        usage_info = {
            "prompt_tokens": int(getattr(usage, "input_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage, "output_tokens", 0) or 0),
        }
        return text, usage_info, response

    extra_kwargs = {}
    if timeout is not None:
        extra_kwargs["timeout"] = timeout
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        **extra_kwargs,
    )
    text = response.choices[0].message.content or ""
    usage = getattr(response, "usage", None)
    usage_info = {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
    }
    return text, usage_info, response
