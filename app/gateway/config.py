"""Portkey gateway config (routing strategy, caching, retries, targets).

Sent with every request in the ``x-portkey-config`` header. An explicit PORTKEY_CONFIG
(saved config id "pc_..." or inline JSON) takes precedence; otherwise, when GROQ_SLUG is
set, the built-in Groq fallback config below is used.
"""

from __future__ import annotations

import json
from typing import Any

from app.config import Settings

# Groq models in fallback order: the 120B model first, the faster 20B model if it fails.
# (Groq retired the Llama 3.x models these replaced.)
GROQ_PRIMARY_MODEL = "openai/gpt-oss-120b"
GROQ_FALLBACK_MODEL = "openai/gpt-oss-20b"


def groq_fallback_config(settings: Settings) -> dict[str, Any]:
    """Fallback across two Groq models (separate Model Catalog providers), with caching and retries.

    - fallback: try targets in order, moving on when one fails;
    - cache "simple": identical requests are answered from Portkey's cache;
    - retry: each target is retried twice on rate limits (429) and overload (503).
    Each target's ``override_params.model`` replaces the model named in the request.
    """
    return {
        "strategy": {"mode": "fallback"},
        "cache": {"mode": "simple"},
        "retry": {
            "attempts": 2,
            "on_status_codes": [429, 503],
        },
        "targets": [
            {"override_params": {"model": f"@{settings.groq_slug}/{GROQ_PRIMARY_MODEL}"}},
            {"override_params": {"model": f"@{settings.groq_slug_2 or settings.groq_slug}/{GROQ_FALLBACK_MODEL}"}},
        ],
    }


def build_gateway_config(settings: Settings) -> dict[str, Any] | str | None:
    """The config to send: PORTKEY_CONFIG if set (id string or parsed JSON), else the Groq
    fallback config if GROQ_SLUG is set, else None (plain gateway call to LLM_MODEL)."""
    if settings.portkey_config and settings.portkey_config.strip():
        config = settings.portkey_config.strip()
        if not config.startswith("{"):
            return config  # saved config id, e.g. "pc_..."
        try:
            return json.loads(config)
        except json.JSONDecodeError as exc:
            raise ValueError(f"PORTKEY_CONFIG is not valid JSON: {exc}") from exc
    if settings.groq_slug:
        return groq_fallback_config(settings)
    return None


def first_target_model(config: dict[str, Any] | str | None) -> str | None:
    """Model of the first target in an inline config (what the request should name)."""
    if not isinstance(config, dict):
        return None
    for target in config.get("targets", []):
        model = (target.get("override_params") or {}).get("model")
        if model:
            return model
    return None
