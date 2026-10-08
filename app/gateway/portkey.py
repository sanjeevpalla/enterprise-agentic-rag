"""Portkey AI gateway for the agent's LLM calls.

Portkey sits between the app and LLM providers: one OpenAI-compatible API for 1,600+ models,
request logs and cost tracking, and gateway features (fallbacks, retries, load balancing,
caching) configured in Portkey instead of in code. Because the gateway speaks the OpenAI
API for every provider, LangChain's ``ChatOpenAI`` is used whatever the underlying model:

    ChatOpenAI(
        model="@google-prod/gemini-3.5-flash",   # Model Catalog slug: @<provider>/<model>
        base_url="https://aigw.portkey.ai/v1",
        api_key=PORTKEY_API_KEY,
        default_headers={"x-portkey-config": "pc_..."},  # optional fallbacks/retries/cache
    )

Provider credentials live in Portkey (Model Catalog), not in this app.
See https://portkey.ai/docs/aigw/integrations/libraries/langchain-python
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.language_models import BaseChatModel

from app.config import Settings, get_settings
from app.gateway.config import build_gateway_config, first_target_model

# Portkey request headers.
CONFIG_HEADER = "x-portkey-config"  # saved config id ("pc_...") or inline JSON config
METADATA_HEADER = "x-portkey-metadata"  # JSON object; filterable in Portkey's logs


def build_portkey_chat_model(
    settings: Settings | None = None,
    model: str | None = None,
    **chat_kwargs: Any,
) -> BaseChatModel:
    """A LangChain chat model that sends every call through the Portkey gateway.

    ``model`` is a Model Catalog slug (``@<provider>/<model>``); defaults to LLM_MODEL.
    When the gateway config has targets (e.g. the Groq fallback config), each target's
    ``override_params.model`` replaces the requested model, so the request names the
    first target's model and LLM_MODEL/PLANNER_MODEL don't apply.
    ``chat_kwargs`` are passed to ``ChatOpenAI`` (e.g. ``http_client`` in tests).
    """
    settings = settings or get_settings()
    if settings.portkey_api_key is None:
        raise ValueError("LLM_PROVIDER=portkey needs PORTKEY_API_KEY (environment or .env)")
    config = build_gateway_config(settings)
    model = first_target_model(config) or model or settings.llm_model
    if not model.startswith("@"):
        raise ValueError(
            f"With LLM_PROVIDER=portkey, models must be Portkey Model Catalog slugs like "
            f"'@google-prod/gemini-3.5-flash' (got {model!r}). Set LLM_MODEL/PLANNER_MODEL, "
            "or GROQ_SLUG to use the Groq fallback config."
        )
    from langchain_openai import ChatOpenAI

    if settings.llm_temperature is not None:
        chat_kwargs.setdefault("temperature", settings.llm_temperature)
    return ChatOpenAI(
        model=model,
        base_url=settings.portkey_base_url,
        api_key=settings.portkey_api_key,
        timeout=settings.llm_timeout,
        max_retries=settings.llm_max_retries,
        default_headers=portkey_headers(settings, config),
        **chat_kwargs,
    )


def portkey_headers(settings: Settings, config: dict[str, Any] | str | None = None) -> dict[str, str]:
    """Static Portkey headers: the gateway config (if any), plus metadata tagging every request."""
    headers = {
        METADATA_HEADER: json.dumps(
            {"app": "enterprise-agentic-rag", "environment": settings.langfuse_environment}
        )
    }
    if config is None:
        config = build_gateway_config(settings)
    if config is not None:
        headers[CONFIG_HEADER] = config if isinstance(config, str) else json.dumps(config)
    return headers
