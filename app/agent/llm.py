"""Chat models used by the agent's nodes."""

from __future__ import annotations

from typing import Any

from langchain_core.exceptions import ModelAPIError, ModelRateLimitError
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable

from app.config import Settings, get_settings

# Transient provider errors worth retrying: rate limits (429) and server overload (503).
RETRYABLE_ERRORS = (ModelRateLimitError, ModelAPIError)


def build_chat_model(settings: Settings | None = None, model: str | None = None) -> BaseChatModel:
    """Chat model from settings; ``model`` overrides LLM_MODEL.

    LLM_PROVIDER=gemini calls Google directly; LLM_PROVIDER=portkey routes through the
    Portkey gateway (see app/gateway).
    """
    settings = settings or get_settings()
    if settings.llm_provider == "portkey":
        from app.gateway import build_portkey_chat_model

        return build_portkey_chat_model(settings, model)
    if settings.google_api_key is None:
        raise ValueError("GOOGLE_API_KEY (or GEMINI_API_KEY) is not set (environment or .env)")
    from langchain_google_genai import ChatGoogleGenerativeAI

    kwargs: dict[str, Any] = {}
    if settings.llm_temperature is not None:  # newer Gemini models use fixed sampling and ignore it
        kwargs["temperature"] = settings.llm_temperature
    return ChatGoogleGenerativeAI(
        model=model or settings.llm_model,
        google_api_key=settings.google_api_key,
        timeout=settings.llm_timeout,
        max_retries=settings.llm_max_retries,
        **kwargs,
    )


def with_retry(runnable: Runnable) -> Runnable:
    """Retry rate-limit/overload errors with backoff (~5s, 10s, 20s), e.g. on free-tier quotas."""
    return runnable.with_retry(
        retry_if_exception_type=RETRYABLE_ERRORS,
        stop_after_attempt=4,
        wait_exponential_jitter=True,
        exponential_jitter_params={"initial": 5, "max": 30},
    )
