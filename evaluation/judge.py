"""The judge LLM that scores answers for deepeval's LLM-as-a-judge metrics.

By default the judge goes through the app's own LLM setup (``app.agent.llm.build_chat_model``),
so it uses the same provider and keys as the agent (Portkey → Groq, or Gemini) and no extra
API key is needed. ``EVAL_JUDGE_MODEL`` picks a different model on that provider: prefer a
model other than the agent's, since a model tends to rate its own answers higher.

``provider="gemini"`` (``--judge gemini``) calls Gemini directly with the app's GOOGLE_API_KEY
instead, whatever LLM_PROVIDER is. ``requests_per_minute`` paces the calls for rate-limited
tiers (Gemini's free tier allows a handful per minute; deepeval otherwise fires them in parallel).

``--judge native`` (run_eval.py) skips this wrapper and lets deepeval use the model configured
with its own CLI (``deepeval set-openai`` / ``set-gemini`` / ..., or OPENAI_API_KEY).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Literal

from deepeval.models import DeepEvalBaseLLM
from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter
from pydantic import BaseModel

logger = logging.getLogger(__name__)

DEFAULT_GEMINI_JUDGE = "gemini-3.6-flash"


class AppJudgeLLM(DeepEvalBaseLLM):
    """deepeval model backed by a LangChain chat model built from the app's settings."""

    def __init__(
        self,
        model: str | None = None,
        provider: Literal["gemini", "portkey"] | None = None,
        requests_per_minute: float | None = None,
    ) -> None:
        from app.agent.llm import RETRYABLE_ERRORS, build_chat_model, with_retry
        from app.config import get_settings

        settings = get_settings()
        if provider is not None and provider != settings.llm_provider:
            settings = settings.model_copy(update={"llm_provider": provider})
        default_model = DEFAULT_GEMINI_JUDGE if provider == "gemini" else settings.llm_model
        self._model_name = model or os.getenv("EVAL_JUDGE_MODEL") or default_model
        self._chat: BaseChatModel = build_chat_model(settings, model=self._model_name)
        if requests_per_minute:
            self._chat.rate_limiter = InMemoryRateLimiter(
                requests_per_second=requests_per_minute / 60, check_every_n_seconds=0.5, max_bucket_size=1
            )
        self._with_retry = with_retry
        self._retryable = RETRYABLE_ERRORS
        # Groq's gpt-oss models (through Portkey) need a JSON-schema response format, as in the agent.
        self._structured_method = "json_schema" if settings.llm_provider == "portkey" else None
        super().__init__(self._model_name)

    def load_model(self) -> BaseChatModel:
        return self._chat

    def _structured(self, schema: type[BaseModel]) -> Any:
        kwargs = {"method": self._structured_method} if self._structured_method else {}
        return self._with_retry(self._chat.with_structured_output(schema, **kwargs))

    def generate(self, prompt: str, schema: type[BaseModel] | None = None) -> Any:
        if schema is not None:
            try:
                return self._structured(schema).invoke(prompt)
            except self._retryable:
                raise  # out of retries on a rate limit / outage: a text call would fail the same way
            except Exception as exc:  # schema the provider rejects: deepeval parses JSON from text
                logger.debug("Structured output failed (%s); falling back to text", exc)
        return self._with_retry(self._chat).invoke(prompt).text

    async def a_generate(self, prompt: str, schema: type[BaseModel] | None = None) -> Any:
        if schema is not None:
            try:
                return await self._structured(schema).ainvoke(prompt)
            except self._retryable:
                raise
            except Exception as exc:
                logger.debug("Structured output failed (%s); falling back to text", exc)
        return (await self._with_retry(self._chat).ainvoke(prompt)).text

    def get_model_name(self) -> str:
        return self._model_name
