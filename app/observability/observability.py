"""Langfuse tracing behind a small wrapper that becomes a no-op when Langfuse isn't configured."""

from __future__ import annotations

import logging
from contextlib import contextmanager, nullcontext
from typing import Any, ContextManager, Iterator

from langfuse import Langfuse, propagate_attributes

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


class _NoopObservation:
    """Stands in for a Langfuse observation when tracing is disabled."""

    def update(self, **_: Any) -> _NoopObservation:
        return self


class Tracer:
    """Creates Langfuse observations, or does nothing if no Langfuse keys are configured.

    Call sites use the same code either way:

        with tracer.observation("parse", input={...}) as span:
            ...
            span.update(output={...})
    """

    def __init__(self, settings: Settings | None = None) -> None:
        settings = settings or get_settings()
        self._client: Langfuse | None = None
        self._public_key = settings.langfuse_public_key
        if settings.langfuse_enabled:
            self._client = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key.get_secret_value(),
                base_url=settings.langfuse_base_url,
                environment=settings.langfuse_environment,
                sample_rate=settings.langfuse_sample_rate,
            )
            logger.info("Langfuse tracing enabled (%s)", settings.langfuse_base_url)

    @classmethod
    def disabled(cls) -> Tracer:
        """A tracer that records nothing, regardless of settings."""
        tracer = cls.__new__(cls)
        tracer._client = None
        tracer._public_key = None
        return tracer

    @property
    def enabled(self) -> bool:
        return self._client is not None

    @contextmanager
    def observation(self, name: str, as_type: str = "span", **kwargs: Any) -> Iterator[Any]:
        """Start an observation nested under the current one (or a new trace at top level).

        ``kwargs`` are passed to Langfuse: ``input``, ``output``, ``metadata``,
        ``model``, ``usage_details``, ``level``, ``status_message``, ...
        """
        if self._client is None:
            yield _NoopObservation()
            return
        with self._client.start_as_current_observation(name=name, as_type=as_type, **kwargs) as span:
            yield span

    def attributes(
        self,
        session_id: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, str] | None = None,
    ) -> ContextManager[Any]:
        """Apply session/tags/metadata to every trace started inside this block."""
        if self._client is None:
            return nullcontext()
        return propagate_attributes(session_id=session_id, tags=tags, metadata=metadata)

    def langchain_callbacks(self) -> list[Any]:
        """Callbacks that trace LangChain/LangGraph runs (LLM calls with token usage) to Langfuse."""
        if self._client is None:
            return []
        from langfuse.langchain import CallbackHandler

        return [CallbackHandler(public_key=self._public_key)]

    def flush(self) -> None:
        """Send buffered events. Short-lived processes must call this before exiting."""
        if self._client is not None:
            self._client.flush()
