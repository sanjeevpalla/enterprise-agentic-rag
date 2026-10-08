"""Input, retrieval and output guardrails (Guardrails AI)."""

from typing import Any

__all__ = [
    "GuardResult",
    "GuardrailEvent",
    "InputGuard",
    "OutputGuard",
    "RAGGuardrails",
    "RetrievalGuard",
]


def __getattr__(name: str) -> Any:
    # Imported lazily: loading Guardrails and its validator models is slow, and only
    # needed when guardrails are enabled.
    if name in __all__:
        from app.guardrails import guards

        return getattr(guards, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
