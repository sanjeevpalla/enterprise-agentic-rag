"""Input, retrieval and output guardrails for the RAG agent, built on Guardrails AI.

    INPUT      block: prompt injection (patterns; optional DetectJailbreak model), toxic language
               redact: PII, secrets                          → before the planner/LLM sees the text
    RETRIEVAL  drop: chunks below the relevance floor, chunks containing injected instructions
               redact: PII, secrets in chunk text            → before chunks reach the LLM
    OUTPUT     block: toxic language
               redact/fix: PII, secrets, citations to sources that don't exist

Validators run locally (no Guardrails-hosted inference) and each model is loaded once and
shared between guards.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from langchain_core.documents import Document

from app.config import Settings, get_settings
from app.guardrails.telemetry import disable_guardrails_telemetry

disable_guardrails_telemetry()  # before any Guardrails object is created

from guardrails import Guard, OnFailAction  # noqa: E402
from guardrails.validator_base import Validator  # noqa: E402

from app.guardrails.validators import CitationCheck, InjectionPatterns, SecretsPresent  # noqa: E402

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GuardrailEvent:
    stage: str  # "input" | "retrieval" | "output"
    validator: str
    action: str  # "blocked" | "redacted" | "dropped" | "fixed"
    detail: str = ""

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass
class GuardResult:
    text: str  # possibly redacted/fixed
    blocked: bool = False
    error: bool = False  # a validator raised; see the "error" event
    events: list[GuardrailEvent] = field(default_factory=list)


def _ensure_nltk_data() -> None:
    """ToxicLanguage's sentence mode needs NLTK's punkt_tab tokenizer (recent NLTK versions)."""
    import nltk

    try:
        nltk.data.find("tokenizers/punkt_tab/english/")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)


class _Validators:
    """Validator instances, created once (model loading happens here)."""

    def __init__(self, settings: Settings) -> None:
        started = time.perf_counter()
        _ensure_nltk_data()
        noop, fix = OnFailAction.NOOP, OnFailAction.FIX
        self.injection = InjectionPatterns(on_fail=noop) if settings.guardrails_injection_patterns else None
        self.jailbreak = None
        if settings.guardrails_jailbreak_model:
            from guardrails_ai.detect_jailbreak import DetectJailbreak

            self.jailbreak = DetectJailbreak(
                threshold=settings.guardrails_jailbreak_threshold, use_local=True, on_fail=noop
            )
        from guardrails_ai.detect_pii import DetectPII
        from guardrails_ai.toxic_language import ToxicLanguage

        # "full": score the whole message (short user input); "sentence": per sentence (answers).
        self.toxic_full = ToxicLanguage(
            threshold=settings.guardrails_toxicity_threshold, validation_method="full", use_local=True, on_fail=noop
        )
        self.toxic_sentence = ToxicLanguage(
            threshold=settings.guardrails_toxicity_threshold, validation_method="sentence", use_local=True, on_fail=noop
        )
        self.pii = DetectPII(pii_entities=list(settings.guardrails_pii_entities), use_local=True, on_fail=fix)
        self.secrets = SecretsPresent(use_local=True, on_fail=fix)
        self.citations = CitationCheck(on_fail=fix)
        logger.info("Guardrails validators loaded in %.1fs", time.perf_counter() - started)


def _run(guard: Guard, text: str, stage: str, blocking: set[str], metadata: dict[str, Any] | None = None) -> GuardResult:
    """Validate ``text``; fixes (redactions) are applied, failures of ``blocking`` validators block.

    A validator crash never fails the request: it's logged and reported as an "error" event,
    and the caller decides (retrieval drops the chunk; input/output continue unchecked).
    """
    try:
        outcome = guard.validate(text, metadata=metadata or {})
    except Exception as exc:
        logger.exception("Guardrail validation failed at %s stage", stage)
        return GuardResult(text=text, error=True,
                           events=[GuardrailEvent(stage, "Guard", "error", f"{type(exc).__name__}: {exc}"[:200])])
    result = GuardResult(text=outcome.validated_output if isinstance(outcome.validated_output, str) else text)
    for summary in outcome.validation_summaries or []:
        if summary.validator_status != "fail":
            continue
        name = summary.validator_name
        is_blocking = name in blocking
        result.blocked |= is_blocking
        action = "blocked" if is_blocking else ("fixed" if name == "CitationCheck" else "redacted")
        detail = (summary.failure_reason or "").splitlines()[0][:200]
        result.events.append(GuardrailEvent(stage, name, action, detail))
    return result


class InputGuard:
    """Checks the user's message before the planner sees it."""

    BLOCKING = {"InjectionPatterns", "DetectJailbreak", "ToxicLanguage"}

    def __init__(self, validators: _Validators) -> None:
        chain = [v for v in (validators.injection, validators.jailbreak) if v is not None]
        self.guard = Guard().use(*chain, validators.toxic_full, validators.pii, validators.secrets)

    def check(self, text: str) -> GuardResult:
        return _run(self.guard, text, "input", self.BLOCKING)


class RetrievalGuard:
    """Filters retrieved chunks before they reach the LLM."""

    BLOCKING = {"InjectionPatterns"}  # a chunk with planted instructions is dropped, not redacted

    def __init__(self, validators: _Validators, min_rerank_score: float | None) -> None:
        chain = [validators.injection] if validators.injection is not None else []
        self.guard = Guard().use(*chain, validators.pii, validators.secrets)
        self.min_rerank_score = min_rerank_score

    def filter(self, documents: list[Document]) -> tuple[list[Document], list[GuardrailEvent]]:
        kept: list[Document] = []
        events: list[GuardrailEvent] = []
        for doc in documents:
            name = doc.metadata.get("source", "?").replace("\\", "/").rsplit("/", 1)[-1]
            score = doc.metadata.get("rerank_score")
            if self.min_rerank_score is not None and score is not None and score < self.min_rerank_score:
                events.append(GuardrailEvent("retrieval", "RelevanceFloor", "dropped",
                                             f"{name}: rerank score {score} < {self.min_rerank_score}"))
                continue
            result = _run(self.guard, doc.page_content, "retrieval", self.BLOCKING)
            events.extend(GuardrailEvent(e.stage, e.validator, "dropped" if e.action == "blocked" else e.action,
                                         f"{name}: {e.detail}") for e in result.events)
            if result.blocked or result.error:  # fail closed: an unchecked chunk isn't passed on
                continue
            kept.append(Document(page_content=result.text, metadata=doc.metadata) if result.text != doc.page_content else doc)
        return kept, events


class OutputGuard:
    """Checks the answer before it's returned to the user."""

    BLOCKING = {"ToxicLanguage"}

    def __init__(self, validators: _Validators) -> None:
        self.guard = Guard().use(validators.toxic_sentence, validators.pii, validators.secrets, validators.citations)

    def check(self, text: str, num_sources: int = 0) -> GuardResult:
        return _run(self.guard, text, "output", self.BLOCKING, metadata={"num_sources": num_sources})


class RAGGuardrails:
    """The three guards, sharing one set of loaded validators."""

    def __init__(self, settings: Settings | None = None) -> None:
        settings = settings or get_settings()
        validators = _Validators(settings)
        self.input = InputGuard(validators)
        self.retrieval = RetrievalGuard(validators, settings.retrieval_min_rerank_score)
        self.output = OutputGuard(validators)
