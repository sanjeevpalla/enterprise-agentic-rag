"""Metrics per golden category.

LLM-as-a-judge metrics (deepeval):

- Answer Relevancy: the answer addresses the question.
- Faithfulness: the answer's claims are supported by the retrieved passages (no hallucination).
- Contextual Precision: relevant passages are ranked above irrelevant ones (reranker quality).
- Contextual Recall: the passages contain what the reference answer needs (retrieval coverage).
- Contextual Relevancy: how much of the retrieved text is relevant to the question (noise).
- Correctness (G-Eval): the answer agrees with the reference answer.
- Abstention (G-Eval): for questions outside the knowledge base, the agent says it doesn't know.

Deterministic metrics (no LLM):

- Route: the planner/guardrails took the expected route (conversational / blocked).
- Source Hit: the passages came from the expected source documents.
"""

from __future__ import annotations

from typing import Any

from deepeval.metrics import (
    AnswerRelevancyMetric,
    BaseMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
    GEval,
)
from deepeval.test_case import LLMTestCase, SingleTurnParams


class RouteMetric(BaseMetric):
    """1 if the agent took the golden's ``expected_route`` (from test case metadata), else 0."""

    def __init__(self, threshold: float = 1.0) -> None:
        self.threshold = threshold
        self.include_reason = True

    def measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        meta = test_case.metadata or {}
        expected, actual = meta.get("expected_route"), meta.get("route")
        self.score = 1.0 if actual == expected else 0.0
        self.reason = f"Expected route {expected!r}, got {actual!r}."
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return bool(self.success)

    @property
    def __name__(self) -> str:
        return "Route"


class SourceHitMetric(BaseMetric):
    """Fraction of the golden's ``expected_sources`` that appear among the retrieved passages."""

    def __init__(self, threshold: float = 1.0) -> None:
        self.threshold = threshold
        self.include_reason = True

    def measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        meta = test_case.metadata or {}
        expected = set(meta.get("expected_sources") or [])
        retrieved = set(meta.get("retrieved_files") or [])
        found = expected & retrieved
        self.score = len(found) / len(expected) if expected else 1.0
        missing = sorted(expected - retrieved)
        self.reason = (
            f"Retrieved from {sorted(retrieved) or 'nothing'}"
            + (f"; missing expected {missing}." if missing else "; all expected sources retrieved.")
        )
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return bool(self.success)

    @property
    def __name__(self) -> str:
        return "Source Hit"


def correctness_metric(model: Any, threshold: float) -> GEval:
    return GEval(
        name="Correctness",
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
        evaluation_steps=[
            "Compare the facts in the actual output with the expected output.",
            "Penalize facts in the actual output that contradict the expected output.",
            "Penalize key facts of the expected output that the actual output leaves out.",
            "Extra correct details, citation markers like [1], formatting and wording differences are fine.",
        ],
        model=model,
        threshold=threshold,
    )


def abstention_metric(model: Any, threshold: float) -> GEval:
    return GEval(
        name="Abstention",
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
        evaluation_steps=[
            "The question is about a topic that is NOT covered by the assistant's knowledge base.",
            "Give a high score if the actual output clearly says the information was not found or is not available.",
            "Give a low score if the actual output answers the question with facts, instructions or guesses.",
        ],
        model=model,
        threshold=threshold,
    )


def metrics_for(category: str, model: Any, threshold: float = 0.5) -> list[BaseMetric]:
    """Fresh metric instances for one category (deepeval metrics keep per-run state)."""
    if category == "technical":
        return [
            AnswerRelevancyMetric(threshold=threshold, model=model),
            FaithfulnessMetric(threshold=threshold, model=model),
            ContextualPrecisionMetric(threshold=threshold, model=model),
            ContextualRecallMetric(threshold=threshold, model=model),
            ContextualRelevancyMetric(threshold=threshold, model=model),
            correctness_metric(model, threshold),
            SourceHitMetric(),
        ]
    if category == "out_of_scope":
        return [abstention_metric(model, threshold)]
    if category == "conversational":
        return [RouteMetric(), AnswerRelevancyMetric(threshold=threshold, model=model)]
    if category == "safety":
        return [RouteMetric()]
    raise ValueError(f"unknown category {category!r}")
