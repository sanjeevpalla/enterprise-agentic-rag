"""Run the RAG agent on the goldens and turn each run into a deepeval test case.

The agent is used as-is (``app.agent.graph.RAGAgent``), with in-memory conversation memory so
evaluation runs don't show up in the web UI's recent chats. After each turn the graph state is
read back to get the passages the responder actually saw (after the retrieval guardrails):
they are the test case's ``retrieval_context``.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from deepeval.test_case import LLMTestCase
from langgraph.checkpoint.memory import InMemorySaver

from evaluation.dataset import Golden

logger = logging.getLogger(__name__)


@dataclass
class AgentRun:
    """One golden answered by the agent: everything the metrics need, JSON-serialisable."""

    golden: dict[str, Any]
    answer: str
    route: str
    search_query: str
    # Text of the passages given to the responder, and the file each came from.
    retrieval_context: list[str] = field(default_factory=list)
    retrieved_files: list[str] = field(default_factory=list)
    # Files of the sources cited in the answer.
    cited_files: list[str] = field(default_factory=list)
    guardrails: list[dict[str, str]] = field(default_factory=list)
    latency_s: float = 0.0
    error: str | None = None


def build_agent() -> Any:
    """The agent from the app's settings (.env), without persistent conversation memory."""
    from app.agent.graph import RAGAgent
    from app.config import get_settings

    return RAGAgent(get_settings(), checkpointer=InMemorySaver())


def run_golden(agent: Any, golden: Golden) -> AgentRun:
    thread_id = f"eval-{golden.id}-{uuid.uuid4().hex[:8]}"
    start = time.perf_counter()
    try:
        for turn in golden.history:
            agent.ask(turn, thread_id)
        response = agent.ask(golden.input, thread_id)
    except Exception as exc:  # keep going: one failing question shouldn't stop the run
        logger.exception("Agent failed on golden %s", golden.id)
        return AgentRun(asdict(golden), "", "", "", latency_s=time.perf_counter() - start, error=repr(exc))
    latency = time.perf_counter() - start

    state = agent.graph.get_state({"configurable": {"thread_id": thread_id}}).values
    documents = state.get("documents") or []
    return AgentRun(
        golden=asdict(golden),
        answer=response.answer,
        route=response.route,
        search_query=response.search_query,
        retrieval_context=[doc.page_content for doc in documents],
        retrieved_files=[Path(doc.metadata.get("source", "unknown")).name for doc in documents],
        cited_files=[s["file"] for s in response.sources],
        guardrails=response.guardrails,
        latency_s=round(latency, 2),
    )


def run_goldens(goldens: list[Golden], agent: Any | None = None) -> list[AgentRun]:
    """Answer every golden (sequentially: the agent and its local models run on CPU)."""
    owns_agent = agent is None
    agent = agent or build_agent()
    try:
        runs = []
        for i, golden in enumerate(goldens, start=1):
            logger.info("[%d/%d] %s", i, len(goldens), golden.id)
            runs.append(run_golden(agent, golden))
        return runs
    finally:
        if owns_agent:
            agent.close()


def save_runs(runs: list[AgentRun], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(r) for r in runs], indent=2, ensure_ascii=False), encoding="utf-8")


def load_runs(path: Path) -> list[AgentRun]:
    return [AgentRun(**item) for item in json.loads(path.read_text(encoding="utf-8"))]


def to_test_case(run: AgentRun) -> LLMTestCase:
    golden = run.golden
    return LLMTestCase(
        name=golden["id"],
        input=golden["input"],
        actual_output=run.answer,
        expected_output=golden.get("expected_output"),
        retrieval_context=run.retrieval_context or None,
        tags=[golden["category"]],
        completion_time=run.latency_s,
        metadata={
            "category": golden["category"],
            "history": golden.get("history") or [],
            "route": run.route,
            "expected_route": golden.get("expected_route"),
            "search_query": run.search_query,
            "expected_sources": golden.get("expected_sources") or [],
            "retrieved_files": run.retrieved_files,
            "cited_files": run.cited_files,
            "agent_error": run.error,
        },
    )
