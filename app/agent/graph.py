"""The agentic RAG graph and a small chat interface around it.

    START → planner ─┬─ technical ──────→ retriever → responder → END
                     └─ conversational ─────────────→ responder

With guardrails (GUARDRAILS_ENABLED, default on):

    START → input_guard ─┬─ blocked ─────────────────────────────────────────────→ END
                         └─ ok → planner ─┬─ technical → retriever → retrieval_guard ─┐
                                          └─ conversational ──────────────────────────┴→ responder → output_guard → END

The planner is TypeSafe's Jev decision model by default (PLANNER_PROVIDER=jev), or the
Gemini planner (PLANNER_PROVIDER=llm).

Run from the project root:

    python -m app.agent.graph                         # interactive chat (multi-turn)
    python -m app.agent.graph -q "How do I autoscale pods?"
    python -m app.agent.graph --diagram               # print the graph as Mermaid
"""

from __future__ import annotations

import argparse
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from typesafe_sdk import TypeSafeClient

from app.agent.jev import build_typesafe_client
from app.agent.llm import build_chat_model
from app.agent.nodes import (
    InputGuardNode,
    JevPlannerNode,
    OutputGuardNode,
    PlannerNode,
    QueryRewriter,
    ResponderNode,
    RetrievalGuardNode,
    RetrieverNode,
)
from app.agent.state import AgentState
from app.config import Settings, get_settings
from app.logging import setup_logging
from app.observability import Tracer
from app.retrieval import EnterpriseRetriever, build_retriever

logger = logging.getLogger(__name__)


def route_after_input_guard(state: AgentState) -> str:
    """Conditional edge: blocked input ends the turn (the guard already wrote the reply)."""
    return END if state.get("blocked") else "planner"


def route_after_planner(state: AgentState) -> str:
    """Conditional edge: technical queries go through retrieval, conversational ones skip it."""
    return "retriever" if state.get("route") == "technical" else "responder"


def build_graph(
    llm: BaseChatModel,
    retriever: EnterpriseRetriever,
    checkpointer: BaseCheckpointSaver | None = None,
    history_messages: int = 6,
    planner_llm: BaseChatModel | None = None,
    planner: Any | None = None,
    structured_method: str | None = None,
    guardrails: Any | None = None,
) -> CompiledStateGraph:
    """``planner`` overrides the planner node (e.g. JevPlannerNode); default is the LLM planner.
    ``guardrails`` (app.guardrails.RAGGuardrails) adds the input/retrieval/output guard nodes."""
    guards = (
        (InputGuardNode(guardrails.input), RetrievalGuardNode(guardrails.retrieval), OutputGuardNode(guardrails.output))
        if guardrails is not None
        else None
    )
    return _assemble(
        planner or PlannerNode(planner_llm or llm, history_messages, structured_method),
        RetrieverNode(retriever, QueryRewriter(llm, history_messages)),
        ResponderNode(llm, history_messages),
        checkpointer or InMemorySaver(),
        guards,
    )


def _assemble(
    planner: Any,
    retriever: Any,
    responder: Any,
    checkpointer: BaseCheckpointSaver | None,
    guards: tuple[Any, Any, Any] | None = None,
) -> CompiledStateGraph:
    """Wire the nodes into the graph (shared by build_graph and diagram)."""
    builder = StateGraph(AgentState)
    builder.add_node("planner", planner)
    builder.add_node("retriever", retriever)
    builder.add_node("responder", responder)
    planner_targets = {"retriever": "retriever", "responder": "responder"}

    if guards is None:
        builder.add_edge(START, "planner")
        builder.add_edge("retriever", "responder")
        builder.add_edge("responder", END)
    else:
        input_guard, retrieval_guard, output_guard = guards
        builder.add_node("input_guard", input_guard)
        builder.add_node("retrieval_guard", retrieval_guard)
        builder.add_node("output_guard", output_guard)
        builder.add_edge(START, "input_guard")
        builder.add_conditional_edges("input_guard", route_after_input_guard, {"planner": "planner", END: END})
        builder.add_edge("retriever", "retrieval_guard")
        builder.add_edge("retrieval_guard", "responder")
        builder.add_edge("responder", "output_guard")
        builder.add_edge("output_guard", END)

    builder.add_conditional_edges("planner", route_after_planner, planner_targets)
    # The checkpointer stores each thread's state, so follow-up questions see the history.
    return builder.compile(checkpointer=checkpointer)


def diagram(guardrails: bool = True) -> str:
    """The graph as Mermaid, without building the LLM, retriever, guardrails or Qdrant connection."""
    def placeholder(state: AgentState) -> dict:
        return {}

    guards = (placeholder, placeholder, placeholder) if guardrails else None
    return _assemble(placeholder, placeholder, placeholder, None, guards).get_graph().draw_mermaid()


@dataclass
class AgentResponse:
    answer: str
    route: str
    search_query: str = ""
    sources: list[dict[str, Any]] = field(default_factory=list)
    thread_id: str = ""
    # What the guardrails did this turn: [{"stage", "validator", "action", "detail"}, ...]
    guardrails: list[dict[str, str]] = field(default_factory=list)


class RAGAgent:
    """Builds the graph from settings and answers questions per conversation thread."""

    def __init__(
        self,
        settings: Settings | None = None,
        llm: BaseChatModel | None = None,
        planner_llm: BaseChatModel | None = None,
        retriever: EnterpriseRetriever | None = None,
        tracer: Tracer | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
        jev: TypeSafeClient | None = None,
        guardrails: Any | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.tracer = tracer or Tracer(self.settings)
        self.jev = jev or self._build_jev()
        self.retriever = retriever or build_retriever(self.settings, tracer=self.tracer)
        llm = llm or build_chat_model(self.settings)
        if planner_llm is None and self.settings.planner_model:
            planner_llm = build_chat_model(self.settings, model=self.settings.planner_model)
        planner = (
            JevPlannerNode(
                self.jev,
                self.settings.agent_history_messages,
                self.settings.planner_min_confidence,
                self.tracer,
            )
            if self.jev is not None
            else None
        )
        self.graph = build_graph(
            llm,
            self.retriever,
            checkpointer,
            self.settings.agent_history_messages,
            planner_llm=planner_llm,
            planner=planner,
            # Through Portkey, models vary by config (e.g. Groq Llama): tool calling is the
            # portable way to get the planner's structured output.
            structured_method="function_calling" if self.settings.llm_provider == "portkey" else None,
            guardrails=guardrails if guardrails is not None else self._build_guardrails(),
        )

    def _build_guardrails(self) -> Any | None:
        if not self.settings.guardrails_enabled:
            return None
        from app.guardrails import RAGGuardrails  # loads the validator models

        return RAGGuardrails(self.settings)

    def _build_jev(self) -> TypeSafeClient | None:
        if self.settings.planner_provider != "jev":
            return None
        return build_typesafe_client(self.settings)

    def ask(self, question: str, thread_id: str | None = None) -> AgentResponse:
        thread_id = thread_id or uuid.uuid4().hex
        config = {
            "configurable": {"thread_id": thread_id},
            "callbacks": self.tracer.langchain_callbacks(),
            "run_name": "rag-agent",
        }
        # One Langfuse session per conversation thread.
        with self.tracer.attributes(session_id=thread_id, tags=["agent"]):
            state = self.graph.invoke({"messages": [HumanMessage(question)]}, config=config)
        return AgentResponse(
            answer=state["messages"][-1].text,
            route=state.get("route", ""),
            search_query=state.get("search_query", ""),
            sources=state.get("sources", []),
            thread_id=thread_id,
            guardrails=state.get("guardrail_events", []),
        )

    def mermaid(self) -> str:
        return self.graph.get_graph().draw_mermaid()

    def close(self) -> None:
        self.retriever.close()
        if self.jev is not None:
            self.jev.close()
        self.tracer.flush()


def _print_response(response: AgentResponse) -> None:
    route = response.route + (f" | search: {response.search_query!r}" if response.search_query else "")
    print(f"\n[{route}]\n\n{response.answer}\n")
    if response.sources:
        print("Sources:")
        for s in response.sources:
            details = ", ".join(x for x in (s["location"], s["section"]) if x)
            print(f"  [{s['number']}] {s['file']}" + (f" ({details})" if details else ""))
        print()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with the enterprise RAG agent.")
    parser.add_argument("-q", "--question", help="Ask one question and exit")
    parser.add_argument("--diagram", action="store_true", help="Print the graph as Mermaid and exit")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.diagram:  # structure only: no models, API keys or Qdrant needed
        print(diagram())
        return 0

    settings = get_settings()
    setup_logging(settings)
    try:
        agent = RAGAgent(settings)
    except ValueError as exc:  # missing API key, collection missing/mismatched, ...
        logger.error("%s", exc)
        return 2

    try:
        if args.question:
            _print_response(agent.ask(args.question))
            return 0

        thread_id = uuid.uuid4().hex
        print("Enterprise RAG agent. Ask a question (empty line or Ctrl+C to quit).")
        while True:
            try:
                question = input("\nyou> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not question:
                break
            _print_response(agent.ask(question, thread_id))
        return 0
    finally:
        agent.close()


if __name__ == "__main__":
    raise SystemExit(main())
