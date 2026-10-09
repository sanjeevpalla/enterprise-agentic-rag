"""The agentic RAG graph and a small chat interface around it.

    START → planner ─┬─ technical ──────→ retriever → responder → END
                     └─ conversational ─────────────→ responder

With guardrails (GUARDRAILS_ENABLED, default on):

    START → input_guard ─┬─ blocked ─────────────────────────────────────────────→ END
                         └─ ok → planner ─┬─ technical → retriever → retrieval_guard ─┐
                                          └─ conversational ──────────────────────────┴→ responder → output_guard → END

With GROUNDING_CHECK_ENABLED (default on), a grounding node runs right after the responder
(responder → grounding → output_guard / END): it removes statements of technical answers that
the source passages don't support.

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
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessageChunk, HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from typesafe_sdk import TypeSafeClient

from app.agent.jev import build_typesafe_client
from app.agent.llm import build_chat_model
from app.agent.nodes import (
    GroundingCheckNode,
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
from app.memory import build_checkpointer
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
    grounding: bool = True,
) -> CompiledStateGraph:
    """``planner`` overrides the planner node (e.g. JevPlannerNode); default is the LLM planner.
    ``guardrails`` (app.guardrails.RAGGuardrails) adds the input/retrieval/output guard nodes.
    ``grounding`` adds the grounding check after the responder."""
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
        GroundingCheckNode(llm, structured_method) if grounding else None,
    )


def _assemble(
    planner: Any,
    retriever: Any,
    responder: Any,
    checkpointer: BaseCheckpointSaver | None,
    guards: tuple[Any, Any, Any] | None = None,
    grounding: Any | None = None,
) -> CompiledStateGraph:
    """Wire the nodes into the graph (shared by build_graph and diagram)."""
    builder = StateGraph(AgentState)
    builder.add_node("planner", planner)
    builder.add_node("retriever", retriever)
    builder.add_node("responder", responder)
    planner_targets = {"retriever": "retriever", "responder": "responder"}
    # The answer leaves the responder through the grounding check, when there is one.
    answered = "responder"
    if grounding is not None:
        builder.add_node("grounding", grounding)
        builder.add_edge("responder", "grounding")
        answered = "grounding"

    if guards is None:
        builder.add_edge(START, "planner")
        builder.add_edge("retriever", "responder")
        builder.add_edge(answered, END)
    else:
        input_guard, retrieval_guard, output_guard = guards
        builder.add_node("input_guard", input_guard)
        builder.add_node("retrieval_guard", retrieval_guard)
        builder.add_node("output_guard", output_guard)
        builder.add_edge(START, "input_guard")
        builder.add_conditional_edges("input_guard", route_after_input_guard, {"planner": "planner", END: END})
        builder.add_edge("retriever", "retrieval_guard")
        builder.add_edge("retrieval_guard", "responder")
        builder.add_edge(answered, "output_guard")
        builder.add_edge("output_guard", END)

    builder.add_conditional_edges("planner", route_after_planner, planner_targets)
    # The checkpointer stores each thread's state, so follow-up questions see the history.
    return builder.compile(checkpointer=checkpointer)


def diagram(guardrails: bool = True, grounding: bool = True) -> str:
    """The graph as Mermaid, without building the LLM, retriever, guardrails or Qdrant connection."""
    def placeholder(state: AgentState) -> dict:
        return {}

    guards = (placeholder, placeholder, placeholder) if guardrails else None
    return _assemble(
        placeholder, placeholder, placeholder, None, guards, placeholder if grounding else None
    ).get_graph().draw_mermaid()


def _progress_status(node: str, update: dict[str, Any]) -> str | None:
    """Progress message after a graph node finishes (streamed to the user), or None."""
    if node == "planner":
        return "Searching the knowledge base…" if update.get("route") == "technical" else "Writing a reply…"
    if node == "retriever":
        count = len(update.get("documents") or [])
        return f"Found {count} relevant passage{'s' if count != 1 else ''}, writing the answer…" if count else None
    if node == "responder":
        return "Checking the answer against its sources…" if update.get("sources") else "Checking the answer…"
    return None


@dataclass
class AgentResponse:
    answer: str
    route: str
    search_query: str = ""
    sources: list[dict[str, Any]] = field(default_factory=list)
    thread_id: str = ""
    # What the guardrails did this turn: [{"stage", "validator", "action", "detail"}, ...]
    guardrails: list[dict[str, str]] = field(default_factory=list)
    # The question as stored after the input guardrails (redacted / blocked placeholder).
    question: str = ""


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
        # Conversation memory: persisted to MEMORY_DB_PATH unless a checkpointer is injected.
        self._owns_checkpointer = checkpointer is None
        self.checkpointer = checkpointer or build_checkpointer(self.settings.memory_db_path)
        self.graph = build_graph(
            llm,
            self.retriever,
            self.checkpointer,
            self.settings.agent_history_messages,
            planner_llm=planner_llm,
            planner=planner,
            # Through Portkey, use a JSON-schema response format: Groq's gpt-oss models don't
            # support forced tool calls, which LangChain's default method relies on.
            structured_method="json_schema" if self.settings.llm_provider == "portkey" else None,
            guardrails=guardrails if guardrails is not None else self._build_guardrails(),
            grounding=self.settings.grounding_check_enabled,
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

    def _config(self, thread_id: str) -> dict[str, Any]:
        return {
            "configurable": {"thread_id": thread_id},
            "callbacks": self.tracer.langchain_callbacks(),
            "run_name": "rag-agent",
        }

    def ask(self, question: str, thread_id: str | None = None) -> AgentResponse:
        thread_id = thread_id or uuid.uuid4().hex
        # One Langfuse session per conversation thread.
        with self.tracer.attributes(session_id=thread_id, tags=["agent"]):
            state = self.graph.invoke({"messages": [HumanMessage(question)]}, config=self._config(thread_id))
        return self._response(state, thread_id, question)

    def stream(self, question: str, thread_id: str | None = None) -> Iterator[dict[str, Any]]:
        """Run one turn, yielding events as they happen:

        - ``{"type": "status", "text": ...}``: progress ("Searching the knowledge base…");
        - ``{"type": "token", "text": ...}``: the next piece of the answer as the responder writes it;
        - ``{"type": "done", "response": AgentResponse}``: last event, the final turn result.

        The output guardrails check the complete answer, so the final ``response.answer`` can
        differ from the streamed tokens (redacted, citation fixed, or blocked): show it instead.
        Consume the whole generator in one thread (tracing context is thread-bound).
        """
        thread_id = thread_id or uuid.uuid4().hex
        config = self._config(thread_id)
        with self.tracer.attributes(session_id=thread_id, tags=["agent", "stream"]):
            for mode, data in self.graph.stream(
                {"messages": [HumanMessage(question)]}, config=config, stream_mode=["messages", "updates"]
            ):
                if mode == "messages":
                    chunk, metadata = data
                    # Only the responder's tokens: the planner/rewriter also call the LLM.
                    if metadata.get("langgraph_node") == "responder" and isinstance(chunk, AIMessageChunk):
                        if chunk.text:
                            yield {"type": "token", "text": chunk.text}
                else:
                    for node, update in data.items():
                        status = _progress_status(node, update or {})
                        if status:
                            yield {"type": "status", "text": status}
            state = self.graph.get_state(config).values
        yield {"type": "done", "response": self._response(state, thread_id, question)}

    @staticmethod
    def _response(state: dict[str, Any], thread_id: str, question: str) -> AgentResponse:
        return AgentResponse(
            answer=state["messages"][-1].text,
            route=state.get("route", ""),
            search_query=state.get("search_query", ""),
            sources=state.get("sources", []),
            thread_id=thread_id,
            guardrails=state.get("guardrail_events", []),
            question=next((m.text for m in reversed(state["messages"]) if m.type == "human"), question),
        )

    def forget(self, thread_id: str) -> None:
        """Delete a conversation's memory (its checkpoints)."""
        self.checkpointer.delete_thread(thread_id)

    def mermaid(self) -> str:
        return self.graph.get_graph().draw_mermaid()

    def close(self) -> None:
        self.retriever.close()
        if self._owns_checkpointer and hasattr(self.checkpointer, "conn"):
            self.checkpointer.conn.close()
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
