"""Graph state shared by the agent's nodes."""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

Route = Literal["conversational", "technical"]


class AgentState(TypedDict, total=False):
    # Full conversation; add_messages appends instead of overwriting, so the checkpointer
    # keeps history across turns of the same thread.
    messages: Annotated[list[AnyMessage], add_messages]
    # Set by the planner each turn ("blocked" when the input guard stopped the turn).
    route: Route | Literal["blocked"]
    plan_reason: str
    search_query: str
    # Set by the retriever (reset to [] by the planner on every turn).
    documents: list[Document]
    # Set by the responder: the citations used in the answer.
    sources: list[dict[str, Any]]
    # Guardrails (reset by the input guard every turn): what was blocked/redacted/dropped.
    blocked: bool
    guardrail_events: list[dict[str, str]]
