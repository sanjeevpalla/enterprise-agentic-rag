"""Agentic RAG: planner → (retriever) → responder, as a LangGraph graph."""

from typing import Any

__all__ = ["AgentResponse", "RAGAgent", "build_graph", "diagram", "route_after_planner"]


def __getattr__(name: str) -> Any:
    # Imported lazily so `python -m app.agent.graph` doesn't load graph.py twice.
    if name in __all__:
        from app.agent import graph

        return getattr(graph, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
