"""Retrieval over the Qdrant index built by ingestion."""

from typing import Any

__all__ = ["EnterpriseRetriever", "Reranker", "SearchFilters", "build_retriever"]


def __getattr__(name: str) -> Any:
    # Imported lazily so `python -m app.retrieval.retriever` doesn't load retriever.py twice.
    if name in __all__:
        from app.retrieval import retriever

        return getattr(retriever, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
