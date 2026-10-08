"""Retrieval: hybrid search in Qdrant (dense + BM25, RRF fusion) → de-duplicate → cross-encoder rerank.

Run from the project root:

    python -m app.retrieval.retriever "How do I autoscale pods?"
    python -m app.retrieval.retriever "cron schedule syntax" -k 3 --source-type true_data
    python -m app.retrieval.retriever "memory paging" --file-type pdf --no-rerank
    python -m app.retrieval.retriever "kubectl rollout undo" --mode dense   # compare with dense only

``EnterpriseRetriever`` is a LangChain ``BaseRetriever``, so it plugs straight into
chains and agents (``retriever.invoke(query)``).
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from pydantic import ConfigDict
from qdrant_client import QdrantClient, models

from app.config import Settings, get_settings
from app.ingestion.embeddings import SPARSE_VECTOR_NAME, build_embeddings, build_sparse_embeddings
from app.ingestion.ingestion import build_qdrant_client, embedding_model_id, sparse_model_id
from app.logging import setup_logging
from app.observability import Tracer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchFilters:
    """Metadata filters; each field matches any of its values, and all fields must match."""

    source_types: tuple[str, ...] = ()  # e.g. ("true_data",)
    file_types: tuple[str, ...] = ()  # e.g. ("pdf", "docx")
    sources: tuple[str, ...] = ()  # absolute file paths

    def to_qdrant(self) -> models.Filter | None:
        conditions = [
            models.FieldCondition(key=f"metadata.{key}", match=models.MatchAny(any=list(values)))
            for key, values in (("source_type", self.source_types), ("file_type", self.file_types), ("source", self.sources))
            if values
        ]
        return models.Filter(must=conditions) if conditions else None

    def describe(self) -> dict[str, list[str]]:
        return {k: list(v) for k, v in vars(self).items() if v}


class Reranker:
    """Cross-encoder reranker run locally with FastEmbed.

    A cross-encoder reads the query and each candidate together, so it judges relevance
    far better than vector similarity, but it's too slow to run over the whole collection.
    Hence: vector search for ``fetch_k`` candidates, rerank those, keep ``top_k``.
    """

    def __init__(self, model_name: str, cache_dir: Path | None = None, threads: int | None = None) -> None:
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.model_name = model_name
        logger.info("Loading reranker %s (first run downloads it)", model_name)
        self._model = TextCrossEncoder(
            model_name=model_name,
            cache_dir=str(cache_dir) if cache_dir else None,
            threads=threads,
        )

    def rerank(self, query: str, documents: list[Document]) -> list[tuple[Document, float]]:
        """Return ``documents`` paired with relevance scores, most relevant first."""
        if not documents:
            return []
        scores = self._model.rerank(query, [doc.page_content for doc in documents])
        return sorted(zip(documents, (float(s) for s in scores)), key=lambda pair: pair[1], reverse=True)


class EnterpriseRetriever(BaseRetriever):
    """Retrieves chunks for a query: Qdrant search, de-duplication, optional rerank.

    ``search_mode`` is "hybrid" (dense + BM25, fused with Reciprocal Rank Fusion) or "dense".
    Each returned Document's metadata gains ``rank``, ``search_mode``, ``search_score``
    (cosine similarity in dense mode, RRF score in hybrid mode) and, when reranking,
    ``rerank_score``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    vector_store: QdrantVectorStore
    search_mode: str = "dense"
    reranker: Reranker | None = None
    tracer: Tracer | None = None
    top_k: int = 5
    fetch_k: int = 20
    score_threshold: float | None = None
    filters: SearchFilters | None = None

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return self.search(query)

    def search(self, query: str, k: int | None = None, filters: SearchFilters | None = None) -> list[Document]:
        k = k or self.top_k
        filters = filters if filters is not None else self.filters
        tracer = self.tracer or Tracer.disabled()
        started = time.perf_counter()

        with tracer.observation(
            "retrieve",
            as_type="retriever",
            input={"query": query, "k": k, "filters": filters.describe() if filters else {}},
        ) as span:
            fetch_k = max(self.fetch_k, k) if self.reranker else k
            with tracer.observation(f"{self.search_mode}-search", input={"fetch_k": fetch_k}) as search_span:
                hits = self.vector_store.similarity_search_with_score(
                    query,
                    k=fetch_k,
                    filter=filters.to_qdrant() if filters else None,
                    # RRF scores aren't similarities, so a cosine threshold only applies to dense search.
                    score_threshold=self.score_threshold if self.search_mode == "dense" else None,
                )
                candidates = self._dedupe(hits)
                search_span.update(output={"hits": len(hits), "after_dedupe": len(candidates)})

            for doc, score in candidates:
                doc.metadata["search_mode"] = self.search_mode
                doc.metadata["search_score"] = round(score, 4)

            if self.reranker and candidates:
                with tracer.observation("rerank", input={"model": self.reranker.model_name, "candidates": len(candidates)}) as rerank_span:
                    ranked = self.reranker.rerank(query, [doc for doc, _ in candidates])
                    for doc, score in ranked:
                        doc.metadata["rerank_score"] = round(score, 4)
                    results = [doc for doc, _ in ranked[:k]]
                    rerank_span.update(output={"kept": len(results)})
            else:
                results = [doc for doc, _ in candidates[:k]]

            for rank, doc in enumerate(results, start=1):
                doc.metadata["rank"] = rank

            elapsed = round(time.perf_counter() - started, 3)
            span.update(output=[_result_summary(doc) for doc in results])
            logger.info(
                "Retrieved %d chunks in %.2fs for query %r", len(results), elapsed, query[:80],
                extra={"query": query, "results": len(results), "candidates": len(candidates),
                       "duration_seconds": elapsed, "reranked": bool(self.reranker), "search_mode": self.search_mode},
            )
        return results

    @staticmethod
    def _dedupe(hits: list[tuple[Document, float]]) -> list[tuple[Document, float]]:
        """Drop chunks whose text is identical (ignoring whitespace/case) to a higher-scored one.

        The same document often exists in several formats (e.g. .pdf, .html and .txt in
        noisy_data); without this, top results can be the same passage repeated.
        """
        seen: set[str] = set()
        unique = []
        for doc, score in hits:  # already ordered best-first
            key = hashlib.sha1(re.sub(r"\s+", " ", doc.page_content).strip().lower().encode("utf-8")).hexdigest()
            if key not in seen:
                seen.add(key)
                unique.append((doc, score))
        return unique

    def close(self) -> None:
        self.vector_store.client.close()
        if self.tracer is not None:
            self.tracer.flush()


def _result_summary(doc: Document) -> dict[str, Any]:
    meta = doc.metadata
    return {
        "rank": meta.get("rank"),
        "source": Path(meta.get("source", "")).name,
        "page": meta.get("page") or meta.get("slide") or meta.get("sheet"),
        "search_score": meta.get("search_score"),
        "rerank_score": meta.get("rerank_score"),
        "chunk_id": meta.get("chunk_id"),
    }


def _check_collection(
    client: QdrantClient, collection: str, expected_model: str, expected_sparse_model: str | None
) -> None:
    """Fail fast if the collection is missing, was embedded with a different model, or
    lacks the BM25 vectors that hybrid search needs.

    Querying with a different embedding model than the one used at ingestion returns
    plausible-looking but meaningless results, so this is an error, not a warning.
    """
    if not client.collection_exists(collection):
        raise ValueError(f"Qdrant collection {collection!r} does not exist. Run ingestion first.")
    if expected_sparse_model:
        sparse_config = client.get_collection(collection).config.params.sparse_vectors or {}
        if SPARSE_VECTOR_NAME not in sparse_config:
            raise ValueError(
                f"Collection {collection!r} has no BM25 keyword vectors, so hybrid search isn't possible. "
                "Re-ingest into a new QDRANT_COLLECTION with RETRIEVAL_MODE=hybrid, or set RETRIEVAL_MODE=dense."
            )
    points, _ = client.scroll(collection, limit=1, with_payload=True, with_vectors=False)
    if not points:
        raise ValueError(f"Qdrant collection {collection!r} is empty. Run ingestion first.")
    indexed_with = (points[0].payload or {}).get("metadata", {}).get("embedding_model")
    if indexed_with != expected_model:
        raise ValueError(
            f"Collection {collection!r} was embedded with {indexed_with or 'an unrecorded model'}, "
            f"but queries would use {expected_model}. Set EMBEDDING_PROVIDER/EMBEDDING_MODEL to match."
        )
    sparse_with = (points[0].payload or {}).get("metadata", {}).get("sparse_model")
    if expected_sparse_model and sparse_with != expected_sparse_model:
        raise ValueError(
            f"Collection {collection!r} has keyword vectors from {sparse_with or 'an unrecorded model'}, "
            f"but queries would use {expected_sparse_model}. Set SPARSE_MODEL to match."
        )


def build_retriever(
    settings: Settings | None = None,
    *,
    rerank: bool | None = None,
    mode: str | None = None,
    filters: SearchFilters | None = None,
    tracer: Tracer | None = None,
) -> EnterpriseRetriever:
    """Build a retriever from settings. Uses the same embedding model and collection as ingestion.

    ``mode`` overrides RETRIEVAL_MODE for this retriever (e.g. "dense" on a hybrid
    collection, to compare results).
    """
    settings = settings or get_settings()
    if mode is not None:
        settings = settings.model_copy(update={"retrieval_mode": mode})
    embeddings = build_embeddings(settings)
    sparse_embeddings = build_sparse_embeddings(settings)
    client = build_qdrant_client(settings)
    try:
        _check_collection(client, settings.qdrant_collection, embedding_model_id(settings), sparse_model_id(settings))
    except Exception:
        client.close()
        raise

    use_rerank = settings.rerank_enabled if rerank is None else rerank
    reranker = (
        Reranker(settings.rerank_model, cache_dir=settings.embedding_cache_dir, threads=settings.embedding_threads)
        if use_rerank
        else None
    )
    return EnterpriseRetriever(
        vector_store=QdrantVectorStore(
            client=client,
            collection_name=settings.qdrant_collection,
            embedding=embeddings,
            sparse_embedding=sparse_embeddings,
            sparse_vector_name=SPARSE_VECTOR_NAME,
            retrieval_mode=RetrievalMode.HYBRID if sparse_embeddings is not None else RetrievalMode.DENSE,
        ),
        search_mode=settings.retrieval_mode,
        reranker=reranker,
        tracer=tracer or Tracer(settings),
        top_k=settings.retrieval_top_k,
        fetch_k=settings.retrieval_fetch_k,
        score_threshold=settings.retrieval_score_threshold,
        filters=filters,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search the RAG index.")
    parser.add_argument("query", help="Question or search text")
    parser.add_argument("-k", type=int, help="Number of results (default: RETRIEVAL_TOP_K)")
    parser.add_argument("--source-type", action="append", default=[], help="Filter, e.g. true_data (repeatable)")
    parser.add_argument("--file-type", action="append", default=[], help="Filter, e.g. pdf (repeatable)")
    parser.add_argument("--no-rerank", action="store_true", help="Skip the cross-encoder rerank")
    parser.add_argument("--mode", choices=["dense", "hybrid"], help="Override RETRIEVAL_MODE")
    parser.add_argument("--full", action="store_true", help="Print full chunk text instead of a preview")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = get_settings()
    setup_logging(settings)
    filters = SearchFilters(source_types=tuple(args.source_type), file_types=tuple(args.file_type))
    try:
        retriever = build_retriever(
            settings, rerank=False if args.no_rerank else None, mode=args.mode, filters=filters
        )
    except ValueError as exc:
        logger.error("%s", exc)
        return 2
    try:
        results = retriever.search(args.query, k=args.k)
    finally:
        retriever.close()

    print(f"\n{len(results)} results for: {args.query}\n")
    for doc in results:
        meta = doc.metadata
        location = " ".join(
            f"{key} {meta[key]}" for key in ("page", "slide", "sheet") if meta.get(key) is not None
        )
        scores = f"{meta['search_mode']} {meta['search_score']:.3f}"
        if "rerank_score" in meta:
            scores += f" | rerank {meta['rerank_score']:.3f}"
        text = doc.page_content if args.full else re.sub(r"\s+", " ", doc.page_content)[:300]
        print(f"[{meta['rank']}] {Path(meta.get('source', '?')).name} {location} ({meta.get('source_type')}) — {scores}")
        print(f"    {text}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
