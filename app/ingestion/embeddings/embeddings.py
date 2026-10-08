"""Embedding models behind LangChain's ``Embeddings`` interface.

Default is a local open-source model run with FastEmbed (ONNX on CPU): no API key,
no rate limits, no per-token cost. Gemini remains available as a hosted option.
Ingestion and retrieval must use the same model, so both build it via ``build_embeddings``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_qdrant import SparseEmbeddings

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

# Many open-source retrieval models are trained with task prefixes and lose accuracy
# without them: (document_prefix, query_prefix). Models not listed use no prefix.
KNOWN_PREFIXES: dict[str, tuple[str, str]] = {
    "nomic-ai/nomic-embed-text-v1.5": ("search_document: ", "search_query: "),
    "nomic-ai/nomic-embed-text-v1.5-Q": ("search_document: ", "search_query: "),
    "nomic-ai/nomic-embed-text-v1": ("search_document: ", "search_query: "),
    "BAAI/bge-small-en-v1.5": ("", "Represent this sentence for searching relevant passages: "),
    "BAAI/bge-base-en-v1.5": ("", "Represent this sentence for searching relevant passages: "),
    "BAAI/bge-large-en-v1.5": ("", "Represent this sentence for searching relevant passages: "),
    "intfloat/multilingual-e5-large": ("passage: ", "query: "),
    "snowflake/snowflake-arctic-embed-m-long": ("", "Represent this sentence for searching relevant passages: "),
}


class FastEmbedEmbeddings(Embeddings):
    """Local open-source embedding model via FastEmbed.

    Args:
        model_name: Any model from ``fastembed.TextEmbedding.list_supported_models()``.
        document_prefix / query_prefix: Task prefixes; ``None`` uses ``KNOWN_PREFIXES``.
        batch_size: Texts per ONNX inference batch.
        cache_dir: Where the model is downloaded (once) and cached.
        threads: CPU threads for inference; ``None`` uses all cores.
    """

    def __init__(
        self,
        model_name: str,
        document_prefix: str | None = None,
        query_prefix: str | None = None,
        batch_size: int = 32,
        cache_dir: Path | None = None,
        threads: int | None = None,
    ) -> None:
        # Model files cache fine without symlinks; silence Hugging Face's Windows warning about them.
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        # Imported here so the Gemini provider doesn't need fastembed/onnxruntime loaded.
        from fastembed import TextEmbedding

        known_document, known_query = KNOWN_PREFIXES.get(model_name, ("", ""))
        self.model_name = model_name
        self.document_prefix = known_document if document_prefix is None else document_prefix
        self.query_prefix = known_query if query_prefix is None else query_prefix
        self.batch_size = batch_size
        logger.info("Loading embedding model %s (first run downloads it)", model_name)
        self._model = TextEmbedding(
            model_name=model_name,
            cache_dir=str(cache_dir) if cache_dir else None,
            threads=threads,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prefixed = [self.document_prefix + text for text in texts]
        return [vector.tolist() for vector in self._model.embed(prefixed, batch_size=self.batch_size)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.embed([self.query_prefix + text]))).tolist()


# Name of the BM25 sparse vector in Qdrant collections (langchain-qdrant's default).
SPARSE_VECTOR_NAME = "langchain-sparse"


def build_sparse_embeddings(settings: Settings | None = None) -> SparseEmbeddings | None:
    """BM25 keyword vectors for hybrid search, or None in dense mode.

    BM25 matches exact terms (error codes, flags, product names) that dense embeddings
    can miss. Qdrant applies the IDF part server-side (collection uses Modifier.IDF).
    """
    settings = settings or get_settings()
    if settings.retrieval_mode != "hybrid":
        return None
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    from langchain_qdrant import FastEmbedSparse

    return FastEmbedSparse(
        model_name=settings.sparse_model,
        cache_dir=str(settings.embedding_cache_dir),
        threads=settings.embedding_threads,
    )


def build_embeddings(settings: Settings | None = None) -> Embeddings:
    """Create the configured embedding model (``EMBEDDING_PROVIDER``)."""
    settings = settings or get_settings()

    if settings.embedding_provider == "fastembed":
        return FastEmbedEmbeddings(
            model_name=settings.embedding_model,
            document_prefix=settings.embedding_document_prefix,
            query_prefix=settings.embedding_query_prefix,
            batch_size=settings.embedding_batch_size,
            cache_dir=settings.embedding_cache_dir,
            threads=settings.embedding_threads,
        )

    if settings.embedding_provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        if settings.google_api_key is None:
            raise ValueError("GOOGLE_API_KEY (or GEMINI_API_KEY) is not set (environment or .env)")
        # No task_type: the integration uses RETRIEVAL_DOCUMENT for embed_documents
        # and RETRIEVAL_QUERY for embed_query, which is what retrieval needs.
        return GoogleGenerativeAIEmbeddings(
            model=settings.embedding_model,
            google_api_key=settings.google_api_key,
            output_dimensionality=settings.embedding_dimensions,
        )

    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {settings.embedding_provider!r}")
