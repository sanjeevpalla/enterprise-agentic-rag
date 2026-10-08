"""Split loaded documents into chunks sized for embedding and retrieval."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterable, Literal

import tiktoken
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
    TextSplitter,
)

# file_type values (set by the loaders) whose content is Markdown.
MARKDOWN_FILE_TYPES = frozenset({"pdf", "md", "markdown"})


@dataclass(frozen=True)
class ChunkingConfig:
    """Settings shared by all splitters.

    Args:
        chunk_size: Maximum chunk length, in ``length_unit``.
        chunk_overlap: Overlap between neighbouring chunks, in ``length_unit``.
        length_unit: Measure length in ``"tokens"`` (matches embedding-model
            limits) or ``"characters"``.
        encoding_name: tiktoken encoding used when ``length_unit="tokens"``.
        min_chunk_chars: Chunks shorter than this after stripping are dropped
            (stray separators, page numbers and the like).
        markdown_headers: Markdown heading levels to split on, as
            ``(marker, metadata_key)`` pairs.
    """

    chunk_size: int = 512
    chunk_overlap: int = 64
    length_unit: Literal["tokens", "characters"] = "tokens"
    encoding_name: str = "o200k_base"
    min_chunk_chars: int = 10
    markdown_headers: tuple[tuple[str, str], ...] = field(
        default=(("#", "h1"), ("##", "h2"), ("###", "h3"))
    )

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")


class BaseSplitter(ABC):
    """Abstract base class for splitters.

    Subclasses implement ``_split`` for a single document. This class handles
    dropping tiny chunks and stamping chunk metadata.
    """

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config = config or ChunkingConfig()
        self._encoding = (
            tiktoken.get_encoding(self.config.encoding_name)
            if self.config.length_unit == "tokens"
            else None
        )

    def split(self, documents: Iterable[Document]) -> list[Document]:
        chunks: list[Document] = []
        for document in documents:
            pieces = [
                piece
                for piece in self._split(document)
                if len(piece.page_content.strip()) >= self.config.min_chunk_chars
            ]
            chunks.extend(self._annotate(pieces))
        return chunks

    @abstractmethod
    def _split(self, document: Document) -> list[Document]:
        """Split one document. Returned chunks carry the parent's metadata."""

    def _length(self, text: str) -> int:
        """Length of ``text`` in the configured unit."""
        if self._encoding is not None:
            return len(self._encoding.encode(text, disallowed_special=()))
        return len(text)

    def _text_splitter(self, chunk_size: int | None = None) -> TextSplitter:
        chunk_size = chunk_size or self.config.chunk_size
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=min(self.config.chunk_overlap, chunk_size // 2),
            length_function=self._length,
        )

    @staticmethod
    def _annotate(chunks: list[Document]) -> list[Document]:
        for index, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = index
            chunk.metadata["chunk_count"] = len(chunks)
            chunk.metadata["chunk_id"] = _chunk_id(chunk)
        return chunks


class RecursiveSplitter(BaseSplitter):
    """Split on paragraphs, then lines, then words, until chunks fit."""

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        super().__init__(config)
        self._splitter = self._text_splitter()

    def _split(self, document: Document) -> list[Document]:
        return self._splitter.split_documents([document])


class MarkdownSplitter(BaseSplitter):
    """Split Markdown on headings first, then size-split each section.

    Chunks never straddle two sections. Each chunk records its heading path in
    metadata (``h1``/``h2``/``h3``) and starts with it as a breadcrumb line
    (e.g. ``Guide > Install``), so every chunk of a long section keeps its
    context when embedded. The breadcrumb counts toward ``chunk_size``.
    """

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        super().__init__(config)
        self._header_keys = [key for _, key in self.config.markdown_headers]
        self._header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=list(self.config.markdown_headers),
            strip_headers=True,
        )
        self._splitter_for_size = lru_cache(maxsize=64)(self._text_splitter)

    def _split(self, document: Document) -> list[Document]:
        chunks: list[Document] = []
        for section in self._header_splitter.split_text(document.page_content):
            metadata = {**document.metadata, **section.metadata}
            breadcrumb = " > ".join(
                metadata[key] for key in self._header_keys if metadata.get(key)
            )
            prefix = f"{breadcrumb}\n\n" if breadcrumb else ""
            # Leave room for the breadcrumb, but never squeeze the body below half the budget.
            budget = max(self.config.chunk_size - self._length(prefix), self.config.chunk_size // 2)
            splitter = self._splitter_for_size(budget)
            for text in splitter.split_text(section.page_content):
                chunks.append(Document(page_content=prefix + text, metadata=dict(metadata)))
        return chunks


class DocumentSplitter(BaseSplitter):
    """Route each document to the right splitter based on its ``file_type``.

    Markdown-producing sources (PDFs, ``.md`` files) use ``MarkdownSplitter``;
    everything else uses ``RecursiveSplitter``.
    """

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        super().__init__(config)
        self._markdown = MarkdownSplitter(self.config)
        self._recursive = RecursiveSplitter(self.config)

    def _split(self, document: Document) -> list[Document]:
        if document.metadata.get("file_type") in MARKDOWN_FILE_TYPES:
            return self._markdown._split(document)
        return self._recursive._split(document)


def _chunk_id(chunk: Document) -> str:
    """Deterministic ID, so re-ingesting a file overwrites its chunks rather than duplicating them."""
    metadata = chunk.metadata
    key = "|".join(
        str(metadata.get(part, ""))
        for part in ("source", "page", "sheet", "slide", "section_index", "chunk_index")
    )
    return hashlib.sha256(f"{key}|{chunk.page_content}".encode("utf-8")).hexdigest()[:32]
