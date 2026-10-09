"""Split loaded documents into chunks sized for embedding and retrieval."""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterable, Literal

import tiktoken
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter

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

    Every chunk records where it came from, for citations:

    - ``h1``/``h2``/``h3`` (and ``section_heading``): its heading path, when the loader or the
      Markdown splitter found headings. The path is also prepended to the chunk text as a
      breadcrumb line (e.g. ``Guide > Install``), so each chunk keeps its context when embedded.
    - ``start_index``/``end_index``: the chunk's character span in its loaded document's
      text (one page/slide/sheet/section), and ``body_offset``: where that span starts in the
      chunk text (after the breadcrumb). The source viewer uses them to join neighbouring
      chunks exactly.
    """

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config = config or ChunkingConfig()
        self._encoding = (
            tiktoken.get_encoding(self.config.encoding_name)
            if self.config.length_unit == "tokens"
            else None
        )
        self._header_keys = [key for _, key in self.config.markdown_headers]
        self._splitter_for_size = lru_cache(maxsize=64)(self._text_splitter)

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

    def _breadcrumb(self, metadata: dict) -> str:
        """``h1 > h2 > h3`` (plus a deeper ``section_heading``), or "" without headings."""
        path = [metadata[key] for key in self._header_keys if metadata.get(key)]
        heading = metadata.get("section_heading")
        if heading and (not path or path[-1] != heading):
            path.append(heading)
        return " > ".join(path)

    def _split_section(self, body: str, base_offset: int, metadata: dict) -> list[Document]:
        """Size-split ``body`` (found at ``base_offset`` in the loaded document's text) into
        chunks that start with the section's breadcrumb and record their span."""
        breadcrumb = self._breadcrumb(metadata)
        prefix = f"{breadcrumb}\n\n" if breadcrumb else ""
        # Leave room for the breadcrumb, but never squeeze the body below half the budget.
        budget = max(self.config.chunk_size - self._length(prefix), self.config.chunk_size // 2)
        chunks: list[Document] = []
        cursor = 0
        for text in self._splitter_for_size(budget).split_text(body):
            # Pieces are verbatim, in order, possibly overlapping: search from the last start.
            found = body.find(text, cursor)
            spans: dict = {}
            if found >= 0:
                cursor = found + 1
                start = base_offset + found
                spans = {"start_index": start, "end_index": start + len(text), "body_offset": len(prefix)}
            chunks.append(Document(page_content=prefix + text, metadata={**metadata, **spans}))
        return chunks

    @staticmethod
    def _annotate(chunks: list[Document]) -> list[Document]:
        for index, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = index
            chunk.metadata["chunk_count"] = len(chunks)
            chunk.metadata["chunk_id"] = _chunk_id(chunk)
        return chunks


class RecursiveSplitter(BaseSplitter):
    """Split on paragraphs, then lines, then words, until chunks fit.

    Heading metadata set by the loader (Word/HTML sections) becomes each chunk's breadcrumb.
    """

    def _split(self, document: Document) -> list[Document]:
        return self._split_section(document.page_content, 0, dict(document.metadata))


class MarkdownSplitter(BaseSplitter):
    """Split Markdown on headings first, then size-split each section.

    Chunks never straddle two sections. Each chunk records its heading path in
    metadata (``h1``/``h2``/``h3``) and starts with it as a breadcrumb line
    (e.g. ``Guide > Install``). The breadcrumb counts toward ``chunk_size``.

    Section text is kept verbatim (indentation, blank lines), and ``#`` lines inside
    fenced code blocks (shell/YAML comments) are not headings.
    """

    def _split(self, document: Document) -> list[Document]:
        levels = {marker: key for marker, key in self.config.markdown_headers}
        chunks: list[Document] = []
        for start, end, headings in _markdown_sections(document.page_content, levels):
            metadata = {**document.metadata, **headings}
            chunks.extend(self._split_section(document.page_content[start:end], start, metadata))
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


_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t#]*$")
_FENCE = re.compile(r"^[ \t]*(`{3,}|~{3,})")


def _markdown_sections(text: str, levels: dict[str, str]) -> list[tuple[int, int, dict[str, str]]]:
    """``(start, end, headings)`` of each section's body in ``text``: the text between one
    heading line (of a level in ``levels``, e.g. {"#": "h1"}) and the next. Heading lines
    themselves are excluded; ``#`` lines inside fenced code blocks are ignored."""
    sections: list[tuple[int, int, dict[str, str]]] = []
    current: dict[str, str] = {}
    start = 0
    fence: str | None = None
    offset = 0
    for line in text.splitlines(keepends=True):
        line_start, offset = offset, offset + len(line)
        stripped = line.rstrip("\r\n")
        fence_match = _FENCE.match(stripped)
        if fence_match:
            marker = fence_match.group(1)
            if fence is None:
                fence = marker[0] * 3
            elif marker.startswith(fence):
                fence = None
            continue
        heading = None if fence else _HEADING.match(stripped)
        if not heading or heading.group(1) not in levels:
            continue
        sections.append((start, line_start, dict(current)))
        key = levels[heading.group(1)]
        depth = list(levels.values()).index(key)
        # A heading resets its own and deeper levels; "**Bold**" PDF headings lose the markup.
        current = {k: v for k, v in current.items() if list(levels.values()).index(k) < depth}
        current[key] = heading.group(2).strip().strip("*_").strip()
        start = offset
    sections.append((start, len(text), dict(current)))
    return [(s, e, h) for s, e, h in sections if text[s:e].strip()]


def _chunk_id(chunk: Document) -> str:
    """Deterministic ID, so re-ingesting a file overwrites its chunks rather than duplicating them."""
    metadata = chunk.metadata
    key = "|".join(
        str(metadata.get(part, ""))
        for part in ("source", "page", "sheet", "slide", "section_index", "chunk_index")
    )
    return hashlib.sha256(f"{key}|{chunk.page_content}".encode("utf-8")).hexdigest()[:32]
