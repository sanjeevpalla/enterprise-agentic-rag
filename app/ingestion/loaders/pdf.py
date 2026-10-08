"""PDF file loader backed by LangChain's PyMuPDF4LLM integration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_pymupdf4llm import PyMuPDF4LLMLoader

from app.ingestion.loaders.base import BaseLoader, Document, LoaderError

# PyMuPDF metadata keys kept (when non-empty), mapped to our metadata names.
_METADATA_KEYS = {
    "title": "title",
    "author": "author",
    "subject": "subject",
    "keywords": "keywords",
    "creationdate": "created",
}


class PDFLoader(BaseLoader):
    """Load ``.pdf`` files as Markdown, so headings and tables keep their structure.

    Args:
        split_pages: If True, return one Document per page; otherwise one
            Document for the whole file.
        password: Password for encrypted PDFs.
        **pymupdf4llm_kwargs: Extra options forwarded to ``pymupdf4llm.to_markdown``
            (e.g. ``table_strategy``, ``ignore_images``).
    """

    supported_extensions = (".pdf",)

    def __init__(
        self,
        split_pages: bool = True,
        password: str | None = None,
        **pymupdf4llm_kwargs: Any,
    ) -> None:
        self.split_pages = split_pages
        self.password = password
        self.pymupdf4llm_kwargs = pymupdf4llm_kwargs

    def _load(self, path: Path) -> list[Document]:
        loader = PyMuPDF4LLMLoader(
            path,
            password=self.password,
            mode="page" if self.split_pages else "single",
            **self.pymupdf4llm_kwargs,
        )
        documents = [
            Document(
                page_content=doc.page_content.strip(),
                metadata=self._normalize_metadata(doc.metadata, path),
            )
            for doc in loader.load()
            if doc.page_content.strip()
        ]
        if not documents:
            # Usually a scanned PDF: the pages are images and need OCR.
            raise LoaderError(f"No extractable text found in {path} (scanned PDF?)")
        return documents

    def _normalize_metadata(self, raw: dict[str, Any], path: Path) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "source": str(path),
            "file_type": "pdf",
            "total_pages": raw.get("total_pages"),
        }
        for raw_key, key in _METADATA_KEYS.items():
            value = raw.get(raw_key)
            if value:
                metadata[key] = str(value).strip()
        if self.split_pages and "page" in raw:
            metadata["page"] = raw["page"] + 1  # PyMuPDF pages are 0-based
        return metadata
