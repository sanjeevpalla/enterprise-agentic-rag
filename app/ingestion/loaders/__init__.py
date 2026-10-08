"""Document loaders and a factory that picks one by file extension."""

from __future__ import annotations

from pathlib import Path

from app.ingestion.loaders.base import BaseLoader, Document, LoaderError
from app.ingestion.loaders.html import HTMLLoader
from app.ingestion.loaders.office import DocxLoader, OfficeLoader, PptxLoader, XlsxLoader
from app.ingestion.loaders.pdf import PDFLoader
from app.ingestion.loaders.text import TextLoader

_LOADER_CLASSES: tuple[type[BaseLoader], ...] = (
    TextLoader,
    PDFLoader,
    HTMLLoader,
    DocxLoader,
    XlsxLoader,
    PptxLoader,
)


def get_loader(path: str | Path) -> BaseLoader:
    """Return a default-configured loader for ``path`` based on its extension."""
    suffix = Path(path).suffix.lower()
    for loader_cls in _LOADER_CLASSES:
        if suffix in loader_cls.supported_extensions:
            return loader_cls()
    raise LoaderError(f"No loader registered for '{suffix}' files")


def load_document(path: str | Path) -> list[Document]:
    """Load any supported file with its default loader."""
    return get_loader(path).load(path)


__all__ = [
    "BaseLoader",
    "Document",
    "LoaderError",
    "TextLoader",
    "PDFLoader",
    "HTMLLoader",
    "DocxLoader",
    "XlsxLoader",
    "PptxLoader",
    "OfficeLoader",
    "get_loader",
    "load_document",
]
