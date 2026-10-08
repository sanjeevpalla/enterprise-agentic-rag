"""Base abstractions shared by all document loaders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

# Loaders return LangChain documents so output feeds straight into LangChain
# splitters, vector stores and retrievers.
from langchain_core.documents import Document


class LoaderError(Exception):
    """Raised when a loader cannot read or parse a source."""


class BaseLoader(ABC):
    """Abstract base class for file loaders.

    Subclasses declare the extensions they support and implement ``_load``.
    Path validation and multi-file loading are handled here.
    """

    supported_extensions: tuple[str, ...] = ()

    def load(self, path: str | Path) -> list[Document]:
        """Load a single file and return its documents."""
        file_path = self._validate_path(path)
        try:
            return self._load(file_path)
        except LoaderError:
            raise
        except Exception as exc:
            raise LoaderError(f"Failed to load {file_path}: {exc}") from exc

    def load_many(self, paths: Iterable[str | Path]) -> list[Document]:
        """Load several files and return all documents in order."""
        documents: list[Document] = []
        for path in paths:
            documents.extend(self.load(path))
        return documents

    def supports(self, path: str | Path) -> bool:
        return Path(path).suffix.lower() in self.supported_extensions

    def _validate_path(self, path: str | Path) -> Path:
        file_path = Path(path)
        if not file_path.is_file():
            raise LoaderError(f"File not found: {file_path}")
        if not self.supports(file_path):
            raise LoaderError(
                f"{type(self).__name__} does not support '{file_path.suffix}' files "
                f"(supported: {', '.join(self.supported_extensions)})"
            )
        return file_path

    @abstractmethod
    def _load(self, path: Path) -> list[Document]:
        """Read ``path`` and return its documents. Path is already validated."""
