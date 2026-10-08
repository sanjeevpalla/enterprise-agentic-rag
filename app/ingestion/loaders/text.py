"""Plain-text file loader (txt, markdown, rst, logs)."""

from __future__ import annotations

import re
from pathlib import Path

from charset_normalizer import from_bytes

from app.ingestion.loaders.base import BaseLoader, Document, LoaderError


class TextLoader(BaseLoader):
    """Load plain-text files.

    Args:
        encoding: Encoding used to decode the file. ``None`` tries UTF-8 first
            and falls back to charset detection.
        preferred_encodings: Tie-breakers for charset detection. Short legacy
            files often fit several code pages equally well; a listed encoding
            wins any tie.
        strip_whitespace: Collapse runs of blank lines and trailing spaces.
    """

    supported_extensions = (".txt", ".md", ".markdown", ".rst", ".log")

    def __init__(
        self,
        encoding: str | None = None,
        preferred_encodings: tuple[str, ...] = ("cp1252",),
        strip_whitespace: bool = True,
    ) -> None:
        self.encoding = encoding
        self.preferred_encodings = preferred_encodings
        self.strip_whitespace = strip_whitespace

    def _load(self, path: Path) -> list[Document]:
        text, encoding = self._decode(path.read_bytes())
        if self.strip_whitespace:
            text = self._normalize(text)
        if not text:
            raise LoaderError(f"No extractable text found in {path}")

        metadata = {
            "source": str(path),
            "file_type": path.suffix.lower().lstrip("."),
            "encoding": encoding,
        }
        return [Document(page_content=text, metadata=metadata)]

    def _decode(self, raw: bytes) -> tuple[str, str]:
        if self.encoding:
            return raw.decode(self.encoding), self.encoding
        try:
            return raw.decode("utf-8-sig"), "utf-8"
        except UnicodeDecodeError:
            matches = list(from_bytes(raw))
            if not matches:
                raise LoaderError("Unable to detect file encoding")
            lowest_chaos = min(match.chaos for match in matches)
            tied = [match for match in matches if match.chaos == lowest_chaos]
            best = next(
                (match for match in tied if match.encoding in self.preferred_encodings),
                tied[0],
            )
            return str(best), best.encoding

    @staticmethod
    def _normalize(text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = "\n".join(line.rstrip() for line in text.split("\n"))
        return re.sub(r"\n{3,}", "\n\n", text).strip()
