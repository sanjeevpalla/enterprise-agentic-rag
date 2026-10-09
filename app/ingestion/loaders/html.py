"""HTML file loader built on BeautifulSoup."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup, Tag

from app.ingestion.loaders.base import BaseLoader, Document, LoaderError

HEADING_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6")
DEFAULT_REMOVE_TAGS = (
    "script", "style", "noscript", "iframe", "svg", "canvas",
    "nav", "header", "footer", "aside", "form",
)


# Heading markers for _split_sections: "\x00<level>\x00<heading text>\x00" (NUL never occurs in HTML text).
_MARK = "\x00"
_MARKER = re.compile(r"\n?\x00([1-6])\x00(.*?)\x00\n?")


class HTMLLoader(BaseLoader):
    """Load readable text and metadata from ``.html`` / ``.htm`` files.

    Args:
        encoding: Encoding used to read the file. ``None`` lets BeautifulSoup
            detect it from the raw bytes (honours ``<meta charset>``).
        parser: BeautifulSoup parser name, e.g. ``"html.parser"`` or ``"lxml"``.
        remove_tags: Tags stripped (with their contents) before text extraction.
        split_by_headings: If True (default), return one Document per heading section
            instead of one Document for the whole page. Each records its heading path as
            ``h1``/``h2``/``h3`` metadata (deeper headings: ``section_heading`` only), which
            citations show and the splitter prepends to every chunk.
        min_section_length: Sections shorter than this (in characters) are dropped
            when splitting by headings.
    """

    supported_extensions = (".html", ".htm")

    def __init__(
        self,
        encoding: str | None = None,
        parser: str = "html.parser",
        remove_tags: tuple[str, ...] = DEFAULT_REMOVE_TAGS,
        split_by_headings: bool = True,
        min_section_length: int = 20,
    ) -> None:
        self.encoding = encoding
        self.parser = parser
        self.remove_tags = remove_tags
        self.split_by_headings = split_by_headings
        self.min_section_length = min_section_length

    def load_from_string(self, html: str, source: str = "<string>") -> list[Document]:
        """Parse an in-memory HTML string (e.g. a scraped page)."""
        return self._parse(BeautifulSoup(html, self.parser), source)

    def _load(self, path: Path) -> list[Document]:
        raw = path.read_bytes()
        soup = BeautifulSoup(raw, self.parser, from_encoding=self.encoding)
        documents = self._parse(soup, str(path))
        if not documents:
            raise LoaderError(f"No extractable text found in {path}")
        return documents

    def _parse(self, soup: BeautifulSoup, source: str) -> list[Document]:
        base_metadata = self._extract_metadata(soup, source)
        self._clean(soup)
        body = soup.body or soup

        if self.split_by_headings:
            return self._split_sections(body, base_metadata)

        text = self._normalize(body.get_text(separator="\n"))
        return [Document(page_content=text, metadata=base_metadata)] if text else []

    def _extract_metadata(self, soup: BeautifulSoup, source: str) -> dict[str, Any]:
        metadata: dict[str, Any] = {"source": source, "file_type": "html"}

        if soup.title and soup.title.string:
            metadata["title"] = soup.title.string.strip()

        html_tag = soup.find("html")
        if isinstance(html_tag, Tag) and html_tag.get("lang"):
            metadata["language"] = html_tag["lang"]

        for meta in soup.find_all("meta"):
            name = (meta.get("name") or meta.get("property") or "").lower()
            content = meta.get("content")
            if not content:
                continue
            if name in ("description", "og:description"):
                metadata.setdefault("description", content.strip())
            elif name == "keywords":
                metadata["keywords"] = [k.strip() for k in content.split(",") if k.strip()]
            elif name == "author":
                metadata["author"] = content.strip()

        return metadata

    def _clean(self, soup: BeautifulSoup) -> None:
        for tag in soup.find_all(self.remove_tags):
            tag.decompose()

    def _split_sections(self, body: Tag, base_metadata: dict[str, Any]) -> list[Document]:
        """Split the page's full text at its headings (same text as the whole-page mode)."""
        # Swap each heading for a marker line, so one get_text() keeps all the page's text
        # (not only <p>/<li>/...) in order, then cut it at the markers.
        for heading in body.find_all(HEADING_TAGS):
            text = heading.get_text(separator=" ", strip=True)
            marker = f"\n{_MARK}{heading.name[1]}{_MARK}{text}{_MARK}\n" if text else "\n"
            heading.replace_with(marker)
        pieces = _MARKER.split(body.get_text(separator="\n"))

        # pieces = [text before the first heading, level, heading, text, level, heading, text, ...]
        sections: list[tuple[dict[str, str], str]] = [({}, pieces[0])]
        path_by_level: dict[int, str] = {}
        for i in range(1, len(pieces) - 1, 3):
            level, heading, text = int(pieces[i]), pieces[i + 1], pieces[i + 2]
            path_by_level = {lvl: t for lvl, t in path_by_level.items() if lvl < level}
            path_by_level[level] = heading
            headings = {f"h{lvl}": t for lvl, t in path_by_level.items() if lvl <= 3}
            sections.append(({**headings, "section_heading": heading}, text))

        documents: list[Document] = []
        for index, (headings, text) in enumerate(sections):
            content = self._normalize(text)
            if len(content) < self.min_section_length:
                continue
            metadata = {**base_metadata, **headings, "section_index": index}
            documents.append(Document(page_content=content, metadata=metadata))
        return documents

    @staticmethod
    def _normalize(text: str) -> str:
        lines = (re.sub(r"[ \t\xa0]+", " ", line).strip() for line in text.splitlines())
        text = "\n".join(lines)
        return re.sub(r"\n{3,}", "\n\n", text).strip()
