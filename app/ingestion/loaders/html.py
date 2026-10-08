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


class HTMLLoader(BaseLoader):
    """Load readable text and metadata from ``.html`` / ``.htm`` files.

    Args:
        encoding: Encoding used to read the file. ``None`` lets BeautifulSoup
            detect it from the raw bytes (honours ``<meta charset>``).
        parser: BeautifulSoup parser name, e.g. ``"html.parser"`` or ``"lxml"``.
        remove_tags: Tags stripped (with their contents) before text extraction.
        split_by_headings: If True, return one Document per heading section
            instead of one Document for the whole page.
        min_section_length: Sections shorter than this (in characters) are dropped
            when splitting by headings.
    """

    supported_extensions = (".html", ".htm")

    def __init__(
        self,
        encoding: str | None = None,
        parser: str = "html.parser",
        remove_tags: tuple[str, ...] = DEFAULT_REMOVE_TAGS,
        split_by_headings: bool = False,
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
        """Group text under the nearest preceding heading."""
        sections: list[tuple[str | None, list[str]]] = [(None, [])]

        for element in body.find_all([*HEADING_TAGS, "p", "li", "pre", "td", "blockquote"]):
            # Skip nested blocks already covered by an ancestor (e.g. <p> inside <li>).
            if element.find_parent(["p", "li", "pre", "td", "blockquote"]):
                continue
            text = element.get_text(separator=" ", strip=True)
            if not text:
                continue
            if element.name in HEADING_TAGS:
                sections.append((text, []))
            else:
                sections[-1][1].append(text)

        documents: list[Document] = []
        for index, (heading, parts) in enumerate(sections):
            content = self._normalize("\n".join(parts))
            if len(content) < self.min_section_length:
                continue
            if heading:
                content = f"{heading}\n\n{content}"
            metadata = {**base_metadata, "section_index": index, "section_heading": heading}
            documents.append(Document(page_content=content, metadata=metadata))
        return documents

    @staticmethod
    def _normalize(text: str) -> str:
        lines = (re.sub(r"[ \t\xa0]+", " ", line).strip() for line in text.splitlines())
        text = "\n".join(lines)
        return re.sub(r"\n{3,}", "\n\n", text).strip()
