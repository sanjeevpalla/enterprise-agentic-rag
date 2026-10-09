"""Source viewer support: a cited chunk in its surrounding text, and what the answer drew from it.

- ``source_passage``: the chunk plus its neighbours from the same file (and page/slide/sheet),
  in reading order, with the overlap between consecutive chunks trimmed, so it reads as one text.
- ``highlight_ranges``: the sentences of the chunk the answer's citing sentences draw on, by
  word overlap (answers paraphrase their sources closely, and quote commands/config verbatim).
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Any

from langchain_qdrant import QdrantVectorStore
from qdrant_client import models

# Location keys: neighbours must share these with the cited chunk (same page, slide or sheet).
_LOCATION_KEYS = ("page", "slide", "sheet")
_MAX_NEIGHBOURS_SCANNED = 2000
_MAX_OVERLAP_CHARS = 2000


def _location(meta: dict[str, Any]) -> str | None:
    return next((f"{key} {meta[key]}" for key in _LOCATION_KEYS if meta.get(key) is not None), None)


def _section(meta: dict[str, Any]) -> str | None:
    return " > ".join(meta[h] for h in ("h1", "h2", "h3") if meta.get(h)) or meta.get("section_heading")


def _order(meta: dict[str, Any]) -> tuple[int, int]:
    return int(meta.get("section_index") or 0), int(meta.get("chunk_index") or 0)


def _overlap(left: str, right: str) -> int:
    """Length of the longest suffix of ``left`` that is a prefix of ``right`` (chunk overlap)."""
    for size in range(min(len(left), len(right), _MAX_OVERLAP_CHARS), 15, -1):
        if left.endswith(right[:size]):
            return size
    return 0


def source_passage(vector_store: QdrantVectorStore, chunk_id: str, context: int = 2) -> dict[str, Any] | None:
    """The chunk ``chunk_id`` with up to ``context`` neighbours on each side, or None if unknown.

    Returns ``{"file", "source", "location", "section", "source_type", "chunk_id",
    "passages": [{"chunk_id", "heading", "content", "cited"}]}``; exactly one passage is
    ``cited``. ``heading`` is the section path where a new section starts (None otherwise).
    """
    client, collection = vector_store.client, vector_store.collection_name
    content_key, meta_key = vector_store.content_payload_key, vector_store.metadata_payload_key
    points = client.retrieve(collection, ids=[str(uuid.UUID(hex=chunk_id))], with_payload=True, with_vectors=False)
    if not points:
        return None
    payload = points[0].payload or {}
    meta = payload.get(meta_key) or {}
    target = {"chunk_id": chunk_id, "content": payload.get(content_key, ""), "cited": True, "meta": meta}

    chunks = [target]
    source = meta.get("source")
    if context > 0 and source:
        # Filter in Qdrant on the source only: it's the indexed field (a Qdrant server rejects
        # filters on unindexed fields such as page/slide/sheet). The location is matched here.
        neighbours, _ = client.scroll(
            collection,
            scroll_filter=models.Filter(must=[
                models.FieldCondition(key=f"{meta_key}.source", match=models.MatchValue(value=source)),
            ]),
            limit=_MAX_NEIGHBOURS_SCANNED, with_payload=True, with_vectors=False,
        )
        same_place = [
            p.payload or {} for p in neighbours
            if all((((p.payload or {}).get(meta_key) or {}).get(key)) == meta.get(key) for key in _LOCATION_KEYS)
        ]
        ordered = sorted(same_place, key=lambda p: _order(p.get(meta_key) or {}))
        ids = [(p.get(meta_key) or {}).get("chunk_id") for p in ordered]
        if chunk_id in ids:
            at = ids.index(chunk_id)
            window = ordered[max(0, at - context): at + context + 1]
            chunks = [
                target if (p.get(meta_key) or {}).get("chunk_id") == chunk_id
                else {"chunk_id": (p.get(meta_key) or {}).get("chunk_id"), "content": p.get(content_key, ""),
                      "cited": False, "meta": p.get(meta_key) or {}}
                for p in window
            ]

    _join(chunks)

    return {
        "file": Path(source or "unknown").name,
        "source": source,
        "location": _location(meta),
        "section": _section(meta),
        "source_type": meta.get("source_type"),
        "chunk_id": chunk_id,
        "passages": [{k: c.get(k) for k in ("chunk_id", "heading", "content", "cited")}
                     for c in chunks if c["content"].strip() or c["cited"]],
    }


def _body_offset(chunk: dict[str, Any]) -> int | None:
    """Where the chunk's text span starts in its content (after the breadcrumb), if recorded."""
    meta = chunk["meta"]
    if meta.get("start_index") is None or meta.get("end_index") is None:
        return None
    return int(meta.get("body_offset") or 0)


def _join(chunks: list[dict[str, Any]]) -> None:
    """Make consecutive chunks read as one text, in place, keeping the cited chunk intact:
    drop the overlap the splitter repeats between neighbours, and a neighbour's breadcrumb
    when it repeats the previous chunk's (same section)."""
    breadcrumbs = [c["content"][: _body_offset(c) or 0] for c in chunks]
    # Overlap of each chunk with the previous one: exact from the recorded spans when both
    # are in the same section (ingested with spans), else by matching text (older data).
    overlaps = [0]
    for prev, cur in zip(chunks, chunks[1:]):
        same_section = prev["meta"].get("section_index") == cur["meta"].get("section_index")
        if same_section and _body_offset(prev) is not None and _body_offset(cur) is not None:
            overlaps.append(max(0, int(prev["meta"]["end_index"]) - int(cur["meta"]["start_index"])))
        else:
            overlaps.append(_overlap(prev["content"], cur["content"]))

    at = next(i for i, c in enumerate(chunks) if c["cited"])
    for i in range(1, len(chunks)):
        cut = overlaps[i]
        if not cut:
            continue
        if i > at:  # after the cited chunk: drop the repeated start of this chunk's text
            body = _body_offset(chunks[i]) or 0
            chunks[i]["content"] = chunks[i]["content"][:body] + chunks[i]["content"][body + cut:]
        else:       # up to the cited chunk: drop the repeated end of the previous chunk
            chunks[i - 1]["content"] = chunks[i - 1]["content"][: len(chunks[i - 1]["content"]) - cut]

    # Move each breadcrumb out of the text into "heading"; a neighbour continuing the previous
    # chunk's section gets none (the cited chunk always shows its section).
    for i, chunk in enumerate(chunks):
        crumb = breadcrumbs[i]
        if crumb and chunk["content"].startswith(crumb):
            chunk["content"] = chunk["content"][len(crumb):]
        repeated = i > 0 and crumb == breadcrumbs[i - 1]
        chunk["heading"] = (crumb.strip() or None) if chunk["cited"] or not repeated else None


# ---------------------------------------------------------------------------- highlighting

_STOPWORDS = frozenset("""
a about above after again against all also am an and any are as at be because been before being below
between both but by can could did do does doing down during each few for from further had has have having
he her here hers him his how i if in into is it its itself just let may me might more most must my no nor
not now of off on once only or other our ours out over own same set she should so some such than that the
their theirs them then there these they this those through to too under until up use used uses using very
via was we were what when where which while who whom why will with would you your yours yes value values
""".split())

_WORD = re.compile(r"[a-z0-9_][a-z0-9_\-./]*[a-z0-9_]|[a-z0-9]")
_CITATION = re.compile(r"\[\d{1,2}\]")
# Sentences and lines of the passage, with their offsets.
_SEGMENT = re.compile(r"[^\n.!?]+(?:[.!?]+(?=\s|$)|$)|[^\n]+", re.M)


def _words(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if (len(w) > 2 or any(c.isdigit() for c in w)) and w not in _STOPWORDS]


def _citing_text(answer: str, citation: int | None) -> str:
    """The answer sentences/lines that cite ``[citation]`` (the whole answer if none do)."""
    if citation is None:
        return answer
    units = re.split(r"(?<=[.!?])\s+|\n+", answer)
    marker = f"[{citation}]"
    citing = [u for u in units if marker in u]
    # Code blocks and list items right before a citation often carry the cited details too.
    return "\n".join(citing) if citing else answer


def highlight_ranges(text: str, answer: str, citation: int | None = None) -> list[tuple[int, int]]:
    """Character ranges of ``text`` (a source passage) that the answer draws on."""
    answer_words = set(_words(_CITATION.sub(" ", _citing_text(answer, citation))))
    if not answer_words:
        return []
    segments = []
    for match in _SEGMENT.finditer(text):
        words = _words(match.group())
        if words:
            start = match.start() + (len(match.group()) - len(match.group().lstrip()))
            segments.append((words, start, match.start() + len(match.group().rstrip())))
    # Words found in many sentences of the passage (its topic: "job", "pod") say little about
    # which sentence the answer used: weight each word by 1 / number of sentences containing it.
    frequency: dict[str, int] = {}
    for words, _, _ in segments:
        for w in set(words):
            frequency[w] = frequency.get(w, 0) + 1

    scored: list[tuple[float, int, bool, int, int]] = []
    for words, start, end in segments:
        weights = [1 / frequency[w] for w in words]
        matched = [(w, weight) for w, weight in zip(words, weights) if w in answer_words]
        share = sum(weight for _, weight in matched) / sum(weights)
        distinctive = any(frequency[w] <= 2 for w, _ in matched)
        scored.append((share, len(matched), distinctive, start, end))

    ranges = [(s, e) for share, hits, distinctive, s, e in scored
              if distinctive and ((hits >= 3 and share >= 0.45) or (hits >= 2 and share >= 0.7))]
    if not ranges and scored:  # at least point at the closest sentence, if it's a fair match
        share, hits, distinctive, s, e = max(scored, key=lambda x: (x[0] * x[1], x[1]))
        if hits >= 2 and share >= 0.3:
            ranges = [(s, e)]
    return sorted(ranges)
