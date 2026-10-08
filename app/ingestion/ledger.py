"""Ingestion ledger: which files are in which collection, identified by content hash.

The ledger lets the pipeline:
  * skip files whose content hasn't changed since they were indexed (no re-embedding),
  * skip byte-identical copies of a file that is already indexed under another path,
  * find files that were deleted, moved or renamed, so their stale points can be removed.

It's a small SQLite database. That's fine for one machine; for several workers or
ephemeral containers (e.g. AWS ECS/Lambda), implement the same methods on a shared
database such as Postgres.
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ingested_files (
    collection      TEXT NOT NULL,
    source          TEXT NOT NULL,
    content_hash    TEXT NOT NULL,
    source_type     TEXT,
    chunks          INTEGER NOT NULL,
    embedding_model TEXT NOT NULL,
    ingested_at     TEXT NOT NULL,
    PRIMARY KEY (collection, source)
);
CREATE INDEX IF NOT EXISTS idx_ingested_files_hash ON ingested_files (collection, content_hash);
"""


def file_hash(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """SHA-256 of the file's bytes, read in 1 MB blocks so large files don't load into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class LedgerEntry:
    collection: str
    source: str
    content_hash: str
    source_type: str | None
    chunks: int
    embedding_model: str
    ingested_at: str


class IngestionLedger:
    """SQLite-backed record of files indexed into each Qdrant collection."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._conn = sqlite3.connect(path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)

    def get(self, collection: str, source: str) -> LedgerEntry | None:
        row = self._conn.execute(
            "SELECT * FROM ingested_files WHERE collection = ? AND source = ?", (collection, source)
        ).fetchone()
        return LedgerEntry(**row) if row else None

    def sources_with_hash(self, collection: str, content_hash: str) -> list[str]:
        rows = self._conn.execute(
            "SELECT source FROM ingested_files WHERE collection = ? AND content_hash = ?",
            (collection, content_hash),
        ).fetchall()
        return [row["source"] for row in rows]

    def sources(self, collection: str) -> list[str]:
        rows = self._conn.execute("SELECT source FROM ingested_files WHERE collection = ?", (collection,))
        return [row["source"] for row in rows]

    def record(
        self,
        collection: str,
        source: str,
        content_hash: str,
        source_type: str | None,
        chunks: int,
        embedding_model: str,
    ) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO ingested_files VALUES (?, ?, ?, ?, ?, ?, ?)",
                (collection, source, content_hash, source_type, chunks, embedding_model,
                 datetime.now(timezone.utc).isoformat()),
            )

    def remove(self, collection: str, source: str) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM ingested_files WHERE collection = ? AND source = ?", (collection, source))

    def clear(self, collection: str) -> int:
        with self._conn:
            return self._conn.execute("DELETE FROM ingested_files WHERE collection = ?", (collection,)).rowcount

    def close(self) -> None:
        self._conn.close()
