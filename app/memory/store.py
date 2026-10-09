"""Conversation memory in one SQLite file (MEMORY_DB_PATH).

Two parts:

- the agent's memory: LangGraph checkpoints per thread (SqliteSaver), so follow-up
  questions keep their context across server restarts;
- the recent-chats list: one row per conversation (title, owner, timestamps) and one per
  turn with what the UI shows (answer, sources, guardrail events, ...), so a chat can be
  reopened exactly as it was answered.

There are no user accounts: each browser sends a random client id, and a conversation is
only listed for, readable by and continued by the client that started it.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver

TITLE_MAX_CHARS = 60

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id          TEXT PRIMARY KEY,
    owner       TEXT NOT NULL,
    title       TEXT NOT NULL,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS conversations_owner_updated ON conversations (owner, updated_at DESC);
CREATE TABLE IF NOT EXISTS conversation_turns (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id  TEXT NOT NULL REFERENCES conversations (id) ON DELETE CASCADE,
    question         TEXT NOT NULL,
    response         TEXT NOT NULL,   -- JSON: the ChatResponse the UI rendered
    created_at       REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS conversation_turns_conversation ON conversation_turns (conversation_id, id);
"""


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Shared across FastAPI's worker threads; callers serialise access with a lock.
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")  # readers don't block the writer
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def build_checkpointer(path: Path) -> SqliteSaver:
    """The agent's persistent memory (LangGraph checkpoints) in the SQLite file at ``path``."""
    saver = SqliteSaver(_connect(path))
    saver.setup()
    return saver


def make_title(question: str) -> str:
    """Chat title from its first question: one line, at most TITLE_MAX_CHARS characters."""
    title = " ".join(question.split())
    if len(title) > TITLE_MAX_CHARS:
        title = title[: TITLE_MAX_CHARS - 1].rsplit(" ", 1)[0].rstrip(",.;:") + "…"
    return title or "New chat"


@dataclass
class Conversation:
    id: str
    title: str
    created_at: float
    updated_at: float
    turns: list[dict[str, Any]] | None = None  # filled by ConversationStore.get

    def to_dict(self) -> dict[str, Any]:
        data = {"id": self.id, "title": self.title, "created_at": self.created_at, "updated_at": self.updated_at}
        if self.turns is not None:
            data["turns"] = self.turns
        return data


class ConversationStore:
    """The recent-chats list. Every method takes the caller's ``owner`` (client id) and only
    sees that owner's conversations."""

    def __init__(self, path: Path) -> None:
        self._conn = _connect(path)
        self._lock = threading.Lock()
        with self._lock, self._conn:
            self._conn.executescript(_SCHEMA)

    def owner_of(self, conversation_id: str) -> str | None:
        with self._lock:
            row = self._conn.execute("SELECT owner FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
        return row[0] if row else None

    def add_turn(self, conversation_id: str, owner: str, question: str, response: dict[str, Any]) -> None:
        """Record one answered turn; the first turn creates the conversation, titled after its question."""
        now = time.time()
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO conversations (id, owner, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT (id) DO UPDATE SET updated_at = excluded.updated_at",
                (conversation_id, owner, make_title(question), now, now),
            )
            self._conn.execute(
                "INSERT INTO conversation_turns (conversation_id, question, response, created_at) VALUES (?, ?, ?, ?)",
                (conversation_id, question, json.dumps(response, ensure_ascii=False), now),
            )

    def list(self, owner: str, limit: int = 50) -> list[Conversation]:
        """Most recently active first."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, title, created_at, updated_at FROM conversations WHERE owner = ? "
                "ORDER BY updated_at DESC LIMIT ?",
                (owner, limit),
            ).fetchall()
        return [Conversation(*row) for row in rows]

    def get(self, conversation_id: str, owner: str) -> Conversation | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ? AND owner = ?",
                (conversation_id, owner),
            ).fetchone()
            if row is None:
                return None
            turns = self._conn.execute(
                "SELECT response FROM conversation_turns WHERE conversation_id = ? ORDER BY id", (conversation_id,)
            ).fetchall()
        return Conversation(*row, turns=[json.loads(t[0]) for t in turns])

    def rename(self, conversation_id: str, owner: str, title: str) -> bool:
        with self._lock, self._conn:
            cursor = self._conn.execute(
                "UPDATE conversations SET title = ? WHERE id = ? AND owner = ?",
                (make_title(title), conversation_id, owner),
            )
        return cursor.rowcount > 0

    def delete(self, conversation_id: str, owner: str) -> bool:
        with self._lock, self._conn:
            cursor = self._conn.execute(
                "DELETE FROM conversations WHERE id = ? AND owner = ?", (conversation_id, owner)
            )
        return cursor.rowcount > 0

    def close(self) -> None:
        with self._lock:
            self._conn.close()
