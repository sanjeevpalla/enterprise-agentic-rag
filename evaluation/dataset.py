"""The golden dataset: questions with the expected answer, sources and route.

Each entry in goldens.json has:

- ``id``: stable name, used in reports;
- ``category``: ``technical`` (answer from the knowledge base), ``out_of_scope`` (not in the
  knowledge base: the agent should say so), ``conversational`` (small talk, no retrieval) or
  ``safety`` (the input guardrails should block it);
- ``input``: the question; ``history``: optional earlier user turns in the same conversation
  (tests follow-up rewriting);
- ``expected_output``: a reference answer (technical / out_of_scope);
- ``expected_sources``: file names the answer should come from (technical);
- ``expected_route``: optional, the planner route the agent should take.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

Category = Literal["technical", "out_of_scope", "conversational", "safety"]
CATEGORIES: tuple[Category, ...] = ("technical", "out_of_scope", "conversational", "safety")

DEFAULT_GOLDENS = Path(__file__).with_name("goldens.json")


@dataclass
class Golden:
    id: str
    category: Category
    input: str
    history: list[str] = field(default_factory=list)
    expected_output: str | None = None
    expected_sources: list[str] = field(default_factory=list)
    expected_route: str | None = None

    def __post_init__(self) -> None:
        if self.category not in CATEGORIES:
            raise ValueError(f"golden {self.id!r}: unknown category {self.category!r}")
        if self.category == "technical" and not self.expected_output:
            raise ValueError(f"golden {self.id!r}: technical goldens need an expected_output")


def load_goldens(
    path: Path = DEFAULT_GOLDENS,
    categories: list[str] | None = None,
    ids: list[str] | None = None,
) -> list[Golden]:
    goldens = [Golden(**item) for item in json.loads(path.read_text(encoding="utf-8"))]
    if categories:
        goldens = [g for g in goldens if g.category in categories]
    if ids:
        goldens = [g for g in goldens if g.id in ids]
    return goldens
