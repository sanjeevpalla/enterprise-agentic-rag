"""The goldens as pytest tests, for deepeval's test runner (one test per golden):

    uv run --group eval --env-file evaluation/deepeval.env deepeval test run evaluation/test_rag.py
    uv run --group eval --env-file evaluation/deepeval.env deepeval test run evaluation/test_rag.py -k technical

evaluation/deepeval.env sets the deepeval options that evaluation/__init__.py sets for
run_eval.py (deepeval's pytest plugin loads before that package).

Uses the same agent, metrics and judge as run_eval.py (EVAL_JUDGE_MODEL picks the judge model).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from deepeval import assert_test

from evaluation.dataset import Golden, load_goldens
from evaluation.metrics import metrics_for
from evaluation.runner import build_agent, run_golden, to_test_case

GOLDENS = load_goldens()


@pytest.fixture(scope="session")
def agent() -> Iterator[Any]:
    agent = build_agent()
    yield agent
    agent.close()


@pytest.fixture(scope="session")
def judge() -> Any:
    from evaluation.judge import AppJudgeLLM

    return AppJudgeLLM()


@pytest.mark.parametrize("golden", GOLDENS, ids=[f"{g.category}-{g.id}" for g in GOLDENS])
def test_golden(golden: Golden, agent: Any, judge: Any) -> None:
    run = run_golden(agent, golden)
    assert run.error is None, f"agent failed: {run.error}"
    assert_test(to_test_case(run), metrics_for(golden.category, judge))
