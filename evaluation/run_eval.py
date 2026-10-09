"""Evaluate the RAG agent on the golden dataset with deepeval.

Run from the project root (the app's .env configures the agent and, by default, the judge):

    uv run --group eval python -m evaluation.run_eval
    uv run --group eval python -m evaluation.run_eval --category technical --limit 5
    uv run --group eval python -m evaluation.run_eval --judge-model @groq/openai/gpt-oss-20b
    uv run --group eval python -m evaluation.run_eval --outputs evaluation/results/<run>/outputs.json

Each run writes evaluation/results/<timestamp>/ with outputs.json (the agent's answers and
retrieved passages, reusable with --outputs to re-score without re-running the agent),
report.json (every metric score and reason) and summary.md.
Exit code: 0 if every test case passed, 1 otherwise (usable as a CI gate).
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig, DisplayConfig, ErrorConfig

from evaluation.dataset import CATEGORIES, DEFAULT_GOLDENS, load_goldens
from evaluation.metrics import metrics_for
from evaluation.runner import AgentRun, load_runs, run_goldens, save_runs, to_test_case

logger = logging.getLogger("evaluation")

RESULTS_DIR = Path(__file__).with_name("results")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the RAG agent with deepeval.")
    parser.add_argument("--goldens", type=Path, default=DEFAULT_GOLDENS, help="Golden dataset (JSON)")
    parser.add_argument("--category", action="append", choices=CATEGORIES, help="Only these categories (repeatable)")
    parser.add_argument("--id", action="append", dest="ids", help="Only these golden ids (repeatable)")
    parser.add_argument("--limit", type=int, help="At most this many goldens")
    parser.add_argument("--outputs", type=Path, help="Score saved agent outputs instead of running the agent")
    parser.add_argument(
        "--judge", choices=("app", "gemini", "native"), default="app",
        help="app: the app's LLM setup (default); gemini: Gemini with the app's GOOGLE_API_KEY; "
        "native: deepeval's own configured model",
    )
    parser.add_argument(
        "--judge-model",
        help="Judge model: for --judge app a model slug (default EVAL_JUDGE_MODEL or LLM_MODEL; a saved "
        "PORTKEY_CONFIG overrides it), for --judge gemini a Gemini model name (default: deepeval's)",
    )
    parser.add_argument(
        "--judge-rpm", type=float,
        help="Pace judge calls to this many per minute (e.g. 5 for Gemini's free tier); default unlimited",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Pass threshold of the LLM metrics")
    parser.add_argument("--concurrency", type=int, default=2, help="Test cases scored in parallel (rate limits!)")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    return parser.parse_args(argv)


def build_judge(args: argparse.Namespace) -> Any:
    if args.judge == "native":
        return None  # deepeval's default model
    from evaluation.judge import AppJudgeLLM

    provider = "gemini" if args.judge == "gemini" else None
    return AppJudgeLLM(args.judge_model, provider=provider, requests_per_minute=args.judge_rpm)


def score(runs: list[AgentRun], judge: Any, threshold: float, concurrency: int) -> list[dict[str, Any]]:
    """Evaluate each category with its metrics; one flat list of per-test-case results."""
    by_category: dict[str, list[AgentRun]] = defaultdict(list)
    for run in runs:
        by_category[run.golden["category"]].append(run)

    results = []
    for category, category_runs in by_category.items():
        logger.info("Scoring %d %s case(s)", len(category_runs), category)
        evaluation = evaluate(
            test_cases=[to_test_case(r) for r in category_runs],
            metrics=metrics_for(category, judge, threshold),
            async_config=AsyncConfig(max_concurrent=concurrency),
            display_config=DisplayConfig(print_results=False, inspect_after_run=False),
            # A metric that errors (e.g. no retrieved passages) is reported, not fatal.
            error_config=ErrorConfig(ignore_errors=True),
        )
        tests = {test.name: test for test in evaluation.test_results}
        for run in category_runs:
            # deepeval leaves out test cases it cancelled (e.g. timeouts): report them as failed.
            test = tests.get(run.golden["id"])
            results.append({
                "id": run.golden["id"],
                "category": category,
                "success": bool(test and test.success and not run.error),
                "scoring_error": None if test else "Not scored: deepeval returned no result (cancelled or timed out)",
                "input": run.golden["input"],
                "history": run.golden.get("history") or [],
                "expected_output": run.golden.get("expected_output"),
                "expected_sources": run.golden.get("expected_sources") or [],
                "expected_route": run.golden.get("expected_route"),
                "answer": run.answer,
                "route": run.route,
                "search_query": run.search_query,
                "retrieved_files": run.retrieved_files,
                "cited_files": run.cited_files,
                "latency_s": run.latency_s,
                "agent_error": run.error,
                "metrics": [
                    {
                        "name": m.name,
                        "score": m.score,
                        "success": m.success,
                        "threshold": m.threshold,
                        "reason": m.reason,
                        "error": m.error,
                    }
                    for m in (test.metrics_data if test else None) or []
                ],
            })
    order = {r.golden["id"]: i for i, r in enumerate(runs)}
    return sorted(results, key=lambda r: order[r["id"]])


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    per_metric: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        for metric in result["metrics"]:
            per_metric[metric["name"]].append(metric)
    metrics = {}
    for name, values in per_metric.items():
        scores = [m["score"] for m in values if m["score"] is not None and not m["error"]]
        metrics[name] = {
            "mean": round(sum(scores) / len(scores), 3) if scores else None,
            "pass_rate": round(sum(bool(m["success"]) for m in values) / len(values), 3),
            "n": len(values),
            "errors": sum(bool(m["error"]) for m in values),
        }
    latencies = sorted(r["latency_s"] for r in results)
    return {
        "cases": len(results),
        "passed": sum(r["success"] for r in results),
        "pass_rate": round(sum(r["success"] for r in results) / len(results), 3) if results else None,
        "median_latency_s": latencies[len(latencies) // 2] if latencies else None,
        "metrics": metrics,
    }


def summary_markdown(summary: dict[str, Any], results: list[dict[str, Any]], judge_name: str) -> str:
    lines = [
        "# RAG evaluation",
        "",
        f"Judge: `{judge_name}` · cases: {summary['cases']} · passed: {summary['passed']} "
        f"({summary['pass_rate']:.0%}) · median latency: {summary['median_latency_s']}s",
        "",
        "| Metric | Mean score | Pass rate | Cases | Errors |",
        "|---|---|---|---|---|",
    ]
    for name, m in summary["metrics"].items():
        mean = "–" if m["mean"] is None else f"{m['mean']:.2f}"
        lines.append(f"| {name} | {mean} | {m['pass_rate']:.0%} | {m['n']} | {m['errors']} |")
    lines += ["", "| Case | Category | Result | Failed metrics |", "|---|---|---|---|"]
    for r in results:
        failed = [
            f"{m['name']} ({'error' if m['error'] else format(m['score'], '.2f')})"
            for m in r["metrics"] if not m["success"]
        ]
        if r["agent_error"]:
            failed.insert(0, "agent error")
        if r["scoring_error"]:
            failed.insert(0, "not scored")
        lines.append(f"| {r['id']} | {r['category']} | {'✅' if r['success'] else '❌'} | {', '.join(failed)} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    logger.setLevel(logging.INFO)

    run_dir = args.results_dir / datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.outputs:
        runs = load_runs(args.outputs)
        wanted = {g.id for g in load_goldens(args.goldens, args.category, args.ids)}
        runs = [r for r in runs if r.golden["id"] in wanted]
    else:
        goldens = load_goldens(args.goldens, args.category, args.ids)
        runs = run_goldens(goldens[: args.limit] if args.limit else goldens)
        save_runs(runs, run_dir / "outputs.json")
        logger.info("Agent outputs saved to %s", run_dir / "outputs.json")
    if args.limit:
        runs = runs[: args.limit]
    if not runs:
        logger.error("No goldens selected")
        return 2

    judge = build_judge(args)
    judge_name = judge.get_model_name() if judge is not None else "deepeval default"
    results = score(runs, judge, args.threshold, args.concurrency)
    summary = summarize(results)

    run_dir.mkdir(parents=True, exist_ok=True)
    report = {"judge": judge_name, "threshold": args.threshold, "summary": summary, "results": results}
    (run_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown = summary_markdown(summary, results, judge_name)
    (run_dir / "summary.md").write_text(markdown, encoding="utf-8")
    print("\n" + markdown)
    print(f"Full report: {run_dir / 'report.json'}")
    return 0 if summary["passed"] == summary["cases"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
