# 🧪 Evaluation

Offline evaluation of the RAG agent with [deepeval](https://deepeval.com). The agent runs on a golden dataset of questions about the documents in `DATA/true_data`. Each answer is then scored on retrieval quality, answer quality, abstention, routing and guardrails.

The app is used as-is: the evaluation calls `RAGAgent` from `app/agent/graph.py` with the same `.env` and reads the passages the responder saw back from the graph state. Nothing in `app/` is evaluation-specific.

## 🚀 Running it

Prerequisites: the app works (`.env` filled in, documents ingested; see the main README).

```bash
uv sync --group eval                                   # installs deepeval (dependency group "eval")

uv run --group eval python -m evaluation.run_eval      # all goldens
uv run --group eval python -m evaluation.run_eval --category technical --limit 5
uv run --group eval python -m evaluation.run_eval --id arch-etcd --id oos-istio

# Score saved answers again without re-running the agent (e.g. after changing the judge or metrics)
uv run --group eval python -m evaluation.run_eval --outputs evaluation/results/<run>/outputs.json

# The same goldens as pytest tests, through deepeval's test runner
uv run --group eval --env-file evaluation/deepeval.env deepeval test run evaluation/test_rag.py
```

Each `run_eval` run writes `evaluation/results/<timestamp>/` (git-ignored):

| File | Contents |
|---|---|
| `outputs.json` | The agent's answer, route, search query, retrieved passages and cited files per golden |
| `report.json` | Every metric's score, pass/fail and the judge's reason |
| `summary.md` | Mean score and pass rate per metric, and the failed metrics per case |

The exit code is `0` only if every case passed, so the command can gate CI.

### 📊 Viewing results in the web UI

Start the app (`uv run uvicorn app.main:app --port 9000`, then http://127.0.0.1:9000/) and open **Evaluation** in the sidebar. Pick a run to see its pass rate, latency and judge; a score bar per metric with a tick at its pass threshold; and every case, filterable by category or failures, with the question, answer, reference answer, route, retrieved and expected sources, and each metric's score and the judge's reason.

The view is read-only: it reads `report.json` files from `EVAL_RESULTS_DIR` (default `evaluation/results`) through `GET /evaluations` and `GET /evaluations/{run_id}`. New runs show up when you reopen the view.

| Option | Default | Meaning |
|---|---|---|
| `--category` | all | `technical`, `out_of_scope`, `conversational`, `safety` (repeatable) |
| `--id` | all | Only these golden ids (repeatable) |
| `--limit` | – | At most N goldens |
| `--outputs` | – | Re-score a saved `outputs.json` instead of running the agent |
| `--judge` | `app` | `app`: the judge uses the app's LLM setup; `native`: deepeval's own configured model |
| `--judge-model` | `EVAL_JUDGE_MODEL`, else `LLM_MODEL` | Judge model for `--judge app` |
| `--threshold` | `0.5` | Pass threshold of the LLM-judged metrics |
| `--concurrency` | `2` | Test cases scored in parallel; keep it low on rate-limited LLM tiers |

## ⚖️ The judge

The LLM-as-a-judge metrics need a judge model. By default (`--judge app`) `evaluation/judge.py` builds it with the app's own `build_chat_model`, so it uses the same provider and keys as the agent (Portkey → Groq, or Gemini) and needs no extra API key.

A model tends to rate its own answers favourably, so for reported numbers use a different judge model than the agent:

```bash
EVAL_JUDGE_MODEL=@google-prod/gemini-3.5-flash uv run --group eval python -m evaluation.run_eval
```

To use deepeval's built-in providers instead, pass `--judge native` and configure deepeval as usual (`deepeval set-openai ...`, `deepeval set-gemini ...`, or keys such as `OPENAI_API_KEY`). deepeval reads dotenv files from **`evaluation/.env`**, not the project root (see below).

## 📏 Metrics

| Category | Metric | Kind | What it checks |
|---|---|---|---|
| technical | Answer Relevancy | LLM | The answer addresses the question |
| technical | Faithfulness | LLM | Every claim is supported by the retrieved passages (hallucination) |
| technical | Contextual Precision | LLM | Relevant passages are ranked above irrelevant ones (reranker) |
| technical | Contextual Recall | LLM | The passages contain what the reference answer needs (retrieval coverage) |
| technical | Contextual Relevancy | LLM | How much of the retrieved text is relevant (noise sent to the responder) |
| technical | Correctness (G-Eval) | LLM | The answer agrees with the reference answer |
| technical | Source Hit | rule | The passages came from the expected document(s) |
| out_of_scope | Abstention (G-Eval) | LLM | The agent says it doesn't know instead of answering from general knowledge |
| conversational | Route, Answer Relevancy | rule, LLM | Planner routes small talk away from retrieval; reply is relevant |
| safety | Route | rule | Input guardrails block the prompt (`route == "blocked"`) |

The retrieval context is the set of passages the responder received, after reranking and the retrieval guardrails. `safety` cases fail when `GUARDRAILS_ENABLED=false`, because nothing blocks the prompt.

## 📝 The golden dataset

`goldens.json` holds 22 cases: 17 technical (2 of them follow-ups that test query rewriting), 2 out-of-scope, 1 conversational and 2 safety. The technical cases cover every document in `DATA/true_data`. Each entry looks like this:

```json
{
  "id": "followup-hpa-metrics",
  "category": "technical",
  "history": ["What does the Horizontal Pod Autoscaler do?"],
  "input": "Which metrics can it scale on?",
  "expected_output": "CPU utilization, memory usage, or custom metrics ...",
  "expected_sources": ["pods_autoscale.html"]
}
```

`history` holds earlier user turns that run in the same conversation before `input`. `expected_route` is used by the conversational and safety cases. To add a case, append an entry; reference answers should come from the documents, not from general knowledge.

## 🗂️ Files

```
evaluation/
├── run_eval.py     CLI: run the agent, score with deepeval, write the report
├── test_rag.py     Same evaluation as pytest tests (deepeval test run)
├── goldens.json    Golden dataset
├── dataset.py      Golden loading and validation
├── runner.py       Runs the agent per golden → deepeval LLMTestCase
├── metrics.py      Metric set per category, Route and Source Hit metrics
├── judge.py        Judge LLM built from the app's settings
├── deepeval.env    deepeval settings for `deepeval test run` (passed with uv --env-file)
└── results/        Run outputs (git-ignored)
```

## ⚙️ deepeval settings set by this package

`evaluation/__init__.py` sets these before deepeval is imported. Each one only applies if the variable isn't already set in the environment. Under `deepeval test run`, deepeval's pytest plugin loads first, so `evaluation/deepeval.env` sets the same values through `uv run --env-file`.

| Variable | Value | Why |
|---|---|---|
| `ENV_DIR_PATH` | `evaluation/` | deepeval auto-loads `.env` files. Loading the app's root `.env` would copy its empty values and inline comments into the process environment, and deepeval's own settings reject an empty `PORTKEY_BASE_URL`. |
| `DEEPEVAL_TELEMETRY_OPT_OUT` | `YES` | No usage telemetry from evaluation runs |
| `DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE` | `900` | Multi-step metrics (Faithfulness, Contextual *) exceed deepeval's 180 s default once rate-limit backoff kicks in |

## 🧾 Notes

- Agent runs use in-memory conversation memory, so evaluation questions don't appear in the web UI's recent chats. Langfuse tracing still applies if it's configured.
- A full run makes about 100 judge calls on top of the agent's own calls. Rate-limited free tiers will be slow; lower `--concurrency` if you see timeouts.
- Adding the `eval` group pinned `click<8.4` (a deepeval requirement), which resolves `huggingface-hub` 1.13 and `transformers` 5.17 in the shared lockfile.
