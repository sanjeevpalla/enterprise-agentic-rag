"""Offline evaluation of the RAG agent with deepeval (see evaluation/README.md)."""

import os
from pathlib import Path

# Set before deepeval is imported:
# - no usage telemetry from evaluation runs;
# - deepeval loads dotenv files from evaluation/ (evaluation/.env), not the project root. The
#   app reads the root .env itself; deepeval loading it too would copy it into the process
#   environment (empty values and inline comments included) and fail its own settings validation;
# - a 15-minute budget per metric (default 180s): with rate-limit backoff, multi-step metrics
#   like Faithfulness on free-tier LLM quotas take longer.
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
os.environ.setdefault("ENV_DIR_PATH", str(Path(__file__).resolve().parent))
os.environ.setdefault("DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE", "900")
