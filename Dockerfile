# Enterprise Agentic RAG: API + web UI.
#
#   docker build -t enterprise-agentic-rag .
#   docker run --rm -p 9000:9000 --env-file .env -v rag-state:/app/state enterprise-agentic-rag
#
# Then open http://127.0.0.1:9000/. Configuration comes from environment variables (--env-file .env);
# the image contains no secrets. Set QDRANT_URL to a Qdrant server: the embedded store
# (QDRANT_PATH) serialises every request, so it's only for trying things out.
#
# Ingest documents with the same image, mounting the folder to index:
#
#   docker run --rm --env-file .env -v rag-state:/app/state -v "$PWD/DATA:/app/DATA:ro" \
#       enterprise-agentic-rag python -m app.ingestion.ingestion --data-dir DATA/true_data

ARG PYTHON_VERSION=3.13

# ---------------------------------------------------------------------------- build
FROM ghcr.io/astral-sh/uv:python${PYTHON_VERSION}-bookworm-slim AS build

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# Dependencies only (the project itself isn't a package), so this layer is reused until
# pyproject.toml or uv.lock change.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --frozen --no-dev --no-install-project

# Bake the default local models into the image, so containers start without downloading:
# embedding (EMBEDDING_MODEL), BM25 (SPARSE_MODEL), reranker (RERANK_MODEL), and NLTK's
# tokenizer for the toxicity guardrail. Other models set via env are downloaded at startup.
ENV FASTEMBED_CACHE=/app/.cache/fastembed \
    NLTK_DATA=/app/.cache/nltk_data
RUN /app/.venv/bin/python -c "\
from fastembed import TextEmbedding, SparseTextEmbedding; \
from fastembed.rerank.cross_encoder import TextCrossEncoder; \
import nltk; \
TextEmbedding('nomic-ai/nomic-embed-text-v1.5', cache_dir='$FASTEMBED_CACHE'); \
SparseTextEmbedding('Qdrant/bm25', cache_dir='$FASTEMBED_CACHE'); \
TextCrossEncoder('jinaai/jina-reranker-v1-turbo-en', cache_dir='$FASTEMBED_CACHE'); \
nltk.download('punkt_tab', download_dir='$NLTK_DATA', quiet=True)"

# ---------------------------------------------------------------------------- runtime
FROM python:${PYTHON_VERSION}-slim-bookworm

RUN groupadd --system app && useradd --system --gid app --home-dir /app app

WORKDIR /app

COPY --from=build --chown=app:app /app/.venv /app/.venv
COPY --from=build --chown=app:app /app/.cache /app/.cache
COPY --chown=app:app app ./app
COPY --chown=app:app ui ./ui
COPY --chown=app:app evaluation ./evaluation

# Everything the app writes goes under /app/state (mount a volume there to keep it):
# conversation memory, ingestion ledger and parsed chunks, embedded Qdrant, eval reports.
# Logs go to stdout only. HF_HOME: models the guardrails download (toxicity classifier).
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    EMBEDDING_CACHE_DIR=/app/.cache/fastembed \
    NLTK_DATA=/app/.cache/nltk_data \
    HF_HOME=/app/state/huggingface \
    MEMORY_DB_PATH=/app/state/memory/memory.sqlite \
    PROCESSED_DATA_DIR=/app/state/processed_data \
    LEDGER_PATH=/app/state/processed_data/ingestion_ledger.db \
    QDRANT_PATH=/app/state/qdrant_data \
    EVAL_RESULTS_DIR=/app/state/evaluation_results \
    LOG_TO_FILE=false

RUN mkdir -p /app/state && chown app:app /app/state
VOLUME ["/app/state"]

USER app
EXPOSE 9000

# Startup loads several models, hence the long start period.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:9000/health', timeout=4)"]

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9000"]
