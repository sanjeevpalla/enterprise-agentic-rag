"""Application settings, read from environment variables or a ``.env`` file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # env_ignore_empty: a key left empty in .env (e.g. "RETRIEVAL_TOP_K=") means "use the default".
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", env_ignore_empty=True
    )

    # Embeddings. Changing provider/model changes the vectors: use a new QDRANT_COLLECTION.
    # "fastembed": local open-source model (no key, no rate limits). "gemini": Google API.
    embedding_provider: Literal["fastembed", "gemini"] = "fastembed"
    # 8192-token context, so 512-token chunks are never truncated; 768 dims.
    embedding_model: str = "nomic-ai/nomic-embed-text-v1.5"
    embedding_batch_size: int = 32
    # fastembed only. Prefixes default to the model's known ones (see app/ingestion/embeddings).
    embedding_document_prefix: str | None = None
    embedding_query_prefix: str | None = None
    embedding_cache_dir: Path = Path(".cache/fastembed")
    embedding_threads: int | None = None
    # gemini only. Key is read from GOOGLE_API_KEY or GEMINI_API_KEY; dims 768/1536/3072.
    google_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("google_api_key", "gemini_api_key")
    )
    embedding_dimensions: int = 768

    # Qdrant: set QDRANT_URL for a server; otherwise an embedded on-disk store at QDRANT_PATH.
    qdrant_url: str | None = None
    qdrant_api_key: SecretStr | None = None
    qdrant_path: Path = Path("qdrant_data")
    qdrant_collection: str = "enterprise_docs"
    # Seconds per request to a Qdrant server. Upserts of vector batches to small/remote
    # clusters can exceed the client's few-second default.
    qdrant_timeout: int = 60

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 64

    # Agent LLM. "gemini": direct to Google (GOOGLE_API_KEY). "portkey": through the Portkey
    # gateway (PORTKEY_API_KEY); LLM_MODEL/PLANNER_MODEL are then Model Catalog slugs such as
    # "@google-prod/gemini-3.5-flash". Free-tier quotas are per model, so a separate
    # PLANNER_MODEL (e.g. a flash-lite model) also spreads the request budget.
    llm_provider: Literal["gemini", "portkey"] = "gemini"
    portkey_api_key: SecretStr | None = None
    portkey_base_url: str = "https://aigw.portkey.ai/v1"
    # Optional gateway config: a saved config id ("pc_...") or inline JSON (fallbacks, retries, cache).
    portkey_config: str | None = None
    # Built-in Groq fallback config (used when PORTKEY_CONFIG is empty): Model Catalog
    # provider slugs (without "@") for llama-3.3-70b-versatile and llama-3.1-8b-instant.
    groq_slug: str | None = None
    groq_slug_2: str | None = None  # defaults to GROQ_SLUG
    llm_model: str = "gemini-3.5-flash"
    planner_model: str | None = None  # None = same as llm_model
    # None = model default (newer Gemini models use fixed sampling and ignore temperature).
    llm_temperature: float | None = None
    llm_timeout: int = 60
    llm_max_retries: int = 2
    # Planner: "jev" = TypeSafe's Jev decision model (typed choice with calibrated
    # probabilities, no text generation); "llm" = the Gemini planner (also writes the query).
    planner_provider: Literal["jev", "llm"] = "jev"
    typesafe_api_key: SecretStr | None = None
    typesafe_base_url: str = "https://api.typesafe.ai"
    jev_model: str = "jev-latest"
    jev_timeout: float = 10.0
    jev_max_retries: int = 2
    # Below this confidence the planner routes to "technical" (retrieving is the safe side).
    planner_min_confidence: float = 0.6
    # Earlier messages the planner/responder see, for follow-up questions.
    agent_history_messages: int = 6

    # Search mode, used by ingestion (what vectors to store) and retrieval (how to search).
    # "hybrid": dense embedding + BM25 keyword vectors, fused with RRF. "dense": embedding only.
    # Changing it needs a new QDRANT_COLLECTION (re-ingest).
    retrieval_mode: Literal["dense", "hybrid"] = "hybrid"
    sparse_model: str = "Qdrant/bm25"

    # Retrieval: fetch RETRIEVAL_FETCH_K candidates (per search, before fusion), rerank them
    # with a cross-encoder, return the best RETRIEVAL_TOP_K.
    retrieval_top_k: int = 5
    retrieval_fetch_k: int = 20
    # Minimum cosine similarity for a candidate, dense mode only (None = no cut-off).
    # Hybrid scores are rank-fusion scores, not similarities, so no threshold applies there.
    retrieval_score_threshold: float | None = None
    rerank_enabled: bool = True
    # 8K-token context, so long chunks aren't truncated; ~0.15 GB, runs locally on CPU.
    rerank_model: str = "jinaai/jina-reranker-v1-turbo-en"

    # Guardrails (Guardrails AI validators, run locally; see app/guardrails).
    guardrails_enabled: bool = True
    # Injection-pattern validator on user input and retrieved chunks (fast, rule-based).
    guardrails_injection_patterns: bool = True
    # DetectJailbreak ML model (PyTorch, ~4s load). Off by default: on this project's test
    # prompts it scored attacks and benign questions alike (0.72-0.80), so it added no signal.
    guardrails_jailbreak_model: bool = False
    guardrails_jailbreak_threshold: float = 0.81
    guardrails_toxicity_threshold: float = 0.5
    # Presidio entity types to redact (input, retrieved chunks, output). IP addresses and
    # names are left out on purpose: technical docs legitimately contain both.
    guardrails_pii_entities: list[str] = ["EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD", "US_SSN", "IBAN_CODE"]
    # Retrieved chunks below this rerank score are dropped as irrelevant (None = keep all).
    # Calibrated on true_data (jina-reranker-v1-turbo-en): relevant queries' top chunk scored
    # >= -0.62, off-topic queries' best chunk <= -2.12. Re-check if the data or reranker changes.
    retrieval_min_rerank_score: float | None = -2.0

    # Local copy of parsed chunks
    processed_data_dir: Path = Path("processed_data")
    # Ingestion ledger (SQLite): content hashes of indexed files, for skip/dedupe/prune.
    ledger_path: Path = Path("processed_data/ingestion_ledger.db")

    # Application logging: console always; rotating files under log_dir when log_to_file.
    # On AWS set LOG_TO_FILE=false and LOG_FORMAT=json (stdout goes to CloudWatch Logs).
    log_level: str = "INFO"
    log_to_file: bool = True
    log_dir: Path = Path("logs")
    log_format: Literal["text", "json"] = "text"
    log_max_bytes: int = 10 * 1024 * 1024
    log_backup_count: int = 5

    # Langfuse observability: tracing is enabled only when both keys are set.
    langfuse_public_key: str | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_base_url: str = "https://cloud.langfuse.com"
    langfuse_environment: str = "development"
    langfuse_sample_rate: float = 1.0

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
