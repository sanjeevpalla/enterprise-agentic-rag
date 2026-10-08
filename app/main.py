"""HTTP API for the enterprise agentic RAG.

Run from the project root:

    uv run uvicorn app.main:app --reload            # development
    uv run uvicorn app.main:app --host 0.0.0.0      # serve on the network

Then open http://127.0.0.1:8000/ (redirects to the docs). With --host 0.0.0.0, Uvicorn prints
"http://0.0.0.0:8000": that's the bind address (all interfaces), not a URL a browser can open;
use 127.0.0.1 on this machine, or the machine's IP/hostname from elsewhere.

Endpoints (interactive docs at /docs):

    POST /chat     ask the agent; pass thread_id back for follow-up questions
    POST /search   retrieval only (hybrid search + rerank), for debugging retrieval
    GET  /health   liveness and configuration summary
    GET  /graph    the agent graph as Mermaid
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager, nullcontext
from pathlib import Path
from typing import Any, ContextManager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from app.agent.graph import RAGAgent, diagram
from app.config import Settings, get_settings
from app.logging import setup_logging
from app.retrieval import SearchFilters

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------- schemas


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000, examples=["How do I limit Kubernetes job retries?"])
    thread_id: str | None = Field(
        default=None, max_length=128, description="Conversation id; omit to start a new conversation"
    )


class Source(BaseModel):
    number: int
    file: str
    location: str | None = None
    section: str | None = None
    source_type: str | None = None
    chunk_id: str | None = None
    rerank_score: float | None = None


class GuardrailEvent(BaseModel):
    stage: str = Field(description='"input", "retrieval" or "output"')
    validator: str
    action: str = Field(description='"blocked", "redacted", "dropped" or "fixed"')
    detail: str = ""


class ChatResponse(BaseModel):
    answer: str
    route: str = Field(description='"technical" (answered from the knowledge base), "conversational", '
                                   'or "blocked" (stopped by the input guardrails)')
    search_query: str = Field(description="Query used for retrieval; empty for conversational turns")
    sources: list[Source]
    thread_id: str = Field(description="Send this back with the next question to continue the conversation")
    guardrails: list[GuardrailEvent] = Field(description="What the guardrails blocked, redacted, dropped or fixed")
    duration_seconds: float


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    k: int | None = Field(default=None, ge=1, le=50, description="Number of results (default RETRIEVAL_TOP_K)")
    source_types: list[str] = Field(default_factory=list, examples=[["true_data"]])
    file_types: list[str] = Field(default_factory=list, examples=[["pdf", "docx"]])


class SearchResult(BaseModel):
    rank: int
    content: str
    file: str
    source: str | None = None
    source_type: str | None = None
    file_type: str | None = None
    location: str | None = None
    search_mode: str | None = None
    search_score: float | None = None
    rerank_score: float | None = None
    chunk_id: str | None = None


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    duration_seconds: float


# ---------------------------------------------------------------------------- app


def create_app(
    settings: Settings | None = None,
    agent_factory: Callable[[Settings], RAGAgent] = RAGAgent,
) -> FastAPI:
    """Build the API. ``agent_factory`` is injectable for tests."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        setup_logging(settings)
        logger.info("Starting API: loading models and connecting to Qdrant")
        # Startup fails loudly on a bad configuration (missing keys, wrong/missing collection).
        agent = agent_factory(settings)
        app.state.agent = agent
        # Embedded (on-disk) Qdrant isn't built for concurrent access: serialise requests in
        # that mode. With a Qdrant server (QDRANT_URL) requests run concurrently.
        app.state.agent_lock = threading.Lock() if not settings.qdrant_url else None
        logger.info("API ready")
        try:
            yield
        finally:
            agent.close()
            logger.info("API stopped")

    app = FastAPI(
        title="Enterprise Agentic RAG",
        version="0.1.0",
        description="Planner → (retriever) → responder agent over the enterprise knowledge base.",
        lifespan=lifespan,
    )

    def agent_access(request: Request) -> tuple[RAGAgent, ContextManager[Any]]:
        lock = request.app.state.agent_lock
        return request.app.state.agent, (lock if lock is not None else nullcontext())

    # Endpoints are sync (`def`): FastAPI runs them in its thread pool, so the blocking
    # agent/model calls don't stall the event loop.

    @app.post("/chat", response_model=ChatResponse, tags=["agent"])
    def chat(body: ChatRequest, request: Request) -> ChatResponse:
        agent, guard = agent_access(request)
        started = time.perf_counter()
        try:
            with guard:
                response = agent.ask(body.question, body.thread_id)
        except Exception as exc:
            logger.exception("Chat request failed")
            raise HTTPException(status_code=500, detail="The agent failed to answer; see server logs.") from exc
        return ChatResponse(
            answer=response.answer,
            route=response.route,
            search_query=response.search_query,
            sources=[Source(**{k: s.get(k) for k in Source.model_fields}) for s in response.sources],
            thread_id=response.thread_id,
            guardrails=[GuardrailEvent(**event) for event in response.guardrails],
            duration_seconds=round(time.perf_counter() - started, 3),
        )

    @app.post("/search", response_model=SearchResponse, tags=["retrieval"])
    def search(body: SearchRequest, request: Request) -> SearchResponse:
        agent, guard = agent_access(request)
        filters = SearchFilters(source_types=tuple(body.source_types), file_types=tuple(body.file_types))
        started = time.perf_counter()
        try:
            with guard:
                documents = agent.retriever.search(body.query, k=body.k, filters=filters)
        except Exception as exc:
            logger.exception("Search request failed")
            raise HTTPException(status_code=500, detail="Search failed; see server logs.") from exc
        results = []
        for doc in documents:
            meta = doc.metadata
            location = next(
                (f"{key} {meta[key]}" for key in ("page", "slide", "sheet") if meta.get(key) is not None), None
            )
            results.append(SearchResult(
                rank=meta.get("rank", len(results) + 1),
                content=doc.page_content,
                file=Path(meta.get("source", "unknown")).name,
                source=meta.get("source"),
                source_type=meta.get("source_type"),
                file_type=meta.get("file_type"),
                location=location,
                search_mode=meta.get("search_mode"),
                search_score=meta.get("search_score"),
                rerank_score=meta.get("rerank_score"),
                chunk_id=meta.get("chunk_id"),
            ))
        return SearchResponse(query=body.query, results=results, duration_seconds=round(time.perf_counter() - started, 3))

    @app.get("/health", tags=["ops"])
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "collection": settings.qdrant_collection,
            "qdrant": "server" if settings.qdrant_url else "embedded",
            "retrieval_mode": settings.retrieval_mode,
            "embedding_model": settings.embedding_model,
            "planner": settings.planner_provider,
            "llm_provider": settings.llm_provider,
            "llm_model": settings.llm_model,
            "guardrails": settings.guardrails_enabled,
        }

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse(url="/docs")

    @app.get("/graph", tags=["agent"])
    def graph() -> dict[str, str]:
        return {"mermaid": diagram()}

    return app


app = create_app()
