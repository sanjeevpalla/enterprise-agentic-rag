"""HTTP API for the enterprise agentic RAG.

Run from the project root:

    uv run uvicorn app.main:app --reload            # development
    uv run uvicorn app.main:app --host 0.0.0.0      # serve on the network

Then open http://127.0.0.1:8000/ for the web UI (API docs at /docs). With --host 0.0.0.0, Uvicorn prints
"http://0.0.0.0:8000": that's the bind address (all interfaces), not a URL a browser can open;
use 127.0.0.1 on this machine, or the machine's IP/hostname from elsewhere.

Endpoints (interactive docs at /docs):

    POST /chat          ask the agent; pass thread_id back for follow-up questions
    POST /chat/stream   same, streamed as NDJSON events (progress, answer tokens, final response)
    GET  /conversations          recent chats of this client (X-Client-Id header), newest first
    GET  /conversations/{id}     one chat with its turns, to reopen it
    PATCH/DELETE /conversations/{id}   rename / delete a chat (and the agent's memory of it)
    POST /sources/passage        a cited chunk in context, with what the answer drew from it highlighted
    POST /search        retrieval only (hybrid search + rerank), for debugging retrieval
    GET  /health        liveness and configuration summary
    GET  /graph         the agent graph as Mermaid
"""

from __future__ import annotations

import json
import logging
import queue
import re
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager, nullcontext
from pathlib import Path
from typing import Any, ContextManager

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.agent.graph import AgentResponse, RAGAgent, diagram
from app.config import Settings, get_settings
from app.gateway import build_gateway_config
from app.logging import setup_logging
from app.memory import ConversationStore
from app.retrieval import SearchFilters
from app.retrieval.passage import highlight_ranges, source_passage

logger = logging.getLogger(__name__)

UI_DIR = Path(__file__).resolve().parent.parent / "ui"  # <project root>/ui: index.html + static/


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
    question: str = Field(description="The question as processed by the input guardrails (PII/secrets redacted)")
    duration_seconds: float


class ConversationSummary(BaseModel):
    id: str = Field(description="The conversation's thread_id")
    title: str
    created_at: float = Field(description="Unix time")
    updated_at: float = Field(description="Unix time of the last turn")


class ConversationDetail(ConversationSummary):
    turns: list[ChatResponse] = Field(description="Every answered turn, oldest first, as /chat returned it")


class RenameRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)


class PassageRequest(BaseModel):
    chunk_id: str = Field(pattern=r"^[0-9a-f]{32}$", description="A source's chunk_id from a chat answer")
    answer: str = Field(default="", max_length=20000, description="The answer citing it, to highlight what it used")
    citation: int | None = Field(default=None, ge=1, le=99, description="The source's number in that answer")
    context: int = Field(default=2, ge=0, le=5, description="Neighbouring chunks to include on each side")


class Passage(BaseModel):
    chunk_id: str | None
    heading: str | None = Field(default=None, description="Section path, where a new section starts")
    content: str
    cited: bool = Field(description="True for the cited chunk; the others are its neighbours")
    highlights: list[tuple[int, int]] = Field(
        default_factory=list, description="[start, end) character ranges the answer draws on (cited chunk only)"
    )


class PassageResponse(BaseModel):
    file: str
    location: str | None = None
    section: str | None = None
    source_type: str | None = None
    chunk_id: str
    passages: list[Passage] = Field(description="In reading order")


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

CLIENT_HEADER = "X-Client-Id"
_CLIENT_ID = re.compile(r"[A-Za-z0-9_-]{8,64}")


def client_id(request: Request) -> str:
    """The calling browser's id (random, generated by the UI): conversations belong to it.
    Not authentication: anyone who knows an id can use it. Clients without one share "anonymous"."""
    value = request.headers.get(CLIENT_HEADER, "")
    if not value:
        return "anonymous"
    if not _CLIENT_ID.fullmatch(value):
        raise HTTPException(status_code=400, detail=f"Invalid {CLIENT_HEADER} header")
    return value


def to_chat_response(response: AgentResponse, started: float) -> ChatResponse:
    return ChatResponse(
        answer=response.answer,
        route=response.route,
        search_query=response.search_query,
        sources=[Source(**{k: s.get(k) for k in Source.model_fields}) for s in response.sources],
        thread_id=response.thread_id,
        guardrails=[GuardrailEvent(**event) for event in response.guardrails],
        question=response.question,
        duration_seconds=round(time.perf_counter() - started, 3),
    )


def effective_llm(settings: Settings) -> dict[str, Any]:
    """The model the agent actually calls, plus fallbacks. Through Portkey, a gateway config
    with targets overrides LLM_MODEL (e.g. the Groq fallback config), and a saved config id
    ("pc_...") decides the model inside Portkey."""
    if settings.llm_provider != "portkey":
        return {"llm_model": settings.llm_model, "llm_fallbacks": []}
    config = build_gateway_config(settings)
    if isinstance(config, str):
        return {"llm_model": f"set by Portkey config {config}", "llm_fallbacks": []}
    models = [
        (target.get("override_params") or {}).get("model")
        for target in (config or {}).get("targets", [])
    ]
    models = [m for m in models if m] or [settings.llm_model]
    return {"llm_model": models[0], "llm_fallbacks": models[1:]}


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
        app.state.conversations = ConversationStore(settings.memory_db_path)
        # Embedded (on-disk) Qdrant isn't built for concurrent access: serialise requests in
        # that mode. With a Qdrant server (QDRANT_URL) requests run concurrently.
        app.state.agent_lock = threading.Lock() if not settings.qdrant_url else None
        logger.info("API ready")
        try:
            yield
        finally:
            agent.close()
            app.state.conversations.close()
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

    def check_thread(request: Request, thread_id: str | None) -> str:
        """The caller's client id, after checking it may continue ``thread_id``."""
        owner = client_id(request)
        if thread_id:
            current = request.app.state.conversations.owner_of(thread_id)
            if current is not None and current != owner:
                raise HTTPException(status_code=404, detail="Conversation not found")
        return owner

    def record(request: Request, owner: str, response: ChatResponse) -> None:
        """Add the answered turn to the recent-chats list (failures only logged: the answer stands)."""
        try:
            request.app.state.conversations.add_turn(
                response.thread_id, owner, response.question, response.model_dump()
            )
        except Exception:
            logger.exception("Could not save the conversation turn")

    # Endpoints are sync (`def`): FastAPI runs them in its thread pool, so the blocking
    # agent/model calls don't stall the event loop.

    @app.post("/chat", response_model=ChatResponse, tags=["agent"])
    def chat(body: ChatRequest, request: Request) -> ChatResponse:
        owner = check_thread(request, body.thread_id)
        agent, guard = agent_access(request)
        started = time.perf_counter()
        try:
            with guard:
                response = agent.ask(body.question, body.thread_id)
        except Exception as exc:
            logger.exception("Chat request failed")
            raise HTTPException(status_code=500, detail="The agent failed to answer; see server logs.") from exc
        result = to_chat_response(response, started)
        record(request, owner, result)
        return result

    @app.post(
        "/chat/stream",
        tags=["agent"],
        response_class=StreamingResponse,
        responses={200: {"content": {"application/x-ndjson": {}}, "description": (
            "Newline-delimited JSON events: "
            '{"type": "status", "text": ...} progress, '
            '{"type": "token", "text": ...} answer pieces as they are generated, then '
            '{"type": "done", "response": ChatResponse} (show response.answer: the output '
            'guardrails may have changed it) or {"type": "error", "detail": ...}.'
        )}},
    )
    def chat_stream(body: ChatRequest, request: Request) -> StreamingResponse:
        owner = check_thread(request, body.thread_id)
        agent, guard = agent_access(request)
        started = time.perf_counter()
        events: queue.Queue[dict[str, Any] | None] = queue.Queue()

        # The whole graph run happens on one worker thread (tracing context is thread-bound);
        # the response generator just forwards its events.
        def run() -> None:
            try:
                with guard:
                    for event in agent.stream(body.question, body.thread_id):
                        if event["type"] == "done":
                            result = to_chat_response(event["response"], started)
                            record(request, owner, result)
                            event = {"type": "done", "response": result.model_dump()}
                        events.put(event)
            except Exception:
                logger.exception("Streaming chat request failed")
                events.put({"type": "error", "detail": "The agent failed to answer; see server logs."})
            finally:
                events.put(None)

        threading.Thread(target=run, name="chat-stream", daemon=True).start()

        def ndjson() -> Iterator[str]:
            while (event := events.get()) is not None:
                yield json.dumps(event, ensure_ascii=False) + "\n"

        # X-Accel-Buffering: stop reverse proxies (nginx) from buffering the stream.
        return StreamingResponse(ndjson(), media_type="application/x-ndjson",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # ------------------------------------------------------------ recent chats

    @app.get("/conversations", response_model=list[ConversationSummary], tags=["conversations"])
    def list_conversations(request: Request, limit: int = 50) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 200))
        return [c.to_dict() for c in request.app.state.conversations.list(client_id(request), limit)]

    @app.get("/conversations/{conversation_id}", response_model=ConversationDetail, tags=["conversations"])
    def get_conversation(conversation_id: str, request: Request) -> dict[str, Any]:
        conversation = request.app.state.conversations.get(conversation_id, client_id(request))
        if conversation is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return conversation.to_dict()

    @app.patch("/conversations/{conversation_id}", status_code=204, tags=["conversations"])
    def rename_conversation(conversation_id: str, body: RenameRequest, request: Request) -> Response:
        if not request.app.state.conversations.rename(conversation_id, client_id(request), body.title):
            raise HTTPException(status_code=404, detail="Conversation not found")
        return Response(status_code=204)

    @app.delete("/conversations/{conversation_id}", status_code=204, tags=["conversations"])
    def delete_conversation(conversation_id: str, request: Request) -> Response:
        """Delete the chat from the list and the agent's memory of it."""
        if not request.app.state.conversations.delete(conversation_id, client_id(request)):
            raise HTTPException(status_code=404, detail="Conversation not found")
        try:
            request.app.state.agent.forget(conversation_id)
        except Exception:
            logger.exception("Could not delete the agent memory of conversation %s", conversation_id)
        return Response(status_code=204)

    @app.post("/sources/passage", response_model=PassageResponse, tags=["retrieval"])
    def passage(body: PassageRequest, request: Request) -> PassageResponse:
        agent, guard = agent_access(request)
        try:
            with guard:
                found = source_passage(agent.retriever.vector_store, body.chunk_id, body.context)
        except Exception as exc:
            logger.exception("Source passage lookup failed")
            raise HTTPException(status_code=500, detail="Couldn't load the source; see server logs.") from exc
        if found is None:
            raise HTTPException(status_code=404, detail="Source not found (re-ingested or removed?)")
        passages = [
            Passage(**p, highlights=highlight_ranges(p["content"], body.answer, body.citation)
                    if p["cited"] and body.answer else [])
            for p in found["passages"]
        ]
        return PassageResponse(**{k: v for k, v in found.items() if k not in ("passages", "source")}, passages=passages)

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
            **effective_llm(settings),
            "guardrails": settings.guardrails_enabled,
        }

    @app.get("/graph", tags=["agent"])
    def graph() -> dict[str, str]:
        return {"mermaid": diagram()}

    # Web UI: a single page (no build step) that calls the API above. ui/index.html loads its
    # assets from ui/static/ by relative path (static/app.js), so it also works opened from disk.
    app.mount("/static", StaticFiles(directory=UI_DIR / "static"), name="static")

    @app.get("/", include_in_schema=False)
    def ui() -> FileResponse:
        return FileResponse(UI_DIR / "index.html")

    return app


app = create_app()
