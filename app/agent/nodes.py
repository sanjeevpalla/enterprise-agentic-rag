"""The agent's graph nodes: planner → (retriever) → responder.

Each node is a callable class: dependencies (LLM, retriever) are injected once, and
``__call__`` takes the graph state and returns the state updates.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from typesafe_sdk import Choice, TypeSafeClient, TypeSafeError

from app.agent.llm import RETRYABLE_ERRORS, with_retry
from app.agent.prompts import (
    BLOCKED_MESSAGE_PLACEHOLDER,
    GROUNDING_EMPTY_ANSWER,
    GROUNDING_SYSTEM,
    INPUT_BLOCKED_ANSWER,
    OUTPUT_BLOCKED_ANSWER,
    JEV_ROUTE_CRITERIA,
    JEV_ROUTE_INSTRUCTIONS,
    LLM_ERROR_ANSWER,
    LLM_UNAVAILABLE_ANSWER,
    NO_RESULTS_ANSWER,
    PLANNER_SYSTEM,
    QUERY_REWRITE_SYSTEM,
    RESPONDER_CONVERSATIONAL_SYSTEM,
    RESPONDER_TECHNICAL_SYSTEM,
)
from app.agent.state import AgentState, Route
from app.observability import Tracer
from app.retrieval import EnterpriseRetriever

logger = logging.getLogger(__name__)


def _latest_user_message(state: AgentState) -> str:
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage):
            return message.text
    return ""


def _recent_history(state: AgentState, limit: int) -> list[AnyMessage]:
    """The last ``limit`` messages before the latest user message (for follow-ups)."""
    messages = [m for m in state.get("messages", []) if isinstance(m, (HumanMessage, AIMessage))]
    return messages[:-1][-limit:] if limit > 0 else []


# Messages that are only a greeting, thanks, farewell or acknowledgement: routed as
# conversational without asking a model (faster, and works while the planner model is down).
# Anything with more content ("hi, how do I scale pods?") doesn't match and goes to the planner.
_SMALL_TALK = re.compile(
    r"(?:hi|hii+|hello+|hey+|hiya|yo|greetings|howdy|good (?:morning|afternoon|evening|day)"
    r"|thanks?|thank you|thx|ty|cheers|much appreciated|appreciate it"
    r"|bye|goodbye|see you|see ya|later|good night"
    r"|ok|okay|k|cool|great|nice|awesome|perfect|got it|sounds good"
    r"|how are you|how are you doing|how's it going|what's up|sup"
    r"|who are you|what are you|what can you do)"
    r"(?: (?:there|all|everyone|team|bot|assistant|again|so much|a lot|very much|buddy|mate))*"
)


def _is_small_talk(message: str) -> bool:
    """True if the message is nothing but small talk (case, punctuation and emoji ignored)."""
    text = re.sub(r"[^\w\s']", " ", message.lower().replace("’", "'"))
    return bool(_SMALL_TALK.fullmatch(" ".join(text.split())))


def _small_talk_plan() -> dict[str, Any]:
    return {"route": "conversational", "plan_reason": "small talk (rule)", "search_query": "",
            "documents": [], "sources": []}


class QueryPlan(BaseModel):
    """The planner's decision for one user message."""

    route: Route = Field(description='"technical" (needs the knowledge base) or "conversational"')
    search_query: str = Field(description="Standalone knowledge-base search query; empty if conversational")
    reason: str = Field(description="One sentence explaining the route")


class PlannerNode:
    """Classifies the latest user message and, if technical, writes a standalone search query."""

    def __init__(
        self, llm: BaseChatModel, history_messages: int = 6, structured_method: str | None = None
    ) -> None:
        # structured_method picks how the model returns the plan ("json_schema" or
        # "function_calling"); support varies by model behind a gateway. None = LangChain's default.
        kwargs = {"method": structured_method} if structured_method else {}
        self.planner = with_retry(llm.with_structured_output(QueryPlan, **kwargs))
        self.history_messages = history_messages

    def __call__(self, state: AgentState) -> dict[str, Any]:
        question = _latest_user_message(state)
        if _is_small_talk(question):
            logger.info("Planner: conversational (small talk rule)", extra={"route": "conversational"})
            return _small_talk_plan()
        messages = [
            SystemMessage(PLANNER_SYSTEM),
            *_recent_history(state, self.history_messages),
            HumanMessage(question),
        ]
        try:
            plan: QueryPlan = self.planner.invoke(messages)
        except Exception:
            # Retrieving unnecessarily costs a little latency; answering a technical question
            # without the knowledge base risks a made-up answer. So fail towards "technical".
            logger.exception("Planner failed; defaulting to technical route")
            plan = QueryPlan(route="technical", search_query=question, reason="planner error fallback")

        search_query = plan.search_query.strip() or question
        logger.info(
            "Planner: %s (%s)", plan.route, plan.reason,
            extra={"route": plan.route, "search_query": search_query if plan.route == "technical" else None},
        )
        return {
            "route": plan.route,
            "plan_reason": plan.reason,
            "search_query": search_query if plan.route == "technical" else "",
            # Clear the previous turn's results: state persists across turns of a thread.
            "documents": [],
            "sources": [],
        }


class JevPlannerNode:
    """Routes the latest user message with TypeSafe's Jev decision model.

    Jev answers a typed choice question (technical vs conversational) with calibrated
    probabilities, in one fast pass and without generating text. The recent conversation
    is part of the state, so follow-ups to a technical topic are classified correctly.
    Low-confidence or failed decisions route to "technical", the safe side. Jev writes no
    search query; the retriever node prepares it (see QueryRewriter).
    """

    def __init__(
        self,
        client: TypeSafeClient,
        history_messages: int = 6,
        min_confidence: float = 0.6,
        tracer: Tracer | None = None,
    ) -> None:
        self.client = client
        self.questions = {"route": Choice(instructions=JEV_ROUTE_INSTRUCTIONS, criteria=JEV_ROUTE_CRITERIA)}
        self.history_messages = history_messages
        self.min_confidence = min_confidence
        self.tracer = tracer or Tracer.disabled()

    def __call__(self, state: AgentState) -> dict[str, Any]:
        question = _latest_user_message(state)
        if _is_small_talk(question):
            logger.info("Planner (Jev): conversational (small talk rule)", extra={"route": "conversational"})
            return _small_talk_plan()
        decision_state =_conversation_state(_recent_history(state, self.history_messages), question)
        with self.tracer.observation(
            "jev-route", as_type="tool", input={"state": decision_state, "criteria": list(JEV_ROUTE_CRITERIA)}
        ) as span:
            try:
                response = self.client.system_one(state=decision_state, questions=self.questions)
                answer = response.answers["route"]
                confident = answer.confidence >= self.min_confidence and answer.choice in JEV_ROUTE_CRITERIA
                route: Route = answer.choice if confident else "technical"
                reason = (
                    f"Jev chose {answer.choice} (confidence {answer.confidence:.2f})"
                    + ("" if confident else f"; below {self.min_confidence}, using technical")
                )
                span.update(output={"choice": answer.choice, "confidence": answer.confidence,
                                    "probabilities": dict(answer.probabilities), "route": route},
                            metadata={"model": response.model, "usage": response.usage.model_dump()})
            except (TypeSafeError, KeyError) as exc:  # SDK errors (after its retries) or no "route" answer
                logger.error("Jev planner failed (%s); defaulting to technical route", exc)
                route, reason = "technical", "Jev error fallback"
                span.update(level="ERROR", status_message=str(exc))

        logger.info("Planner (Jev): %s (%s)", route, reason, extra={"route": route})
        return {
            "route": route,
            "plan_reason": reason,
            "search_query": "",  # the retriever node writes it
            "documents": [],
            "sources": [],
        }


class QueryRewriter:
    """Turns a follow-up question into a standalone search query using the conversation.

    Only calls the LLM when there is earlier conversation to resolve; a first question
    is searched exactly as asked.
    """

    def __init__(self, llm: BaseChatModel, history_messages: int = 6) -> None:
        self.llm = with_retry(llm)
        self.history_messages = history_messages

    def rewrite(self, state: AgentState) -> str:
        question = _latest_user_message(state)
        history = _recent_history(state, self.history_messages)
        if not history:
            return question
        try:
            reply = self.llm.invoke([SystemMessage(QUERY_REWRITE_SYSTEM), *history, HumanMessage(question)])
            return reply.text.strip().strip('"').strip() or question
        except Exception:
            logger.exception("Query rewrite failed; searching the question as asked")
            return question


class RetrieverNode:
    """Fetches knowledge-base chunks (hybrid search + rerank).

    Uses the planner's search query when it wrote one (LLM planner); otherwise prepares
    one with the QueryRewriter (Jev planner, which doesn't generate text).
    """

    def __init__(self, retriever: EnterpriseRetriever, rewriter: QueryRewriter | None = None) -> None:
        self.retriever = retriever
        self.rewriter = rewriter

    def __call__(self, state: AgentState) -> dict[str, Any]:
        query = state.get("search_query")
        if not query:
            query = self.rewriter.rewrite(state) if self.rewriter else _latest_user_message(state)
        try:
            documents = self.retriever.search(query)
        except Exception:
            logger.exception("Retrieval failed for %r", query)
            documents = []
        return {"documents": documents, "search_query": query}


# Some models (e.g. gpt-oss) cite in their native style, 【1】 or 【1†L3-L5】: the rest of the
# pipeline (source filtering, CitationCheck, grounding, the UI) expects [1].
_NATIVE_CITATION = re.compile(r"【\s*(\d{1,2})(?:\s*†[^】]*)?\s*】")


def _normalize_citations(text: str) -> str:
    return _NATIVE_CITATION.sub(lambda m: f"[{m.group(1)}]", text)


class ResponderNode:
    """Writes the final answer: grounded with citations for technical queries, a direct reply otherwise."""

    def __init__(self, llm: BaseChatModel, history_messages: int = 6) -> None:
        self.llm = with_retry(llm)
        self.history_messages = history_messages

    def _generate(self, messages: list[AnyMessage]) -> tuple[str | None, str]:
        """(LLM reply text, None) or (None, fallback answer) if the call failed: a "try again"
        answer for transient errors (still failing after retries), an admin-facing one otherwise
        (invalid API key, unknown model, ...)."""
        try:
            return _normalize_citations(self.llm.invoke(messages).text), ""
        except RETRYABLE_ERRORS:
            logger.exception("Responder LLM call failed (rate limit/overload)")
            return None, LLM_UNAVAILABLE_ANSWER
        except Exception:
            logger.exception("Responder LLM call failed")
            return None, LLM_ERROR_ANSWER

    def __call__(self, state: AgentState) -> dict[str, Any]:
        question = _latest_user_message(state)
        history = _recent_history(state, self.history_messages)

        if state.get("route") == "conversational":
            reply, fallback = self._generate(
                [SystemMessage(RESPONDER_CONVERSATIONAL_SYSTEM), *history, HumanMessage(question)]
            )
            return {"messages": [AIMessage(reply or fallback)], "sources": []}

        documents = state.get("documents") or []
        if not documents:
            # Fixed answer instead of asking the LLM: without sources it could only guess.
            return {"messages": [AIMessage(NO_RESULTS_ANSWER)], "sources": []}

        sources = [_source_entry(number, doc) for number, doc in enumerate(documents, start=1)]
        system = RESPONDER_TECHNICAL_SYSTEM.format(context=_format_context(documents, sources))
        answer, fallback = self._generate([SystemMessage(system), *history, HumanMessage(question)])
        if answer is None:
            return {"messages": [AIMessage(fallback)], "sources": []}
        cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
        return {
            "messages": [AIMessage(answer)],
            # Only the sources the answer actually cites (all of them if it cites none).
            "sources": [s for s in sources if s["number"] in cited] or sources,
        }


class InputGuardNode:
    """Input guardrails: block injection/toxic messages, redact PII and secrets.

    Redacted text replaces the user's message in the conversation (same message id), so the
    planner, LLM, logs and later turns never see the original. A blocked message is replaced
    by a placeholder, so an injection attempt can't resurface through the history.
    """

    def __init__(self, guard: Any) -> None:  # app.guardrails.InputGuard
        self.guard = guard

    def __call__(self, state: AgentState) -> dict[str, Any]:
        last = next(m for m in reversed(state["messages"]) if isinstance(m, HumanMessage))
        result = self.guard.check(last.text)
        events = [e.to_dict() for e in result.events]
        update: dict[str, Any] = {"blocked": result.blocked, "guardrail_events": events, "documents": [], "sources": []}
        if result.blocked:
            logger.warning("Input blocked by guardrails: %s", [e["validator"] for e in events],
                           extra={"guardrail_events": events})
            update.update(
                route="blocked", plan_reason="blocked by input guardrails", search_query="",
                messages=[HumanMessage(BLOCKED_MESSAGE_PLACEHOLDER, id=last.id), AIMessage(INPUT_BLOCKED_ANSWER)],
            )
        elif result.text != last.text:
            update["messages"] = [HumanMessage(result.text, id=last.id)]
        if events:
            logger.info("Input guardrails: %s", [(e["validator"], e["action"]) for e in events])
        return update


class RetrievalGuardNode:
    """Retrieval guardrails: drop irrelevant chunks and chunks with planted instructions, redact PII/secrets."""

    def __init__(self, guard: Any) -> None:  # app.guardrails.RetrievalGuard
        self.guard = guard

    def __call__(self, state: AgentState) -> dict[str, Any]:
        documents, events = self.guard.filter(state.get("documents") or [])
        new_events = [e.to_dict() for e in events]
        if new_events:
            logger.info("Retrieval guardrails: %s", [(e["validator"], e["action"]) for e in new_events])
        return {"documents": documents, "guardrail_events": [*state.get("guardrail_events", []), *new_events]}


class UnsupportedClaim(BaseModel):
    quote: str = Field(description="The unsupported sentence or bullet item, copied exactly from the answer")
    reason: str = Field(description="Why the passages don't support it, in a few words")


class GroundingReport(BaseModel):
    unsupported: list[UnsupportedClaim] = Field(description="Empty if every statement is supported")


_FENCE_BLOCK = re.compile(r"```.*?(?:```|\Z)", re.S)
_LIST_MARKER = re.compile(r"^\s*(?:[-*•]|\d+[.)])?\s*$")


def _remove_claims(answer: str, quotes: list[str]) -> tuple[str, list[str]]:
    """Remove each quoted sentence from the answer (never inside code blocks); drop lines
    left empty. Returns the new text and the quotes actually removed."""
    removed: list[str] = []
    for quote in dict.fromkeys(q.strip() for q in quotes if q and q.strip()):
        at = answer.find(quote)
        if at < 0:
            continue  # not verbatim: leave the answer alone rather than guess
        fences = [m.span() for m in _FENCE_BLOCK.finditer(answer)]
        if any(start <= at < end for start, end in fences):
            continue
        end = at + len(quote)
        while end < len(answer) and answer[end] in " \t":
            end += 1
        answer = answer[:at] + answer[end:]
        removed.append(quote)
    lines = [line for line in answer.split("\n") if not (line.strip() and _LIST_MARKER.match(line))]
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(line.rstrip() for line in lines)).strip()
    return text, removed


class GroundingCheckNode:
    """Removes statements a technical answer makes that its source passages don't support.

    The prompt tells the responder to use only the passages, but models still add facts they
    know (e.g. a default value the documents never mention), and CitationCheck only verifies
    that cited numbers exist. A second, focused LLM pass lists the unsupported sentences; they
    are cut from the answer and reported as guardrail events. Code blocks are left as written.
    If the check itself fails, the answer is kept unchanged.
    """

    def __init__(self, llm: BaseChatModel, structured_method: str | None = None) -> None:
        kwargs = {"method": structured_method} if structured_method else {}
        self.checker = with_retry(llm.with_structured_output(GroundingReport, **kwargs))

    def __call__(self, state: AgentState) -> dict[str, Any]:
        documents = state.get("documents") or []
        last = state["messages"][-1] if state.get("messages") else None
        if state.get("route") != "technical" or not documents or not isinstance(last, AIMessage):
            return {}
        answer = last.text
        if answer in (NO_RESULTS_ANSWER, LLM_UNAVAILABLE_ANSWER, LLM_ERROR_ANSWER):
            return {}

        sources = [_source_entry(number, doc) for number, doc in enumerate(documents, start=1)]
        messages = [
            SystemMessage(GROUNDING_SYSTEM.format(context=_format_context(documents, sources))),
            HumanMessage(f"Answer to verify:\n\n{answer}"),
        ]
        try:
            report: GroundingReport = self.checker.invoke(messages)
        except Exception:
            logger.exception("Grounding check failed; keeping the answer unchanged")
            return {}

        reasons = {c.quote.strip(): c.reason for c in report.unsupported}
        text, removed = _remove_claims(answer, list(reasons))
        if not removed:
            if report.unsupported:
                logger.info("Grounding check: %d claim(s) flagged but not found verbatim", len(report.unsupported))
            return {}

        logger.warning("Grounding check removed %d unsupported statement(s)", len(removed))
        events = [
            {"stage": "output", "validator": "Grounding", "action": "dropped", "detail": f"{q} — {reasons[q]}"}
            for q in removed
        ]
        if not re.search(r"\w", _FENCE_BLOCK.sub("", text)):  # nothing but code (or nothing) left
            text, kept_sources = GROUNDING_EMPTY_ANSWER, []
        else:
            cited = {int(n) for n in re.findall(r"\[(\d+)\]", text)}
            current = state.get("sources") or []
            kept_sources = [s for s in current if s["number"] in cited] or current
        return {
            "messages": [AIMessage(text, id=last.id)],
            "sources": kept_sources,
            "guardrail_events": [*state.get("guardrail_events", []), *events],
        }


class OutputGuardNode:
    """Output guardrails: block toxic answers, redact PII/secrets, remove citations to non-existent sources."""

    def __init__(self, guard: Any) -> None:  # app.guardrails.OutputGuard
        self.guard = guard

    def __call__(self, state: AgentState) -> dict[str, Any]:
        last = state["messages"][-1]
        if not isinstance(last, AIMessage):
            return {}
        # The responder numbers the (guarded) documents 1..N, so valid citations are [1]..[N].
        result = self.guard.check(last.text, num_sources=len(state.get("documents") or []))
        new_events = [e.to_dict() for e in result.events]
        update: dict[str, Any] = {"guardrail_events": [*state.get("guardrail_events", []), *new_events]}
        if result.blocked:
            logger.warning("Answer blocked by output guardrails: %s", [e["validator"] for e in new_events])
            update.update(messages=[AIMessage(OUTPUT_BLOCKED_ANSWER, id=last.id)], sources=[])
        elif result.text != last.text:
            update["messages"] = [AIMessage(result.text, id=last.id)]
        if new_events:
            logger.info("Output guardrails: %s", [(e["validator"], e["action"]) for e in new_events])
        return update


def _conversation_state(history: list[AnyMessage], question: str, max_chars: int = 600) -> dict[str, Any]:
    """Structured state for Jev: recent turns (long replies trimmed) plus the latest message."""
    conversation = []
    for message in history:
        text = re.sub(r"\s+", " ", message.text).strip()
        conversation.append({
            "role": "user" if isinstance(message, HumanMessage) else "assistant",
            "content": text[:max_chars] + ("..." if len(text) > max_chars else ""),
        })
    return {"conversation": conversation, "latest_user_message": question}


def _source_entry(number: int, doc: Document) -> dict[str, Any]:
    meta = doc.metadata
    location = next(
        (f"{label} {meta[key]}" for key, label in (("page", "page"), ("slide", "slide"), ("sheet", "sheet")) if meta.get(key)),
        None,
    )
    section = " > ".join(meta[h] for h in ("h1", "h2", "h3") if meta.get(h)) or meta.get("section_heading")
    return {
        "number": number,
        "file": Path(meta.get("source", "unknown")).name,
        "source": meta.get("source"),
        "location": location,
        "section": section,
        "source_type": meta.get("source_type"),
        "chunk_id": meta.get("chunk_id"),
        "rerank_score": meta.get("rerank_score"),
    }


def _format_context(documents: list[Document], sources: list[dict[str, Any]]) -> str:
    blocks = []
    for doc, source in zip(documents, sources):
        label = ", ".join(x for x in (source["file"], source["location"], source["section"]) if x)
        blocks.append(f"[{source['number']}] ({label})\n{doc.page_content}")
    return "\n\n".join(blocks)
