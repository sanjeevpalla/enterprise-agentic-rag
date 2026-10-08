"""Custom Guardrails AI validators for the RAG pipeline."""

from __future__ import annotations

import locale
import re
from typing import Any

from guardrails.validator_base import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)

# Common prompt-injection / jailbreak phrasings. Rule-based on purpose: it is fast, has no
# model to load, and is predictable. Patterns are kept specific to avoid flagging ordinary
# technical text (e.g. "ignore the warnings in the logs" is fine).
INJECTION_PATTERNS: dict[str, str] = {
    "override instructions": r"\b(ignore|disregard|forget|override|bypass)\b[^.\n]{0,40}\b(previous|prior|above|earlier|all|your|the system|system)\b[^.\n]{0,20}\b(instructions?|prompts?|rules|guidelines|directions)\b",
    "reveal system prompt": r"\b(reveal|show|print|repeat|output|display|leak|tell me)\b[^.\n]{0,40}\b(system prompt|system message|hidden instructions|initial instructions|your instructions|your prompt)\b",
    "persona jailbreak": r"\b(you are now|from now on,? you are|act as|pretend (to be|you are)|roleplay as)\b[^.\n]{0,60}\b(DAN|unfiltered|uncensored|jailbroken|without (any )?(rules|restrictions|limits|filters)|no (rules|restrictions|limits))\b",
    "named jailbreak": r"\b(DAN mode|do anything now|developer mode|jailbreak mode|god mode)\b",
    "no restrictions": r"\b(without|ignore|no longer (bound|restricted) by)\b[^.\n]{0,20}\b(any )?(restrictions|filters|safety|guardrails|content policy|ethical guidelines)\b",
    "fake system tag": r"(<\s*/?\s*(system|instructions?|context)\s*>|\[/?(SYSTEM|INST)\]|BEGIN SYSTEM PROMPT|###\s*(system|instruction)s?\s*:)",
}
_COMPILED = {name: re.compile(pattern, re.IGNORECASE) for name, pattern in INJECTION_PATTERNS.items()}


@register_validator(name="rag/injection_patterns", data_type="string")
class InjectionPatterns(Validator):
    """Flags text containing common prompt-injection phrasings.

    Used on user input (direct injection) and on retrieved chunks (indirect injection:
    instructions planted in a document that the LLM would otherwise read as context).
    """

    def _validate(self, value: Any, metadata: dict[str, Any]) -> ValidationResult:
        text = str(value)
        matched = [name for name, pattern in _COMPILED.items() if pattern.search(text)]
        if not matched:
            return PassResult()
        return FailResult(
            error_message=f"Possible prompt injection: {', '.join(matched)}",
            metadata={"matched": matched},
        )


_CITATION = re.compile(r"\[(\d+)\]")


@register_validator(name="rag/citations", data_type="string")
class CitationCheck(Validator):
    """Checks that an answer only cites sources that were actually provided.

    Needs ``metadata={"num_sources": N}``. Citations like ``[7]`` when only 5 sources were
    given are hallucinated; the fix removes them, keeping valid ones.
    """

    def _validate(self, value: Any, metadata: dict[str, Any]) -> ValidationResult:
        text = str(value)
        num_sources = int(metadata.get("num_sources", 0))
        invalid = sorted({int(n) for n in _CITATION.findall(text) if not 1 <= int(n) <= num_sources})
        if not invalid:
            return PassResult()
        fixed = _CITATION.sub(lambda m: m.group(0) if 1 <= int(m.group(1)) <= num_sources else "", text)
        return FailResult(
            error_message=f"Answer cites sources that don't exist: {invalid} (only {num_sources} provided)",
            fix_value=re.sub(r"[ \t]{2,}", " ", fixed),
            metadata={"invalid_citations": invalid},
        )


from guardrails_ai.secrets_present import SecretsPresent as _SecretsPresent  # noqa: E402


class SecretsPresent(_SecretsPresent):
    """SecretsPresent, fixed for non-UTF-8 systems and with whole-token masking.

    The upstream validator writes the text to a temp file in the system's default encoding
    (cp1252 on Windows), so any character outside it (e.g. an emoji in a document) crashed
    validation. We scan a copy where such characters are replaced by "?" (same length and
    lines; secrets are ASCII), then mask the original text. Upstream masking can also leave
    most of a secret visible (GitHub token "ghp_A1b2..." became "********_A1b2..."), so every
    token the scanner touched is replaced entirely by "<SECRET>".
    """

    def validate(self, value: Any, metadata: dict[str, Any]) -> ValidationResult:
        text = str(value)
        encoding = locale.getpreferredencoding(False)
        scannable = text.encode(encoding, errors="replace").decode(encoding)
        result = super().validate(scannable, metadata)
        if isinstance(result, FailResult):
            masked = (result.fix_value or scannable).rstrip("\n")
            original, scanned, fixed = (re.split(r"(\s+)", s) for s in (text, scannable, masked))
            if len(original) == len(scanned) == len(fixed):
                result.fix_value = "".join(o if s == f else "<SECRET>" for o, s, f in zip(original, scanned, fixed))
        return result
