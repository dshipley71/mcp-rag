from __future__ import annotations

import asyncio
import json
from typing import Any
from urllib import error, request

from src.models import AnswerResult, RetrievedChunk


SYSTEM_PROMPT = (
    "You are a grounded RAG answer generator. "
    "Answer only from the supplied context. "
    "If the context is insufficient, say so briefly. "
    "Do not use outside knowledge."
)


def _build_context(chunks: list[RetrievedChunk]) -> str:
    parts: list[str] = []
    for chunk in chunks:
        parts.append(f"[Citation: {chunk.chunk_id}]\n{chunk.text.strip()}")
    return "\n\n".join(parts)


def _local_fallback_answer(chunks: list[RetrievedChunk]) -> AnswerResult:
    if not chunks:
        return AnswerResult(answer="", citations=[], status="no_evidence")

    for chunk in chunks:
        text = chunk.text.strip()
        if text:
            return AnswerResult(
                answer=text[:500].strip(),
                citations=[chunk.chunk_id],
                status="answered",
            )

    return AnswerResult(answer="", citations=[], status="no_evidence")


def _parse_bridge_response(payload: dict[str, Any]) -> str:
    message = payload.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()

    if isinstance(payload.get("response"), str):
        return payload["response"].strip()

    return ""


def _call_bridge_sync(bridge_url: str, model: str, query: str, context: str) -> str:
    url = bridge_url.rstrip("/") + "/api/chat"
    body = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Use the following retrieved context to answer the question.\n\n"
                    f"Question:\n{query}\n\n"
                    f"Context:\n{context}\n"
                ),
            },
        ],
        "options": {
            "temperature": 0,
        },
    }

    req = request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with request.urlopen(req, timeout=90) as resp:
        raw = resp.read().decode("utf-8")

    payload = json.loads(raw)
    return _parse_bridge_response(payload)


async def generate_answer(query: str, chunks: list[RetrievedChunk], runtime=None) -> AnswerResult:
    """
    Generate an answer using Ollama MCP Bridge when configured.

    The bridge is expected to be running separately and configured to use
    Ollama Cloud. If the bridge is unavailable during tests or offline runs,
    fall back to a deterministic context-only local response.
    """
    if not chunks:
        return AnswerResult(answer="", citations=[], status="no_evidence")

    citations = [chunk.chunk_id for chunk in chunks[:3]]

    if runtime is None:
        return _local_fallback_answer(chunks[:3])

    bridge_url = getattr(runtime, "ollama_bridge_url", "").strip()
    model = getattr(runtime, "ollama_model", "").strip()

    if not bridge_url or not model:
        return _local_fallback_answer(chunks[:3])

    context = _build_context(chunks[:3])

    try:
        answer_text = await asyncio.to_thread(_call_bridge_sync, bridge_url, model, query, context)
    except (error.URLError, TimeoutError, ValueError, json.JSONDecodeError):
        return _local_fallback_answer(chunks[:3])

    if not answer_text:
        return _local_fallback_answer(chunks[:3])

    return AnswerResult(
        answer=answer_text,
        citations=citations,
        status="answered",
    )
