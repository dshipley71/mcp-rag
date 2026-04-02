from __future__ import annotations

import json
from typing import Any

from src.models import RetrievedChunk
from src.utils import safe_getattr


def _extract_structured_payload(tool_result: Any) -> Any:
    """
    VelociRAG may return JSON via structuredContent or as text content.
    """
    structured_content = safe_getattr(tool_result, "structuredContent", None)
    if structured_content is not None:
        return structured_content

    content = safe_getattr(tool_result, "content", None)
    if content:
        for block in content:
            structured = safe_getattr(block, "structuredContent", None)
            if structured is not None:
                return structured

            text = safe_getattr(block, "text", None)
            if isinstance(text, str):
                text = text.strip()
                if not text:
                    continue
                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    return parsed

    return None


def _normalize_search_hits(payload: Any) -> list[dict[str, Any]]:
    """
    Normalize the observed VelociRAG search payload:
    {
      "error": "...",
      "results": [...],
      "total_results": 0,
      "search_time_ms": 0
    }
    """
    if not isinstance(payload, dict):
        return []

    items = payload.get("results", [])
    if not isinstance(items, list):
        return []

    normalized: list[dict[str, Any]] = []

    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue

        doc_id = (
            item.get("doc_id")
            or item.get("id")
            or item.get("source_id")
            or item.get("path")
            or f"result-{idx}"
        )

        text = (
            item.get("text")
            or item.get("content")
            or item.get("chunk_text")
            or item.get("snippet")
            or ""
        )

        score = item.get("score", 0.0)

        metadata = {}
        if isinstance(item.get("metadata"), dict):
            metadata = item["metadata"].copy()

        if "path" in item and "path" not in metadata:
            metadata["path"] = item["path"]
        if "source" in item and "source" not in metadata:
            metadata["source"] = item["source"]

        normalized.append(
            {
                "doc_id": str(doc_id),
                "text": str(text),
                "score": float(score) if isinstance(score, (int, float)) else 0.0,
                "metadata": metadata,
            }
        )

    normalized.sort(key=lambda x: x["score"], reverse=True)
    return normalized


async def health_check_velocirag(runtime) -> bool:
    """
    Treat any parseable health payload dictionary as healthy unless it reports an error.
    """
    try:
        result = await runtime.velocirag.call_tool("health", {})
    except Exception:
        return False

    payload = _extract_structured_payload(result)
    if not isinstance(payload, dict):
        return False

    error = payload.get("error")
    if isinstance(error, str) and error.strip():
        return False

    return True


async def run_bm25_search(runtime, query: str, top_k: int = 20) -> list[dict[str, Any]]:
    return await _run_velocirag_search(runtime, query=query, top_k=top_k)


async def run_vector_search(runtime, query: str, top_k: int = 20) -> list[dict[str, Any]]:
    return await _run_velocirag_search(runtime, query=query, top_k=top_k)


async def _run_velocirag_search(runtime, query: str, top_k: int = 20) -> list[dict[str, Any]]:
    """
    Real MCP search call to VelociRAG using the observed payload shape.
    """
    if not query.strip():
        return []

    result = await runtime.velocirag.call_tool(
        "search",
        {
            "query": query,
            "limit": top_k,
        },
    )

    payload = _extract_structured_payload(result)
    if not isinstance(payload, dict):
        return []

    items = payload.get("results", [])
    if not isinstance(items, list) or not items:
        return []

    return _normalize_search_hits(payload)


async def fetch_documents(runtime, search_hits):
    """
    Fetch documents using Filesystem MCP.

    Uses metadata path if available.
    Falls back to VelociRAG text if needed.
    """

    chunks = []

    for hit in search_hits:
        metadata = hit.get("metadata", {})
        path = metadata.get("path")

        text = ""

        if path:
            try:
                result = await runtime.filesystem.call_tool(
                    "read_text_file",
                    {"path": path}
                )

                if hasattr(result, "content"):
                    parts = []
                    for block in result.content:
                        if hasattr(block, "text") and block.text:
                            parts.append(block.text)
                    text = "\n".join(parts)

            except Exception:
                text = ""

        # fallback to VelociRAG text
        if not text:
            text = hit.get("text", "")

        if text.strip():
            chunks.append(
                RetrievedChunk(
                    chunk_id=hit["doc_id"],
                    text=text,
                    score=hit.get("score", 0.0),
                    metadata=metadata,
                )
            )

    return chunks


def rerank_candidates(query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    _ = query
    return sorted(chunks, key=lambda x: x.score, reverse=True)
