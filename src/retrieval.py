from __future__ import annotations

from typing import Any

from src.models import RetrievedChunk
from src.utils import safe_getattr, try_parse_json_text


def _extract_text_blocks(tool_result: Any) -> list[str]:
    """
    MCP tool results commonly expose a .content list containing typed blocks.
    We only read text-like blocks and ignore everything else.
    """
    content = safe_getattr(tool_result, "content", None)
    if not content:
        return []

    texts: list[str] = []
    for block in content:
        text = safe_getattr(block, "text", None)
        if isinstance(text, str) and text.strip():
            texts.append(text)

        structured = safe_getattr(block, "structuredContent", None)
        if isinstance(structured, str) and structured.strip():
            texts.append(structured)

    return texts


def _extract_structured_payload(tool_result: Any) -> Any:
    """
    Try the most likely structured locations first, then fall back to parsing
    the first text block as JSON.
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

    text_blocks = _extract_text_blocks(tool_result)
    for text in text_blocks:
        parsed = try_parse_json_text(text)
        if parsed is not None:
            return parsed

    return None


def _normalize_search_hits(payload: Any) -> list[dict[str, Any]]:
    """
    Convert likely VelociRAG search payload shapes into a stable internal format.

    Supported common shapes:
    - {"results": [...]}
    - {"documents": [...]}
    - [...]
    """
    if payload is None:
        return []

    if isinstance(payload, dict):
        if isinstance(payload.get("results"), list):
            items = payload["results"]
        elif isinstance(payload.get("documents"), list):
            items = payload["documents"]
        else:
            return []
    elif isinstance(payload, list):
        items = payload
    else:
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
    Real MCP call. Returns False on any unexpected response.
    """
    try:
        result = await runtime.velocirag.call_tool("health", {})
    except Exception:
        return False

    payload = _extract_structured_payload(result)
    if isinstance(payload, dict):
        status = str(payload.get("status", "")).lower()
        if status in {"ok", "healthy", "ready"}:
            return True

    text = "\n".join(_extract_text_blocks(result)).lower()
    return any(token in text for token in ("ok", "healthy", "ready"))


async def run_bm25_search(runtime, query: str, top_k: int = 20) -> list[dict[str, Any]]:
    """
    Logical BM25 stage preserved for the routing contract.
    Backed by VelociRAG MCP search in v1.
    """
    return await _run_velocirag_search(runtime, query=query, top_k=top_k)


async def run_vector_search(runtime, query: str, top_k: int = 20) -> list[dict[str, Any]]:
    """
    Logical vector stage preserved for the routing contract.
    Backed by the same VelociRAG MCP search in v1.
    """
    return await _run_velocirag_search(runtime, query=query, top_k=top_k)


async def _run_velocirag_search(runtime, query: str, top_k: int = 20) -> list[dict[str, Any]]:
    """
    Real MCP search call to VelociRAG.
    This is intentionally strict: if the payload shape is unknown or empty,
    return [] and let the orchestrator fail closed.
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
    return _normalize_search_hits(payload)


def fetch_documents(search_hits: list[dict[str, Any]]) -> list[RetrievedChunk]:
    """
    Filesystem integration is deferred.
    For now, treat the grounded text returned by VelociRAG search as the
    document-fetch stage input. If the search results have no text, return [].
    """
    chunks: list[RetrievedChunk] = []

    for hit in search_hits:
        text = hit.get("text", "")
        if not text.strip():
            continue

        chunks.append(
            RetrievedChunk(
                chunk_id=hit["doc_id"],
                text=text,
                score=hit.get("score", 0.0),
                metadata=hit.get("metadata", {}),
            )
        )

    return chunks


def rerank_candidates(query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """
    Deterministic local rerank placeholder.
    VelociRAG may already return reranked results, so v1 preserves ordering
    by score and does not introduce a second model here.
    """
    _ = query
    return sorted(chunks, key=lambda x: x.score, reverse=True)
