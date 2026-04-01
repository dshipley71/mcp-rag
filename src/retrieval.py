from __future__ import annotations

from typing import Any

from src.models import RetrievedChunk
from src.utils import safe_getattr


def _extract_structured_payload(tool_result: Any) -> Any:
    """
    VelociRAG returns structuredContent cleanly in the inspection output.
    Prefer that directly.
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
    Treat a valid structured health payload as healthy, even if there are zero docs.
    """
    try:
        result = await runtime.velocirag.call_tool("health", {})
    except Exception:
        return False

    payload = _extract_structured_payload(result)
    if not isinstance(payload, dict):
        return False

    required_keys = {
        "total_documents",
        "total_chunks",
        "model_name",
        "db_path",
        "components",
    }
    return required_keys.issubset(payload.keys())


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

    total_results = payload.get("total_results", 0)
    if not isinstance(total_results, int) or total_results <= 0:
        return []

    return _normalize_search_hits(payload)


def fetch_documents(search_hits: list[dict[str, Any]]) -> list[RetrievedChunk]:
    """
    Filesystem integration is still deferred.
    For now, treat VelociRAG search result text as the fetched content.
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
    _ = query
    return sorted(chunks, key=lambda x: x.score, reverse=True)
