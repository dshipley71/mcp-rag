from __future__ import annotations

from pathlib import Path
from typing import Any

from src.models import RetrievedChunk
from src.utils import safe_getattr, try_parse_json_text


def _extract_structured_payload(tool_result: Any) -> Any:
    """
    Prefer MCP structuredContent directly, but fall back to JSON encoded text blocks.

    VelociRAG MCP often returns payloads like:
      CallToolResult(content=[TextContent(text='{"results":[...]}')])
    rather than populating structuredContent.
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
                parsed = try_parse_json_text(text)
                if parsed is not None:
                    return parsed

    return None


def _normalize_search_hits(payload: Any) -> list[dict[str, Any]]:
    """
    Normalize VelociRAG search payload into a stable internal format.
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
            or item.get("file_path")
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

        metadata: dict[str, Any] = {}
        if isinstance(item.get("metadata"), dict):
            metadata = item["metadata"].copy()

        if "file_path" in item:
            metadata["file_path"] = str(item["file_path"])

        if "graph_connections" in item and "graph_connections" not in metadata:
            metadata["graph_connections"] = item["graph_connections"]

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
    Accept a valid VelociRAG health payload whether it arrives as structuredContent
    or JSON text. A zero-document index is still a healthy service; it just has no data.
    """
    try:
        result = await runtime.velocirag.call_tool("health", {})
    except Exception:
        return False

    payload = _extract_structured_payload(result)
    if not isinstance(payload, dict):
        return False

    if isinstance(payload.get("error"), str) and payload["error"].strip():
        return False

    required_keys = {"total_documents", "total_chunks"}
    return required_keys.issubset(payload.keys())


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

    if isinstance(payload.get("error"), str) and payload["error"].strip():
        return []

    items = payload.get("results", [])
    if not isinstance(items, list) or not items:
        return []

    return _normalize_search_hits(payload)


async def fetch_documents(runtime, search_hits: list[dict[str, Any]]) -> list[RetrievedChunk]:
    """
    Fetch documents using Filesystem MCP when file_path metadata is available.

    Order of preference:
    1. Filesystem read_text_file using runtime.docs_dir / file_path
    2. Fall back to the content returned by VelociRAG search
    """
    chunks: list[RetrievedChunk] = []

    for hit in search_hits:
        metadata = hit.get("metadata", {})
        file_path = metadata.get("file_path")
        text = ""

        if file_path:
            try:
                resolved_path = Path(runtime.docs_dir) / file_path
                result = await runtime.filesystem.call_tool(
                    "read_text_file",
                    {"path": str(resolved_path)},
                )

                content = safe_getattr(result, "content", None)
                if content:
                    parts: list[str] = []
                    for block in content:
                        block_text = safe_getattr(block, "text", None)
                        if isinstance(block_text, str) and block_text.strip():
                            parts.append(block_text)
                    text = "\n".join(parts).strip()
                    metadata["path"] = str(resolved_path)
            except Exception:
                text = ""

        if not text:
            text = str(hit.get("text", "")).strip()

        if text:
            chunks.append(
                RetrievedChunk(
                    chunk_id=str(hit["doc_id"]),
                    text=text,
                    score=float(hit.get("score", 0.0)),
                    metadata=metadata,
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
