from __future__ import annotations

"""
Deterministic ingestion helpers for explicit setup-time indexing.
"""

import hashlib
from pathlib import Path
from typing import Any

from src.models import IngestionResult
from src.utils import safe_getattr, try_parse_json_text

DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 150


def _ensure_allowed_path(path: Path, filesystem_root: str) -> Path:
    root = Path(filesystem_root).resolve()
    resolved = path.resolve()

    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"Path is outside allowed ingestion root: {resolved}") from exc

    return resolved


def _extract_structured_payload(tool_result: Any) -> Any:
    if isinstance(tool_result, dict):
        return tool_result

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
                if isinstance(parsed, dict):
                    return parsed

    return None


def _normalize_parsed_text(payload: Any) -> tuple[str, dict[str, Any]]:
    if not isinstance(payload, dict):
        raise RuntimeError("Document parser returned an invalid payload")

    if isinstance(payload.get("error"), str) and payload["error"].strip():
        raise RuntimeError(f"Document parser failed: {payload['error']}")

    metadata_value = payload.get("metadata")
    metadata = metadata_value.copy() if isinstance(metadata_value, dict) else {}

    if isinstance(payload.get("text"), str) and payload["text"].strip():
        return payload["text"].strip(), metadata

    if isinstance(payload.get("content"), str) and payload["content"].strip():
        return payload["content"].strip(), metadata

    elements = payload.get("elements")
    if isinstance(elements, list):
        lines: list[str] = []
        element_types: list[str] = []
        for element in elements:
            if not isinstance(element, dict):
                continue
            text = element.get("text")
            if isinstance(text, str) and text.strip():
                lines.append(text.strip())
            element_type = element.get("type")
            if isinstance(element_type, str) and element_type.strip():
                element_types.append(element_type.strip())

        normalized = "\n\n".join(lines).strip()
        if normalized:
            if element_types:
                metadata["element_types"] = sorted(set(element_types))
            return normalized, metadata

    raise RuntimeError("Document parser returned no text content")


def _normalize_tool_names(list_tools_result: Any) -> set[str]:
    names: set[str] = set()
    if isinstance(list_tools_result, list):
        for item in list_tools_result:
            if isinstance(item, str):
                names.add(item)
            elif isinstance(item, dict) and isinstance(item.get("name"), str):
                names.add(item["name"])
    return names


def _split_paragraphs(text: str) -> list[str]:
    paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
    if paragraphs:
        return paragraphs
    return [text.strip()] if text.strip() else []


def _chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[str]:
    text = text.strip()
    if not text:
        return []

    paragraphs = _split_paragraphs(text)
    if len(text) <= chunk_size and len(paragraphs) <= 1:
        return [text]

    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current.strip())
            tail = current[-overlap:].strip() if overlap > 0 else ""
            current = f"{tail}\n\n{paragraph}".strip() if tail else paragraph
        else:
            start = 0
            while start < len(paragraph):
                end = min(start + chunk_size, len(paragraph))
                piece = paragraph[start:end].strip()
                if piece:
                    chunks.append(piece)
                if end >= len(paragraph):
                    current = ""
                    break
                start = max(end - overlap, start + 1)

    if current.strip():
        chunks.append(current.strip())

    return [chunk for chunk in chunks if chunk]


async def _call_parser(runtime, path: str) -> Any:
    parser_tools = _normalize_tool_names(await runtime.document_parser.list_tools())

    if "parse_file" in parser_tools:
        return await runtime.document_parser.call_tool("parse_file", {"path": path})
    if "parse" in parser_tools:
        return await runtime.document_parser.call_tool("parse", {"path": path})
    if "partition" in parser_tools:
        return await runtime.document_parser.call_tool("partition", {"path": path})

    raise RuntimeError("Document parser MCP does not expose a parse tool")


async def ingest_file(path: str, runtime) -> IngestionResult:
    """
    Ingestion flow: filesystem -> document parser -> deterministic chunks -> VelociRAG add_document.
    """
    await runtime.connect()
    await runtime.connect_ingestion()

    root = getattr(runtime, "filesystem_root", ".")
    resolved_path = str(_ensure_allowed_path(Path(path), root))
    filename = Path(resolved_path).name

    fs_result = await runtime.filesystem.call_tool("read_file", {"path": resolved_path})
    if safe_getattr(fs_result, "isError", False):
        raise RuntimeError(f"Filesystem MCP failed to read file: {resolved_path}")

    parser_result = await _call_parser(runtime, resolved_path)
    if safe_getattr(parser_result, "isError", False):
        raise RuntimeError(f"Document parser MCP call failed for file: {resolved_path}")

    parsed_payload = _extract_structured_payload(parser_result)
    normalized_text, parser_metadata = _normalize_parsed_text(parsed_payload)
    chunks = _chunk_text(normalized_text)
    if not chunks:
        raise RuntimeError(f"Document parser returned empty content for file: {resolved_path}")

    file_hash = hashlib.sha1(resolved_path.encode("utf-8")).hexdigest()[:12]
    chunk_ids: list[str] = []

    for idx, chunk_text in enumerate(chunks):
        chunk_id = f"{file_hash}-chunk-{idx:04d}"
        chunk_metadata: dict[str, Any] = {
            "source_path": resolved_path,
            "file_name": filename,
            "file_type": Path(filename).suffix.lower().lstrip("."),
            "parser_used": "document_parser",
            "chunk_index": idx,
            "total_chunks": len(chunks),
        }
        chunk_metadata.update(parser_metadata)

        add_result = await runtime.velocirag.call_tool(
            "add_document",
            {
                "content": chunk_text,
                "source": resolved_path,
                "doc_id": chunk_id,
                "metadata": chunk_metadata,
            },
        )
        if safe_getattr(add_result, "isError", False):
            raise RuntimeError(f"VelociRAG indexing failed for file: {resolved_path}")

        chunk_ids.append(chunk_id)

    result_metadata: dict[str, Any] = {
        "source_path": resolved_path,
        "file_name": filename,
        "file_type": Path(filename).suffix.lower().lstrip("."),
        "parser_used": "document_parser",
        "chunk_count": len(chunks),
        "chunk_ids": chunk_ids,
    }
    result_metadata.update(parser_metadata)

    return IngestionResult(
        path=resolved_path,
        doc_id=chunk_ids[0],
        status="ingested",
        metadata=result_metadata,
    )


async def ingest_directory(path: str, runtime) -> list[IngestionResult]:
    """
    Deterministic ingestion for all files in a directory tree.
    """
    root = Path(path).resolve()
    if not root.exists() or not root.is_dir():
        raise RuntimeError(f"Ingestion directory does not exist: {root}")

    files = sorted([item for item in root.rglob("*") if item.is_file()], key=lambda x: str(x))
    results: list[IngestionResult] = []
    for file_path in files:
        result = await ingest_file(str(file_path), runtime)
        results.append(result)

    return results
