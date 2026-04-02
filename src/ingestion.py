from __future__ import annotations

"""
Deterministic ingestion helpers for explicit setup-time indexing.
"""

from pathlib import Path
from typing import Any

from src.models import IngestionResult
from src.utils import safe_getattr, try_parse_json_text


def _ensure_allowed_path(path: Path, filesystem_root: str) -> Path:
    root = Path(filesystem_root).resolve()
    resolved = path.resolve()

    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"Path is outside allowed ingestion root: {resolved}") from exc

    return resolved


def _extract_structured_payload(tool_result: Any) -> Any:
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

    if isinstance(payload.get("text"), str) and payload["text"].strip():
        return payload["text"].strip(), payload.get("metadata", {}) if isinstance(payload.get("metadata"), dict) else {}

    if isinstance(payload.get("content"), str) and payload["content"].strip():
        return payload["content"].strip(), payload.get("metadata", {}) if isinstance(payload.get("metadata"), dict) else {}

    elements = payload.get("elements")
    if isinstance(elements, list):
        lines: list[str] = []
        file_type = None
        for element in elements:
            if not isinstance(element, dict):
                continue
            text = element.get("text")
            if isinstance(text, str) and text.strip():
                lines.append(text.strip())
            if file_type is None and isinstance(element.get("type"), str):
                file_type = element["type"]

        normalized = "\n".join(lines).strip()
        if normalized:
            metadata: dict[str, Any] = {}
            if file_type:
                metadata["file_type"] = file_type
            return normalized, metadata

    raise RuntimeError("Document parser returned no text content")


async def _call_parser(runtime, path: str) -> Any:
    parser_tools = await runtime.document_parser.list_tools()

    if "parse_file" in parser_tools:
        return await runtime.document_parser.call_tool("parse_file", {"path": path})
    if "parse" in parser_tools:
        return await runtime.document_parser.call_tool("parse", {"path": path})
    if "partition" in parser_tools:
        return await runtime.document_parser.call_tool("partition", {"path": path})

    raise RuntimeError("Document parser MCP does not expose a parse tool")


async def ingest_file(path: str, runtime) -> IngestionResult:
    """
    Ingestion flow: filesystem -> document parser -> VelociRAG add_document.
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

    metadata: dict[str, Any] = {
        "original_path": resolved_path,
        "file_name": filename,
        "file_type": Path(filename).suffix.lower().lstrip("."),
    }
    metadata.update(parser_metadata)

    add_result = await runtime.velocirag.call_tool(
        "add_document",
        {
            "content": normalized_text,
            "source": resolved_path,
            "metadata": metadata,
        },
    )
    if safe_getattr(add_result, "isError", False):
        raise RuntimeError(f"VelociRAG indexing failed for file: {resolved_path}")

    payload = _extract_structured_payload(add_result)
    doc_id = filename
    if isinstance(payload, dict) and isinstance(payload.get("doc_id"), str):
        doc_id = payload["doc_id"]

    return IngestionResult(
        path=resolved_path,
        doc_id=doc_id,
        status="ingested",
        metadata=metadata,
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
