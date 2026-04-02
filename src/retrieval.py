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
