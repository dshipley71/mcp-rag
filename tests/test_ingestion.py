from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.ingestion import ingest_file


class FakeFilesystem:
    async def call_tool(self, name, args):
        assert name == "read_file"
        assert args["path"].endswith("sample.md")
        return SimpleNamespace(isError=False)


class FakeParser:
    def __init__(self, payload=None, should_fail=False):
        self.payload = payload or {"text": "default parser text"}
        self.should_fail = should_fail

    async def list_tools(self):
        return ["parse_file"]

    async def call_tool(self, name, args):
        assert name == "parse_file"
        if self.should_fail:
            raise RuntimeError("parse failure")
        return self.payload


class FakeVelociRAG:
    def __init__(self):
        self.calls = []

    async def call_tool(self, name, args):
        assert name == "add_document"
        self.calls.append(args)
        return SimpleNamespace(isError=False)


class FakeRuntime:
    def __init__(self, parser):
        self.filesystem_root = "/tmp"
        self.docs_dir = "/tmp"
        self.filesystem = FakeFilesystem()
        self.document_parser = parser
        self.velocirag = FakeVelociRAG()

    async def connect(self):
        return None

    async def connect_ingestion(self):
        return None


@pytest.mark.asyncio
async def test_ingestion_chunks_and_adds_metadata(tmp_path):
    sample = tmp_path / "sample.md"
    sample.write_text("placeholder", encoding="utf-8")

    parser_payload = {
        "text": "Paragraph one with enough text to keep.\n\nParagraph two with more text to force chunking.",
        "metadata": {"mime_type": "text/markdown"},
    }
    runtime = FakeRuntime(FakeParser(payload=parser_payload))
    runtime.filesystem_root = str(tmp_path)
    runtime.docs_dir = str(tmp_path)

    result = await ingest_file(str(sample), runtime)

    assert result.status == "ingested"
    assert result.metadata["chunk_count"] == 1
    assert len(runtime.velocirag.calls) == 1
    first_call = runtime.velocirag.calls[0]
    assert first_call["metadata"]["source_path"] == str(sample.resolve())
    assert first_call["metadata"]["file_name"] == "sample.md"
    assert first_call["metadata"]["chunk_index"] == 0
    assert first_call["metadata"]["total_chunks"] == 1
    assert first_call["metadata"]["mime_type"] == "text/markdown"


@pytest.mark.asyncio
async def test_ingestion_multiple_chunks(tmp_path):
    sample = tmp_path / "sample.md"
    sample.write_text("placeholder", encoding="utf-8")

    long_text = (
        ("A" * 900)
        + "\n\n"
        + ("B" * 900)
        + "\n\n"
        + ("C" * 900)
    )
    runtime = FakeRuntime(FakeParser(payload={"text": long_text}))
    runtime.filesystem_root = str(tmp_path)
    runtime.docs_dir = str(tmp_path)

    result = await ingest_file(str(sample), runtime)

    assert result.metadata["chunk_count"] >= 2
    assert len(runtime.velocirag.calls) == result.metadata["chunk_count"]
    assert runtime.velocirag.calls[0]["doc_id"].endswith("chunk-0000")
    assert runtime.velocirag.calls[-1]["metadata"]["total_chunks"] == result.metadata["chunk_count"]


@pytest.mark.asyncio
async def test_ingestion_fails_closed_on_parser_error(tmp_path):
    sample = tmp_path / "sample.md"
    sample.write_text("placeholder", encoding="utf-8")

    runtime = FakeRuntime(FakeParser(should_fail=True))
    runtime.filesystem_root = str(tmp_path)
    runtime.docs_dir = str(tmp_path)

    with pytest.raises(RuntimeError, match="parse failure"):
        await ingest_file(str(sample), runtime)
