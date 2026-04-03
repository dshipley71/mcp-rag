import pytest


class MockParser:
    async def list_tools(self):
        return [{"name": "parse_file"}]

    async def call_tool(self, name, args):
        if args.get("fail"):
            raise RuntimeError("parse failure")
        return {"text": "parsed content"}


class MockRuntime:
    def __init__(self):
        self.document_parser = MockParser()


@pytest.mark.asyncio
async def test_parser_success():
    runtime = MockRuntime()

    result = await runtime.document_parser.call_tool(
        "parse_file", {"path": "test.txt"}
    )

    assert "parsed content" in result["text"]


@pytest.mark.asyncio
async def test_parser_failure():
    runtime = MockRuntime()

    with pytest.raises(RuntimeError):
        await runtime.document_parser.call_tool(
            "parse_file", {"fail": True}
        )from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.ingestion import ingest_file


class FakeClient:
    def __init__(self, tools, tool_outputs):
        self._tools = tools
        self._tool_outputs = tool_outputs

    async def list_tools(self):
        return self._tools

    async def call_tool(self, name, arguments):
        return self._tool_outputs[name]


class FakeRuntime:
    def __init__(self, parser_payload, add_payload=None, filesystem_root="."):
        self.filesystem_root = filesystem_root
        self.filesystem = FakeClient(
            tools=["read_file"],
            tool_outputs={
                "read_file": SimpleNamespace(isError=False),
            },
        )
        self.document_parser = FakeClient(
            tools=["parse"],
            tool_outputs={
                "parse": parser_payload,
            },
        )
        self.velocirag = FakeClient(
            tools=["add_document"],
            tool_outputs={
                "add_document": add_payload or SimpleNamespace(structuredContent={"doc_id": "doc-123"}, isError=False),
            },
        )

    async def connect(self):
        return None

    async def connect_ingestion(self):
        return None


@pytest.mark.asyncio
async def test_ingest_file_indexes_parsed_content(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("hello", encoding="utf-8")

    parser_payload = SimpleNamespace(
        structuredContent={
            "text": "Normalized parser output",
            "metadata": {"parser": "unstructured"},
        },
        isError=False,
    )

    result = await ingest_file(
        str(file_path),
        FakeRuntime(parser_payload, filesystem_root=str(tmp_path)),
    )

    assert result.status == "ingested"
    assert result.doc_id == "doc-123"
    assert result.metadata["original_path"].endswith("sample.txt")
    assert result.metadata["file_name"] == "sample.txt"


@pytest.mark.asyncio
async def test_ingest_file_fails_closed_on_parser_error(tmp_path):
    file_path = tmp_path / "broken.pdf"
    file_path.write_text("data", encoding="utf-8")

    parser_payload = SimpleNamespace(
        structuredContent={"error": "unsupported format"},
        isError=False,
    )

    with pytest.raises(RuntimeError, match="Document parser failed"):
        await ingest_file(
            str(file_path),
            FakeRuntime(parser_payload, filesystem_root=str(tmp_path)),
        )


@pytest.mark.asyncio
async def test_ingest_file_rejects_path_outside_allowed_root(tmp_path):
    allowed_root = tmp_path / "allowed"
    disallowed_root = tmp_path / "disallowed"
    allowed_root.mkdir()
    disallowed_root.mkdir()

    outside_file = disallowed_root / "outside.txt"
    outside_file.write_text("content", encoding="utf-8")

    parser_payload = SimpleNamespace(
        structuredContent={"text": "Should not be reached"},
        isError=False,
    )

    runtime = FakeRuntime(
        parser_payload=parser_payload,
        filesystem_root=str(allowed_root),
    )

    with pytest.raises(RuntimeError, match="outside allowed ingestion root"):
        await ingest_file(str(outside_file), runtime)
