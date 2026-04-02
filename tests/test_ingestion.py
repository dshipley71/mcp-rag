from __future__ import annotations

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
    def __init__(self, parser_payload, add_payload=None):
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

    result = await ingest_file(str(file_path), FakeRuntime(parser_payload))

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
        await ingest_file(str(file_path), FakeRuntime(parser_payload))
