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
        )
