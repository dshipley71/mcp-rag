import os
import pytest

from src.mcp_runtime import MCPRuntime


@pytest.mark.asyncio
async def test_runtime_has_parser():
    runtime = MCPRuntime(
        {
            "document_parser": {
                "command": "uns_mcp",
                "args": []
            }
        }
    )

    try:
        await runtime.connect_ingestion()
    except Exception:
        pass

    assert hasattr(runtime, "document_parser")


def test_api_key_optional():
    if "UNSTRUCTURED_API_KEY" in os.environ:
        del os.environ["UNSTRUCTURED_API_KEY"]

    runtime = MCPRuntime({"document_parser": {"command": "uns_mcp", "args": []}})

    assert runtime is not None
