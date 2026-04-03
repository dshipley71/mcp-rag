import pytest

from src.config import load_catalog
from src.mcp_runtime import MCPRuntime


class FakeClient:
    def __init__(self, command, args, env=None):
        self.command = command
        self.args = args
        self.env = env or {}
        self.connected = False

    async def connect(self):
        self.connected = True

    async def list_tools(self):
        return ["parse_file"]

    async def close(self):
        self.connected = False


@pytest.mark.asyncio
async def test_runtime_has_parser(monkeypatch, tmp_path):
    monkeypatch.setenv("MCP_FILESYSTEM_ROOT", str(tmp_path))
    monkeypatch.setattr(MCPRuntime, "_build_client", lambda self, cfg: FakeClient(cfg["command"], cfg.get("args", []), cfg.get("env", {})))

    catalog = load_catalog("mcp_catalog.yaml")
    runtime = MCPRuntime(catalog=catalog)

    await runtime.connect_ingestion()

    assert runtime.document_parser is not None
    assert runtime.document_parser.command


def test_api_key_optional(monkeypatch):
    monkeypatch.delenv("UNSTRUCTURED_API_KEY", raising=False)

    runtime = MCPRuntime(
        {
            "filesystem_root": "./docs",
            "document_parser": {"command": "unstructured-mcp", "args": []},
        }
    )

    assert runtime is not None
    assert runtime.catalog["document_parser"]["command"] == "unstructured-mcp"


@pytest.mark.asyncio
async def test_runtime_passes_optional_api_key(monkeypatch, tmp_path):
    monkeypatch.setenv("MCP_FILESYSTEM_ROOT", str(tmp_path))
    monkeypatch.setenv("UNSTRUCTURED_API_KEY", "test-key")
    monkeypatch.setattr(MCPRuntime, "_build_client", lambda self, cfg: FakeClient(cfg["command"], cfg.get("args", []), cfg.get("env", {})))

    catalog = load_catalog("mcp_catalog.yaml")
    runtime = MCPRuntime(catalog=catalog)

    await runtime.connect_ingestion()

    assert runtime.document_parser is not None
    assert runtime.document_parser.env["UNSTRUCTURED_API_KEY"] == "test-key"
