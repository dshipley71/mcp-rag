import os
from typing import Optional

from src.mcp_client import MCPToolClient


class MCPRuntime:
    def __init__(self, catalog: dict = None, docs_dir: str = None):
        """
        Backward-compatible constructor.

        Supports:
        - MCPRuntime(catalog=...)
        - MCPRuntime(docs_dir=...)  (legacy notebook usage)
        """

        # Backward compatibility mode
        if catalog is None:
            catalog = self._build_default_catalog(docs_dir)

        self.catalog = catalog

        self.retrieval = None
        self.filesystem = None
        self.document_parser = None

    def _build_default_catalog(self, docs_dir: str):
        """
        Minimal default catalog for notebook compatibility.
        Keeps behavior deterministic.
        """

        if docs_dir is None:
            docs_dir = "./docs"

        return {
            "filesystem": {
                "command": "filesystem-mcp",
                "args": [docs_dir],
            },
            "document_parser": {
                "command": "unstructured-mcp",
                "args": [],
            },
            "retrieval": {
                "command": "velocirag-mcp",
                "args": [],
            },
        }

    async def connect(self):
        """
        Existing query-time connections ONLY.
        DO NOT modify behavior.
        """
        self.retrieval = MCPToolClient(self.catalog["retrieval"])
        await self.retrieval.connect()

        self.filesystem = MCPToolClient(self.catalog["filesystem"])
        await self.filesystem.connect()

    async def connect_ingestion(self):
        """
        NEW: Connect parser + validate tools.
        Deterministic. Fail-fast.
        """

        parser_cfg = self.catalog.get("document_parser")
        if not parser_cfg:
            raise RuntimeError("document_parser not defined in catalog")

        # Optional API key pass-through
        env = dict(os.environ)
        api_key = env.get("UNSTRUCTURED_API_KEY")

        if api_key:
            parser_cfg = dict(parser_cfg)
            parser_cfg["env"] = parser_cfg.get("env", {})
            parser_cfg["env"]["UNSTRUCTURED_API_KEY"] = api_key

        self.document_parser = MCPToolClient(parser_cfg)
        await self.document_parser.connect()

        # Validate tools explicitly
        tools = await self.document_parser.list_tools()
        tool_names = {t["name"] for t in tools}

        valid_tools = {"parse_file", "parse", "partition"}
        if not tool_names.intersection(valid_tools):
            raise RuntimeError(
                f"Unstructured MCP missing required parse tool. Found: {tool_names}"
            )

    async def close(self):
        if self.retrieval:
            await self.retrieval.close()

        if self.filesystem:
            await self.filesystem.close()

        if self.document_parser:
            await self.document_parser.close()
