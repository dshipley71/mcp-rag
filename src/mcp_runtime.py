from __future__ import annotations

import os
from typing import Any, Optional


class MCPRuntime:
    def __init__(self, catalog: Optional[dict[str, Any]] = None, docs_dir: Optional[str] = None):
        """
        Backward-compatible runtime.

        Supports:
        - MCPRuntime(catalog=...) using an executable runtime catalog
        - MCPRuntime(docs_dir=...) for legacy notebook usage
        """
        if catalog is None:
            catalog = self._build_default_catalog(docs_dir)

        self.catalog = catalog
        self.filesystem_root = str(catalog.get("filesystem_root") or docs_dir or "./docs")
        self.docs_dir = self.filesystem_root

        self.filesystem = None
        self.document_parser = None
        self.retrieval = None
        self.velocirag = None
        self.llm_generate = None

    def _build_default_catalog(self, docs_dir: Optional[str]) -> dict[str, Any]:
        if docs_dir is None:
            docs_dir = "./docs"

        return {
            "filesystem_root": docs_dir,
            "filesystem": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", docs_dir],
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

    def _build_client(self, cfg: dict[str, Any]):
        from src.mcp_client import MCPToolClient
        if not isinstance(cfg, dict):
            raise RuntimeError("MCP server config must be a dictionary")

        command = cfg.get("command")
        args = cfg.get("args", [])
        env = cfg.get("env", {})

        if not isinstance(command, str) or not command.strip():
            raise RuntimeError("MCP server config missing command")
        if not isinstance(args, list):
            raise RuntimeError("MCP server config args must be a list")
        if not isinstance(env, dict):
            raise RuntimeError("MCP server config env must be a dictionary")

        return MCPToolClient(command=command, args=args, env=env)

    async def connect(self):
        """
        Query-time connections only.
        """
        if self.retrieval is None:
            retrieval_cfg = self.catalog.get("retrieval")
            if retrieval_cfg is None:
                raise RuntimeError("retrieval not defined in catalog")
            self.retrieval = self._build_client(retrieval_cfg)
            await self.retrieval.connect()
            self.velocirag = self.retrieval

        if self.filesystem is None:
            filesystem_cfg = self.catalog.get("filesystem")
            if filesystem_cfg is None:
                raise RuntimeError("filesystem not defined in catalog")
            self.filesystem = self._build_client(filesystem_cfg)
            await self.filesystem.connect()

    async def connect_ingestion(self):
        """
        Connect parser + validate explicit parse tools.
        """
        if self.document_parser is not None:
            return

        parser_cfg = self.catalog.get("document_parser")
        if parser_cfg is None:
            raise RuntimeError("document_parser not defined in catalog")

        cfg = dict(parser_cfg)
        env = dict(cfg.get("env", {}))

        api_key = os.environ.get("UNSTRUCTURED_API_KEY")
        if api_key:
            env["UNSTRUCTURED_API_KEY"] = api_key

        if env:
            cfg["env"] = env

        self.document_parser = self._build_client(cfg)
        await self.document_parser.connect()

        tool_names = set(await self.document_parser.list_tools())
        valid_tools = {"parse_file", "parse", "partition"}
        if not tool_names.intersection(valid_tools):
            raise RuntimeError(
                f"Unstructured MCP missing required parse tool. Found: {sorted(tool_names)}"
            )

    async def close(self):
        for client_name in ("retrieval", "filesystem", "document_parser", "llm_generate"):
            client = getattr(self, client_name, None)
            if client is not None:
                await client.close()
                setattr(self, client_name, None)

        self.velocirag = None
