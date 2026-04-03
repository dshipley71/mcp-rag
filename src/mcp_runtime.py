import os
from typing import Any, Optional

from src.config import load_catalog


class MCPRuntime:
    def __init__(self, catalog: dict = None, docs_dir: str = None):
        """
        Supports:
        - MCPRuntime(catalog=...)
        - MCPRuntime(docs_dir=...) for legacy notebook usage
        """
        if catalog is None:
            if docs_dir:
                os.environ.setdefault("MCP_FILESYSTEM_ROOT", docs_dir)
                os.environ.setdefault("DOCS_DIR", docs_dir)
            catalog = load_catalog("mcp_catalog.yaml")

        self.catalog = catalog
        self.filesystem_root = catalog.get("filesystem_root", docs_dir or "./docs")
        self.docs_dir = catalog.get("docs_dir", self.filesystem_root)

        self.retrieval: Optional[Any] = None
        self.velocirag: Optional[Any] = None
        self.filesystem: Optional[Any] = None
        self.document_parser: Optional[Any] = None

    def _build_client(self, cfg: dict):
        if not isinstance(cfg, dict):
            raise RuntimeError("Invalid MCP client config")

        command = cfg.get("command")
        if not isinstance(command, str) or not command.strip():
            raise RuntimeError("MCP client config missing command")

        args = cfg.get("args", [])
        if not isinstance(args, list):
            raise RuntimeError("MCP client args must be a list")

        env = cfg.get("env", {})
        if not isinstance(env, dict):
            raise RuntimeError("MCP client env must be a dictionary")

        from src.mcp_client import MCPToolClient

        merged_env = {**os.environ, **{str(k): str(v) for k, v in env.items()}}
        return MCPToolClient(command=command, args=[str(a) for a in args], env=merged_env)

    async def connect(self):
        retrieval_cfg = self.catalog.get("retrieval")
        if retrieval_cfg is None:
            raise RuntimeError("retrieval not defined in catalog")
        self.retrieval = self._build_client(retrieval_cfg)
        await self.retrieval.connect()
        self.velocirag = self.retrieval

        filesystem_cfg = self.catalog.get("filesystem")
        if filesystem_cfg is None:
            raise RuntimeError("filesystem not defined in catalog")
        self.filesystem = self._build_client(filesystem_cfg)
        await self.filesystem.connect()

    async def connect_ingestion(self):
        parser_cfg = self.catalog.get("document_parser")
        if not parser_cfg:
            raise RuntimeError("document_parser not defined in catalog")

        merged_cfg = dict(parser_cfg)
        merged_env = dict(parser_cfg.get("env", {}))
        api_key = os.environ.get("UNSTRUCTURED_API_KEY")
        if api_key:
            merged_env["UNSTRUCTURED_API_KEY"] = api_key
        merged_cfg["env"] = merged_env

        self.document_parser = self._build_client(merged_cfg)
        await self.document_parser.connect()

        tools = await self.document_parser.list_tools()
        tool_names = set(tools)
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
