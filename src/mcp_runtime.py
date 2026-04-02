from __future__ import annotations

import os
from pathlib import Path

from src.mcp_client import MCPToolClient


class MCPRuntime:
    """
    Runtime holder for external integrations.

    v2 keeps retrieval on direct MCP stdio and routes answer generation through
    the Ollama MCP Bridge HTTP API. The bridge must itself be configured to use
    Ollama Cloud.
    """

    def __init__(
        self,
        db_dir: str = "./.rag/velocirag",
        filesystem_root: str = ".",
        filesystem_command: str = "mcp-server-filesystem",
        filesystem_args: list[str] | None = None,
        document_parser_command: str = "uns-mcp",
        document_parser_args: list[str] | None = None,
    ) -> None:
        self.db_dir = str(Path(db_dir).resolve())
        self.filesystem_root = str(Path(filesystem_root).resolve())

        self.velocirag = MCPToolClient(
            command="velocirag",
            args=["mcp", "--db", self.db_dir],
            env={"VELOCIRAG_DB": self.db_dir},
        )

        self.ollama_bridge_url = os.environ.get("OLLAMA_BRIDGE_URL", "http://127.0.0.1:8000")
        self.ollama_model = os.environ.get("OLLAMA_CLOUD_MODEL", "gpt-oss:120b")
        self.ollama_mode = os.environ.get("OLLAMA_MODE", "cloud_only")

    async def connect(self) -> None:
        if self._velocirag_connected:
            return

        await self.velocirag.connect()
        tools = await self.velocirag.list_tools()
        required = {"search", "health"}
        missing = required - set(tools)
        if missing:
            raise RuntimeError(f"VelociRAG missing required MCP tools: {sorted(missing)}")

        if self.ollama_mode != "cloud_only":
            raise RuntimeError(
                "OLLAMA_MODE must be 'cloud_only' for this project. "
                "Configure the Ollama MCP Bridge to use Ollama Cloud."
            )

    async def close(self) -> None:
        if self._ingestion_connected:
            await self.document_parser.close()
            await self.filesystem.close()
            self._ingestion_connected = False

        if self._velocirag_connected:
            await self.velocirag.close()
            self._velocirag_connected = False
