from __future__ import annotations

from pathlib import Path

from src.mcp_client import MCPToolClient


class MCPRuntime:
    """
    Runtime for MCP servers.

    v1:
    - VelociRAG (retrieval)
    - Filesystem (document access)

    Unstructured will be added later.
    """

    def __init__(self, db_dir: str, docs_dir: str) -> None:
        self.db_dir = str(Path(db_dir).resolve())
        self.docs_dir = str(Path(docs_dir).resolve())

        # -------------------------
        # VelociRAG MCP
        # -------------------------
        self.velocirag = MCPToolClient(
            command="velocirag",
            args=["mcp", "--db", self.db_dir],
            env={"VELOCIRAG_DB": self.db_dir},
        )

        # -------------------------
        # Filesystem MCP
        # -------------------------
        self.filesystem = MCPToolClient(
            command="npx",
            args=[
                "-y",
                "@modelcontextprotocol/server-filesystem",
                self.docs_dir,
            ],
        )

    async def connect(self) -> None:
        await self.velocirag.connect()
        await self.filesystem.connect()

        # Validate tools
        vr_tools = set(await self.velocirag.list_tools())
        fs_tools = set(await self.filesystem.list_tools())

        required_vr = {"search", "health"}
        required_fs = {
            "read_text_file",
            "read_multiple_files",
            "search_files",
            "get_file_info",
            "list_allowed_directories",
        }

        missing_vr = required_vr - vr_tools
        missing_fs = required_fs - fs_tools

        if missing_vr:
            raise RuntimeError(f"VelociRAG missing tools: {missing_vr}")

        if missing_fs:
            raise RuntimeError(f"Filesystem missing tools: {missing_fs}")

    async def close(self) -> None:
        await self.velocirag.close()
        await self.filesystem.close()
