from __future__ import annotations

from src.mcp_client import MCPToolClient


class MCPRuntime:
    """
    Runtime holder for MCP server connections.
    Filesystem is intentionally deferred in this pass.
    """

    def __init__(self, db_dir: str = "./.rag/velocirag") -> None:
        self.db_dir = db_dir
        self.velocirag = MCPToolClient(
            command="velocirag",
            args=["mcp", "--db", self.db_dir],
            env={"VELOCIRAG_DB": self.db_dir},
        )

    async def connect(self) -> None:
        await self.velocirag.connect()

        tools = await self.velocirag.list_tools()
        required = {"search", "health"}
        missing = required - set(tools)
        if missing:
            raise RuntimeError(f"VelociRAG missing required MCP tools: {sorted(missing)}")

    async def close(self) -> None:
        await self.velocirag.close()
