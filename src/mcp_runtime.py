from __future__ import annotations

from pathlib import Path

from src.mcp_client import MCPToolClient


class MCPRuntime:
    """
    Runtime holder for MCP server connections.

    Query-time dependencies are connected by `connect()`.
    Ingestion dependencies are connected explicitly by `connect_ingestion()`.
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

        fs_args = filesystem_args if filesystem_args is not None else [self.filesystem_root]
        self.filesystem = MCPToolClient(
            command=filesystem_command,
            args=fs_args,
        )

        parser_args = document_parser_args if document_parser_args is not None else []
        self.document_parser = MCPToolClient(
            command=document_parser_command,
            args=parser_args,
        )

        self._velocirag_connected = False
        self._ingestion_connected = False

    async def connect(self) -> None:
        if self._velocirag_connected:
            return

        await self.velocirag.connect()
        tools = await self.velocirag.list_tools()
        required = {"search", "health"}
        missing = required - set(tools)
        if missing:
            raise RuntimeError(f"VelociRAG missing required MCP tools: {sorted(missing)}")

        self._velocirag_connected = True

    async def connect_ingestion(self) -> None:
        """
        Explicitly connect ingestion MCP dependencies.
        """
        if self._ingestion_connected:
            return

        await self.filesystem.connect()
        await self.document_parser.connect()

        fs_tools = set(await self.filesystem.list_tools())
        if "read_file" not in fs_tools:
            raise RuntimeError("Filesystem MCP missing required tool: read_file")

        parser_tools = set(await self.document_parser.list_tools())
        if not ({"parse", "parse_file", "partition"} & parser_tools):
            raise RuntimeError("Document parser MCP missing parse tool (expected one of: parse, parse_file, partition)")

        self._ingestion_connected = True

    async def close(self) -> None:
        if self._ingestion_connected:
            await self.document_parser.close()
            await self.filesystem.close()
            self._ingestion_connected = False

        if self._velocirag_connected:
            await self.velocirag.close()
            self._velocirag_connected = False
