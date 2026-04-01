from __future__ import annotations

from contextlib import AsyncExitStack
from typing import Any, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPToolClient:
    """
    Minimal stdio MCP client wrapper.
    Keeps a single warm connection to one local MCP server process.
    """

    def __init__(
        self,
        command: str,
        args: list[str],
        env: Optional[dict[str, str]] = None,
    ) -> None:
        self.command = command
        self.args = args
        self.env = env or {}
        self._stack = AsyncExitStack()
        self.session: Optional[ClientSession] = None

    async def connect(self) -> None:
        params = StdioServerParameters(
            command=self.command,
            args=self.args,
            env=self.env,
        )
        read_stream, write_stream = await self._stack.enter_async_context(stdio_client(params))
        self.session = await self._stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await self.session.initialize()

    async def list_tools(self) -> list[str]:
        if self.session is None:
            raise RuntimeError("MCP client is not connected")

        result = await self.session.list_tools()
        return [tool.name for tool in result.tools]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        if self.session is None:
            raise RuntimeError("MCP client is not connected")

        return await self.session.call_tool(name, arguments)

    async def close(self) -> None:
        await self._stack.aclose()
        self.session = None
