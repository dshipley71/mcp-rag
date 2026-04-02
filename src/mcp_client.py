from __future__ import annotations

from contextlib import AsyncExitStack
from typing import Any, Optional
import io
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPToolClient:
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
        self._errlog = None

    def _open_errlog(self):
        """
        Prefer normal stderr when it supports fileno().
        Fall back to os.devnull in notebook environments like Colab.
        """
        try:
            sys.stderr.fileno()
            return sys.stderr
        except (io.UnsupportedOperation, AttributeError):
            return open(os.devnull, "w", encoding="utf-8")

    async def connect(self) -> None:
        params = StdioServerParameters(
            command=self.command,
            args=self.args,
            env=self.env,
        )

        self._errlog = self._open_errlog()

        read_stream, write_stream = await self._stack.enter_async_context(
            stdio_client(params, errlog=self._errlog)
        )
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
        try:
            await self._stack.aclose()
        except BaseException:
            # Colab/Jupyter async subprocess teardown can raise
            # CancelledError / cancel-scope runtime errors on shutdown.
            # Best-effort cleanup is enough here.
            pass
    
        self.session = None
    
        if self._errlog is not None and self._errlog is not sys.stderr:
            try:
                self._errlog.close()
            except BaseException:
                pass
            self._errlog = None
                self._errlog = None
