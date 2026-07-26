"""MCP (Model Context Protocol) client adapter — turns whatever tools a
remote MCP server advertises into `kel.agents.Tool`s in one call, instead
of hand-writing a custom integration adapter per server. Requires
`pip install kel[mcp]` (the official `mcp` SDK).

MCP's client API is entirely async and expects one long-lived session per
connection (the `mcp` server process, for the stdio transport, stays
alive across calls). `kel.agents.Tool.fn` is a plain sync callable, so
`MCPToolset` runs the session's event loop on a background thread for
its lifetime and bridges each call across via
`asyncio.run_coroutine_threadsafe`, rather than reconnecting (and
respawning the server subprocess) on every single tool call.

Same DI seam as every other adapter in kel: pass `session=` (anything
with sync `list_tools()`/`call_tool()` methods) to bypass the real
connection entirely for testing, same shape as the provider adapters'
`client=`.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

from kel.agents.tool import Tool


class MCPToolset:
    """Connects to one MCP server and exposes every tool it advertises as
    a `kel.agents.Tool`. Use as a context manager, or call `.close()`
    when done, to shut the background session down cleanly."""

    def __init__(self, server_params: Any = None, *, session: Any = None):
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._client_cm: Any = None
        self._session_cm: Any = None

        if session is not None:
            self._session = session
            self._owns_connection = False
            return

        if server_params is None:
            raise ValueError("must provide either `server_params` or `session`")
        try:
            import mcp  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "The mcp package is required to use MCPToolset. Install it with `pip install kel[mcp]`."
            ) from exc

        self._owns_connection = True
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self._session = self._run(self._open_session(server_params))

    def _run(self, coro: Any) -> Any:
        assert self._loop is not None
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    async def _open_session(self, server_params: Any) -> Any:
        from mcp import ClientSession
        from mcp.client.stdio import stdio_client

        self._client_cm = stdio_client(server_params)
        read, write = await self._client_cm.__aenter__()
        self._session_cm = ClientSession(read, write)
        session = await self._session_cm.__aenter__()
        await session.initialize()
        return session

    def list_tools(self) -> list[Any]:
        if self._owns_connection:
            return self._run(self._session.list_tools()).tools
        return self._session.list_tools()

    def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        if self._owns_connection:
            result = self._run(self._session.call_tool(name, arguments))
            return "\n".join(part.text for part in result.content if hasattr(part, "text"))
        return self._session.call_tool(name, arguments)

    def as_kel_tools(self) -> list[Tool]:
        def make_fn(tool_name: str):
            def fn(tool_input: dict[str, Any]) -> str:
                return self.call_tool(tool_name, tool_input)

            return fn

        return [
            Tool(
                name=mcp_tool.name,
                description=mcp_tool.description or "",
                input_schema=mcp_tool.inputSchema or {"type": "object"},
                fn=make_fn(mcp_tool.name),
            )
            for mcp_tool in self.list_tools()
        ]

    def close(self) -> None:
        if not self._owns_connection:
            return

        async def _close() -> None:
            if self._session_cm is not None:
                await self._session_cm.__aexit__(None, None, None)
            if self._client_cm is not None:
                await self._client_cm.__aexit__(None, None, None)

        try:
            self._run(_close())
        finally:
            assert self._loop is not None and self._thread is not None
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=2)

    def __enter__(self) -> MCPToolset:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def mcp_tools_from_server(server_params: Any) -> list[Tool]:
    """Connect to an MCP server and return its tools as `kel.agents.Tool`s
    in one call. Keeps the connection alive for the process's lifetime
    (no explicit close) — use `MCPToolset` directly if you need to close
    the connection deterministically."""
    return MCPToolset(server_params).as_kel_tools()
