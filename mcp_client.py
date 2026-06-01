"""MCP client utilities for talking to the local mcp-server-openfda subprocess.

This module wraps Anthropic's MCP Python SDK client-side helpers so the rest
of the app can ignore subprocess management and JSON-RPC framing. It exposes:

    - mcp_session(): async context manager that spawns the MCP server,
      runs the MCP handshake, and yields an initialized ClientSession.
    - mcp_tool_to_anthropic(): convert one MCP Tool object into the dict
      shape the Anthropic Messages API expects.

The server is spawned fresh for each chat turn (see chat.py). That keeps
the lifecycle simple at the cost of a one-time startup per turn.
"""

from __future__ import annotations

import shutil
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def _resolve_server_command() -> str:
    """Return the absolute path to the mcp-server-openfda console script.

    The mcp-server-openfda package installs a `mcp-server-openfda` console
    script into the active venv's bin/ directory when pip-installed. We use
    shutil.which to resolve the full path so the subprocess spawn doesn't
    depend on PATH being set up correctly at the moment of spawn — which
    matters on Streamlit Community Cloud and other hosted environments
    where the subprocess inherits a stripped-down PATH.
    """
    path = shutil.which("mcp-server-openfda")
    if path is None:
        raise RuntimeError(
            "Cannot find the `mcp-server-openfda` console script on PATH. "
            "Install it with: "
            "uv add 'mcp-server-openfda @ git+https://github.com/Janiumang/mcp-server-openfda.git'"
        )
    return path


@asynccontextmanager
async def mcp_session() -> AsyncIterator[ClientSession]:
    """Spawn the mcp-server-openfda subprocess and yield an initialized session.

    Use as:
        async with mcp_session() as session:
            tools = await session.list_tools()
            result = await session.call_tool("ping", arguments={})

    The subprocess is torn down cleanly when the context exits (or when
    the outer task is cancelled). Each invocation creates a fresh process,
    so this is suitable for per-turn use, not for hot paths.
    """
    server_params = StdioServerParameters(
        command=_resolve_server_command(),
        args=[],
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


def mcp_tool_to_anthropic(tool: Any) -> dict[str, Any]:
    """Convert one MCP Tool object into Anthropic tool-use schema.

    Differences between the two schemas:
        - MCP uses `inputSchema` (camelCase); Anthropic uses `input_schema`.
        - MCP returns Tool objects with .name/.description/.inputSchema
          attributes; Anthropic wants a plain dict.

    The JSON Schema for parameters is otherwise compatible enough that
    Claude can use it directly.
    """
    return {
        "name": tool.name,
        "description": tool.description or "",
        "input_schema": tool.inputSchema,
    }
