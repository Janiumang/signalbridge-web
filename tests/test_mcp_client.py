"""MCP client: openFDA key reaches the server subprocess; nothing else leaks."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import mcp_client


def test_env_forwards_only_safe_vars_and_openfda_key(monkeypatch):
    captured = {}

    @asynccontextmanager
    async def fake_stdio(params):
        captured["env"] = params.env
        raise RuntimeError("stop")
        yield  # pragma: no cover

    monkeypatch.setattr(mcp_client, "stdio_client", fake_stdio)
    monkeypatch.setattr(mcp_client, "_resolve_server_command", lambda: "/bin/true")
    monkeypatch.setenv("OPENFDA_API_KEY", "fda-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")

    async def go():
        try:
            async with mcp_client.mcp_session():
                pass
        except RuntimeError:
            pass

    asyncio.run(go())
    assert captured["env"]["OPENFDA_API_KEY"] == "fda-key"
    assert "ANTHROPIC_API_KEY" not in captured["env"]
