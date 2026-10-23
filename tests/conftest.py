"""Shared fixtures for signalbridge-web tests.

No network: the Anthropic client and the MCP server are replaced by fakes,
so the suite runs offline and in CI.
"""
from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class Usage:
    input_tokens = 10
    output_tokens = 5
    cache_read_input_tokens = 0


def text(t: str) -> NS:
    return NS(type="text", text=t)


def tool_use(i: str, name: str, inp: dict | None = None) -> NS:
    return NS(type="tool_use", id=i, name=name, input=inp or {})


def reply(content: list, stop: str) -> NS:
    return NS(content=content, stop_reason=stop, usage=Usage())


class FakeSession:
    """Stands in for an MCP ClientSession."""

    def __init__(self, results: dict[str, str] | None = None):
        self.results = results or {}
        self.calls: list[tuple[str, dict]] = []

    async def list_tools(self):
        tools = [
            NS(name=n, description=f"{n} tool", inputSchema={"type": "object", "properties": {}})
            for n in ("ping", "count_adverse_events", "calculate_reporting_odds_ratio", "get_drug_label")
        ]
        return NS(tools=tools)

    async def call_tool(self, name, arguments=None):
        self.calls.append((name, arguments or {}))
        return NS(content=[NS(text=self.results.get(name, '{"ok": true}'))])


@pytest.fixture
def fake_llm(monkeypatch):
    """Install a scripted Anthropic client + fake MCP session into chat.py.

    Usage: llm = fake_llm([reply(...), reply(...)]); then call run_chat_turn.
    llm.requests holds every kwargs dict sent to messages.create.
    """
    import chat

    def install(script, results=None):
        state = NS(requests=[], gate_requests=[], gate_answer="yes", session=FakeSession(results))

        class Messages:
            async def create(self, **kw):
                if kw.get("model") == chat.GATE_MODEL and kw.get("max_tokens") == 3:
                    state.gate_requests.append(kw)
                    if state.gate_answer == "error":
                        raise RuntimeError("gate down")
                    return reply([text(state.gate_answer)], "end_turn")
                state.requests.append(kw)
                return script[len(state.requests) - 1]

        class Client:
            def __init__(self, api_key):
                self.messages = Messages()

        @asynccontextmanager
        async def fake_session():
            yield state.session

        monkeypatch.setattr(chat, "AsyncAnthropic", Client)
        monkeypatch.setattr(chat, "mcp_session", fake_session)
        return state

    return install
