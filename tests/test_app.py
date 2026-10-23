"""App-level behaviour through Streamlit's test harness (no network)."""
from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]

SCRIPT = f'''
import sys; sys.path.insert(0, {str(ROOT)!r})
import builtins
seen = builtins.__dict__.setdefault("_SB_SEEN", [])
async def fake(user_message, history, api_key, model):
    seen.append((user_message, api_key, model))
    if user_message == "boom-auth":
        import anthropic, httpx
        req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        raise ExceptionGroup("tg", [anthropic.AuthenticationError("bad", response=httpx.Response(401, request=req), body=None)])
    return "answer", [], {{}}
import app
app.run_chat_turn = fake  # patched on the app module only; restored by the fixture
app.main()
'''


import pytest


@pytest.fixture(autouse=True)
def _restore_app_module():
    import importlib
    yield
    import app, chat
    app.run_chat_turn = chat.run_chat_turn
    importlib.reload(chat)


def _app(monkeypatch, **env):
    import builtins
    builtins.__dict__["_SB_SEEN"] = []
    for k, v in {"ANTHROPIC_API_KEY": "sk-owner", "SESSION_QUESTION_LIMIT": "2", "DAILY_QUESTION_LIMIT": "100", **env}.items():
        monkeypatch.setenv(k, v)
    import importlib
    import app
    importlib.reload(app)  # settings are read at import, as on a real (re)start
    at = AppTest.from_string(SCRIPT, default_timeout=30)
    at.run()
    return at, builtins.__dict__["_SB_SEEN"]


def test_home_then_session_limit(monkeypatch):
    at, seen = _app(monkeypatch)
    at.chat_input(key="home_ask").set_value("q1").run()
    at.chat_input(key="chat_ask").set_value("q2").run()
    at.chat_input(key="chat_ask").set_value("q3").run()
    assert [s[0] for s in seen] == ["q1", "q2"]
    assert any("free questions" in w.value for w in at.warning)
    assert all(s[1] == "sk-owner" for s in seen)
    assert not at.exception


def test_example_button_runs_question(monkeypatch):
    at, seen = _app(monkeypatch)
    at.button(key="example_0").click().run()
    assert seen and "pembrolizumab" in seen[0][0]


def test_no_owner_key_blocks_free_tier(monkeypatch):
    at, seen = _app(monkeypatch, ANTHROPIC_API_KEY="")
    at.chat_input(key="home_ask").set_value("q").run()
    assert not seen
    assert any("isn't available" in w.value for w in at.warning)


def test_auth_error_is_unwrapped_and_friendly(monkeypatch):
    at, _ = _app(monkeypatch)
    at.chat_input(key="home_ask").set_value("boom-auth").run()
    msgs = [e.value for e in at.error]
    assert msgs and "isn't available" in msgs[0] and "ExceptionGroup" not in msgs[0]


def test_ui_text_hides_internals():
    """Visitors must not see the AI vendor, model, MCP or tool names."""
    import re, pathlib
    src = pathlib.Path(__file__).resolve().parent.parent
    banned = re.compile(r"Anthropic|Claude|MCP|openFDA|mcp-server|sk-ant", re.IGNORECASE)
    for name in ("app.py", "visuals.py"):
        text = (src / name).read_text()
        # Only user-facing string literals matter; strip comments/docstrings crudely.
        code = "\n".join(l for l in text.splitlines() if not l.strip().startswith("#"))
        code = re.sub(r'"""[\s\S]*?"""', "", code)
        strings = re.findall(r'"([^"\n]*)"|\'([^\'\n]*)\'', code)
        hits = [a or b for a, b in strings if banned.search(a or b)]
        hits = [h for h in hits if not h.startswith(("OPENFDA_API_KEY", "SIGNALBRIDGE_", "ANTHROPIC_API_KEY"))
                and "import" not in h and ".py" not in h and h != "mcp-server-openfda"]
        assert not hits, (name, hits)
