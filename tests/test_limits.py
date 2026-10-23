"""Per-visitor 24-hour limit: survives refresh, resists spoofing, rolls over."""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def test_forged_left_xff_entry_is_ignored():
    import app
    h = {"X-Forwarded-For": "8.8.8.8, 34.120.1.1, 10.0.0.5"}
    assert app.client_ip_from(h, "10.0.0.9") == "34.120.1.1"


def test_direct_ip_used_when_no_header_and_private_ignored():
    import app
    assert app.client_ip_from({}, "52.1.2.3") == "52.1.2.3"
    assert app.client_ip_from({}, "10.1.2.3") is None
    assert app.client_ip_from({"X-Forwarded-For": "garbage, 192.168.1.4"}, None) is None


def test_rolling_24h_window():
    import app
    key = "ip:window-test"
    t0 = 1_000_000.0
    for i in range(3):
        app.record_visitor(key, now=t0 + i)
    assert app.visitor_used(key, now=t0 + 10) == 3
    assert app.visitor_used(key, now=t0 + app.VISITOR_WINDOW_SECONDS + 1) == 1
    assert app.visitor_used(key, now=t0 + app.VISITOR_WINDOW_SECONDS + 10) == 0


SCRIPT = f'''
import sys; sys.path.insert(0, {str(ROOT)!r})
import builtins
seen = builtins.__dict__.setdefault("_SB_LIMIT_SEEN", [])
async def fake(user_message, history, api_key, model):
    seen.append(user_message)
    return "answer", [], {{}}
import app
app.run_chat_turn = fake
app.visitor_key = lambda: "ip:same-visitor"
app.main()
'''


@pytest.fixture
def fresh_app(monkeypatch):
    import builtins
    builtins.__dict__["_SB_LIMIT_SEEN"] = []
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-owner")
    monkeypatch.setenv("VISITOR_QUESTION_LIMIT", "2")
    monkeypatch.setenv("DAILY_QUESTION_LIMIT", "100")
    import app
    importlib.reload(app)
    app._visitor_log.clear()
    yield builtins.__dict__["_SB_LIMIT_SEEN"]
    import chat
    importlib.reload(app)
    app.run_chat_turn = chat.run_chat_turn


def test_refresh_does_not_reset_the_limit(fresh_app):
    first = AppTest.from_string(SCRIPT, default_timeout=30)
    first.run()
    first.chat_input(key="home_ask").set_value("q1").run()
    first.chat_input(key="chat_ask").set_value("q2").run()
    assert fresh_app == ["q1", "q2"]

    refreshed = AppTest.from_string(SCRIPT, default_timeout=30)  # new session, same visitor
    refreshed.run()
    refreshed.chat_input(key="home_ask").set_value("q3").run()
    assert fresh_app == ["q1", "q2"]
    assert any("last 24 hours" in w.value for w in refreshed.warning)
    assert any("0 of 2" in i.value for i in refreshed.sidebar.info)
