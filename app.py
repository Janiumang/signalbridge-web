"""SignalBridge openFDA - Streamlit chat interface.

Browser-based chat for FDA pharmacovigilance questions. Wires user input
through Claude (Anthropic API) and routes any tool calls to the local
mcp-server-openfda subprocess. The MCP server stays unchanged - this
app is a real MCP client talking the protocol, not a shortcut around it.

API key + cost limits
---------------------
By default the app runs on the OWNER's Anthropic key, read from Streamlit
secrets (or environment variables) so visitors need nothing to start:

    ANTHROPIC_API_KEY        required for the free tier
    SIGNALBRIDGE_MODEL       optional, default "claude-haiku-5-5"
    SESSION_QUESTION_LIMIT   optional, default 10 (per browser session)
    DAILY_QUESTION_LIMIT     optional, default 200 (whole app, per UTC day)

Visitors who paste their own key in the sidebar bypass both limits and
are billed on their own account.

The daily counter lives in server memory, so it resets if the app
restarts. The hard stop on spend is the monthly limit you set in the
Anthropic Console - always set one.

Run locally:
    streamlit run app.py
"""

from __future__ import annotations

import asyncio
import datetime
import logging
import re
import os
import threading
from typing import Any

import streamlit as st

from chat import DEFAULT_MODEL, run_chat_turn
import feedback
import visuals


EXAMPLE_PROMPTS: tuple[str, ...] = (
    "What does the FDA label for pembrolizumab say about immune-mediated pneumonitis?",
    "Show me the most common adverse events reported with metformin in patients over 70.",
    "What's the year-over-year trend in semaglutide adverse event reports since 2020?",
    "Is pneumonitis reported disproportionately with pembrolizumab? Give me the ROR.",
)


AVATARS = {"user": ":material/person:", "assistant": ":material/monitor_heart:"}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _config(name: str, default: Any = None) -> Any:
    """Read a setting from Streamlit secrets, then env vars, then default."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        # No secrets.toml present (common in local dev) - fall through.
        pass
    return os.environ.get(name, default)


OWNER_API_KEY: str = _config("ANTHROPIC_API_KEY", "") or ""
MODEL: str = _config("SIGNALBRIDGE_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL
SESSION_QUESTION_LIMIT: int = int(_config("SESSION_QUESTION_LIMIT", 10))
DAILY_QUESTION_LIMIT: int = int(_config("DAILY_QUESTION_LIMIT", 200))

# Forwarded to the MCP server subprocess (see mcp_client.py).
_openfda_key = _config("OPENFDA_API_KEY", "")
if _openfda_key:
    os.environ["OPENFDA_API_KEY"] = str(_openfda_key)


# ---------------------------------------------------------------------------
# App-wide daily counter (shared across all visitors on this server)
# ---------------------------------------------------------------------------

@st.cache_resource
def _daily_counter() -> dict[str, Any]:
    return {"date": "", "count": 0, "lock": threading.Lock()}


def _today() -> str:
    return datetime.datetime.now(datetime.timezone.utc).date().isoformat()


def _daily_used() -> int:
    c = _daily_counter()
    with c["lock"]:
        if c["date"] != _today():
            c["date"], c["count"] = _today(), 0
        return c["count"]


def _increment_daily() -> None:
    c = _daily_counter()
    with c["lock"]:
        if c["date"] != _today():
            c["date"], c["count"] = _today(), 0
        c["count"] += 1


# ---------------------------------------------------------------------------
# Error reporting
# ---------------------------------------------------------------------------

def _root_error(e: BaseException) -> BaseException:
    """Unwrap anyio/asyncio ExceptionGroups to the first real error.

    Anything raised inside the MCP session (including Anthropic API errors)
    surfaces as 'ExceptionGroup: unhandled errors in a TaskGroup', which
    hides the cause.
    """
    while isinstance(e, BaseExceptionGroup) and e.exceptions:
        e = e.exceptions[0]
    return e


def _friendly_error(e: BaseException) -> str:
    """Visitor-facing message; never leaks keys or stack traces."""
    import anthropic

    msg = str(e).lower()
    if isinstance(e, anthropic.AuthenticationError):
        if _using_own_key():
            return "Your Anthropic API key was rejected. Check it in the sidebar."
        return "SignalBridge's AI service isn't available right now (key problem). Please try again later."
    if isinstance(e, anthropic.PermissionDeniedError) or "credit balance" in msg:
        return "SignalBridge's AI service is paused (usage limit reached). Please try again later."
    if isinstance(e, anthropic.NotFoundError):
        return "The configured AI model isn't available. The site owner needs to update SIGNALBRIDGE_MODEL."
    if isinstance(e, anthropic.BadRequestError):
        detail = ""
        try:
            detail = e.body.get("error", {}).get("message", "")  # type: ignore[union-attr]
        except Exception:
            detail = ""
        return "The AI service rejected the request" + (f": {detail[:200]}" if detail else ".") + " Please try again."
    if isinstance(e, anthropic.RateLimitError):
        return "Lots of questions right now - please wait a minute and try again."
    if isinstance(e, anthropic.APIStatusError) and getattr(e, "status_code", 0) >= 500:
        return "The AI service had a temporary problem. Please try again in a minute."
    if isinstance(e, anthropic.APIConnectionError):
        return "Couldn't reach the AI service. Please try again in a minute."
    if "mcp-server-openfda" in msg or "connection closed" in msg:
        return "The FDA search engine failed to start. Please try again; if it keeps happening the site owner will fix it."
    return f"Something went wrong ({type(e).__name__}). Please try again."


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def init_state() -> None:
    """Seed Streamlit session state on first load."""
    if "messages" not in st.session_state:
        # Each entry: {"role": "user"|"assistant", "content": str,
        # "tool_calls": list[dict] (optional, assistant only)}
        st.session_state.messages = []
    if "visitor_api_key" not in st.session_state:
        st.session_state.visitor_api_key = ""
    if "questions_asked" not in st.session_state:
        st.session_state.questions_asked = 0


def _using_own_key() -> bool:
    return bool(st.session_state.visitor_api_key)


def _active_api_key() -> str:
    return st.session_state.visitor_api_key or OWNER_API_KEY


def _limit_message() -> str | None:
    """Return a message if this visitor can't ask another free question."""
    if _using_own_key():
        return None
    if not OWNER_API_KEY:
        return (
            "The free tier isn't configured on this server. Add your own "
            "Anthropic API key in the sidebar to chat."
        )
    if st.session_state.questions_asked >= SESSION_QUESTION_LIMIT:
        return (
            f"You've used all {SESSION_QUESTION_LIMIT} free questions for this "
            "session. Add your own Anthropic API key in the sidebar to keep going."
        )
    if _daily_used() >= DAILY_QUESTION_LIMIT:
        return (
            "SignalBridge has reached its free question limit for today. "
            "Please come back tomorrow, or add your own Anthropic API key in "
            "the sidebar."
        )
    return None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_status(slot: Any) -> None:
    """Free-tier / own-key status line in the sidebar."""
    if _using_own_key():
        slot.success("Using your own API key - no question limit.")
    elif OWNER_API_KEY:
        remaining = max(0, SESSION_QUESTION_LIMIT - st.session_state.questions_asked)
        slot.info(f"Free tier: {remaining} of {SESSION_QUESTION_LIMIT} questions left this session.")


def render_sidebar() -> Any:
    """Sidebar: free-tier status, optional own key, clear chat, about."""
    with st.sidebar:
        st.html(visuals.brand_bar().replace("sb-bar", "sb-bar sb-side"))

        # Filled at the end of the run so the count reflects this turn.
        status_slot = st.empty()

        with st.expander("Use your own Anthropic API key (optional)"):
            key = st.text_input(
                "Anthropic API key",
                type="password",
                value=st.session_state.visitor_api_key,
                label_visibility="collapsed",
                placeholder="sk-ant-...",
                help=(
                    "Removes the free-tier limit. The key is kept only in "
                    "this browser session - never logged, never sent anywhere "
                    "except Anthropic's API."
                ),
            )
            if key != st.session_state.visitor_api_key:
                st.session_state.visitor_api_key = key.strip()
                st.rerun()

        with st.expander("Get updates · request a feature"):
            feedback.render_form("sidebar", st.session_state.questions_asked)

        st.markdown("---")

        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.markdown("---")
        st.markdown(
            "**About**\n\n"
            "Browser chat interface for "
            "[mcp-server-openfda](https://github.com/Janiumang/mcp-server-openfda), "
            "part of the SignalBridge AI family of regulatory-data MCP "
            "connectors. Uses only public openFDA data. MIT licensed."
        )
        st.caption(
            "Not a validated pharmacovigilance system. FAERS counts and "
            "disproportionality statistics are signals for review, not "
            "evidence of causality."
        )
    return status_slot


# ---------------------------------------------------------------------------
# Message + tool-call rendering
# ---------------------------------------------------------------------------

def render_assistant(content: str, tool_calls: list[dict]) -> None:
    """Plan rows -> visual result cards -> written answer."""
    if tool_calls:
        st.html(visuals.plan_rows(tool_calls))
        for card in visuals.result_cards(tool_calls):
            st.html(card)
    # Model output: drop markdown images so injected content can't load remote URLs.
    st.markdown(re.sub(r"!\[[^\]]*\]\([^)]*\)", "", content))


def render_message(msg: dict) -> None:
    """Render a single message from history."""
    with st.chat_message(msg["role"], avatar=AVATARS[msg["role"]]):
        if msg["role"] == "assistant":
            render_assistant(msg["content"], msg.get("tool_calls") or [])
        else:
            st.markdown(msg["content"])


def render_examples() -> None:
    """Show clickable example prompts when the conversation is empty."""
    if st.session_state.messages:
        return
    st.html('<div class="sb-eyebrow" style="margin-top:1.4rem">Or try one of these</div>')
    cols = st.columns(2)
    for i, prompt in enumerate(EXAMPLE_PROMPTS):
        if cols[i % 2].button(prompt, key=f"example_{i}", use_container_width=True):
            st.session_state.pending_input = prompt
            st.rerun()


# ---------------------------------------------------------------------------
# Chat turn handler
# ---------------------------------------------------------------------------

def _history_for_api() -> list[dict]:
    """Strip UI-only fields from session messages before sending to the API."""
    return [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
    ]


def handle_user_input(text: str) -> None:
    """Run one chat turn end-to-end and update session state."""
    blocked = _limit_message()
    if blocked:
        st.warning(blocked)
        return

    history = _history_for_api()
    st.session_state.messages.append({"role": "user", "content": text})

    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown(text)

    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        with st.spinner("Querying openFDA via the MCP server..."):
            try:
                response_text, tool_calls, _usage = asyncio.run(
                    run_chat_turn(
                        user_message=text,
                        history=history,
                        api_key=_active_api_key(),
                        model=MODEL,
                    )
                )
            except Exception as e:
                root = _root_error(e)
                # Full detail goes to the server log (Manage app -> logs).
                logging.exception("Chat turn failed: %r", root)
                st.error(_friendly_error(root))
                st.session_state.messages.pop()
                return

        render_assistant(response_text, tool_calls)

    if not _using_own_key():
        st.session_state.questions_asked += 1
        _increment_daily()

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response_text,
            "tool_calls": tool_calls,
        }
    )


def render_signup() -> None:
    """Main-page sign-up: a card on the home screen, an expander after chats."""
    if st.session_state.get("feedback_submitted"):
        return
    if not st.session_state.messages:
        st.html(
            '<div class="sb-eyebrow" style="margin-top:2.2rem">Help shape SignalBridge</div>'
            '<div class="sb-h">Tell us what to build next.</div>'
        )
        with st.container(border=True):
            feedback.render_form("home", st.session_state.questions_asked)
    else:
        with st.expander("Found this useful? Get updates and tell us what to build next"):
            feedback.render_form("after_answer", st.session_state.questions_asked)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="SignalBridge openFDA",
        layout="centered",
    )

    init_state()
    status_slot = render_sidebar()

    st.html(visuals.GLOBAL_CSS)
    st.html(visuals.brand_bar())

    # Home screen = no conversation yet and nothing queued to answer.
    is_home = not st.session_state.messages and "pending_input" not in st.session_state

    if is_home:
        st.html(visuals.hero())
        # Inline question box right under the headline (a chat_input inside a
        # container renders in place instead of pinning to the bottom).
        with st.container():
            home_text = st.chat_input(
                "Ask a pharmacovigilance question...", key="home_ask", max_chars=1000
            )
        if home_text:
            st.session_state.pending_input = home_text
            st.rerun()
        render_examples()
        render_signup()
    else:
        for msg in st.session_state.messages:
            render_message(msg)

        if "pending_input" in st.session_state:
            prompt = st.session_state.pending_input
            del st.session_state.pending_input
            handle_user_input(prompt)

        render_signup()

        if user_text := st.chat_input("Ask a pharmacovigilance question...", key="chat_ask", max_chars=1000):
            handle_user_input(user_text)

    render_status(status_slot)


if __name__ == "__main__":
    main()
