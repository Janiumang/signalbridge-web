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
import collections
import datetime
import hashlib
import ipaddress
import secrets
import time
import uuid
import logging
import re
import os
import threading
from typing import Any

import streamlit as st

from chat import DEFAULT_MODEL, OUT_OF_SCOPE_REPLY, run_chat_turn
import feedback
import visuals


EXAMPLE_PROMPTS: tuple[str, ...] = (
    "What does the FDA label for pembrolizumab say about immune-mediated pneumonitis?",
    "Show me the most common adverse events reported with metformin in patients over 70.",
    "What's the year-over-year trend in semaglutide adverse event reports since 2020?",
    "Is pneumonitis reported disproportionately with pembrolizumab? Give me the ROR.",
)


AVATARS = {"user": ":material/person:", "assistant": ":material/monitor_heart:"}


logger = logging.getLogger(__name__)

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
# Free questions per visitor (anonymised IP) per rolling 24 hours. Survives
# page refreshes and new tabs; defaults to the old per-session number.
VISITOR_QUESTION_LIMIT: int = int(_config("VISITOR_QUESTION_LIMIT", SESSION_QUESTION_LIMIT))
VISITOR_WINDOW_SECONDS = 24 * 60 * 60
# Salt for hashing IPs. Set IP_HASH_SALT in secrets to keep hashes stable
# across restarts; otherwise a random per-process salt is used.
_IP_SALT: str = str(_config("IP_HASH_SALT", "") or secrets.token_hex(16))
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
# Per-visitor rolling 24-hour limit (anonymised IP)
# ---------------------------------------------------------------------------

def client_ip_from(headers: Any, direct_ip: str | None) -> str | None:
    """Best guess at the visitor's real IP.

    X-Forwarded-For is "client, proxy1, proxy2"; a visitor can forge the
    LEFT part, but each trusted proxy appends what it actually saw. So take
    the right-most public address. Falls back to the socket address.
    """
    xff = ""
    try:
        xff = headers.get("X-Forwarded-For") or headers.get("x-forwarded-for") or ""
    except Exception:
        xff = ""
    for part in reversed([p.strip() for p in xff.split(",") if p.strip()]):
        try:
            addr = ipaddress.ip_address(part)
        except ValueError:
            continue
        if addr.is_global:
            return str(addr)
    if direct_ip:
        try:
            if ipaddress.ip_address(direct_ip).is_global:
                return direct_ip
        except ValueError:
            pass
    return None


def visitor_key() -> str:
    """Salted hash of the visitor's IP; per-session id if no IP is known."""
    ip = None
    try:
        ip = client_ip_from(st.context.headers, getattr(st.context, "ip_address", None))
    except Exception:
        ip = None
    if ip:
        return "ip:" + hashlib.sha256((_IP_SALT + ip).encode()).hexdigest()[:32]
    if "visitor_fallback_id" not in st.session_state:
        st.session_state.visitor_fallback_id = uuid.uuid4().hex
    return "session:" + st.session_state.visitor_fallback_id


@st.cache_resource
def _visitor_log() -> dict[str, Any]:
    return {"lock": threading.Lock(), "hits": {}}


_MAX_TRACKED_VISITORS = 50_000
REFUSALS_PER_CHARGE = 3  # every 3rd off-topic (refused) question counts as 1
REFUSED_PREFIX = "refused|"


def _prune(hits: "collections.deque[float]", now: float) -> None:
    while hits and now - hits[0] >= VISITOR_WINDOW_SECONDS:
        hits.popleft()


def visitor_used(key: str, now: float | None = None) -> int:
    now = time.time() if now is None else now
    log = _visitor_log()
    with log["lock"]:
        hits = log["hits"].get(key)
        if not hits:
            return 0
        _prune(hits, now)
        return len(hits)


def visitor_resets_at(key: str) -> float | None:
    log = _visitor_log()
    with log["lock"]:
        hits = log["hits"].get(key)
        return hits[0] + VISITOR_WINDOW_SECONDS if hits else None


def record_visitor(key: str, now: float | None = None) -> None:
    now = time.time() if now is None else now
    log = _visitor_log()
    with log["lock"]:
        hits = log["hits"].setdefault(key, collections.deque())
        _prune(hits, now)
        hits.append(now)
        if len(log["hits"]) > _MAX_TRACKED_VISITORS:  # drop idle visitors
            for k in [k for k, v in log["hits"].items() if not v or now - v[-1] >= VISITOR_WINDOW_SECONDS]:
                del log["hits"][k]


def _resets_in_words(key: str) -> str:
    at = visitor_resets_at(key)
    if not at:
        return "in 24 hours"
    mins = max(1, int((at - time.time()) // 60))
    return f"in about {mins // 60} h {mins % 60} min" if mins >= 60 else f"in about {mins} min"


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
            return "Your API key was rejected."
        return "SignalBridge's assistant isn't available right now. Please try again later."
    if isinstance(e, anthropic.PermissionDeniedError) or "credit balance" in msg:
        return "SignalBridge has reached its usage limit for now. Please try again later."
    if isinstance(e, anthropic.NotFoundError):
        return "SignalBridge's assistant is being updated. Please try again later."
    if isinstance(e, anthropic.BadRequestError):
        detail = ""
        try:
            detail = e.body.get("error", {}).get("message", "")  # type: ignore[union-attr]
        except Exception:
            detail = ""
        logger.warning("Bad request: %s", detail[:300])
        return "That question couldn't be processed. Please rephrase it and try again."
    if isinstance(e, anthropic.RateLimitError):
        return "Lots of questions right now - please wait a minute and try again."
    if isinstance(e, anthropic.APIStatusError) and getattr(e, "status_code", 0) >= 500:
        return "SignalBridge had a temporary problem. Please try again in a minute."
    if isinstance(e, anthropic.APIConnectionError):
        return "SignalBridge couldn't complete the search. Please try again in a minute."
    if "mcp-server-openfda" in msg or "connection closed" in msg:
        return "The FDA search didn't start. Please try again in a moment."
    logger.warning("Unhandled error: %s: %s", type(e).__name__, e)
    return "Something went wrong. Please try again."


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
            "SignalBridge isn't available right now. Please try again later."
        )
    key = visitor_key()
    if visitor_used(key) >= VISITOR_QUESTION_LIMIT:
        return (
            f"You've used all {VISITOR_QUESTION_LIMIT} free questions for the last 24 hours. "
            f"Your next free question opens {_resets_in_words(key)}. "
            "Sign up in the sidebar to hear about higher limits."
        )
    if _daily_used() >= DAILY_QUESTION_LIMIT:
        return (
            "SignalBridge has reached its free question limit for today. "
            "Please come back tomorrow."
        )
    return None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_status(slot: Any) -> None:
    """Free-tier / own-key status line in the sidebar."""
    if _using_own_key():
        slot.success("No question limit on this session.")
    elif OWNER_API_KEY:
        remaining = max(0, VISITOR_QUESTION_LIMIT - visitor_used(visitor_key()))
        slot.info(f"Free tier: {remaining} of {VISITOR_QUESTION_LIMIT} questions left in the last 24 hours.")


def render_sidebar() -> Any:
    """Sidebar: free-tier status, optional own key, clear chat, about."""
    with st.sidebar:
        st.html(visuals.brand_bar().replace("sb-bar", "sb-bar sb-side"))

        # Filled at the end of the run so the count reflects this turn.
        status_slot = st.empty()

        with st.expander("Get updates · request a feature"):
            feedback.render_form("sidebar", st.session_state.questions_asked)

        st.markdown("---")

        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.markdown("---")
        st.markdown(
            "**About**\n\n"
            "SignalBridge searches the FDA's public adverse event database "
            "(FAERS), drug labels and recalls, and runs standard "
            "disproportionality statistics on what it finds. Every answer is "
            "built from a live search - nothing is answered from memory."
        )
        st.caption(
            "Free-tier limits use an anonymised hash of your IP address, kept "
            "in memory for 24 hours and never logged. "
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
        with st.spinner("Searching FDA adverse event data..."):
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
        key = visitor_key()
        if response_text == OUT_OF_SCOPE_REPLY:
            # Off-topic questions are cheap (refused by the topic gate), so
            # only every REFUSALS_PER_CHARGE-th one uses a free question.
            record_visitor(REFUSED_PREFIX + key)
            if visitor_used(REFUSED_PREFIX + key) % REFUSALS_PER_CHARGE == 0:
                record_visitor(key)
            st.caption(
                f"Off-topic questions: every {REFUSALS_PER_CHARGE}rd one uses a free question."
            )
        else:
            st.session_state.questions_asked += 1
            record_visitor(key)
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
        page_title="SignalBridge",
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
