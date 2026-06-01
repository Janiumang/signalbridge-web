"""SignalBridge openFDA - Streamlit chat interface.

Browser-based chat for FDA pharmacovigilance questions. Wires user input
through Claude (Anthropic API) and routes any tool calls to the local
mcp-server-openfda subprocess. The MCP server stays unchanged - this
app is a real MCP client talking the protocol, not a shortcut around it.

Run locally:
    streamlit run app.py
"""

from __future__ import annotations

import asyncio
import json

import streamlit as st

from chat import run_chat_turn


EXAMPLE_PROMPTS: tuple[str, ...] = (
    "What does the FDA label for pembrolizumab say about immune-mediated pneumonitis?",
    "Show me the most common adverse events reported with metformin in patients over 70.",
    "What's the year-over-year trend in semaglutide adverse event reports since 2020?",
    "What FDA recalls have happened for tirzepatide? What was the most severe?",
)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def init_state() -> None:
    """Seed Streamlit session state on first load."""
    if "messages" not in st.session_state:
        # Each entry: {"role": "user"|"assistant", "content": str,
        # "tool_calls": list[dict] (optional, assistant only)}
        st.session_state.messages = []
    if "api_key" not in st.session_state:
        st.session_state.api_key = ""


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> None:
    """Sidebar: API key input, clear-chat button, about-the-project blurb."""
    with st.sidebar:
        st.markdown("## SignalBridge")
        st.caption("openFDA pharmacovigilance chat")

        st.markdown("### Anthropic API key")
        key = st.text_input(
            "Anthropic API key",
            type="password",
            value=st.session_state.api_key,
            label_visibility="collapsed",
            placeholder="sk-ant-...",
            help=(
                "Get a key at console.anthropic.com. The key is stored only "
                "in this browser session - never logged, never sent anywhere "
                "except Anthropic's API."
            ),
        )
        if key != st.session_state.api_key:
            st.session_state.api_key = key

        if not st.session_state.api_key:
            st.info("Add your API key above to start chatting.")

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
            "Behind the scenes this app spawns the MCP server as a "
            "subprocess and talks the MCP protocol over stdio - the same "
            "way Claude Desktop does."
        )


# ---------------------------------------------------------------------------
# Message + tool-call rendering
# ---------------------------------------------------------------------------

def render_tool_calls(tool_calls: list[dict]) -> None:
    """Render the expandable tool-call log under an assistant message."""
    if not tool_calls:
        return
    with st.expander(f"Tools called ({len(tool_calls)})", expanded=False):
        for i, call in enumerate(tool_calls, start=1):
            st.markdown(f"**{i}. `{call['name']}`**")
            st.caption("Input")
            st.code(json.dumps(call["input"], indent=2), language="json")
            st.caption("Result preview")
            st.code(call["result_preview"], language="json")


def render_message(msg: dict) -> None:
    """Render a single message from history."""
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("tool_calls"):
            render_tool_calls(msg["tool_calls"])


def render_examples() -> None:
    """Show clickable example prompts when the conversation is empty."""
    if st.session_state.messages:
        return
    st.markdown("**Try one of these:**")
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
    if not st.session_state.api_key:
        st.error("Add your Anthropic API key in the sidebar first.")
        return

    history = _history_for_api()
    st.session_state.messages.append({"role": "user", "content": text})

    with st.chat_message("user"):
        st.markdown(text)

    with st.chat_message("assistant"):
        with st.spinner("Querying openFDA via the MCP server..."):
            try:
                response_text, tool_calls = asyncio.run(
                    run_chat_turn(
                        user_message=text,
                        history=history,
                        api_key=st.session_state.api_key,
                    )
                )
            except Exception as e:
                st.error(f"{type(e).__name__}: {e}")
                # Roll back the user message so the chat input is empty
                # and the user can retry.
                st.session_state.messages.pop()
                return

        st.markdown(response_text)
        render_tool_calls(tool_calls)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response_text,
            "tool_calls": tool_calls,
        }
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="SignalBridge openFDA",
        layout="centered",
    )

    init_state()
    render_sidebar()

    st.title("SignalBridge openFDA")
    st.caption(
        "Ask Claude questions about FDA pharmacovigilance data - adverse "
        "events, drug labels, recalls - powered by openFDA and the "
        "mcp-server-openfda MCP connector."
    )

    for msg in st.session_state.messages:
        render_message(msg)

    render_examples()

    # Handle example-prompt click from previous rerun
    if "pending_input" in st.session_state:
        prompt = st.session_state.pending_input
        del st.session_state.pending_input
        handle_user_input(prompt)

    if user_text := st.chat_input("Ask a pharmacovigilance question..."):
        handle_user_input(user_text)


if __name__ == "__main__":
    main()
