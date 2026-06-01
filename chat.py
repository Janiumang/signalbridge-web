"""Anthropic chat loop with MCP tool routing.

One public entrypoint: `run_chat_turn()` takes a user message plus prior
conversation history, spawns one mcp-server-openfda subprocess for the
turn, and runs the Anthropic tool-use loop until Claude returns a final
text answer. Returns the assistant's text plus a structured log of every
tool call made along the way so the UI can show what happened.
"""

from __future__ import annotations

from typing import Any

from anthropic import Anthropic

from mcp_client import mcp_session, mcp_tool_to_anthropic


# Stable Anthropic model identifier. Sonnet 4.6 is the documented current
# model and a good balance of speed, cost, and tool-use reliability for
# this PV use case. Override in run_chat_turn if needed.
DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS = 4096


SYSTEM_PROMPT = """You are SignalBridge, a pharmacovigilance assistant with direct access to FDA openFDA tools.

You have these tools available:
- search_drug_adverse_events: FAERS adverse event reports with drug name, date range, age range, country, and optional MedDRA reaction filters.
- count_adverse_events: Aggregate counts across six pivots (reaction, country, year, reporter_qualification, concomitant_drug, seriousness_subtype).
- count_reactions: Shortcut for count_adverse_events with pivot="reaction".
- get_drug_label: Most recent FDA drug label with PV-essential sections (boxed_warning, adverse_reactions, warnings_and_cautions, etc.).
- search_drug_recalls: FDA drug recalls with classification (Class I/II/III) and status filters.

When the user asks about a specific drug, reaction, recall, or label content, use these tools. Synthesize the tool results into a clear, well-sourced answer. Cite specific numbers, dates, and percentages from the tool output. Do not invent data.

The MedDRA reaction filter expects Preferred Terms (e.g., "Pneumonitis", not "lung inflammation"). The tool descriptions remind you of this; pass through MedDRA-style terms when possible.

If a tool returns an error or empty result, explain it to the user — do not silently work around it.

Be concise. Use markdown formatting (lists, bold) when it helps readability."""


async def run_chat_turn(
    user_message: str,
    history: list[dict[str, Any]],
    api_key: str,
    model: str = DEFAULT_MODEL,
) -> tuple[str, list[dict[str, Any]]]:
    """Process one user message through Claude with MCP tool access.

    Args:
        user_message: The new user input for this turn.
        history: Prior conversation in Anthropic Messages API format.
            Each entry is {"role": "user"|"assistant", "content": ...}.
            Pass an empty list for the first turn.
        api_key: User-provided Anthropic API key.
        model: Anthropic model identifier; defaults to claude-sonnet-4-6.

    Returns:
        Tuple of (assistant_text, tool_calls_log).
            assistant_text: The model's final natural-language answer.
            tool_calls_log: Ordered list of every tool call made during
                this turn. Each entry is {"name", "input", "result_preview"}.
                The UI uses this to show users what FDA queries actually ran.
    """
    client = Anthropic(api_key=api_key)
    tool_calls_log: list[dict[str, Any]] = []

    async with mcp_session() as session:
        # Discover tools from the MCP server and translate them to the
        # shape Claude's API expects.
        tools_response = await session.list_tools()
        anthropic_tools = [mcp_tool_to_anthropic(t) for t in tools_response.tools]

        # Working messages list. We mutate this as we add assistant
        # responses and tool results across loop iterations.
        messages: list[dict[str, Any]] = list(history) + [
            {"role": "user", "content": user_message}
        ]

        # Tool-use loop: keep calling Claude until it stops asking for
        # tool calls and gives a final text answer.
        while True:
            response = client.messages.create(
                model=model,
                max_tokens=DEFAULT_MAX_TOKENS,
                system=SYSTEM_PROMPT,
                tools=anthropic_tools,
                messages=messages,
            )

            # Always append the assistant turn (including any tool_use blocks)
            # in a serializable form so the next loop iteration can replay it.
            assistant_content = [_block_to_dict(b) for b in response.content]
            messages.append({"role": "assistant", "content": assistant_content})

            if response.stop_reason != "tool_use":
                # No tool call requested — this is the final answer.
                return _extract_text(response.content), tool_calls_log

            # Claude wants one or more tool calls. Run them all, gather
            # results, send back in a single user turn.
            tool_results: list[dict[str, Any]] = []
            for block in response.content:
                if block.type != "tool_use":
                    continue

                try:
                    call_result = await session.call_tool(
                        block.name, arguments=block.input
                    )
                    result_text = _extract_tool_result_text(call_result)
                except Exception as e:
                    result_text = f"[tool call failed: {type(e).__name__}: {e}]"

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_text,
                    }
                )

                tool_calls_log.append(
                    {
                        "name": block.name,
                        "input": dict(block.input) if block.input else {},
                        "result_preview": _truncate(result_text, 800),
                    }
                )

            messages.append({"role": "user", "content": tool_results})


def _block_to_dict(block: Any) -> dict[str, Any]:
    """Convert an Anthropic content block to a serializable dict.

    Required because the messages list is fed back into client.messages.create
    on the next loop iteration — Anthropic accepts the same dict shape it
    emits, but only after we down-convert from SDK objects.
    """
    if block.type == "text":
        return {"type": "text", "text": block.text}
    if block.type == "tool_use":
        return {
            "type": "tool_use",
            "id": block.id,
            "name": block.name,
            "input": block.input,
        }
    # Unknown block types — preserve type tag so we don't lose context.
    return {"type": block.type}


def _extract_text(content: Any) -> str:
    """Concatenate text blocks from a final assistant response."""
    parts = [block.text for block in content if block.type == "text"]
    return "\n\n".join(parts) if parts else "[Claude returned no text content]"


def _extract_tool_result_text(call_result: Any) -> str:
    """Pull the text content out of an MCP tool-call result object."""
    if not call_result.content:
        return ""
    pieces = []
    for item in call_result.content:
        text = getattr(item, "text", None)
        pieces.append(text if text is not None else str(item))
    return "\n".join(pieces)


def _truncate(s: str, max_chars: int) -> str:
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + f"\n... [preview truncated, {len(s):,} chars total]"
