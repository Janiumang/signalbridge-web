"""Anthropic chat loop with MCP tool routing.

One public entrypoint: `run_chat_turn()` takes a user message plus prior
conversation history, spawns one mcp-server-openfda subprocess for the
turn, and runs the Anthropic tool-use loop until Claude returns a final
text answer. Returns the assistant's text plus a structured log of every
tool call made along the way so the UI can show what happened.

Cost controls (the app runs on the owner's API key by default):
    - Model is configurable (SIGNALBRIDGE_MODEL); defaults to the
      cheapest current Claude model.
    - System prompt + tool definitions are prompt-cached, so repeat
      calls in the loop pay ~10% for that fixed prefix.
    - Tool results sent back to the model are capped in size.
    - The tool-use loop is capped at MAX_TOOL_ROUNDS.
    - Only the most recent MAX_HISTORY_MESSAGES of history are resent.
"""

from __future__ import annotations

import json
from typing import Any

from anthropic import AsyncAnthropic

from mcp_client import mcp_session, mcp_tool_to_anthropic


# Cheapest current Claude model; good enough for tool routing + summaries.
# Override with the SIGNALBRIDGE_MODEL secret (e.g. "claude-sonnet-5-5")
# if answer quality needs a bump.
DEFAULT_MODEL = "claude-haiku-5-5"
DEFAULT_MAX_TOKENS = 2048

# Cost guards.
MAX_TOOL_ROUNDS = 6  # model <-> tools round-trips per user question
MAX_TOOL_RESULT_CHARS = 15_000  # ~4K tokens per tool result sent to the model
MAX_HISTORY_MESSAGES = 8  # prior user/assistant messages resent each turn


SYSTEM_PROMPT = """You are SignalBridge, a pharmacovigilance assistant with direct access to FDA openFDA tools.

You have these tools available:
- search_drug_adverse_events: FAERS adverse event reports with drug name, date range, age range, country, and optional reaction filters.
- count_adverse_events: Aggregate counts across six pivots (reaction, country, year, reporter_qualification, concomitant_drug, seriousness_subtype).
- count_reactions: Shortcut for count_adverse_events with pivot="reaction".
- get_drug_label: Most recent FDA drug label with PV-essential sections (boxed_warning, adverse_reactions, warnings_and_cautions, etc.).
- search_drug_recalls: FDA drug recalls with classification (Class I/II/III) and status filters.
- calculate_reporting_odds_ratio: Reporting Odds Ratio (ROR) with 95% CI and a signal flag for a drug-reaction pair - the standard disproportionality screen.

When the user asks about a specific drug, reaction, recall, or label content, use these tools. Synthesize the tool results into a clear, well-sourced answer. Cite specific numbers, dates, and percentages from the tool output. Do not invent data.

Prefer aggregate tools (count_reactions, count_adverse_events, calculate_reporting_odds_ratio) over pulling individual reports, and keep search_drug_adverse_events limits small (10-20) unless the user needs case-level detail.

Reaction filters work best with MedDRA Preferred Terms (e.g., "Pneumonitis"); common lay terms are expanded automatically and the tool output says when that happened.

When you report an ROR, state plainly that it is a spontaneous-reporting signal metric, not evidence of incidence or causality.

If a tool returns an error or empty result, explain it to the user - do not silently work around it.

Be concise. Use markdown formatting (lists, bold) when it helps readability."""


async def run_chat_turn(
    user_message: str,
    history: list[dict[str, Any]],
    api_key: str,
    model: str = DEFAULT_MODEL,
) -> tuple[str, list[dict[str, Any]], dict[str, int]]:
    """Process one user message through Claude with MCP tool access.

    Args:
        user_message: The new user input for this turn.
        history: Prior conversation in Anthropic Messages API format.
            Each entry is {"role": "user"|"assistant", "content": ...}.
            Pass an empty list for the first turn.
        api_key: Anthropic API key (owner's or the visitor's own).
        model: Anthropic model identifier.

    Returns:
        Tuple of (assistant_text, tool_calls_log, usage).
            assistant_text: The model's final natural-language answer.
            tool_calls_log: Ordered list of every tool call made during
                this turn. Each entry is {"name", "input", "result_preview"}.
            usage: Summed token usage for the turn
                {"input_tokens", "output_tokens", "cache_read_input_tokens"}.
    """
    client = AsyncAnthropic(api_key=api_key)
    tool_calls_log: list[dict[str, Any]] = []
    usage = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0}

    async with mcp_session() as session:
        tools_response = await session.list_tools()
        anthropic_tools = [mcp_tool_to_anthropic(t) for t in tools_response.tools]
        # Cache the (fixed) system prompt + tool definitions prefix.
        if anthropic_tools:
            anthropic_tools[-1]["cache_control"] = {"type": "ephemeral"}
        system = [
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ]

        messages: list[dict[str, Any]] = _trim_history(history) + [
            {"role": "user", "content": user_message}
        ]

        for round_number in range(MAX_TOOL_ROUNDS + 1):
            # On the final allowed round, withhold tools so the model must
            # answer with what it already has instead of looping further.
            last_round = round_number == MAX_TOOL_ROUNDS
            kwargs: dict[str, Any] = {
                "model": model,
                "max_tokens": DEFAULT_MAX_TOKENS,
                "system": system,
                "messages": messages,
            }
            if last_round:
                kwargs["system"] = system + [
                    {
                        "type": "text",
                        "text": "Tool budget for this question is used up. "
                        "Answer now using the results you already have.",
                    }
                ]
                # Tool definitions must still be present because the
                # history contains tool_use blocks; forbid new calls instead.
                kwargs["tools"] = anthropic_tools
                kwargs["tool_choice"] = {"type": "none"}
            else:
                kwargs["tools"] = anthropic_tools

            response = await client.messages.create(**kwargs)
            _add_usage(usage, response.usage)

            assistant_content = [_block_to_dict(b) for b in response.content]
            messages.append({"role": "assistant", "content": assistant_content})

            if response.stop_reason != "tool_use":
                return _extract_text(response.content), tool_calls_log, usage

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
                        "content": _truncate(result_text, MAX_TOOL_RESULT_CHARS),
                    }
                )
                tool_calls_log.append(
                    {
                        "name": block.name,
                        "input": dict(block.input) if block.input else {},
                        "result_preview": _truncate(result_text, 800),
                        # Parsed result for the UI's visual cards. Kept only
                        # for small aggregate tools so session state stays light.
                        "data": _parse_small_json(block.name, result_text),
                    }
                )

            messages.append({"role": "user", "content": tool_results})

    return "[No answer produced within the tool budget]", tool_calls_log, usage


_VISUAL_TOOLS = frozenset(
    {"calculate_reporting_odds_ratio", "count_adverse_events", "count_reactions"}
)


def _parse_small_json(tool_name: str, text: str) -> Any:
    """Parse JSON tool output for tools the UI draws; None otherwise."""
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        return None
    if tool_name in _VISUAL_TOOLS:
        return data
    # For other tools keep only the headline totals the plan rows show.
    if isinstance(data, dict):
        return {k: data[k] for k in ("error", "total_matching", "total", "total_results") if k in data}
    return None


def _trim_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only the most recent messages, starting on a user turn."""
    trimmed = list(history[-MAX_HISTORY_MESSAGES:])
    while trimmed and trimmed[0].get("role") != "user":
        trimmed.pop(0)
    return trimmed


def _add_usage(total: dict[str, int], u: Any) -> None:
    if u is None:
        return
    for key in total:
        total[key] += getattr(u, key, 0) or 0


def _block_to_dict(block: Any) -> dict[str, Any]:
    """Convert an Anthropic content block to a serializable dict."""
    if block.type == "text":
        return {"type": "text", "text": block.text}
    if block.type == "tool_use":
        return {
            "type": "tool_use",
            "id": block.id,
            "name": block.name,
            "input": block.input,
        }
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
    return s[:max_chars] + f"\n... [truncated, {len(s):,} chars total]"
