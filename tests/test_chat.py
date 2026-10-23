"""Chat loop: grounding, budgets, retries, history, serialisation."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS

import chat
from conftest import reply, text, tool_use


def tool_results(request):
    """The tool_result blocks sent back to the model (messages are shared by reference)."""
    for m in request["messages"]:
        if m["role"] == "user" and isinstance(m["content"], list) and m["content"] and m["content"][0].get("type") == "tool_result":
            return m["content"]
    return []


def run(msg="q", history=None):
    return asyncio.run(chat.run_chat_turn(msg, history or [], "sk-test"))


def test_never_uses_unsupported_tool_choice(fake_llm):
    llm = fake_llm([reply([tool_use("t1", "ping")], "tool_use"), reply([text("from data")], "end_turn")])
    answer, log, _ = run("what is povetacicept?")
    assert all(r.get("tool_choice", {"type": "none"})["type"] in ("auto", "none") for r in llm.requests)
    assert answer == "from data" and [c["name"] for c in log] == ["ping"]


def test_answer_without_query_is_discarded_and_retried(fake_llm):
    llm = fake_llm([
        reply([text("Povetacicept is a BAFF/APRIL inhibitor...")], "end_turn"),  # from memory
        reply([tool_use("t1", "get_drug_label")], "tool_use"),
        reply([text("No FDA label was found.")], "end_turn"),
    ])
    answer, log, _ = run("what is povetacicept?")
    assert answer == "No FDA label was found." and len(log) == 1
    assert "BAFF" not in str(llm.requests[1]["messages"])           # memory answer never kept
    assert "without querying" in str(llm.requests[1]["system"])


def test_model_that_never_queries_gets_safe_reply(fake_llm):
    fake_llm([reply([text("memory 1")], "end_turn"), reply([text("memory 2")], "end_turn")])
    answer, log, _ = run("q")
    assert answer == chat.NO_QUERY_REPLY and log == []


def test_grounding_and_coverage_rules_in_prompt():
    p = chat.SYSTEM_PROMPT
    assert "GROUNDING" in p and "background knowledge" in p
    assert "VAERS" in p and "Never issue more than 8 tool calls" in p


def test_tool_budget_forces_final_answer(fake_llm):
    loop = [reply([tool_use(f"t{i}", "ping")], "tool_use") for i in range(chat.MAX_TOOL_ROUNDS)]
    llm = fake_llm(loop + [reply([text("final")], "end_turn")])
    answer, log, _ = run()
    assert answer == "final"
    assert llm.requests[-1]["tool_choice"] == {"type": "none"}
    assert len(log) == chat.MAX_TOOL_ROUNDS


def test_cut_off_plan_is_retried_not_executed(fake_llm):
    llm = fake_llm([
        reply([tool_use("bad", "ping")], "max_tokens"),
        reply([tool_use("t1", "ping")], "tool_use"),
        reply([text("ok")], "end_turn"),
    ])
    answer, log, _ = run()
    assert answer == "ok" and len(log) == 1
    assert any("cut off" in str(r["system"]) for r in llm.requests[1:])


def test_empty_reply_gets_helpful_fallback(fake_llm):
    fake_llm([reply([tool_use("t1", "ping")], "tool_use"), reply([text("  ")], "end_turn")])
    answer, _, _ = run()
    assert "couldn't put an answer together" in answer


def test_parallel_calls_keep_order_and_parse_visual_data(fake_llm):
    llm = fake_llm(
        [reply([tool_use("a", "calculate_reporting_odds_ratio", {"drug_name": "x"}),
                tool_use("b", "count_adverse_events", {"pivot": "year"})], "tool_use"),
         reply([text("done")], "end_turn")],
        results={"calculate_reporting_odds_ratio": '{"ror": 2.0, "contingency_table": {}}',
                 "count_adverse_events": '{"pivot": "year", "counts": []}'},
    )
    _, log, _ = run()
    assert [c["name"] for c in log] == ["calculate_reporting_odds_ratio", "count_adverse_events"]
    assert log[0]["data"]["ror"] == 2.0
    assert [r["tool_use_id"] for r in tool_results(llm.requests[1])] == ["a", "b"]


def test_tool_failure_is_reported_to_model_not_raised(fake_llm):
    llm = fake_llm([reply([tool_use("t1", "ping")], "tool_use"), reply([text("explained")], "end_turn")])

    async def boom(name, arguments=None):
        raise RuntimeError("openFDA down")

    llm.session.call_tool = boom
    answer, log, _ = run()
    assert answer == "explained"
    assert "tool call failed" in log[0]["result_preview"]


def test_large_tool_result_is_truncated_before_model(fake_llm):
    big = "x" * (chat.MAX_TOOL_RESULT_CHARS + 5000)
    llm = fake_llm([reply([tool_use("t1", "ping")], "tool_use"), reply([text("ok")], "end_turn")],
                   results={"ping": big})
    run()
    sent = tool_results(llm.requests[1])[0]["content"]
    assert len(sent) < len(big) and "truncated" in sent


def test_prompt_caching_on_system_and_tools(fake_llm):
    llm = fake_llm([reply([tool_use("t1", "ping")], "tool_use"), reply([text("ok")], "end_turn")])
    run()
    req = llm.requests[0]
    assert req["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert req["tools"][-1]["cache_control"] == {"type": "ephemeral"}


def test_history_trimmed_and_starts_with_user():
    hist = [{"role": r, "content": str(i)} for i, r in enumerate(["user", "assistant"] * 10)]
    trimmed = chat._trim_history(hist)
    assert len(trimmed) <= chat.MAX_HISTORY_MESSAGES and trimmed[0]["role"] == "user"
    assert chat._trim_history([{"role": "assistant", "content": "x"}]) == []


def test_block_serialisation():
    assert chat._block_to_dict(text("   ")) is None
    assert chat._block_to_dict(text("hi")) == {"type": "text", "text": "hi"}
    tu = chat._block_to_dict(tool_use("i", "ping", {"a": 1}))
    assert tu == {"type": "tool_use", "id": "i", "name": "ping", "input": {"a": 1}}

    class Thinking:
        type = "thinking"

        def model_dump(self, exclude_none=True):
            return {"type": "thinking", "thinking": "...", "signature": "sig"}

    assert chat._block_to_dict(Thinking())["signature"] == "sig"
    assert chat._block_to_dict(NS(type="mystery")) is None


# --- hard guards -------------------------------------------------------------

def test_off_topic_question_never_reaches_main_model(fake_llm):
    llm = fake_llm([])
    llm.gate_answer = "no"
    answer, log, _ = run("Tell me from your memory about Meta stock")
    assert answer == chat.OUT_OF_SCOPE_REPLY
    assert llm.requests == [] and log == [] and llm.session.calls == []


def test_gate_fails_open_but_prompt_scope_still_present(fake_llm):
    llm = fake_llm([reply([tool_use("t1", "ping")], "tool_use"), reply([text("ok")], "end_turn")])
    llm.gate_answer = "error"
    answer, _, _ = run("metformin reactions")
    assert answer == "ok" and "SCOPE" in chat.SYSTEM_PROMPT


def test_memory_admission_is_replaced(fake_llm):
    fake_llm([reply([tool_use("t1", "ping")], "tool_use"),
              reply([text("I'm answering this from memory, not from a tool. Meta trades...")], "end_turn")])
    answer, _, _ = run("q")
    assert answer == chat.MEMORY_REPLY


def test_query_injection_rejected_before_server(fake_llm):
    evil = 'aspirin" OR patient.drug.openfda.generic_name:"ibuprofen'
    llm = fake_llm([reply([tool_use("t1", "count_adverse_events", {"drug_name": evil})], "tool_use"),
                    reply([text("explained")], "end_turn")])
    _, log, _ = run("q")
    assert llm.session.calls == []
    assert "input rejected" in log[0]["result_preview"]


def test_arg_checks():
    ok, err = chat._check_tool_args({"drug_name": "PEMBROLIZUMAB AND BERAHYALURONIDASE ALFA", "country": "us",
                                     "start_date": "2020-01-01", "limit": 5000, "min_age": 65})
    assert err is None and ok["country"] == "US" and ok["limit"] == 100
    for bad in ({"drug_name": 'a"b'}, {"reaction": "x:y"}, {"firm": "pfizer*"}, {"country": "USA"},
                {"start_date": "not-a-date"}, {"end_date": "2020-13-01"}, {"min_age": -1}, {"drug_name": "x" * 200}):
        assert chat._check_tool_args(bad)[1], bad


def test_rejected_tool_choice_is_retried_without_it(fake_llm):
    import anthropic, httpx
    loop = [reply([tool_use(f"t{i}", "ping")], "tool_use") for i in range(chat.MAX_TOOL_ROUNDS)]
    llm = fake_llm(loop + [reply([text("final")], "end_turn")])
    real_create_calls = llm.requests
    orig = chat.AsyncAnthropic

    class Strict(orig):  # rejects any tool_choice, like some models do
        def __init__(self, api_key):
            super().__init__(api_key)
            inner = self.messages.create

            async def create(**kw):
                if "tool_choice" in kw:
                    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
                    raise anthropic.BadRequestError(
                        'tool_choice: type "none" is not supported', response=httpx.Response(400, request=req), body=None)
                return await inner(**kw)
            self.messages.create = create

    chat.AsyncAnthropic = Strict
    try:
        answer, _, _ = run()
    finally:
        chat.AsyncAnthropic = orig
    assert answer == "final" and "tool_choice" not in real_create_calls[-1]
