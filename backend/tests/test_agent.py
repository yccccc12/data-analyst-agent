"""Tests for the agent loop in agent.py, driven by a scripted fake model (no network)."""
import json

import pytest

from app import agent, config
from tests.conftest import calls, final, tool_call, Response


def outputs(history):
    return [i for i in history if i.get("type") == "function_call_output"]


def test_direct_answer_without_tools(fake_llm):
    fake = fake_llm([final("Hello!")])
    history = []
    out = agent.run(history, "hi")
    assert out == {"answer": "Hello!", "chart_urls": []}
    assert len(fake.inputs) == 1
    assert history[0] == {"role": "system", "content": agent.SYSTEM_PROMPT}
    assert history[1] == {"role": "user", "content": "hi"}
    assert history[-1]["type"] == "message"             # the model's reply is kept for next turn


def test_tool_call_then_answer(fake_llm):
    fake = fake_llm([calls(tool_call("list_tables", {}, "c1")), final("One table: sales.")])
    history = []
    out = agent.run(history, "what tables?")
    assert out["answer"] == "One table: sales."
    assert len(fake.inputs) == 2

    (result,) = outputs(history)
    assert result["call_id"] == "c1"                     # answers the matching call
    assert json.loads(result["output"]) == {"tables": ["sales"]}
    # the second model call saw the tool result
    assert any(i.get("type") == "function_call_output" for i in fake.inputs[1])


def test_reasoning_and_call_items_are_replayed_into_history(fake_llm):
    fake_llm([calls(tool_call("list_tables", {})), final("ok")])
    history = []
    agent.run(history, "q")
    types = [i.get("type") or i.get("role") for i in history]
    assert types == ["system", "user", "reasoning", "function_call", "function_call_output", "message"]


def test_parallel_tool_calls_each_get_an_output(fake_llm):
    fake_llm([
        calls(tool_call("list_tables", {}, "a"), tool_call("describe_table", {"table": "sales"}, "b")),
        final("done"),
    ])
    history = []
    agent.run(history, "q")
    assert [o["call_id"] for o in outputs(history)] == ["a", "b"]


def test_tool_error_is_fed_back_and_the_loop_continues(fake_llm):
    fake = fake_llm([
        calls(tool_call("run_sql", {"query": "DROP TABLE sales"}, "bad")),
        final("I can't modify data."),
    ])
    history = []
    out = agent.run(history, "delete everything")
    assert out["answer"] == "I can't modify data."
    (result,) = outputs(history)
    assert result["output"].startswith("Error: ") and "not allowed" in result["output"]
    assert len(fake.inputs) == 2                          # the model got to react to the error


def test_unknown_tool_does_not_crash(fake_llm):
    fake_llm([calls(tool_call("hack_the_planet", {})), final("sorry")])
    history = []
    assert agent.run(history, "q")["answer"] == "sorry"
    assert "unknown tool" in outputs(history)[0]["output"]


def test_chart_urls_are_collected(fake_llm):
    def run_sql_step(history):
        return calls(tool_call("run_sql", {"query": "SELECT category, SUM(sales) AS s FROM sales GROUP BY 1"}, "q1"))

    def chart_step(history):
        rid = json.loads(outputs(history)[-1]["output"])["result_id"]
        return calls(tool_call("make_chart", {"result_id": rid, "chart_type": "bar", "x": "category", "y": "s", "title": "T"}, "q2"))

    fake_llm([run_sql_step, chart_step, final("Here is the chart.")])
    out = agent.run([], "chart sales by category")
    assert len(out["chart_urls"]) == 1 and out["chart_urls"][0].startswith("/charts/")


def test_failed_chart_call_adds_no_url(fake_llm):
    fake_llm([
        calls(tool_call("make_chart", {"result_id": "nope", "chart_type": "bar", "x": "a", "title": "t"})),
        final("could not chart"),
    ])
    assert agent.run([], "chart")["chart_urls"] == []


def test_system_prompt_is_added_only_once_per_session(fake_llm):
    fake_llm([final("one"), final("two")])
    history = []
    agent.run(history, "first")
    agent.run(history, "second")
    assert sum(1 for i in history if i.get("role") == "system") == 1
    assert [i["content"] for i in history if i.get("role") == "user"] == ["first", "second"]


def test_stops_at_max_iterations(fake_llm, monkeypatch):
    monkeypatch.setattr(config, "MAX_ITERATIONS", 3)
    fake = fake_llm([calls(tool_call("list_tables", {}, f"c{i}")) for i in range(10)])
    history = []
    out = agent.run(history, "loop forever")
    assert len(fake.inputs) == 3
    assert "3 steps" in out["answer"]
    assert history[-1] == {"role": "assistant", "content": out["answer"]}


def test_history_is_rolled_back_if_a_turn_fails(fake_llm):
    def boom(history):
        raise RuntimeError("network down")

    fake_llm([final("first answer"), calls(tool_call("list_tables", {})), boom])
    history = []
    agent.run(history, "ok question")
    before = list(history)

    with pytest.raises(RuntimeError):
        agent.run(history, "this one fails after a tool call")
    assert history == before                              # no dangling function_call left behind


# ---- print_tool_call (project rule: print every tool call) ----

def test_every_tool_call_is_printed(fake_llm, capsys):
    fake_llm([calls(tool_call("list_tables", {}), tool_call("describe_table", {"table": "sales"})), final("x")])
    agent.run([], "q")
    out = capsys.readouterr().out
    assert "list_tables" in out and "describe_table" in out
    assert out.count("[tool call #1]") == 2


def test_print_shows_args_and_result(capsys):
    agent.print_tool_call(2, "run_sql", '{"query": "SELECT 1"}', '{"rows": [[1]]}')
    out = capsys.readouterr().out
    assert "#2" in out and "run_sql" in out and "SELECT 1" in out and '"rows"' in out


def test_print_truncates_long_results(capsys):
    agent.print_tool_call(1, "run_sql", "{}", "x" * 5000)
    out = capsys.readouterr().out
    assert len(out) < 600 and "5000 chars" in out


def test_print_survives_unencodable_characters(monkeypatch):
    import io
    import sys

    # a console that can only show ASCII: a plain print() of "ï" or "☃" would raise
    buffer = io.BytesIO()
    console = io.TextIOWrapper(buffer, encoding="ascii", errors="strict", write_through=True)
    monkeypatch.setattr(sys, "stdout", console)
    agent.print_tool_call(1, "run_sql", "{}", "naïve ☃ text")
    output = buffer.getvalue().decode("ascii")
    assert "run_sql" in output and "na" in output


def test_run_stream_yields_events_in_order(fake_llm):
    from tests.conftest import Item
    thinking = Response([Item(type="reasoning", summary=[{"type": "summary_text", "text": "look first"}]),
                         tool_call("list_tables", {}, "c1")])
    fake_llm([thinking, final("Done.")])
    events = list(agent.run_stream([], "q"))
    assert [e["type"] for e in events] == ["thinking", "tool_call", "tool_result", "answer"]
    assert events[0]["text"] == "look first"
    assert events[1]["arguments"] == {} and events[2]["ok"] is True
    assert events[-1] == {"type": "answer", "text": "Done.", "chart_urls": []}


def test_run_stream_marks_failed_tool_results(fake_llm):
    fake_llm([calls(tool_call("no_such_tool", {}, "c1")), final("Sorry.")])
    result = [e for e in agent.run_stream([], "q") if e["type"] == "tool_result"][0]
    assert result["ok"] is False


def test_closing_the_stream_midway_rolls_the_turn_back(fake_llm):
    fake_llm([calls(tool_call("list_tables", {}, "c1")), final("Done.")])
    history = []
    stream = agent.run_stream(history, "q")
    next(stream)          # tool_call event: the turn is half done
    stream.close()        # client disconnected
    assert history == []


def test_closing_after_the_answer_keeps_the_turn(fake_llm):
    fake_llm([final("Hi.")])
    history = []
    stream = agent.run_stream(history, "q")
    next(stream)
    stream.close()
    assert history[-1]["type"] == "message"
