"""The agent loop.

Idea:
    history = [system prompt, ... earlier turns ..., {"role": "user", "content": question}]
    repeat up to config.MAX_ITERATIONS times:
        response = llm.call_model(history, tools=tools.TOOL_SCHEMAS)
        append response.output items to history
        if the response contains no tool calls:  -> the model is done, return its text
        for each tool call in the response:
            print it, run it with tools.execute_tool(), append the result to history

Response shapes, verified live against DashScope (Qwen, thinking on):

    response.output is a list of items, each with `.type`:

    {"type": "reasoning", "id": "...", "summary": [{"type": "summary_text", "text": "..."}]}
    {"type": "function_call", "id": "...", "call_id": "call_xxx", "name": "list_tables",
     "arguments": "{}", "status": "completed"}          # arguments is a JSON *string*
    {"type": "message", "role": "assistant", "status": "completed",
     "content": [{"type": "output_text", "text": "final answer", "annotations": []}]}

    A turn that wants a tool returns [reasoning, function_call]; after we send the
    result, the next response returns [reasoning, message] (no function_call => done).
    `response.output_text` is a shortcut for the final text.

    Continuing the conversation: append each output item to the history as a plain dict
    (`item.model_dump(exclude_none=True)`, reasoning items included), then answer every
    function_call with:
        {"type": "function_call_output", "call_id": <same call_id>, "output": "<string>"}
"""
import json
from collections.abc import Iterator

from . import config, llm, tools

SYSTEM_PROMPT = """\
You are a data analyst assistant. You answer questions about the data in a SQLite database \
by calling tools, and you can draw charts.

How to work:
1. If you don't know the schema, call list_tables, then describe_table. Never guess table \
or column names.
2. Use run_sql to get answers. Aggregate in SQL (GROUP BY, SUM, COUNT, AVG) instead of \
fetching raw rows; you only see a short preview of each result. The database is read-only, \
and only a single SELECT is allowed.
3. If a tool returns an error, read it, fix your query or arguments, and try again.
4. Draw a chart with make_chart only when the user asks for a chart/plot/graph/visualization, \
using the result_id of the run_sql call that produced the data. Pick the chart type that \
fits (line for trends over time, bar for categories, pie for shares of a whole).
5. Base every number in your answer on tool results. Do not invent figures.

Final answer: be concise. State the key numbers and what they mean in a sentence or two. \
If you made a chart, say what it shows (the image is delivered separately, so don't \
include a link or markdown image).\
"""

MAX_PRINT_CHARS = 300
KEEP_OUTPUT_CHARS = 500  # old tool outputs are cut to this many characters


def print_tool_call(iteration: int, name: str, arguments_json: str, result: str) -> None:
    """Print one tool call: iteration, tool name, arguments, and the truncated result/error."""
    shown = result if len(result) <= MAX_PRINT_CHARS else result[:MAX_PRINT_CHARS] + f"... [{len(result)} chars]"
    text = f"[tool call #{iteration}] {name}({arguments_json})\n    -> {shown}"
    try:
        print(text, flush=True)
    except UnicodeEncodeError:  # e.g. a Windows console that can't show some character
        print(text.encode("ascii", "backslashreplace").decode("ascii"), flush=True)


def _chart_url(tool_name: str, result: str) -> str | None:
    """Return the chart_url from a successful make_chart result, else None."""
    if tool_name != "make_chart" or result.startswith("Error"):
        return None
    try:
        return json.loads(result).get("chart_url")
    except (json.JSONDecodeError, AttributeError):
        return None


def compact_history(history: list, start: int) -> None:
    """Shrink items from finished turns (everything before index `start`) in place.

    Drops old reasoning items and truncates long tool outputs, so a long session does
    not outgrow the context window. Every function_call keeps its function_call_output.
    """
    old = [i for i in history[:start] if i.get("type") != "reasoning"]
    for item in old:
        if item.get("type") == "function_call_output" and len(item["output"]) > KEEP_OUTPUT_CHARS:
            item["output"] = item["output"][:KEEP_OUTPUT_CHARS] + "...[truncated]"
    history[:start] = old


def _thinking_texts(output_item: dict) -> list[str]:
    """Non-empty summary texts of one reasoning item (dict form)."""
    return [p["text"] for p in output_item.get("summary") or [] if p.get("text", "").strip()]


def run_stream(history: list, user_message: str) -> Iterator[dict]:
    """Run the agent loop for one user message, yielding an event dict at each step.

    Events (the contract documented in main.py):
        {"type": "thinking", "text"}        {"type": "tool_call", "id", "name", "arguments"}
        {"type": "tool_result", "id", "ok", "output", "chart_url"}
        {"type": "answer", "text", "chart_urls"}   # always the last event on success
    Failures raise (main.py turns them into an "error" event).

    `history` is the per-session list of Responses-API input items, owned by the caller
    (main.py keeps one per session_id). It is mutated in place so the next call in the
    same session remembers this turn. If the consumer stops early (client disconnect), the
    generator is closed, GeneratorExit lands on a `yield`, and the turn is rolled back too.
    """
    compact_history(history, len(history))  # shrink earlier turns first
    start_len = len(history)  # to roll back if this turn fails halfway
    if not history:
        history.append({"role": "system", "content": SYSTEM_PROMPT})
    history.append({"role": "user", "content": user_message})

    chart_urls: list[str] = []
    try:
        for iteration in range(1, config.MAX_ITERATIONS + 1):
            response = llm.call_model(history, tools=tools.TOOL_SCHEMAS)

            # Keep everything the model produced (reasoning, tool calls, message).
            items = [item.model_dump(exclude_none=True) for item in response.output]
            history.extend(items)
            for item in items:
                if item.get("type") == "reasoning":
                    for text in _thinking_texts(item):
                        yield {"type": "thinking", "text": text}

            calls = [item for item in response.output if item.type == "function_call"]
            if not calls:  # no tool requested -> this is the final answer
                status = getattr(response, "status", "completed")
                if status != "completed" or not response.output_text.strip():
                    raise RuntimeError(f"The model returned no usable answer (status={status}).")
                final = {"type": "answer", "text": response.output_text, "chart_urls": chart_urls}
                break

            # Every function_call needs its own function_call_output (parallel calls).
            for call in calls:
                try:
                    shown_args = json.loads(call.arguments or "{}")
                except json.JSONDecodeError:
                    shown_args = {"_raw": call.arguments}
                yield {"type": "tool_call", "id": call.call_id, "name": call.name, "arguments": shown_args}

                result = tools.execute_tool(call.name, call.arguments)
                print_tool_call(iteration, call.name, call.arguments, result)
                history.append({"type": "function_call_output", "call_id": call.call_id, "output": result})

                url = _chart_url(call.name, result)
                if url:
                    chart_urls.append(url)
                yield {"type": "tool_result", "id": call.call_id, "ok": not result.startswith("Error"),
                       "output": result, "chart_url": url}

        else:  # ran out of iterations without a final answer
            answer = f"I couldn't finish within {config.MAX_ITERATIONS} steps. Try a more specific question."
            history.append({"role": "assistant", "content": answer})
            final = {"type": "answer", "text": answer, "chart_urls": chart_urls}
    except BaseException:  # includes Ctrl-C and GeneratorExit (stream closed early)
        # A dangling function_call without an output would make every later call fail,
        # so drop this turn's partial items before re-raising. (After a normal finish the
        # code never gets here, so a finished turn is never rolled back.)
        del history[start_len:]
        raise
    yield final  # outside the try: a consumer closing after this cannot roll back a finished turn


def run(history: list, user_message: str) -> dict:
    """Run the agent loop to completion. Returns {"answer": str, "chart_urls": list[str]}."""
    final = None
    for event in run_stream(history, user_message):  # consume it all: closing early would roll back
        if event["type"] == "answer":
            final = event
    return {"answer": final["text"], "chart_urls": final["chart_urls"]}


if __name__ == "__main__":
    # Quick CLI chat:  python -m app.agent            (run from backend/)
    #                  python -m app.agent "question"  (one-shot)
    import sys

    session: list = []
    if len(sys.argv) > 1:
        questions = [" ".join(sys.argv[1:])]
    else:
        questions = iter(lambda: input("\nyou> ").strip(), "exit")
    for q in questions:
        if q:
            out = run(session, q)
            print(f"\nagent> {out['answer']}")
            for url in out["chart_urls"]:
                print(f"chart: {url}")
