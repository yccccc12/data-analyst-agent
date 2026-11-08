"""
Tool definitions: what the model is told, and how its requests are executed.

The function bodies are in tool_impl.py:
    list_tables() / describe_table(table) / run_sql(query)
    make_chart(result_id, chart_type, x, title, y=None)
"""
import inspect
import json
import sqlite3

from . import tool_impl

# What the model sees. Responses API format: flat, NOT nested under "function" like
# Chat Completions. The `description` fields act as prompts: the model decides *when* to call a tool from them.
TOOL_SCHEMAS = [
    {
        "type": "function",
        "name": "list_tables",
        "description": "List all tables in the database. Call this first if you don't know the schema.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "type": "function",
        "name": "describe_table",
        "description": "Get the columns and types of a table, plus 3 sample rows. Call before writing SQL.",
        "parameters": {
            "type": "object",
            "properties": {"table": {"type": "string", "description": "Table name from list_tables"}},
            "required": ["table"],
        },
    },
    {
        "type": "function",
        "name": "run_sql",
        "description": (
            "Run one read-only SQLite SELECT query (WITH ... SELECT is also allowed). "
            "Writes are rejected. Returns a result_id, the column names, the row count and a "
            "preview of the first rows only, so aggregate in SQL (GROUP BY, SUM, COUNT) "
            "instead of selecting raw rows. Pass the result_id to make_chart to draw a chart."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A single SQLite SELECT statement. Dates are stored as YYYY-MM-DD text; use strftime() for year/month.",
                }
            },
            "required": ["query"],
        },
    },
    {
        "type": "function",
        "name": "make_chart",
        "description": (
            "Draw a PNG chart from the result of an earlier run_sql call and return its URL. "
            "Run run_sql first and use its result_id. x and y must be column names of that result."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "result_id": {"type": "string", "description": "result_id returned by run_sql"},
                "chart_type": {"type": "string", "enum": ["bar", "line", "pie", "scatter", "hist"]},
                "x": {"type": "string", "description": "Column name for the x axis (labels for bar/line/pie; the values for hist)"},
                "y": {"type": "string", "description": "Column name with the numbers to plot. Not needed for hist."},
                "title": {"type": "string", "description": "Chart title"},
            },
            "required": ["result_id", "chart_type", "x", "title"],  # y is optional
        },
    },
]

# tool name: Python function, so a model request can be routed.
TOOLS = {
    "list_tables": tool_impl.list_tables,
    "describe_table": tool_impl.describe_table,
    "run_sql": tool_impl.run_sql,
    "make_chart": tool_impl.make_chart,
}


def execute_tool(name: str, arguments_json: str) -> str:
    """Run one tool call requested by the model and return a STRING result.

    `arguments_json` is the raw JSON string from the model (not a dict yet).
    Failures never raise: they come back as an "Error: ..." string, which the agent loop
    appends as the tool result so the model can read it and correct itself.
    """
    if name not in TOOLS:
        return f"Error: unknown tool '{name}'. Available tools: {list(TOOLS)}."

    try:
        args = json.loads(arguments_json or "{}")
    except json.JSONDecodeError as e:
        return f"Error: the arguments for {name} are not valid JSON ({e}). Send a JSON object."
    
    if not isinstance(args, dict):
        return f"Error: the arguments for {name} must be a JSON object, got {type(args).__name__}."

    fn = TOOLS[name]
    try:  # bind first, so only a wrong/missing argument name counts as "bad arguments"
        inspect.signature(fn).bind(**args)
    except TypeError as e:
        return f"Error: bad arguments for {name}: {e}"

    try:
        result = fn(**args)
    except (ValueError, KeyError, sqlite3.Error) as e:  # rejected SQL, bad column, bad result_id ...
        return f"Error: {e}"
    except Exception as e:  # safety net: a bug in a tool must not crash the agent loop
        return f"Error: unexpected {type(e).__name__} in {name}: {e}"

    return json.dumps(result, default=str)
