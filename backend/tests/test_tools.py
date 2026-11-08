"""Tests for tools.py (schemas + execute_tool) and the function bodies in tool_impl.py."""
import json

import pytest

from app import config, tool_impl, tools


# ---- schemas ----

def test_every_schema_has_a_function_and_vice_versa():
    assert {s["name"] for s in tools.TOOL_SCHEMAS} == set(tools.TOOLS)


@pytest.mark.parametrize("schema", tools.TOOL_SCHEMAS, ids=lambda s: s["name"])
def test_schema_is_well_formed(schema):
    assert schema["type"] == "function"          # Responses API: flat, not nested under "function"
    assert schema["description"].strip()
    params = schema["parameters"]
    assert params["type"] == "object"
    assert set(params["required"]) <= set(params["properties"])


def test_make_chart_schema_restricts_chart_types():
    props = next(s for s in tools.TOOL_SCHEMAS if s["name"] == "make_chart")["parameters"]["properties"]
    assert set(props["chart_type"]["enum"]) == {"bar", "line", "pie", "scatter", "hist"}


# ---- execute_tool: success ----

def test_list_tables():
    assert json.loads(tools.execute_tool("list_tables", "{}")) == {"tables": ["sales"]}


def test_empty_arguments_string_is_treated_as_no_arguments():
    assert "tables" in json.loads(tools.execute_tool("list_tables", ""))


def test_run_sql_returns_result_id_and_preview():
    out = json.loads(tools.execute_tool("run_sql", json.dumps({"query": "SELECT category FROM sales"})))
    assert out["result_id"].startswith("r_")
    assert out["columns"] == ["category"]
    assert out["row_count"] == config.MAX_ROWS and out["truncated"] is True
    assert len(out["rows"]) == config.PREVIEW_ROWS     # the model only sees a preview


def test_make_chart_from_run_sql_result():
    rid = json.loads(tools.execute_tool(
        "run_sql", json.dumps({"query": "SELECT category, SUM(sales) AS total FROM sales GROUP BY 1"})))["result_id"]
    out = json.loads(tools.execute_tool(
        "make_chart", json.dumps({"result_id": rid, "chart_type": "bar", "x": "category", "y": "total", "title": "t"})))
    assert out["chart_url"].startswith("/charts/") and out["chart_url"].endswith(".png")


# ---- execute_tool: every failure comes back as an "Error: ..." string, never an exception ----

@pytest.mark.parametrize("name, arguments, expected_in_error", [
    ("nope", "{}", "unknown tool"),
    ("run_sql", "not json", "not valid JSON"),
    ("run_sql", "[1, 2]", "JSON object"),
    ("run_sql", "{}", "bad arguments"),                                   # missing argument
    ("run_sql", '{"sql": "SELECT 1"}', "bad arguments"),                  # wrong argument name
    ("run_sql", '{"query": "DROP TABLE sales"}', "not allowed"),  # refused by the authorizer
    ("run_sql", '{"query": "SELECT 1; SELECT 2"}', "one statement"),
    ("run_sql", '{"query": "SELECT nope FROM sales"}', "no such column"), # SQLite error
    ("describe_table", '{"table": "missing"}', "Unknown table"),
    ("make_chart", '{"result_id": "zzz", "chart_type": "bar", "x": "a", "title": "t"}', "Unknown result_id"),
])
def test_failures_are_returned_as_error_strings(name, arguments, expected_in_error):
    out = tools.execute_tool(name, arguments)
    assert isinstance(out, str)
    assert out.startswith("Error")
    assert expected_in_error in out


def test_unexpected_exception_in_a_tool_does_not_escape(monkeypatch):
    def broken():
        raise ZeroDivisionError("boom")

    monkeypatch.setitem(tools.TOOLS, "list_tables", broken)
    out = tools.execute_tool("list_tables", "{}")
    assert out.startswith("Error") and "ZeroDivisionError" in out


def test_non_json_values_are_serialised(monkeypatch):
    import datetime
    monkeypatch.setitem(tools.TOOLS, "list_tables", lambda: {"when": datetime.date(2020, 1, 2)})
    assert json.loads(tools.execute_tool("list_tables", "{}")) == {"when": "2020-01-02"}


# ---- tool_impl details ----

def test_describe_table_columns_and_samples():
    out = tool_impl.describe_table("sales")
    names = [c["name"] for c in out["columns"]]
    assert {"order_date", "sales", "profit", "category"} <= set(names)
    assert len(out["sample_rows"]) == 3


def test_describe_table_rejects_injection_through_the_table_name():
    with pytest.raises(ValueError):
        tool_impl.describe_table('sales"; DROP TABLE sales; --')
    assert tool_impl.list_tables()["tables"] == ["sales"]


def test_typeerror_inside_a_tool_is_not_reported_as_bad_arguments(monkeypatch):
    def boom(): raise TypeError("inside the tool")
    monkeypatch.setitem(tools.TOOLS, "list_tables", boom)
    out = tools.execute_tool("list_tables", "{}")
    assert "bad arguments" not in out and "inside the tool" in out
