"""Tests for the read-only guard in app/db.py: the authorizer (layer 1) and the read-only
connection (layer 2).

Run from backend/:   python -m pytest
"""
import sqlite3

import pytest

from app import config, db

GOOD_QUERIES = [
    "SELECT * FROM sales LIMIT 1",
    "select category from sales limit 1 -- note",
    "WITH t AS (SELECT 1) SELECT * FROM t",
    "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c WHERE x<3) SELECT * FROM c",
    "SELECT 1 AS updated_at",                          # keyword only as part of a word
    "SELECT 1;",                                       # a trailing ; is harmless
    "  select\n 1",                                    # whitespace / newlines / lowercase
    "SELECT replace(product_name, 'a', 'b') FROM sales LIMIT 1",     # replace() is a normal function
    "SELECT strftime('%Y', order_date) y, SUM(sales) FROM sales GROUP BY y",
    # These were false alarms for a keyword/regex check. The authorizer judges what the
    # statement does, not which words appear in it.
    "SELECT COUNT(*) FROM sales WHERE product_name LIKE '%delete%'",
    "SELECT 'drop table sales' AS note",
    "SELECT 1 /* drop table sales */",
    "SELECT * FROM sqlite_master LIMIT 1",             # reading the schema is harmless
]

BAD_QUERIES = [
    "",
    "   ",
    "-- just a comment",
    "/* x */",
    "DROP TABLE sales",
    "DROP INDEX idx_sales_order_date",
    "SELECT 1; DELETE FROM sales",
    "/* hi */ DELETE FROM sales",
    "PRAGMA table_info(sales)",
    "PRAGMA query_only = OFF",                         # can't switch the safety off
    "SELECT '--'; DROP TABLE sales",                   # quoted -- must not hide the DROP
    "SELECT 1 /* ; */ ; DROP TABLE sales",
    "INSERT INTO sales (row_id) VALUES (1)",
    "INSERT OR REPLACE INTO sales (row_id) VALUES (1)",
    "REPLACE INTO sales VALUES (1)",
    "UPDATE sales SET sales = 0",
    "DELETE FROM sales",
    "WITH x AS (SELECT 1) DELETE FROM sales",          # a write hidden behind a WITH
    "CREATE TABLE x(a)",
    "CREATE TEMP TABLE x(a)",
    "CREATE VIEW v AS SELECT 1",
    "CREATE INDEX i ON sales(sales)",
    "ALTER TABLE sales ADD COLUMN z",
    "ATTACH DATABASE ':memory:' AS m",
    "BEGIN",
    "COMMIT",
    "SAVEPOINT a",
    "VACUUM",
]


@pytest.mark.parametrize("sql", GOOD_QUERIES)
def test_good_queries_run(sql):
    db.run_query(sql)  # must not raise


@pytest.mark.parametrize("sql", BAD_QUERIES)
def test_bad_queries_raise_value_error(sql):
    # ValueError specifically: any other exception would escape the error handling in
    # tools.py instead of being reported back to the model as a readable message.
    with pytest.raises(ValueError):
        db.run_query(sql)


def test_the_database_is_untouched_after_every_bad_query():
    for sql in BAD_QUERIES:
        with pytest.raises(ValueError):
            db.run_query(sql)
    assert db._execute("SELECT COUNT(*) FROM sales")["rows"] == [[9994]]
    assert db._execute("SELECT name FROM sqlite_master WHERE type='table'")["rows"] == [["sales"]]


@pytest.mark.parametrize("sql, expected_in_message", [
    ("", "empty"),
    ("-- x", "only comments"),
    ("SELECT 1; DROP TABLE sales", "one statement"),
    ("INSERT INTO sales (row_id) VALUES (1)", "INSERT is not allowed"),
    ("UPDATE sales SET sales = 0", "UPDATE is not allowed"),
    ("DELETE FROM sales", "DELETE is not allowed"),
    ("ALTER TABLE sales ADD COLUMN z", "ALTER is not allowed"),
    ("ATTACH DATABASE ':memory:' AS m", "ATTACH is not allowed"),
    ("PRAGMA table_info(sales)", "PRAGMA is not allowed"),
    ("DROP TABLE sales", "Changing the schema"),       # reported as what was asked, not the internal write
    ("CREATE TABLE x(a)", "Changing the schema"),
    ("BEGIN", "not allowed"),                           # no specific name: generic wording
])
def test_error_messages_tell_the_model_what_was_wrong(sql, expected_in_message):
    with pytest.raises(ValueError) as exc:
        db.run_query(sql)
    message = str(exc.value)
    assert expected_in_message in message
    assert "SELECT" in message                         # and says what IS allowed


@pytest.mark.parametrize("sql, expected_in_message", [
    ("SELEC 1", "syntax error"),
    ("SELECT nope FROM sales", "no such column"),
    ("SELECT * FROM nope", "no such table"),
])
def test_ordinary_sql_mistakes_pass_through_for_the_model_to_fix(sql, expected_in_message):
    # Not a security rejection, so not a ValueError: the original SQLite message is kept.
    with pytest.raises(sqlite3.OperationalError, match=expected_in_message):
        db.run_query(sql)


# ---- internal (trusted) queries skip the authorizer, e.g. describe_table needs PRAGMA ----

def test_internal_queries_may_use_pragma():
    out = db._execute("PRAGMA table_info(sales)")
    assert any(row[1] == "order_date" for row in out["rows"])


# ---- Layer 2: the connection itself must refuse writes, even with the authorizer bypassed ----

@pytest.mark.parametrize("stmt", [
    "DELETE FROM sales",
    "DROP TABLE sales",
    "CREATE TABLE x(a)",
    "UPDATE sales SET sales = 0",
])
def test_connection_is_read_only(stmt):
    with pytest.raises(sqlite3.OperationalError):
        db._execute(stmt)  # untrusted=False: no authorizer on purpose


# ---- Limits ----

def test_row_cap():
    out = db.run_query("SELECT * FROM sales")
    assert len(out["rows"]) == config.MAX_ROWS and out["truncated"] is True


def test_query_timeout(monkeypatch):
    monkeypatch.setattr(config, "QUERY_TIMEOUT_S", 0.2)
    slow = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c) SELECT count(*) FROM c"
    with pytest.raises(sqlite3.OperationalError, match="interrupted"):
        db.run_query(slow)


def test_run_query_returns_rows():
    out = db.run_query("SELECT category, COUNT(*) AS n FROM sales GROUP BY category")
    assert out["columns"] == ["category", "n"]
    assert len(out["rows"]) == 3
    assert out["truncated"] is False
