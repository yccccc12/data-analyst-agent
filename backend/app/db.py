"""Read-only access to the SQLite database.

Two independent layers protect it:
  1. An authorizer (con.set_authorizer): before running a statement, SQLite asks our
     callback about every operation the statement would perform. We only say yes to
     reading data, so INSERT / DROP / ATTACH / PRAGMA ... are refused by SQLite itself,
     however the SQL is written (comments, quotes, odd syntax cannot fool it).
  2. A read-only connection (mode=ro + PRAGMA query_only): even without the authorizer,
     writes fail.
"""

import sqlite3
import time

from . import config

# Operations a model-written query may perform. Everything else is denied.
ALLOWED_ACTIONS = {
    sqlite3.SQLITE_SELECT,     # a SELECT statement
    sqlite3.SQLITE_READ,       # reading a column
    sqlite3.SQLITE_FUNCTION,   # calling a function such as SUM() or strftime()
    sqlite3.SQLITE_RECURSIVE,  # WITH RECURSIVE
}

# Names used in the error message for the most common denied operations.
ACTION_NAMES = {
    sqlite3.SQLITE_INSERT: "INSERT",
    sqlite3.SQLITE_UPDATE: "UPDATE",
    sqlite3.SQLITE_DELETE: "DELETE",
    sqlite3.SQLITE_DROP_TABLE: "DROP",
    sqlite3.SQLITE_CREATE_TABLE: "CREATE",
    sqlite3.SQLITE_ALTER_TABLE: "ALTER",
    sqlite3.SQLITE_ATTACH: "ATTACH",
    sqlite3.SQLITE_PRAGMA: "PRAGMA",
}

# CREATE / DROP / ALTER are carried out as writes to these internal tables, so that is
# what the authorizer sees first; report them as what the user actually asked for.
SCHEMA_TABLES = {"sqlite_master", "sqlite_schema", "sqlite_temp_master", "sqlite_temp_schema"}
SCHEMA_CHANGE = "Changing the schema (CREATE / DROP / ALTER)"


def get_connection() -> sqlite3.Connection:
    """Open the database read-only. Writes fail at the SQLite level."""
    con = sqlite3.connect(f"{config.DB_PATH.as_uri()}?mode=ro", uri=True)
    con.execute("PRAGMA query_only = ON")

    return con


def _execute(sql: str, untrusted: bool = False) -> dict:
    """Run a query on a read-only connection with a time limit and row cap.

    untrusted=True (model-written SQL) also installs the authorizer, and raises
    ValueError, with a message for the model, if the query tries anything but reading.
    untrusted=False is for our own internal queries (e.g. PRAGMA table_info).
    """
    con = get_connection()
    denied = []  # what the authorizer refused, for the error message

    if untrusted:
        def authorizer(action, arg1, arg2, db_name, source):
            if action in ALLOWED_ACTIONS:
                return sqlite3.SQLITE_OK
            if arg1 in SCHEMA_TABLES:
                denied.append(SCHEMA_CHANGE)
            else:
                denied.append(ACTION_NAMES.get(action, "This operation"))
            return sqlite3.SQLITE_DENY

        con.set_authorizer(authorizer)

    deadline = time.monotonic() + config.QUERY_TIMEOUT_S

    # Called every N SQLite VM steps; returning non-zero aborts the query.
    con.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
    try:
        cur = con.execute(sql)
        if cur.description is None:
            if untrusted:  # nothing to run: empty text or only comments
                raise ValueError("The query is empty or contains only comments. Send a single SELECT statement.")
            return {"columns": [], "rows": [], "truncated": False}
        columns = [d[0] for d in cur.description]
        fetched = cur.fetchmany(config.MAX_ROWS + 1)
    except sqlite3.DatabaseError as e:
        if untrusted and denied:
            raise ValueError(
                f"{denied[0]} is not allowed. The database is read-only; send a single SELECT (or WITH ... SELECT) query."
            ) from None
        if untrusted and "one statement" in str(e):
            raise ValueError("Only one statement is allowed. Remove the extra ';' and send a single SELECT query.") from None
        raise  # anything else (e.g. "no such column") is a normal SQL error the model can read
    finally:
        con.close()
    truncated = len(fetched) > config.MAX_ROWS
    rows = [list(r) for r in fetched[: config.MAX_ROWS]]

    return {"columns": columns, "rows": rows, "truncated": truncated}


def run_query(sql: str) -> dict:
    """Run model-written SQL. Raises ValueError (not allowed) or sqlite3.Error (failed)."""
    return _execute(sql, untrusted=True)
