"""The Python functions behind each tool. Plain functions: arguments in, JSON-able dict out."""
from . import charts, config, db, results


def list_tables() -> dict:
    out = db._execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    return {"tables": [r[0] for r in out["rows"]]}


def describe_table(table: str) -> dict:
    known = list_tables()["tables"]

    if table not in known: 
        raise ValueError(f"Unknown table '{table}'. Available tables: {known}.")
    
    cols = db._execute(f'PRAGMA table_info("{table}")')
    sample = db._execute(f'SELECT * FROM "{table}" LIMIT 3')

    return {
        "table": table,
        "columns": [{"name": r[1], "type": r[2]} for r in cols["rows"]],
        "sample_rows": sample["rows"],
        "sample_columns": sample["columns"],
    }


def run_sql(query: str) -> dict:
    out = db.run_query(query)  # authorizer + read-only connection: only reads are allowed
    result_id = results.store(out["columns"], out["rows"])

    return {
        "result_id": result_id,
        "columns": out["columns"],
        "row_count": len(out["rows"]),
        "truncated": out["truncated"],  
        "rows": out["rows"][: config.PREVIEW_ROWS],
    }


def make_chart(result_id: str, chart_type: str, x: str, title: str, y: str | None = None) -> dict:
    data = results.get(result_id)
    name = charts.render_chart(data["columns"], data["rows"], chart_type, x, y, title)
    
    return {"chart_url": f"/charts/{name}"}
