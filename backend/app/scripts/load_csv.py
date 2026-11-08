"""Load the e-commerce (Superstore) CSV into a SQLite database.

Run from the repo root:
    python backend/app/scripts/load_csv.py [path/to/file.csv]

Cleaning done (everything else is loaded as-is):
  - CSV is Windows-1252 encoded.
  - Column names -> snake_case.
  - Dates M/D/YYYY -> ISO YYYY-MM-DD (so SQLite date functions work).
  - postal_code stored as 5-char TEXT (leading zeros were lost in the CSV).
"""
import csv
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_DIR / "backend"))  # so `app.config` imports when run as a script

from app.config import DB_PATH  # noqa: E402  (honours the DB_PATH setting in .env)

DEFAULT_CSV = REPO_DIR / "data" / "e commerce dataset (2).csv"

SCHEMA = """
CREATE TABLE sales (
    row_id        INTEGER PRIMARY KEY,
    order_id      TEXT NOT NULL,   -- one order has several rows (one per product)
    order_date    TEXT NOT NULL,   -- YYYY-MM-DD
    ship_date     TEXT NOT NULL,   -- YYYY-MM-DD
    ship_mode     TEXT NOT NULL,
    customer_id   TEXT NOT NULL,
    customer_name TEXT NOT NULL,
    segment       TEXT NOT NULL,   -- Consumer / Corporate / Home Office
    country       TEXT NOT NULL,
    city          TEXT NOT NULL,
    state         TEXT NOT NULL,
    postal_code   TEXT NOT NULL,
    region        TEXT NOT NULL,   -- East / West / Central / South
    product_id    TEXT NOT NULL,
    category      TEXT NOT NULL,
    sub_category  TEXT NOT NULL,
    product_name  TEXT NOT NULL,
    sales         REAL NOT NULL,   -- revenue for the line, after discount
    quantity      INTEGER NOT NULL,
    discount      REAL NOT NULL,   -- fraction 0..1
    profit        REAL NOT NULL    -- can be negative
);
CREATE INDEX idx_sales_order_date ON sales(order_date);
CREATE INDEX idx_sales_customer ON sales(customer_id);
CREATE INDEX idx_sales_category ON sales(category, sub_category);
"""


def iso(d: str) -> str:
    return datetime.strptime(d, "%m/%d/%Y").date().isoformat()


def convert(r: dict) -> tuple:
    return (
        int(r["Row ID"]), r["Order ID"], iso(r["Order Date"]), iso(r["Ship Date"]),
        r["Ship Mode"], r["Customer ID"], r["Customer Name"], r["Segment"],
        r["Country"], r["City"], r["State"], r["Postal Code"].strip().zfill(5),
        r["Region"], r["Product ID"], r["Category"], r["Sub-Category"], r["Product Name"],
        float(r["Sales"]), int(r["Quantity"]), float(r["Discount"]), float(r["Profit"]),
    )


def main():
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV
    with open(csv_path, encoding="cp1252", newline="") as f:
        rows = [convert(r) for r in csv.DictReader(f)]

    # Build into a temp file, then swap it in, so a failed load never destroys the good DB.
    tmp_path = DB_PATH.with_suffix(".db.tmp")
    tmp_path.unlink(missing_ok=True)
    con = sqlite3.connect(tmp_path)
    try:
        con.executescript(SCHEMA)
        placeholders = ",".join("?" * len(rows[0]))
        con.executemany(f"INSERT INTO sales VALUES ({placeholders})", rows)
        con.commit()
    except Exception:
        con.close()
        tmp_path.unlink(missing_ok=True)
        raise
    con.close()
    os.replace(tmp_path, DB_PATH)

    print(f"Loaded {len(rows)} rows from {csv_path.name} -> {DB_PATH}")


if __name__ == "__main__":
    main()
