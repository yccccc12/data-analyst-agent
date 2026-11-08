"""Tests for charts.py (matplotlib rendering) and results.py (result cache)."""
import pytest

from app import charts, config, results

COLUMNS = ["category", "total"]
ROWS = [["Furniture", 10.0], ["Technology", 25.5], ["Office Supplies", 18.0]]


@pytest.fixture(autouse=True)
def tmp_charts_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CHARTS_DIR", tmp_path)   # don't litter backend/charts/
    return tmp_path


@pytest.mark.parametrize("chart_type, x, y", [
    ("bar", "category", "total"),
    ("line", "category", "total"),
    ("pie", "category", "total"),
    ("scatter", "total", "total"),
    ("hist", "total", None),                              # hist needs no y
])
def test_each_chart_type_writes_a_png(tmp_charts_dir, chart_type, x, y):
    name = charts.render_chart(COLUMNS, ROWS, chart_type, x, y, "Title")
    path = tmp_charts_dir / name
    assert name.endswith(".png")
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.mark.parametrize("kwargs, expected_in_error", [
    (dict(chart_type="donut", x="category", y="total"), "chart_type must be one of"),
    (dict(chart_type="bar", x="zzz", y="total"), "x column 'zzz'"),
    (dict(chart_type="bar", x="category", y="zzz"), "y column 'zzz'"),
    (dict(chart_type="bar", x="category", y=None), "y is required"),
    (dict(chart_type="pie", x="category", y="category"), "NULL or non-numeric"),
])
def test_bad_input_raises_value_error_naming_the_problem(kwargs, expected_in_error):
    with pytest.raises(ValueError) as exc:
        charts.render_chart(COLUMNS, ROWS, title="t", **kwargs)
    assert expected_in_error in str(exc.value)


def test_empty_result_cannot_be_charted():
    with pytest.raises(ValueError, match="no rows"):
        charts.render_chart(COLUMNS, [], "bar", "category", "total", "t")


def test_each_chart_gets_its_own_file(tmp_charts_dir):
    a = charts.render_chart(COLUMNS, ROWS, "bar", "category", "total", "t")
    b = charts.render_chart(COLUMNS, ROWS, "bar", "category", "total", "t")
    assert a != b and len(list(tmp_charts_dir.glob("*.png"))) == 2


# ---- results cache ----

def test_store_and_get_roundtrip():
    rid = results.store(COLUMNS, ROWS)
    assert rid.startswith("r_")
    assert results.get(rid) == {"columns": COLUMNS, "rows": ROWS}


def test_unknown_result_id_has_a_helpful_message():
    with pytest.raises(KeyError, match="Run a query with run_sql first"):
        results.get("r_missing")


def test_cache_evicts_the_oldest_entries(monkeypatch):
    monkeypatch.setattr(results, "_MAX_ENTRIES", 3)
    ids = [results.store(COLUMNS, ROWS) for _ in range(5)]
    with pytest.raises(KeyError):
        results.get(ids[0])                               # oldest evicted
    assert results.get(ids[-1])                           # newest kept


def test_pie_rejects_negative_values():
    with pytest.raises(ValueError, match="non-negative"):
        charts.render_chart(["c", "v"], [["a", 1], ["b", -2]], "pie", "c", "v", "t")


def test_null_values_give_a_clear_error():
    with pytest.raises(ValueError, match="NULL or non-numeric"):
        charts.render_chart(["c", "v"], [["a", None]], "bar", "c", "v", "t")
