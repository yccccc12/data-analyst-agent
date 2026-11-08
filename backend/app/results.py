"""In-memory cache of full query results, keyed by result_id.

The model only ever sees a short preview of a result. make_chart later refers back to the
full data by result_id, so thousands of rows never go through the model's context.
The cache is process-wide (shared across sessions) and lost on restart - fine for learning.
"""
import uuid
from collections import OrderedDict

_MAX_ENTRIES = 50
_cache: "OrderedDict[str, dict]" = OrderedDict()


def store(columns: list[str], rows: list[list]) -> str:
    result_id = "r_" + uuid.uuid4().hex[:8]
    _cache[result_id] = {"columns": columns, "rows": rows}
    while len(_cache) > _MAX_ENTRIES:
        _cache.popitem(last=False)  # evict oldest
    return result_id


def get(result_id: str) -> dict:
    """Return {"columns", "rows"}; raises KeyError with a model-friendly message."""
    if result_id not in _cache:
        raise KeyError(f"Unknown result_id '{result_id}'. Run a query with run_sql first.")
    return _cache[result_id]
