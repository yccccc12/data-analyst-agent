"""Render a PNG chart from a cached query result with matplotlib."""
import uuid

import matplotlib

matplotlib.use("Agg")  # headless backend, must be set before importing pyplot
import matplotlib.pyplot as plt  # noqa: E402

from matplotlib.ticker import FuncFormatter  # noqa: E402

from . import config  # noqa: E402

CHART_TYPES = ("bar", "line", "pie", "scatter", "hist")
_THOUSANDS = FuncFormatter(lambda v, _: f"{v:,.10g}")  # 18000 -> 18,000; 0.5 stays 0.5


def _label(column: str) -> str:
    """total_profit -> Total profit"""
    return column.replace("_", " ").capitalize()


def _numbers(values: list, column: str) -> list[float]:
    """Convert a column to floats, or raise a model-friendly ValueError."""
    try:
        return [float(v) for v in values]
    except (TypeError, ValueError):
        raise ValueError(f"Column '{column}' has NULL or non-numeric values; filter them out in SQL.") from None


def render_chart(columns: list[str], rows: list[list], chart_type: str,
                 x: str, y: str | None, title: str) -> str:
    """Draw the chart and return the saved file name (inside config.CHARTS_DIR).

    Raises ValueError with a model-friendly message on bad input.
    """
    if chart_type not in CHART_TYPES:
        raise ValueError(f"chart_type must be one of {list(CHART_TYPES)}, got '{chart_type}'.")
    if x not in columns:
        raise ValueError(f"x column '{x}' not in result columns {columns}.")
    if chart_type != "hist" and y is None:
        raise ValueError(f"y is required for a {chart_type} chart.")
    if y is not None and y not in columns:
        raise ValueError(f"y column '{y}' not in result columns {columns}.")
    if not rows:
        raise ValueError("The result has no rows, so there is nothing to plot.")

    xs = [r[columns.index(x)] for r in rows]
    ys = [r[columns.index(y)] for r in rows] if y else None
    if chart_type == "hist":
        xs = _numbers(xs, x)
    elif chart_type == "scatter":
        xs = _numbers(xs, x)
        ys = _numbers(ys, y)
    else:
        ys = _numbers(ys, y)
    if chart_type == "pie" and any(v < 0 for v in ys):
        raise ValueError(f"A pie chart needs non-negative values, but '{y}' has negatives. Use a bar chart.")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    try:
        if chart_type == "bar":
            ax.bar([str(v) for v in xs], ys)
            ax.tick_params(axis="x", rotation=45)
        elif chart_type == "line":
            ax.plot([str(v) for v in xs], ys, marker="o")
            ax.tick_params(axis="x", rotation=45)
        elif chart_type == "pie":
            ax.pie(ys, labels=[str(v) for v in xs], autopct="%1.1f%%")
        elif chart_type == "scatter":
            ax.scatter(xs, ys)
        elif chart_type == "hist":
            ax.hist(xs, bins=20)
        if chart_type != "pie":
            ax.set_xlabel(_label(x))
            ax.set_ylabel(_label(y) if y else "Count")
            ax.yaxis.set_major_formatter(_THOUSANDS)
            if chart_type in ("scatter", "hist"):
                ax.xaxis.set_major_formatter(_THOUSANDS)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=0.3)
            ax.set_axisbelow(True)
            if chart_type in ("bar", "line"):
                for tick in ax.get_xticklabels():
                    tick.set_ha("right")  # rotated labels end under their tick, not centred on it
        ax.set_title(title)
        fig.tight_layout()
        name = f"{uuid.uuid4().hex[:12]}.png"
        fig.savefig(config.CHARTS_DIR / name, dpi=110)
    finally:
        plt.close(fig)
    return name
