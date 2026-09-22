"""The drawing for the univariate figures: one page of small plots per call.

Each figure is a grid of small plots, one per column, of one kind of column. A numeric column is a
histogram, or one bar per value when it holds few whole numbers. A categorical column is a bar for
each of its most frequent values. Whatever the data is, a figure is a file, and nothing is shown.
"""

from pathlib import Path
from typing import Any

import numpy as np

from datadoctor.core.config import AnalysisConfig
from datadoctor.viz import figure_path, new_figure, save_figure
from datadoctor.viz.base import PALETTE

COLUMNS = 4
PER_PAGE = 3 * COLUMNS
PAGE_WIDTH = 14.0  # inches
ROW_HEIGHT = 3.2  # inches, for each row of small plots
TITLE_HEIGHT = 0.7  # inches, for the title of the whole figure
MAX_BINS = 50
# A categorical plot has room for at least this many bars, so that one or two categories are not
# drawn as bars as tall as the plot.
MIN_BAR_SLOTS = 3
MAX_TICKS = 5

_TITLE_LENGTH = 28
_LABEL_LENGTH = 22
_KIND_TITLES = {"numeric": "Numeric columns", "categorical": "Categorical columns"}


def save_page(
    kind: str, page: int, pages: int, columns: list[dict[str, Any]], config: AnalysisConfig
) -> Path:
    """Draw up to ``PER_PAGE`` columns of one kind on one figure, save it and return its path."""
    draw = _draw_numeric if kind == "numeric" else _draw_categorical
    rows = -(-len(columns) // COLUMNS)  # only as many rows as the columns need
    size = (PAGE_WIDTH, ROW_HEIGHT * rows + TITLE_HEIGHT)
    with new_figure(rows, COLUMNS, size=size) as (fig, axes):
        axes_list = list(np.atleast_1d(axes).flat)
        for axis, column in zip(axes_list, columns, strict=False):
            draw(axis, column)
            axis.set_title(_cut(column["name"], _TITLE_LENGTH))
            axis.tick_params(labelsize=8)
        for axis in axes_list[len(columns) :]:
            axis.set_visible(False)
        fig.suptitle(f"{_KIND_TITLES[kind]}, page {page} of {pages}", fontsize=14)
    return save_figure(fig, figure_path(config, f"univariate_{kind}_{page}"))


def _draw_numeric(axis, column: dict[str, Any]) -> None:
    discrete = column["discrete"]
    if discrete is not None:
        values, counts = discrete
        axis.bar(range(len(values)), counts, color=PALETTE[0])
        axis.set_xticks(range(len(values)), [f"{value:g}" for value in values])
    else:
        finite, shown = column["values"], column["range_shown"]
        inside = finite
        if shown is not None:
            inside = finite[(finite >= shown["low"]) & (finite <= shown["high"])]
        edges = np.histogram_bin_edges(inside, bins="auto")
        if len(edges) - 1 > MAX_BINS:
            edges = np.histogram_bin_edges(inside, bins=MAX_BINS)
        axis.hist(inside, bins=edges, color=PALETTE[0])
        if shown is not None:
            _draw_overflow(axis, shown, finite, edges)
        _tidy_ticks(axis.xaxis)
    _tidy_ticks(axis.yaxis)
    note = f"n={column['count']:,}, missing {column['missing']:,}"
    if column["infinite"]:
        note += f", infinite {column['infinite']:,}"
    axis.set_xlabel(note, fontsize=8)


def _draw_overflow(axis, shown: dict[str, Any], finite: np.ndarray, edges: np.ndarray) -> None:
    """One orange bar beyond each end of the range for the values that lie outside it.

    A bar stands one bin beyond the last bar of the histogram, so it is set apart from it and there
    is no empty stretch of axis between them. Its position is not a value: the label says how many
    values it stands for, and the range they lie beyond. The label is in the top corner on its
    side, joined to the bar by a thin line, so it never sits on top of the histogram.
    """
    width = edges[1] - edges[0]
    ends = [
        ("above", edges[-1] + width, "above", finite.max(), "max", 0.98, "right"),
        ("below", edges[0] - width, "below", finite.min(), "min", 0.02, "left"),
    ]
    for side, position, word, extreme, name, x_corner, align in ends:
        count = shown[side]
        if not count:
            continue
        axis.bar(position, count, width=width, color=PALETTE[1])
        label = f"{count:,} {word} {_rounded(shown['high' if side == 'above' else 'low'])}"
        label += f"\n({name} {_rounded(extreme)})"
        axis.annotate(
            label,
            xy=(position, count),
            xytext=(x_corner, 0.92),
            textcoords="axes fraction",
            ha=align,
            va="top",
            fontsize=8,
            arrowprops={"arrowstyle": "-", "color": PALETTE[1], "linewidth": 0.8},
        )


def _draw_categorical(axis, column: dict[str, Any]) -> None:
    counts = [entry["count"] for entry in column["top"]]
    labels = [entry.get("label") for entry in column["top"]]
    if column["other_count"]:
        counts.append(column["other_count"])
        labels.append(f"other ({column['other_distinct']:,} more)")
    positions = range(len(counts))
    axis.barh(positions, counts, color=PALETTE[0])
    # The most frequent value on top, and room for a few bars whatever their number.
    axis.set_ylim(max(len(counts), MIN_BAR_SLOTS) - 0.5, -0.5)
    _tidy_ticks(axis.xaxis)
    if column["labels_hidden"]:
        axis.set_yticks([])
        axis.text(
            0.98,
            0.05,
            "labels hidden:\nmay hold personal data",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
        )
    else:
        axis.set_yticks(list(positions), [_cut(str(label), _LABEL_LENGTH) for label in labels])
    note = f"n={column['count']:,}, missing {column['missing']:,}"
    axis.set_xlabel(note, fontsize=8)


def _compact(value: float, _position: int) -> str:
    """A tick label short enough to sit beside its neighbours: 20000 as 20k, 1500000 as 1.5M."""
    if abs(value) >= 1e6:
        return f"{value / 1e6:g}M"
    if abs(value) >= 1e4:
        return f"{value / 1e3:g}k"
    return f"{value:g}"


def _rounded(value: float) -> str:
    """A number to three significant digits, written short: 131953 as 132k, 226.67 as 227."""
    return _compact(float(f"{value:.3g}"), 0)


def _tidy_ticks(axis_object) -> None:
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    axis_object.set_major_locator(MaxNLocator(nbins=MAX_TICKS))
    axis_object.set_major_formatter(FuncFormatter(_compact))


def _cut(text: str, length: int) -> str:
    return text if len(text) <= length else text[: length - 1] + "…"
