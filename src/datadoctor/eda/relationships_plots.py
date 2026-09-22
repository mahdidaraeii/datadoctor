"""The drawing for the relationships figures: a correlation heatmap and a target distribution.

The target distribution reuses ``univariate_plots.draw_numeric``/``draw_categorical``, the same
code that draws each column on its own in S20, so a histogram or a bar chart looks and behaves
the same wherever it appears.
"""

from pathlib import Path
from typing import Any

import numpy as np

from datadoctor.core.config import AnalysisConfig
from datadoctor.eda.univariate_plots import draw_categorical, draw_numeric
from datadoctor.viz import figure_path, new_figure, save_figure

# Heatmap size grows with the number of columns, between these bounds, in inches.
CELL_INCHES = 0.5
MIN_HEATMAP_SIDE = 4.0
MAX_HEATMAP_SIDE = 16.0
# Cells are annotated with their value only up to this many columns; more would be unreadable.
MAX_ANNOTATED = 15
TARGET_FIGURE_SIZE = (8.0, 5.0)

_LABEL_LENGTH = 16


def save_heatmap(names: list[str], matrix: np.ndarray, config: AnalysisConfig) -> Path:
    """Draw a Pearson correlation matrix as a heatmap and save it.

    ``names`` are the display names of the columns behind ``matrix``, in the same order. Cells
    are annotated with their value when the matrix is small enough to read, and left blank (with
    a dash) where there were too few rows in common to compute a correlation.
    """
    n = len(names)
    side = min(MAX_HEATMAP_SIDE, max(MIN_HEATMAP_SIDE, CELL_INCHES * n + 2.5))
    with new_figure(size=(side, side)) as (fig, axis):
        image = axis.imshow(matrix, vmin=-1, vmax=1, cmap="RdBu_r")
        labels = [_cut(name, _LABEL_LENGTH) for name in names]
        axis.set_xticks(range(n), labels, rotation=45, ha="right", fontsize=8)
        axis.set_yticks(range(n), labels, fontsize=8)
        if n <= MAX_ANNOTATED:
            _annotate(axis, matrix)
        fig.colorbar(image, ax=axis, shrink=0.8, label="Pearson correlation")
        axis.set_title("Correlation between numeric columns")
    return save_figure(fig, figure_path(config, "correlation_heatmap"))


def _annotate(axis, values: np.ndarray) -> None:
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            if np.isnan(value):
                axis.text(column, row, "—", ha="center", va="center", fontsize=7)
                continue
            colour = "white" if abs(value) >= 0.5 else "black"
            axis.text(
                column, row, f"{value:.2f}", ha="center", va="center", fontsize=7, color=colour
            )


def save_target_distribution(kind: str, drawing: dict[str, Any], config: AnalysisConfig) -> Path:
    """Draw the target's own distribution: a histogram for a numeric target, bars for classes."""
    with new_figure(size=TARGET_FIGURE_SIZE) as (fig, axis):
        if kind == "numeric":
            draw_numeric(axis, drawing)
        else:
            draw_categorical(axis, drawing)
        axis.set_title(f"Distribution of the target, {drawing['name']}")
        axis.tick_params(labelsize=9)
    return save_figure(fig, figure_path(config, "target_distribution"))


def _cut(text: str, length: int) -> str:
    return text if len(text) <= length else text[: length - 1] + "…"
