"""Figures that are saved as files, with one consistent look.

Figures are built with matplotlib's object-oriented API and rendered by the Agg canvas. Nothing
here uses matplotlib's global figure manager, so no window can open, nothing can be shown, no
backend is switched and no figure is kept alive by a registry. A figure is a file, and its path
goes into ``AnalysisResult.artifacts``.

matplotlib is imported inside the functions, so ``import datadoctor`` stays fast for commands that
draw nothing.
"""

import hashlib
import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from datadoctor.core.config import AnalysisConfig
from datadoctor.core.exceptions import ConfigError

if TYPE_CHECKING:
    from matplotlib.figure import Figure

FIGURE_SIZE = (7.0, 4.5)  # inches
DPI = 150
# The Okabe-Ito palette, which stays distinguishable with the common kinds of color blindness.
PALETTE = ("#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000")

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")
_MAX_STEM = 80


def _style() -> dict[str, Any]:
    """The look of every figure, as matplotlib settings, applied only while one is made or saved."""
    from cycler import cycler

    return {
        "axes.prop_cycle": cycler(color=list(PALETTE)),
        "axes.facecolor": "white",
        "axes.axisbelow": True,
        "axes.grid": True,
        "axes.labelsize": 10,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "figure.facecolor": "white",
        "font.family": "DejaVu Sans",  # bundled with matplotlib, so it is the same everywhere
        "font.size": 10,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "savefig.facecolor": "white",
    }


@contextmanager
def new_figure(
    nrows: int = 1, ncols: int = 1, *, size: tuple[float, float] | None = None
) -> Iterator[tuple["Figure", Any]]:
    """Create a styled figure with its axes, to draw on inside a ``with`` block.

    Whatever is drawn inside the block gets the common look. That includes titles, labels and
    legends, which matplotlib styles from its settings at the moment they are created, so they
    would look different if they were made after the block. The settings are put back when the
    block ends, and nothing global is changed.

    ::

        with new_figure() as (fig, ax):
            ax.hist(values)
            ax.set_title("Age")
        save_figure(fig, path)

    Args:
        nrows: Rows of axes.
        ncols: Columns of axes.
        size: Width and height in inches. Defaults to ``FIGURE_SIZE``.

    Yields:
        The figure and its axes: one ``Axes`` for a single plot, otherwise an array of them.
    """
    from matplotlib import rc_context
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    with rc_context(_style()):
        figure = Figure(figsize=size or FIGURE_SIZE, layout="constrained")
        FigureCanvasAgg(figure)
        yield figure, figure.subplots(nrows, ncols)


def save_figure(fig: "Figure", path: str | Path) -> Path:
    """Write a figure to a PNG file, creating its directory if needed.

    The same figure always gives the same bytes: the matplotlib version that ``savefig`` would
    otherwise write into the file is left out. An existing file with the same name is replaced.

    Args:
        fig: The figure to save, made with ``new_figure``.
        path: Where to write it. The suffix must be ``.png``.

    Returns:
        The path that was written, as given.

    Raises:
        ConfigError: The suffix is not ``.png``.
        OSError: The file or its directory cannot be created.
    """
    from matplotlib import rc_context

    path = Path(path)
    if path.suffix.lower() != ".png":
        raise ConfigError(f"figures are saved as .png files, got {path.name!r}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with rc_context(_style()):
        fig.savefig(path, format="png", dpi=DPI, metadata={"Software": None})
    return path


def figure_path(config: AnalysisConfig, name: str) -> Path:
    """The file a figure called ``name`` is saved to, inside ``config.output_dir``.

    A name often comes from a column of the data, and can hold slashes, dots or any other
    character. It is reduced to letters, digits, dots, underscores and hyphens, so the file always
    lands inside the output directory. Whenever that changes the name, or a long name is cut, a
    short hash of the original is added, so that different names never share a file, while the same
    name always gives the same path.
    """
    stem = _UNSAFE.sub("_", name).strip("._")
    changed = stem != name or len(stem) > _MAX_STEM
    if changed or not stem:
        digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:8]
        stem = f"{stem[:_MAX_STEM] or 'figure'}-{digest}"
    return config.output_dir / f"{stem}.png"
