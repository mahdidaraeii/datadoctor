"""Plotting foundation: styled figures that are saved as files.

- ``new_figure``: a ``with`` block that gives a figure with one consistent look to draw on.
- ``save_figure``: write it to a PNG file and return the path.
- ``figure_path``: the file for a named figure inside ``AnalysisConfig.output_dir``.

Nothing here shows a figure. See ``datadoctor.viz.base``.
"""

from datadoctor.viz.base import figure_path, new_figure, save_figure

__all__ = ["figure_path", "new_figure", "save_figure"]
