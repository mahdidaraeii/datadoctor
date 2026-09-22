"""Exploratory data analysis: distributions and summaries, drawn and saved as files.

``check_univariate`` describes each column on its own. It takes a ``Dataset`` and an
``AnalysisConfig`` and returns an ``AnalysisResult`` whose ``artifacts`` are the saved figures.
"""

from datadoctor.eda.univariate import check_univariate

__all__ = ["check_univariate"]
