"""Exploratory data analysis: distributions and summaries, drawn and saved as files.

``check_univariate`` describes each column on its own. ``check_relationships`` correlates the
numeric columns and compares every feature against the target. ``run_eda`` runs both and merges
their results. Each takes a ``Dataset`` and an ``AnalysisConfig`` and returns an
``AnalysisResult`` whose ``artifacts`` are the saved figures.
"""

from datadoctor.eda.relationships import check_relationships
from datadoctor.eda.run import run_eda
from datadoctor.eda.univariate import check_univariate

__all__ = ["check_relationships", "check_univariate", "run_eda"]
