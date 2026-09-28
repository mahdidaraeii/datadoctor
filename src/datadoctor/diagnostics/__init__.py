"""Diagnostics: leakage and split-strategy checks, the readiness assessment built on them, and
the recommendations engine that ranks everything together.

Each check function takes a ``Dataset`` and an ``AnalysisConfig`` and returns an
``AnalysisResult``, the same shape every analyzer in this project uses.
``build_recommendations`` instead takes the already-computed results from every analyzer and
returns a ranked tuple of ``Recommendation``.
"""

from datadoctor.diagnostics.leakage import check_leakage
from datadoctor.diagnostics.readiness import check_readiness
from datadoctor.diagnostics.recommendations import Recommendation, build_recommendations
from datadoctor.diagnostics.splits import check_split_strategy

__all__ = [
    "Recommendation",
    "build_recommendations",
    "check_leakage",
    "check_readiness",
    "check_split_strategy",
]
