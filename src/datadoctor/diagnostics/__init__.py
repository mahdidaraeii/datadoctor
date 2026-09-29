"""Diagnostics: leakage and split-strategy checks, the readiness assessment built on them, the
recommendations engine that ranks everything together, and ``run_diagnostics``, which runs all
of it plus quality and eda for the ``diagnose`` command.

Each check function takes a ``Dataset`` and an ``AnalysisConfig`` and returns an
``AnalysisResult``, the same shape every analyzer in this project uses.
``build_recommendations`` instead takes the already-computed results from every analyzer and
returns a ranked tuple of ``Recommendation``. ``run_diagnostics`` computes every input itself and
returns an ``AnalysisResult`` scoped to split strategy, leakage and readiness; see its module
docstring for exactly what that scope excludes and why.
``drop_findings_superseded_by_recommendations`` is for a caller (``report``) that combines
findings from several already-merged results and would otherwise show a merged fact twice.
"""

from datadoctor.diagnostics.leakage import check_leakage
from datadoctor.diagnostics.readiness import check_readiness
from datadoctor.diagnostics.recommendations import (
    Recommendation,
    build_recommendations,
    drop_findings_superseded_by_recommendations,
)
from datadoctor.diagnostics.run import run_diagnostics
from datadoctor.diagnostics.splits import check_split_strategy

__all__ = [
    "Recommendation",
    "build_recommendations",
    "check_leakage",
    "check_readiness",
    "check_split_strategy",
    "drop_findings_superseded_by_recommendations",
    "run_diagnostics",
]
