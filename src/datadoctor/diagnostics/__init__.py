"""Diagnostics: leakage and split-strategy checks, and the readiness assessment built on them.

Each function takes a ``Dataset`` and an ``AnalysisConfig`` and returns an ``AnalysisResult``,
the same shape every analyzer in this project uses.
"""

from datadoctor.diagnostics.leakage import check_leakage
from datadoctor.diagnostics.readiness import check_readiness
from datadoctor.diagnostics.splits import check_split_strategy

__all__ = ["check_leakage", "check_readiness", "check_split_strategy"]
