"""Diagnostics: leakage and split-strategy checks, and the readiness assessment built on them.

``check_split_strategy`` takes a ``Dataset`` and an ``AnalysisConfig`` and returns an
``AnalysisResult``, the same shape every analyzer in this project uses.
"""

from datadoctor.diagnostics.splits import check_split_strategy

__all__ = ["check_split_strategy"]
