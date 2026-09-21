"""Privacy diagnostics: columns that appear to hold personal data.

``check_pii`` takes a ``Dataset`` and an ``AnalysisConfig`` and returns an ``AnalysisResult``, like
the data quality analyzers. It never returns or prints the values it matched.
"""

from datadoctor.privacy.pii import check_pii

__all__ = ["check_pii"]
