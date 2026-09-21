"""Data quality diagnostics.

``run_quality_checks`` runs all of the analyzers below and merges their findings. Each analyzer
can also be used alone. They all take a ``Dataset`` and an ``AnalysisConfig`` and return an
``AnalysisResult``.
"""

from datadoctor.quality.constants import check_constants
from datadoctor.quality.dtypes import check_dtypes
from datadoctor.quality.duplicates import check_duplicates
from datadoctor.quality.impossible import check_impossible_values
from datadoctor.quality.missingness import check_missingness
from datadoctor.quality.outliers import check_outliers
from datadoctor.quality.run import run_quality_checks

__all__ = [
    "check_constants",
    "check_dtypes",
    "check_duplicates",
    "check_impossible_values",
    "check_missingness",
    "check_outliers",
    "run_quality_checks",
]
