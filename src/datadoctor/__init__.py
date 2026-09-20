"""DataDoctor: a diagnostic workbench for tabular datasets.

The public API is these five names:

- ``Dataset``: a dataframe plus metadata such as its name, source and target column.
- ``AnalysisConfig``: settings for a run, including the random seed and size thresholds.
- ``AnalysisResult``: what an analysis produced (findings, metrics and saved files).
- ``Finding``: one diagnostic, with separate evidence, interpretation and limitations.
- ``Severity``: how much a finding matters if it is real.

Results serialize to strict JSON with ``to_json`` and ``from_json``.
"""

from datadoctor.core.config import AnalysisConfig
from datadoctor.core.dataset import Dataset
from datadoctor.core.result import AnalysisResult, Finding, Severity

__version__ = "0.1.0"

__all__ = [
    "AnalysisConfig",
    "AnalysisResult",
    "Dataset",
    "Finding",
    "Severity",
]
