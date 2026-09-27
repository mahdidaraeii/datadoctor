"""Synthetic dataset generation for validating diagnostics against known ground truth.

Not part of the CLI surface. Used by this package's own tests and by the leakage and
split-strategy diagnostics' tests, which need datasets whose planted defects are known by
construction rather than inferred after the fact.
"""

from datadoctor.testing.synthetic import SyntheticConfig, SyntheticDataset, make_synthetic_dataset

__all__ = ["SyntheticConfig", "SyntheticDataset", "make_synthetic_dataset"]
