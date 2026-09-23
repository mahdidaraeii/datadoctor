"""Regenerate the bundled sample datasets in data/sample/.

Run from the repository root, with scikit-learn available (it is not a project dependency,
only a one-time tool for this script):

    uv run --with scikit-learn python scripts/make_sample_datasets.py

Two datasets, both small, real and clearly licensed:

- Breast Cancer Wisconsin (Diagnostic), a classification set, from the UCI Machine Learning
  Repository (CC BY 4.0), loaded through scikit-learn's bundled copy of it.
- Wine Quality (white), a regression set, from the UCI Machine Learning Repository (CC BY 4.0),
  downloaded directly since scikit-learn does not bundle it.

See data/sample/README.md for what each file holds and its full source and license.
"""

from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer

HERE = Path(__file__).parent
OUT = HERE.parent / "data" / "sample"

WINE_QUALITY_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
)


def make_breast_cancer() -> None:
    bunch = load_breast_cancer()
    frame = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    # Labels, not the raw 0/1 codes: a categorical target demonstrates classification cleanly,
    # without the "0/1 stored as numbers is typed numeric" case eda/relationships.py handles.
    frame["diagnosis"] = [bunch.target_names[code] for code in bunch.target]
    frame.to_csv(OUT / "breast_cancer.csv", index=False)


def make_wine_quality() -> None:
    frame = pd.read_csv(WINE_QUALITY_URL, sep=";")
    frame.to_csv(OUT / "wine_quality.csv", index=False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    make_breast_cancer()
    make_wine_quality()


if __name__ == "__main__":
    main()
