"""The bundled sample datasets load cleanly with their target set. Not a diagnostic test: these
are meant to be clean, so there is nothing to plant or find, just confirmation that each file and
its target are usable end to end (see tests/test_defects_fixture.py for a file with planted
defects).
"""

from pathlib import Path

import pytest

from datadoctor import load_dataset

SAMPLE_DIR = Path(__file__).parent.parent / "data" / "sample"


@pytest.mark.parametrize(
    ("filename", "target", "shape"),
    [
        ("breast_cancer.csv", "diagnosis", (569, 31)),
        ("wine_quality.csv", "quality", (4898, 12)),
    ],
)
def test_each_sample_dataset_loads_with_its_target_and_no_missing_values(filename, target, shape):
    dataset = load_dataset(SAMPLE_DIR / filename, target=target)

    assert dataset.data.shape == shape
    assert dataset.target == target
    assert dataset.data.isna().sum().sum() == 0
