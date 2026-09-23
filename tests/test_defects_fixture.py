"""An end-to-end ground-truth test: load a real file, then confirm every planted defect in it
is found. Unlike tests/quality/test_run.py's planted_frame, this goes through load_dataset, so
the loader's own type inference and missing-value handling are exercised too, not just the
in-memory analyzers.
"""

from pathlib import Path

from datadoctor import AnalysisConfig, load_dataset
from datadoctor.quality import run_quality_checks

FIXTURE = Path(__file__).parent / "fixtures" / "defects_sample.csv"


def test_the_fixture_is_committed():
    assert FIXTURE.exists(), "regenerate it: uv run python tests/fixtures/make_fixtures.py"


def test_every_planted_defect_is_found_through_the_real_loader():
    dataset = load_dataset(FIXTURE)

    result = run_quality_checks(dataset, AnalysisConfig())

    found = {(f.category, tuple(f.affected_columns)) for f in result.findings}
    expected = {
        ("missingness", ("score",)),
        ("duplicates", ()),
        ("constants", ("flag",)),
        ("outliers", ("weight",)),
        ("dtypes", ("units",)),
        ("impossible_values", ("age",)),
        ("impossible_values", ("signup",)),
        ("privacy", ("contact",)),
    }
    assert expected <= found
