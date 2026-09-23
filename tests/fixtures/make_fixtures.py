"""Regenerate the binary and text fixtures in this directory.

Run from the repository root with the dev extra installed:

    uv run python tests/fixtures/make_fixtures.py

``sample.*`` all hold the same four-row table, so every format can be checked against one
expected frame. The other files exist to trigger one specific behavior each.
"""

import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent

SAMPLE = pd.DataFrame(
    {
        "id": [1, 2, 3, 4],
        "score": [91.5, None, 78.0, 88.25],
        "city": ["Oslo", "Zürich", "Paris", "Rome"],
        "active": [True, False, True, True],
    }
)


def defects_frame() -> pd.DataFrame:
    """100 rows plus 10 duplicated, one planted defect per quality and privacy analyzer.

    Unlike ``tests/quality/test_run.py``'s ``planted_frame``, which builds a frame directly, this
    one is written to a real CSV file, so a test against it also exercises the loader: type
    inference, missing-value tokens and all.

    - missingness: ``score`` is missing for a third of the rows
    - duplicates: the first 10 rows are repeated at the end
    - constants: ``flag`` holds the same value throughout
    - outliers: one extreme ``weight``
    - dtypes: ``units`` stored as text, with a stray non-numeric token
    - impossible_values: two rows with a negative ``age``, two with a ``signup`` date in the
      future
    - privacy: ``contact`` holds email addresses
    """
    n = 100
    rows = []
    for k in range(n):
        age = -3 if k in (1, 2) else 20 + k % 50
        # "err", not "n/a": pandas' default missing-value tokens would turn "n/a" into NaN at
        # load time, leaving a clean numeric column instead of the mixed-type text this plants.
        units = "err" if k in (5, 17) else str(10 + k % 90)
        weight = 900.0 if k == 40 else round(50 + k % 40 * 1.3, 1)
        signup = "2999-01-01" if k in (3, 6) else f"2024-{1 + k % 12:02d}-{1 + k % 28:02d}"
        score = None if k % 3 == 0 else round(k * 1.5, 2)
        city = ["Oslo", "Paris", "Rome", "Lima"][k % 4]
        contact = f"user{k}@example.com"
        rows.append([age, units, "same", signup, score, weight, city, contact])
    frame = pd.DataFrame(
        rows, columns=["age", "units", "flag", "signup", "score", "weight", "city", "contact"]
    )
    return pd.concat([frame, frame.iloc[:10]], ignore_index=True)


def main() -> None:
    SAMPLE.to_csv(HERE / "sample.csv", index=False)
    SAMPLE.to_csv(HERE / "sample.tsv", index=False, sep="\t")
    records = SAMPLE.astype(object).where(SAMPLE.notna(), None).to_dict(orient="records")
    (HERE / "sample.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    SAMPLE.to_parquet(HERE / "sample.parquet", index=False)
    SAMPLE.to_excel(HERE / "sample.xlsx", index=False, sheet_name="data")

    # Two sheets: a caller must choose one.
    with pd.ExcelWriter(HERE / "multisheet.xlsx") as writer:
        SAMPLE.to_excel(writer, index=False, sheet_name="first")
        pd.DataFrame({"z": [10, 20]}).to_excel(writer, index=False, sheet_name="second")

    # pandas would silently rename the second "a" to "a.1".
    pd.DataFrame([[1, 2, 3]], columns=["a", "b", "a"]).to_excel(
        HERE / "duplicate_header.xlsx", index=False
    )

    # A workbook whose only sheet has no cells.
    with pd.ExcelWriter(HERE / "empty_sheet.xlsx") as writer:
        pd.DataFrame().to_excel(writer, index=False, sheet_name="blank")

    defects_frame().to_csv(HERE / "defects_sample.csv", index=False)


if __name__ == "__main__":
    main()
