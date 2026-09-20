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


if __name__ == "__main__":
    main()
