import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_hex
from scipy.stats import skew

from datadoctor import AnalysisConfig, AnalysisResult, Dataset, Severity, load_dataset
from datadoctor.core.exceptions import DatasetError
from datadoctor.eda import check_univariate, univariate
from datadoctor.eda import univariate_plots as plots

N = 100
FIRST_NAMES = ["Anna", "Peter", "Maria", "John", "Sofia", "Lars", "Chloe", "Omar"]
REAL_FILE = Path(__file__).parents[2] / "data" / "telecom_customers.csv"


def planted_frame() -> pd.DataFrame:
    """100 rows. The numbers and counts of every column are known, and listed in the tests."""
    gaps = np.arange(N) * 0.5
    gaps[:10] = np.nan  # ten missing
    gaps[10], gaps[11] = np.inf, -np.inf  # two infinite, so 88 finite values from 6.0 to 49.5
    many = [f"c{k:02d}" for k in range(21)]
    many_counts = [14, 12, 10, 9, 8, 7, 6, 5, 4, 3] + [2] * 11
    return pd.DataFrame(
        {
            "hundred": np.arange(1, N + 1) * 2,  # 2, 4 .. 200: not a run of 1 to 100, an identifier
            "counts": np.arange(N) % 5,
            "gaps": gaps,
            "constant": [7.5] * N,
            "colour": ["red"] * 50 + ["green"] * 30 + ["blue"] * 20,
            "many": np.repeat(many, many_counts),
            "flag": [k % 4 == 0 for k in range(N)],
            "first_name": [FIRST_NAMES[k % 8] for k in range(N)],
            "ident": [f"ID-{k:04d}" for k in range(N)],
            "memo": [f"order number {k} shipped late and was signed for" for k in range(N)],
            "when": [f"2024-{1 + k % 12:02d}-{1 + k % 28:02d}" for k in range(N)],
            "nested": [[k] for k in range(N)],
        }
    )


@pytest.fixture
def figures(monkeypatch):
    """The figures as they were drawn, by file name, so that their contents can be checked."""
    saved = {}
    real = plots.save_figure

    def capture(fig, path):
        saved[Path(path).name] = fig
        return real(fig, path)

    monkeypatch.setattr(plots, "save_figure", capture)
    return saved


@pytest.fixture
def config(tmp_path):
    return AnalysisConfig(output_dir=tmp_path / "plots")


@pytest.fixture(scope="module")
def planted_run(tmp_path_factory):
    """The planted frame, analyzed once: the result, its figures as drawn, and the config."""
    config = AnalysisConfig(output_dir=tmp_path_factory.mktemp("plots"))
    saved = {}
    real = plots.save_figure

    def capture(fig, path):
        saved[Path(path).name] = fig
        return real(fig, path)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(plots, "save_figure", capture)
        result = check_univariate(Dataset(data=planted_frame(), name="planted"), config)
    return result, saved, config


@pytest.fixture
def result(planted_run):
    return planted_run[0]


@pytest.fixture
def planted_figures(planted_run):
    return planted_run[1]


@pytest.fixture
def planted_config(planted_run):
    return planted_run[2]


def entry(result, name):
    return next(c for c in result.metrics["columns"] if c["name"] == name)


def stats(result, name):
    return entry(result, name)["statistics"]


def visible(fig):
    return [axis for axis in fig.axes if axis.get_visible()]


def axis_of(fig, title):
    return next(axis for axis in visible(fig) if axis.get_title() == title)


def numeric_frame(columns, rows=30):
    rs = np.random.RandomState(0)
    return pd.DataFrame({f"n{k:02d}": rs.normal(size=rows) for k in range(columns)})


class TestStatistics:
    def test_the_even_numbers_up_to_two_hundred_have_their_known_summary(self, result):
        # 2, 4 .. 200 is twice 1 to 100: mean 2 * 50.5, sample deviation 2 * sqrt(100 * 101 / 12),
        # and quartiles that interpolate at twice 25.75, 50.5 and 75.25.
        s = stats(result, "hundred")

        assert (s["count"], s["missing"], s["infinite"], s["distinct"]) == (100, 0, 0, 100)
        assert s["mean"] == 101
        assert s["std"] == pytest.approx(2 * math.sqrt(100 * 101 / 12))
        assert (s["min"], s["q1"], s["median"], s["q3"], s["max"]) == (2, 51.5, 101, 150.5, 200)
        assert s["skew"] == pytest.approx(0, abs=1e-12)

    def test_skewness_needs_three_different_values(self, config, figures):
        frame = pd.DataFrame({"two": [1.5, 2.5] * 30, "three": [1.5, 2.5, 4.0] * 20})

        result = check_univariate(Dataset(data=frame, name="t"), config)

        assert stats(result, "two")["skew"] is None
        assert stats(result, "three")["skew"] == pytest.approx(
            skew([1.5, 2.5, 4.0] * 20, bias=False)
        )

    def test_the_deviation_needs_two_values(self, config, figures):
        two = check_univariate(Dataset(data=pd.DataFrame({"n": [1.5, 2.5]}), name="t"), config)
        one = check_univariate(Dataset(data=pd.DataFrame({"n": [1.5]}), name="t"), config)

        assert stats(two, "n")["std"] == pytest.approx(math.sqrt(0.5))
        assert stats(one, "n")["std"] is None

    def test_categories_with_equal_counts_are_ordered_by_label_not_by_the_file(
        self, config, figures
    ):
        frame = pd.DataFrame({"c": ["b"] * 20 + ["a"] * 20})

        result = check_univariate(Dataset(data=frame, name="t"), config)

        assert [t["label"] for t in stats(result, "c")["top"]] == ["a", "b"]

    def test_missing_and_infinite_values_are_counted_and_left_out_of_the_summary(self, result):
        s = stats(result, "gaps")

        assert (s["count"], s["missing"], s["infinite"]) == (88, 10, 2)
        assert (s["min"], s["max"], s["mean"]) == (6.0, 49.5, 27.75)

    def test_a_column_of_one_value_has_no_spread_and_no_skew(self, result):
        s = stats(result, "constant")

        assert (s["std"], s["skew"], s["distinct"]) == (0.0, None, 1)

    def test_categorical_counts_are_ranked_and_the_rest_is_pooled(self, result):
        colour = stats(result, "colour")
        many = stats(result, "many")

        assert colour["top"] == [
            {"label": "red", "count": 50},
            {"label": "green", "count": 30},
            {"label": "blue", "count": 20},
        ]
        assert (colour["count"], colour["distinct"], colour["other_count"]) == (100, 3, 0)
        assert [t["count"] for t in many["top"]] == [14, 12, 10, 9, 8, 7, 6, 5, 4, 3]
        assert (many["distinct"], many["other_count"]) == (21, 22)

    def test_equal_counts_are_ordered_by_label(self, result):
        # c10 to c20 all have 2 values, so the pooled ones are the last of them.
        assert [t["label"] for t in stats(result, "many")["top"]][-3:] == ["c07", "c08", "c09"]

    def test_booleans_are_counted_as_categories(self, result):
        assert stats(result, "flag")["top"] == [
            {"label": "False", "count": 75},
            {"label": "True", "count": 25},
        ]

    def test_columns_that_are_not_drawn_still_get_their_basic_statistics(self, result):
        assert stats(result, "ident") == {"count": 100, "missing": 0, "distinct": 100}
        dates = planted_frame()["when"]
        assert (stats(result, "when")["min"], stats(result, "when")["max"]) == (
            dates.min(),
            dates.max(),
        )


class TestWhatIsDrawn:
    def test_each_column_is_drawn_the_way_its_kind_needs(self, result):
        plots_by_column = {c["name"]: c["plot"] for c in result.metrics["columns"]}

        assert plots_by_column == {
            "hundred": "histogram",
            "counts": "value_bars",
            "gaps": "histogram",
            "constant": "histogram",
            "colour": "category_bars",
            "many": "category_bars",
            "flag": "category_bars",
            "first_name": "category_bars",
            "ident": None,
            "memo": None,
            "when": None,
            "nested": None,
        }

    def test_the_reason_a_column_was_not_drawn_is_recorded(self, result):
        reasons = {c["name"]: c["skipped_reason"] for c in result.metrics["columns"]}

        assert reasons["ident"] == "identifier"
        assert reasons["memo"] == "text"
        assert reasons["when"] == "datetime"
        assert reasons["nested"] == "unknown"
        assert reasons["hundred"] is None
        note = next(f for f in result.findings if f.title == "Some columns were not plotted")
        assert note.severity is Severity.INFO
        assert "4 columns were not plotted" in note.evidence
        assert "identifiers, where every value is different (ident)" in note.evidence

    @pytest.mark.parametrize(("distinct", "expected"), [(15, "value_bars"), (16, "histogram")])
    def test_whole_numbers_get_a_bar_each_up_to_fifteen_distinct_values(
        self, config, figures, distinct, expected
    ):
        frame = pd.DataFrame({"n": np.arange(60) % distinct})

        result = check_univariate(Dataset(data=frame, name="t"), config)

        assert entry(result, "n")["plot"] == expected

    def test_numbers_with_decimals_are_a_histogram_however_few_they_are(self, config, figures):
        frame = pd.DataFrame({"n": [0.5, 1.5, 2.5] * 20})

        assert entry(check_univariate(Dataset(data=frame, name="t"), config), "n")["plot"] == (
            "histogram"
        )


class TestTheFigures:
    def test_the_counts_drawn_are_the_counts_in_the_data(self, result, planted_figures):
        numeric = planted_figures["univariate_numeric_1.png"]

        hundred = axis_of(numeric, "hundred")
        assert sum(p.get_height() for p in hundred.patches) == 100
        counts = axis_of(numeric, "counts")
        assert [p.get_height() for p in counts.patches] == [20] * 5
        assert [t.get_text() for t in counts.get_xticklabels()] == ["0", "1", "2", "3", "4"]
        gaps = axis_of(numeric, "gaps")
        assert sum(p.get_height() for p in gaps.patches) == 88
        assert gaps.get_xlabel() == "n=88, missing 10, infinite 2"
        constant = axis_of(numeric, "constant")
        assert [p.get_height() for p in constant.patches if p.get_height()] == [100]

    def test_the_most_frequent_categories_come_first_and_the_rest_share_one_bar(
        self, result, planted_figures
    ):
        categorical = planted_figures["univariate_categorical_1.png"]

        colour = axis_of(categorical, "colour")
        assert [p.get_width() for p in colour.patches] == [50, 30, 20]
        assert [t.get_text() for t in colour.get_yticklabels()][:3] == ["red", "green", "blue"]
        many = axis_of(categorical, "many")
        assert [p.get_width() for p in many.patches] == [14, 12, 10, 9, 8, 7, 6, 5, 4, 3, 22]
        assert many.get_yticklabels()[-1].get_text() == "other (11 more)"

    def test_the_figure_files_are_pngs_named_for_their_kind_in_the_output_directory(
        self, result, planted_config
    ):
        assert set(result.artifacts) == {"univariate_numeric_1", "univariate_categorical_1"}
        for key, path in result.artifacts.items():
            assert path == planted_config.output_dir / f"{key}.png"
            assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        assert result.metrics["figures"] == [
            {
                "key": "univariate_numeric_1",
                "kind": "numeric",
                "columns": ["hundred", "counts", "gaps", "constant"],
            },
            {
                "key": "univariate_categorical_1",
                "kind": "categorical",
                "columns": ["colour", "many", "flag", "first_name"],
            },
        ]
        assert entry(result, "colour")["figure"] == "univariate_categorical_1"
        assert entry(result, "ident")["figure"] is None

    def test_the_same_data_gives_the_same_files(self, config, tmp_path):
        first = check_univariate(Dataset(data=planted_frame(), name="t"), config)
        again = AnalysisConfig(output_dir=tmp_path / "again")
        second = check_univariate(Dataset(data=planted_frame(), name="t"), again)

        for key in first.artifacts:
            assert first.artifacts[key].read_bytes() == second.artifacts[key].read_bytes()


BLUE, ORANGE = "#0072b2", "#e69f00"


def bars(axis, colour):
    return [p for p in axis.patches if to_hex(p.get_facecolor()) == colour]


def one_column(config, figures, values, name="n"):
    result = check_univariate(Dataset(data=pd.DataFrame({name: values}), name="t"), config)
    return result, axis_of(figures["univariate_numeric_1.png"], name)


class TestHistogramRange:
    """A few extreme values must not squeeze the rest of a histogram into a few bars."""

    def test_a_value_far_above_the_rest_is_one_orange_bar_beyond_the_end_of_the_range(
        self, config, figures
    ):
        # 1 to 99 and 1000: quartiles 25.75 and 75.25, so the fences are 1 and 223.75.
        result, axis = one_column(config, figures, list(range(1, 100)) + [1000])

        assert entry(result, "n")["range_shown"] == {
            "low": 1.0,
            "high": 223.75,
            "below": 0,
            "above": 1,
        }
        blue, orange = bars(axis, BLUE), bars(axis, ORANGE)
        assert sum(p.get_height() for p in blue) == 99
        assert min(p.get_x() for p in blue) == pytest.approx(1)
        last_edge = max(p.get_x() + p.get_width() for p in blue)
        assert last_edge == pytest.approx(99)  # the histogram ends at the largest value inside
        assert [p.get_height() for p in orange] == [1]
        # One bin wide, and one bin beyond the last bar: apart from it, with no empty axis between.
        width = blue[0].get_width()
        assert orange[0].get_width() == pytest.approx(width)
        assert orange[0].get_x() + width / 2 == pytest.approx(last_edge + width)
        assert stats(result, "n")["max"] == 1000  # the statistics still use every value

    def test_the_bar_is_named_in_the_corner_with_the_count_and_the_true_extreme(
        self, config, figures
    ):
        result, axis = one_column(config, figures, list(range(1, 100)) + [1000])

        (label,) = axis.texts
        assert label.get_text() == "1 above 224\n(max 1000)"  # to three significant digits
        assert (label.get_ha(), label.xyann) == ("right", (0.98, 0.92))
        assert label.arrow_patch is not None  # joined to its bar by a line

    def test_a_value_far_below_the_rest_is_drawn_the_same_way_on_the_left(self, config, figures):
        # -1000 and 1 to 99: quartiles 24.75 and 74.25, so the fences are -123.75 and 99.
        result, axis = one_column(config, figures, [-1000] + list(range(1, 100)))

        assert entry(result, "n")["range_shown"] == {
            "low": -123.75,
            "high": 99.0,
            "below": 1,
            "above": 0,
        }
        blue, orange = bars(axis, BLUE), bars(axis, ORANGE)
        assert sum(p.get_height() for p in blue) == 99
        first_edge, width = min(p.get_x() for p in blue), blue[0].get_width()
        assert orange[0].get_x() + width / 2 == pytest.approx(first_edge - width)  # one bin before
        (label,) = axis.texts
        assert label.get_text() == "1 below -124\n(min -1000)"
        assert (label.get_ha(), label.xyann) == ("left", (0.02, 0.92))

    def test_a_large_count_is_written_with_a_thousands_separator(self, config, figures):
        values = list(np.arange(18_000) % 1000) + [100_000] * 2_000

        result, axis = one_column(config, figures, values)

        assert entry(result, "n")["range_shown"]["above"] == 2_000
        assert axis.texts[0].get_text().startswith("2,000 above ")

    def test_values_on_both_sides_get_a_bar_each(self, config, figures):
        # -1000, 1 to 98 and 1000: quartiles 24.75 and 74.25, fences -123.75 and 222.75.
        result, axis = one_column(config, figures, [-1000] + list(range(1, 99)) + [1000])

        shown = entry(result, "n")["range_shown"]
        assert (shown["below"], shown["above"]) == (1, 1)
        assert sorted(t.get_text() for t in axis.texts) == [
            "1 above 223\n(max 1000)",
            "1 below -124\n(min -1000)",
        ]
        assert sum(p.get_height() for p in bars(axis, BLUE)) == 98
        assert len(bars(axis, ORANGE)) == 2

    def test_a_column_with_nothing_beyond_its_fences_is_drawn_whole(self, config, figures):
        result, axis = one_column(config, figures, np.arange(1, 101) * 2)

        assert entry(result, "n")["range_shown"] is None
        assert bars(axis, ORANGE) == []
        assert list(axis.texts) == []
        assert sum(p.get_height() for p in bars(axis, BLUE)) == 100

    def test_a_value_exactly_on_a_fence_is_inside_the_range(self, config, figures):
        # The data ends before the fence on the right, so the highest value is the range end.
        result, axis = one_column(config, figures, [-1000] + list(range(1, 100)))

        assert max(p.get_x() + p.get_width() for p in bars(axis, BLUE)) == pytest.approx(99)
        assert entry(result, "n")["range_shown"]["above"] == 0

    def test_no_range_is_cut_when_the_middle_half_has_no_width(self, config, figures):
        # Ninety zeros and ten values above: the interquartile range is zero, so the fences would
        # be a single point and everything else would count as outside.
        result, axis = one_column(config, figures, [0.0] * 90 + [k * 1.5 for k in range(1, 11)])

        assert entry(result, "n")["range_shown"] is None
        assert bars(axis, ORANGE) == []
        assert sum(p.get_height() for p in bars(axis, BLUE)) == 100

    def test_whole_numbers_drawn_as_bars_per_value_have_no_range_to_cut(self, config, figures):
        values = [1] * 20 + [2] * 20 + [3] * 20 + [1000]

        result, axis = one_column(config, figures, values)

        assert entry(result, "n")["plot"] == "value_bars"
        assert entry(result, "n")["range_shown"] is None
        assert bars(axis, ORANGE) == []

    def test_the_planted_columns_have_nothing_outside_their_fences(self, result):
        numeric = [c for c in result.metrics["columns"] if c["semantic_type"] == "numeric"]

        assert {c["name"]: c["range_shown"] for c in numeric} == dict.fromkeys(
            ["hundred", "counts", "gaps", "constant"]
        )


class TestPresentation:
    def test_a_histogram_never_has_more_than_fifty_bars(self, config, figures):
        values = np.random.RandomState(1).normal(size=20_000)

        check_univariate(Dataset(data=pd.DataFrame({"n": values}), name="t"), config)

        assert len(axis_of(figures["univariate_numeric_1.png"], "n").patches) == 50

    def test_few_categories_leave_room_for_three_bars_so_they_are_not_drawn_huge(
        self, config, figures
    ):
        frame = pd.DataFrame({"one": ["a"] * 60, "many": [f"c{k % 12}" for k in range(60)]})

        check_univariate(Dataset(data=frame, name="t"), config)

        categorical = figures["univariate_categorical_1.png"]
        assert axis_of(categorical, "one").get_ylim() == (2.5, -0.5)
        assert axis_of(categorical, "many").get_ylim() == (10.5, -0.5)  # 10 categories and the rest

    def test_every_categorical_plot_says_how_many_values_it_counts_and_how_many_are_missing(
        self, result, planted_figures
    ):
        categorical = planted_figures["univariate_categorical_1.png"]

        assert axis_of(categorical, "colour").get_xlabel() == "n=100, missing 0"

    def test_long_labels_and_titles_are_cut_short_with_an_ellipsis(self, config, figures):
        long_name = "a_column_whose_name_is_much_too_long_to_fit"
        long_label = "a category label that goes on and on"
        frame = pd.DataFrame({long_name: [long_label, "short"] * 20})

        check_univariate(Dataset(data=frame, name="t"), config)

        axis = visible(figures["univariate_categorical_1.png"])[0]
        assert axis.get_title() == long_name[:27] + "…"
        labels = [t.get_text() for t in axis.get_yticklabels()]
        assert labels[0] == long_label[:21] + "…"
        assert "short" in labels

    def test_a_title_and_a_label_of_exactly_the_limit_are_not_cut(self, config, figures):
        name, label = "n" * 28, "l" * 22
        frame = pd.DataFrame({name: [label, "short"] * 20})

        check_univariate(Dataset(data=frame, name="t"), config)

        axis = visible(figures["univariate_categorical_1.png"])[0]
        assert axis.get_title() == name
        assert axis.get_yticklabels()[0].get_text() == label

    def test_large_numbers_on_the_axes_are_written_short(self, config, figures):
        frame = pd.DataFrame(
            {
                "millions": np.linspace(0, 2_000_000, 200),
                "thousands": np.linspace(20_000, 160_000, 200),
                "small": np.linspace(0.5, 90.5, 200),
            }
        )

        check_univariate(Dataset(data=frame, name="t"), config)

        numeric = figures["univariate_numeric_1.png"]
        numeric.canvas.draw()

        def labels(title):
            return [t.get_text() for t in axis_of(numeric, title).get_xticklabels() if t.get_text()]

        assert {"500k", "1M", "1.5M", "2M"} <= set(labels("millions"))  # from a million, in M
        assert {"40k", "80k", "160k"} <= set(labels("thousands"))  # from ten thousand, in k
        assert {"0", "20", "60", "100"} <= set(labels("small"))  # below that, as they are
        assert not any(text.endswith(("k", "M")) for text in labels("small"))
        # About five ticks, and a few more just outside the view.
        assert all(len(axis_of(numeric, t).get_xticks()) <= 8 for t in ["millions", "thousands"])

    def test_the_note_lists_five_columns_and_counts_the_rest(self, config, figures):
        memos = {
            f"memo{k}": [f"order {j} for memo {k} was shipped late" for j in range(30)]
            for k in range(7)
        }

        result = check_univariate(Dataset(data=pd.DataFrame(memos), name="t"), config)

        note = next(f for f in result.findings if f.title == "Some columns were not plotted")
        assert "7 columns were not plotted" in note.evidence
        assert "free text (memo0, memo1, memo2, memo3, memo4 and 2 more)" in note.evidence


class TestPages:
    def test_twelve_columns_fit_on_one_figure_and_a_thirteenth_starts_the_next(
        self, config, figures
    ):
        result = check_univariate(Dataset(data=numeric_frame(13), name="t"), config)

        assert set(result.artifacts) == {"univariate_numeric_1", "univariate_numeric_2"}
        first, second = figures["univariate_numeric_1.png"], figures["univariate_numeric_2.png"]
        assert (len(visible(first)), len(visible(second))) == (12, 1)
        assert first._suptitle.get_text() == "Numeric columns, page 1 of 2"
        assert second._suptitle.get_text() == "Numeric columns, page 2 of 2"
        assert entry(result, "n12")["figure"] == "univariate_numeric_2"

    def test_a_figure_has_only_the_rows_its_columns_need(self, config, figures):
        check_univariate(Dataset(data=numeric_frame(5), name="t"), config)

        fig = figures["univariate_numeric_1.png"]
        assert fig.get_size_inches()[1] == pytest.approx(2 * 3.2 + 0.7)
        assert len(visible(fig)) == 5

    def test_a_last_figure_that_is_exactly_full_leaves_nothing_over(
        self, config, figures, monkeypatch
    ):
        monkeypatch.setattr(univariate, "MAX_PAGES", 1)

        result = check_univariate(Dataset(data=numeric_frame(12), name="t"), config)

        assert len(result.artifacts) == 1
        assert not [f for f in result.findings if f.title.startswith("Some columns did not fit")]

    def test_one_column_more_than_the_figures_can_hold_is_named(self, config, figures, monkeypatch):
        monkeypatch.setattr(univariate, "MAX_PAGES", 1)

        result = check_univariate(Dataset(data=numeric_frame(13), name="t"), config)

        assert len(result.artifacts) == 1
        assert entry(result, "n12")["skipped_reason"] == "page_limit"
        assert "numeric (n12)" in next(f for f in result.findings if f.category == "eda").evidence

    def test_columns_beyond_the_fourth_figure_are_named_and_kept_in_the_metrics(
        self, config, figures
    ):
        result = check_univariate(Dataset(data=numeric_frame(49), name="t"), config)

        assert len(result.artifacts) == 4
        # The figures count only themselves: four, and not the five that all the columns would need.
        assert figures["univariate_numeric_1.png"]._suptitle.get_text().endswith("page 1 of 4")
        left_over = entry(result, "n48")
        assert (left_over["skipped_reason"], left_over["figure"], left_over["plot"]) == (
            "page_limit",
            None,
            None,
        )
        assert left_over["statistics"]["count"] == 30
        note = next(
            f for f in result.findings if f.title == "Some columns did not fit on the figures"
        )
        assert "at most 4 figures" in note.evidence
        assert "numeric (n48)" in note.evidence


class TestPersonalData:
    def test_labels_of_a_column_that_may_hold_personal_data_are_left_out_everywhere(
        self, result, planted_figures
    ):
        names = entry(result, "first_name")
        axis = axis_of(planted_figures["univariate_categorical_1.png"], "first_name")

        assert names["labels_hidden"] is True
        assert all("label" not in t for t in names["statistics"]["top"])
        assert sorted(t["count"] for t in names["statistics"]["top"]) == [12] * 4 + [13] * 4
        assert list(axis.get_yticks()) == []
        assert "labels hidden" in axis.texts[0].get_text()
        assert not any(name in result.to_json() for name in FIRST_NAMES)

    def test_columns_that_are_not_flagged_keep_their_labels_and_a_note_names_the_others(
        self, result
    ):
        assert entry(result, "colour")["labels_hidden"] is False
        note = next(f for f in result.findings if f.title.startswith("Category labels"))
        assert note.severity is Severity.INFO
        assert "1 categorical column: first_name" in note.evidence


class TestResult:
    def test_the_dataset_is_left_as_it_was(self, config, figures):
        frame = planted_frame()
        before = frame.copy(deep=True)

        check_univariate(Dataset(data=frame, name="t"), config)

        pd.testing.assert_frame_equal(frame, before)

    def test_config_is_recorded_and_the_result_survives_json(self, result, planted_config):
        assert result.config == planted_config
        assert AnalysisResult.from_json(result.to_json()) == result

    def test_a_dataset_without_rows_is_rejected(self, config):
        with pytest.raises(DatasetError, match="cannot explore a dataset with no rows"):
            check_univariate(Dataset(data=pd.DataFrame({"a": []}), name="t"), config)


class TestTheExampleDataset:
    def test_the_telecom_file_is_summarized_and_drawn_on_two_figures(self, config):
        assert REAL_FILE.exists(), f"the committed example dataset is missing: {REAL_FILE}"

        result = check_univariate(load_dataset(REAL_FILE, target="churned"), config)

        assert set(result.artifacts) == {"univariate_numeric_1", "univariate_categorical_1"}
        assert all(path.is_file() for path in result.artifacts.values())
        by_figure = {f["key"]: f["columns"] for f in result.metrics["figures"]}
        assert by_figure == {
            "univariate_numeric_1": [
                "postal_code",
                "age",
                "annual_income",
                "tenure_months",
                "monthly_charges",
                "support_calls",
                "churned",
            ],
            "univariate_categorical_1": [
                "country_code",
                "employment_status",
                "plan_type",
                "region",
                "data_plan",
            ],
        }
        reasons = {c["name"]: c["skipped_reason"] for c in result.metrics["columns"]}
        assert {n: r for n, r in reasons.items() if r} == {
            "customer_id": "identifier",
            "signup_date": "datetime",
            "total_charges": "text",
        }
        charges = stats(result, "monthly_charges")
        assert (charges["min"], charges["max"], charges["count"]) == (-45.0, 704.0, 639)
        assert stats(result, "annual_income")["missing"] == 87
        assert not any(f.category == "privacy" for f in result.findings)

    def test_the_extreme_monthly_charges_are_an_edge_bar_and_not_the_whole_axis(
        self, config, figures
    ):
        # 639 values from -45 to 704. Three lie above the wide fence, so they are one bar, and the
        # rest of the histogram spans about -45 to 227 and not -45 to 704.
        result = check_univariate(load_dataset(REAL_FILE, target="churned"), config)

        charges = entry(result, "monthly_charges")["range_shown"]
        assert charges["low"] == -45.0
        assert charges["high"] == pytest.approx(226.67, abs=0.01)
        assert (charges["below"], charges["above"]) == (0, 3)
        assert entry(result, "annual_income")["range_shown"]["above"] == 2
        assert entry(result, "age")["range_shown"] is None
        axis = axis_of(figures["univariate_numeric_1.png"], "monthly_charges")
        assert [t.get_text() for t in axis.texts] == ["3 above 227\n(max 704)"]
