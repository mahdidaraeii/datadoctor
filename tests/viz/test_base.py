import re
import struct
import subprocess
import sys

import matplotlib
import pytest
from matplotlib.colors import to_hex

from datadoctor import AnalysisConfig
from datadoctor.core.exceptions import ConfigError
from datadoctor.viz import figure_path, new_figure, save_figure

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def drawn(size=None):
    with new_figure(size=size) as (fig, ax):
        ax.plot([0, 1, 2], [1, 3, 2])
        ax.set_title("A title")
    return fig, ax


def png_size(path):
    data = path.read_bytes()
    assert data[:8] == PNG_SIGNATURE
    return struct.unpack(">II", data[16:24])  # width and height, from the IHDR chunk


class TestSaving:
    def test_a_png_appears_on_disk_and_the_returned_path_is_the_one_given(self, tmp_path):
        target = tmp_path / "hist_age.png"
        fig, _ = drawn()

        returned = save_figure(fig, target)

        assert returned == target
        assert target.is_file()
        assert png_size(target) == (1050, 675)  # 7 by 4.5 inches at 150 dots per inch

    def test_the_size_can_be_chosen(self, tmp_path):
        fig, _ = drawn(size=(4, 2))

        save_figure(fig, tmp_path / "small.png")

        assert png_size(tmp_path / "small.png") == (600, 300)

    def test_a_text_path_is_accepted_and_returned_as_a_path(self, tmp_path):
        fig, _ = drawn()

        returned = save_figure(fig, str(tmp_path / "plot.png"))

        assert returned == tmp_path / "plot.png"
        assert returned.is_file()

    def test_the_output_directory_comes_from_the_config_and_is_created_when_absent(self, tmp_path):
        config = AnalysisConfig(output_dir=tmp_path / "results" / "plots")
        target = figure_path(config, "hist_age")
        fig, _ = drawn()
        assert not config.output_dir.exists()

        save_figure(fig, target)

        assert target.parent == config.output_dir
        assert target.is_file()

    def test_a_file_with_the_same_name_is_replaced(self, tmp_path):
        target = tmp_path / "plot.png"
        target.write_bytes(b"old")
        fig, _ = drawn()

        save_figure(fig, target)

        assert png_size(target)[0] > 0

    def test_the_suffix_may_be_in_capitals(self, tmp_path):
        fig, _ = drawn()

        assert save_figure(fig, tmp_path / "PLOT.PNG").is_file()

    @pytest.mark.parametrize("name", ["plot.svg", "plot.pdf", "plot"])
    def test_any_format_but_png_is_refused_and_nothing_is_created(self, tmp_path, name):
        fig, _ = drawn()

        with pytest.raises(ConfigError, match=r"\.png"):
            save_figure(fig, tmp_path / "sub" / name)

        assert not (tmp_path / "sub").exists()


class TestDeterminism:
    def test_the_same_figure_saved_twice_gives_the_same_bytes(self, tmp_path):
        fig, _ = drawn()

        save_figure(fig, tmp_path / "one.png")
        save_figure(fig, tmp_path / "two.png")

        assert (tmp_path / "one.png").read_bytes() == (tmp_path / "two.png").read_bytes()

    def test_two_figures_built_the_same_way_give_the_same_bytes(self, tmp_path):
        save_figure(drawn()[0], tmp_path / "one.png")
        save_figure(drawn()[0], tmp_path / "two.png")

        assert (tmp_path / "one.png").read_bytes() == (tmp_path / "two.png").read_bytes()

    def test_the_file_does_not_say_which_matplotlib_wrote_it(self, tmp_path):
        save_figure(drawn()[0], tmp_path / "plot.png")

        data = (tmp_path / "plot.png").read_bytes()
        assert b"Software" not in data
        assert matplotlib.__version__.encode() not in data


class TestStyle:
    def test_the_look_is_applied_to_the_figure(self):
        fig, ax = drawn()
        ax.plot([0, 1, 2], [2, 1, 3])
        ax.set_xlabel("x")

        colors = [to_hex(line.get_color()) for line in ax.get_lines()]
        assert colors == ["#0072b2", "#e69f00"]  # the first two of the colorblind-safe palette
        assert not ax.spines["top"].get_visible()
        assert not ax.spines["right"].get_visible()
        assert any(line.get_visible() for line in ax.get_xgridlines())
        assert ax.title.get_fontfamily() == ["DejaVu Sans"]
        assert (ax.title.get_fontsize(), ax.title.get_fontweight()) == (12, "bold")
        assert ax.xaxis.label.get_fontsize() == 10
        assert to_hex(fig.get_facecolor()) == "#ffffff"

    def test_the_figure_is_drawn_by_the_agg_canvas_with_a_constrained_layout(self):
        fig, _ = drawn()

        assert type(fig.canvas).__name__ == "FigureCanvasAgg"
        assert type(fig.get_layout_engine()).__name__ == "ConstrainedLayoutEngine"

    def test_the_saved_background_is_white(self, tmp_path):
        from PIL import Image

        save_figure(drawn()[0], tmp_path / "plot.png")

        with Image.open(tmp_path / "plot.png") as image:
            assert image.convert("RGB").getpixel((0, 0)) == (255, 255, 255)

    def test_a_grid_of_axes_shares_the_look(self):
        with new_figure(2, 2) as (fig, axes):
            pass

        assert axes.shape == (2, 2)
        assert all(not axis.spines["top"].get_visible() for axis in axes.flat)

    def test_a_legend_and_labels_made_inside_the_block_have_the_look_too(self):
        with new_figure() as (fig, ax):
            ax.plot([0, 1], [0, 1], label="a")
            legend = ax.legend()

        assert not legend.get_frame_on()

    def test_what_is_made_after_the_block_has_matplotlibs_own_defaults(self):
        with new_figure() as (fig, ax):
            pass

        ax.set_title("late")

        assert ax.title.get_fontweight() == "normal"

    def test_matplotlibs_global_settings_are_left_as_they_were(self, tmp_path):
        before = dict(matplotlib.rcParams)

        with new_figure() as (fig, ax):
            ax.plot([0, 1])
            assert dict(matplotlib.rcParams) != before  # the look applies inside the block
        save_figure(fig, tmp_path / "plot.png")

        assert dict(matplotlib.rcParams) == before


class TestFigurePath:
    config = AnalysisConfig(output_dir="plots")

    def test_a_name_that_is_already_safe_is_used_as_it_is(self):
        assert (
            figure_path(self.config, "hist_age-2.v1")
            == self.config.output_dir / "hist_age-2.v1.png"
        )

    def test_the_same_name_always_gives_the_same_path(self):
        assert figure_path(self.config, "hist/age") == figure_path(self.config, "hist/age")

    def test_a_hostile_name_stays_inside_the_output_directory(self, tmp_path):
        config = AnalysisConfig(output_dir=tmp_path / "plots")

        path = figure_path(config, "../../etc/passwd")

        assert path.parent == config.output_dir
        assert path.resolve().is_relative_to(config.output_dir.resolve())
        assert ".." not in path.name

    def test_names_that_differ_only_in_unsafe_characters_do_not_share_a_file(self):
        paths = {figure_path(self.config, name) for name in ["a/b", "a_b", "a b", "a:b", "a.b"]}

        assert len(paths) == 5

    def test_a_name_of_exactly_eighty_characters_is_kept_and_one_more_is_cut_with_a_hash(self):
        kept = figure_path(self.config, "x" * 80)
        cut = figure_path(self.config, "x" * 81)

        assert kept.name == "x" * 80 + ".png"
        assert re.fullmatch(r"x{80}-[0-9a-f]{8}\.png", cut.name)

    def test_a_long_name_is_cut_and_still_told_apart(self):
        first, second = "x" * 200 + "1", "x" * 200 + "2"

        paths = (figure_path(self.config, first), figure_path(self.config, second))

        assert paths[0] != paths[1]
        assert all(len(p.name) <= 80 + len("-12345678.png") for p in paths)

    def test_names_with_no_usable_characters_still_give_a_file(self):
        path = figure_path(self.config, "///")

        assert path.name.startswith("figure-")
        assert path != figure_path(self.config, "???")

    def test_an_empty_name_still_gives_a_file(self):
        assert figure_path(self.config, "").name.startswith("figure-")

    def test_a_name_that_starts_with_a_dot_does_not_make_a_hidden_file(self):
        assert not figure_path(self.config, ".hidden").name.startswith(".")


class TestNothingIsShown:
    def test_importing_draws_nothing_and_pyplot_is_never_loaded(self, tmp_path):
        code = "\n".join(
            [
                "import sys",
                "import datadoctor.viz",
                "assert 'matplotlib' not in sys.modules, 'importing datadoctor.viz loaded it'",
                "from datadoctor.viz import new_figure, save_figure",
                "with new_figure() as (fig, ax):",
                "    ax.plot([1, 2])",
                "save_figure(fig, sys.argv[1])",
                "assert 'matplotlib.pyplot' not in sys.modules, 'pyplot was loaded'",
            ]
        )

        done = subprocess.run(
            [sys.executable, "-c", code, str(tmp_path / "plot.png")],
            capture_output=True,
            text=True,
            check=False,
        )

        assert done.returncode == 0, done.stderr
        assert (tmp_path / "plot.png").is_file()
