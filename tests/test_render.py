from pathlib import Path

import pandas as pd

from datadoctor.core.config import AnalysisConfig
from datadoctor.core.dataset import Dataset
from datadoctor.core.result import AnalysisResult, Finding, Severity
from datadoctor.render import render_report_html, render_report_markdown

CONFIG = AnalysisConfig(random_seed=7, row_threshold=1000, column_threshold=10)


def finding(
    title="Finding title",
    severity=Severity.HIGH,
    confidence=0.9,
    columns=(),
    evidence="evidence text",
    interpretation="interpretation text",
    limitations="limitations text",
    recommendation=None,
):
    return Finding(
        category="quality",
        severity=severity,
        confidence=confidence,
        title=title,
        evidence=evidence,
        interpretation=interpretation,
        limitations=limitations,
        affected_columns=columns,
        recommendation=recommendation,
    )


def result(findings=(), artifacts=None, config=CONFIG):
    return AnalysisResult(
        findings=findings,
        metrics={
            "profile": {"n_rows": 100, "n_columns": 3},
            "findings_by_severity": {
                severity.value: sum(f.severity is severity for f in findings)
                for severity in Severity
            },
        },
        artifacts=artifacts or {},
        config=config,
    )


class TestTitle:
    def test_a_given_dataset_name_is_used_as_the_title(self):
        html = render_report_html(result(), dataset_name="wine")

        assert "<title>wine</title>" in html
        assert "<h1>wine</h1>" in html

    def test_no_dataset_name_falls_back_to_a_generic_title(self):
        html = render_report_html(result())

        assert "<title>Diagnostic Report</title>" in html
        assert "<h1>Diagnostic Report</h1>" in html


class TestEscaping:
    def test_a_dataset_name_with_html_characters_is_escaped(self):
        html = render_report_html(result(), dataset_name="t<script>&x")

        assert "<script>" not in html
        assert "t&lt;script&gt;&amp;x" in html

    def test_finding_text_with_html_characters_is_escaped_in_every_field(self):
        f = finding(
            title="<b>title</b>",
            columns=("<col>",),
            evidence="ev & <i>idence</i>",
            interpretation="<interp>",
            limitations="<lim>",
            recommendation="<rec>",
        )

        html = render_report_html(result(findings=(f,)))

        for raw in ("<b>title</b>", "<col>", "<i>idence</i>", "<interp>", "<lim>", "<rec>"):
            assert raw not in html
        assert "&lt;b&gt;title&lt;/b&gt;" in html
        assert "&lt;col&gt;" in html
        assert "&lt;rec&gt;" in html

    def test_a_figure_key_with_html_characters_is_escaped_in_alt_text(self):
        artifacts = {"weird<key>": Path("plot.png")}

        html = render_report_html(result(artifacts=artifacts))

        assert 'alt="weird&lt;key&gt;"' in html
        assert "<key>" not in html


class TestFigures:
    def test_figures_use_the_bare_filename_not_the_full_path(self, tmp_path):
        artifacts = {"heatmap": tmp_path / "sub" / "heatmap.png"}

        html = render_report_html(result(artifacts=artifacts))

        assert '<img src="heatmap.png"' in html
        assert str(tmp_path) not in html

    def test_the_report_artifact_itself_is_never_shown_as_a_figure(self, tmp_path):
        artifacts = {"report": tmp_path / "report.md", "heatmap": tmp_path / "heatmap.png"}

        html = render_report_html(result(artifacts=artifacts))

        assert "report.md" not in html
        assert "heatmap.png" in html

    def test_no_figures_section_when_there_are_no_figures(self):
        html = render_report_html(result())

        assert "<h2>Figures</h2>" not in html


class TestSeverityBadges:
    def test_each_severity_gets_its_own_badge_class(self):
        findings = tuple(finding(severity=s, title=f"{s.value} finding") for s in Severity)

        html = render_report_html(result(findings=findings))

        for severity in Severity:
            assert f'class="badge badge-{severity.value}"' in html
            assert f'class="finding severity-{severity.value}"' in html


class TestFieldOmission:
    def test_columns_and_recommendation_are_omitted_when_absent(self):
        f = finding(columns=(), recommendation=None)

        html = render_report_html(result(findings=(f,)))

        assert "<dt>Columns</dt>" not in html
        assert "<dt>Recommendation</dt>" not in html
        assert "<dt>Evidence</dt>" in html
        assert "<dt>Interpretation</dt>" in html
        assert "<dt>Limitations</dt>" in html

    def test_columns_and_recommendation_are_shown_when_present(self):
        f = finding(columns=("age",), recommendation="Fix it.")

        html = render_report_html(result(findings=(f,)))

        assert "<dt>Columns</dt><dd>age</dd>" in html
        assert "<dt>Recommendation</dt><dd>Fix it.</dd>" in html


class TestEmptyFindings:
    def test_no_findings_shows_the_caveat_text_not_a_findings_heading(self):
        html = render_report_html(result())

        assert "No findings from: profile, quality, eda, diagnose." in html
        assert "It does not show that the data is clean or ready to model." in html
        assert "<h2>Findings</h2>" not in html


class TestLimitationsSection:
    def test_the_limitations_paragraph_matches_the_markdown_versions_wording(self):
        dataset = Dataset(data=pd.DataFrame({"a": [1, 2, 3]}), name="t")

        html = render_report_html(result())
        markdown = render_report_markdown(
            dataset,
            CONFIG,
            findings=(),
            findings_by_severity=dict.fromkeys((s.value for s in Severity), 0),
            figures={},
        )

        sentence = (
            "Each finding's own Limitations field states what that specific check did not test"
        )
        assert sentence in html
        assert sentence in markdown
