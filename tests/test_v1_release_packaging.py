import re
from pathlib import Path

import opponent_adjusted


RELEASE_NOTES = Path("docs/releases/v1.0.0.md")
REVIEWER_QUICKSTART = Path("docs/releases/v1_reviewer_quickstart.md")
RELEASE_CHECKLIST = Path("docs/releases/v1_release_checklist.md")
CHANGELOG = Path("CHANGELOG.md")
README = Path("README.md")
PYPROJECT = Path("pyproject.toml")
MAKEFILE = Path("Makefile")


def test_version_marker_is_v1():
    assert opponent_adjusted.__version__ == "1.0.0"
    assert 'version = "1.0.0"' in PYPROJECT.read_text(encoding="utf-8")


def test_release_notes_exist_and_describe_v1_release():
    text = RELEASE_NOTES.read_text(encoding="utf-8")

    assert "# Opponent-Adjusted Football Metrics v1.0.0" in text
    assert "Release date: 2026-06-27" in text
    assert "CxG" in text
    assert "CxA" in text
    assert "Baseline CxT" in text
    assert "Streamlit dashboard v1" in text
    assert "Reviewer Quickstart" in text
    assert "Known Limitations" in text
    assert "Deferred Enhancements" in text


def test_changelog_has_grouped_v1_section():
    text = CHANGELOG.read_text(encoding="utf-8")

    assert "## v1.0.0 - 2026-06-27" in text
    for heading in (
        "### Modelling",
        "### Dashboard",
        "### Documentation",
        "### Testing/CI",
        "### Known Limitations",
        "### Deferred Work",
    ):
        assert heading in text


def test_reviewer_quickstart_and_release_checklist_exist():
    quickstart = REVIEWER_QUICKSTART.read_text(encoding="utf-8")
    checklist = RELEASE_CHECKLIST.read_text(encoding="utf-8")

    assert "5-10 minute review" in quickstart
    assert "poetry install" in quickstart
    assert 'poetry run pytest -v -m "not e2e"' in quickstart
    assert "make dashboard" in quickstart
    assert "missing-output" in quickstart

    assert "V1 Release Checklist" in checklist
    assert "Tests pass" in checklist
    assert "Dashboard runs" in checklist
    assert "Generated outputs are not committed" in checklist
    assert "Create the `v1.0.0` tag" in checklist


def test_readme_is_portfolio_ready_for_v1():
    text = README.read_text(encoding="utf-8")

    assert "v1.0.0 status" in text
    assert "CxG" in text
    assert "CxA" in text
    assert "Baseline CxT" in text
    assert "make dashboard" in text
    assert "v1.0.0 release notes" in text
    assert "v1 reviewer quickstart" in text
    assert "dashboard demo walkthrough" in text
    assert "Generated files under `feature_store/` and `outputs/` are not committed" in text


def test_makefile_has_release_quality_targets():
    text = MAKEFILE.read_text(encoding="utf-8")

    assert re.search(r"^test:", text, flags=re.MULTILINE)
    assert re.search(r"^lint:", text, flags=re.MULTILINE)
    assert re.search(r"^format-check:", text, flags=re.MULTILINE)
    assert re.search(r"^dashboard:", text, flags=re.MULTILINE)


def test_release_docs_do_not_claim_deferred_cxt_features_are_implemented():
    docs = [
        RELEASE_NOTES,
        REVIEWER_QUICKSTART,
        RELEASE_CHECKLIST,
        Path("docs/releases/v1_scope.md"),
        README,
        CHANGELOG,
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in docs).lower()

    prohibited_claims = (
        "cxt+ implemented",
        "contextual cxt implemented",
        "advanced cxt implemented",
        "od-cxt implemented",
        "od-cxt+ implemented",
        "cxt+ is implemented",
        "contextual cxt is implemented",
        "advanced cxt is implemented",
        "od-cxt is implemented",
        "od-cxt+ is implemented",
    )

    for claim in prohibited_claims:
        assert claim not in text
