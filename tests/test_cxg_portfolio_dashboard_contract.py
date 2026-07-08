import json
import importlib.util
import sys
import types
from pathlib import Path

if importlib.util.find_spec("streamlit") is None:
    streamlit_stub = types.ModuleType("streamlit")
    streamlit_stub.set_page_config = lambda **_: None
    streamlit_stub.cache_data = lambda **_: lambda func: func
    sys.modules["streamlit"] = streamlit_stub

from app.streamlit_app import (
    CXA_PORTFOLIO_PLAYERS_PATH,
    CXA_PORTFOLIO_SEQUENCES_PATH,
    CXA_PORTFOLIO_TEAMS_PATH,
    PROJECT_STATUS,
    load_portfolio_scorecard,
    load_portfolio_table,
    load_text_file,
    render_streamlit_image,
)


DASHBOARD_CONTRACT_PATH = Path("configs/dashboard/v1_dashboard_contract.json")
README_PATH = Path("README.md")
MAKEFILE_PATH = Path("Makefile")
APP_PATH = Path("app/streamlit_app.py")


def _contract() -> dict:
    return json.loads(DASHBOARD_CONTRACT_PATH.read_text(encoding="utf-8"))


def test_dashboard_contract_declares_promoted_cxg_portfolio_outputs():
    contract = _contract()
    cxg = contract["metric_sections"]["cxg"]
    inputs = cxg["inputs"]

    assert "promoted_cxg_portfolio" in contract["dashboard_pages"]
    assert cxg["portfolio_summary"]["path"] == ("outputs/portfolio/cxg/cxg_portfolio_summary.md")
    assert inputs["portfolio_scorecard"]["path"] == (
        "outputs/portfolio/cxg/cxg_model_scorecard.json"
    )
    assert inputs["portfolio_team_rankings"]["path"] == (
        "outputs/portfolio/cxg/cxg_team_rankings.csv"
    )
    assert inputs["portfolio_player_rankings"]["path"] == (
        "outputs/portfolio/cxg/cxg_player_rankings.csv"
    )
    assert inputs["portfolio_feature_drivers"]["path"] == (
        "outputs/portfolio/cxg/cxg_feature_driver_summary.csv"
    )
    assert inputs["portfolio_category_insights"]["path"] == (
        "outputs/portfolio/cxg/cxg_category_insights.csv"
    )


def test_portfolio_helpers_handle_missing_outputs(tmp_path: Path):
    assert load_text_file(tmp_path / "missing.md") == ""
    assert load_portfolio_table(tmp_path / "missing.csv").empty
    assert load_portfolio_scorecard(tmp_path / "missing.json") == {}


def test_cxa_portfolio_scorecard_helper_parses_headline_metrics(tmp_path: Path):
    path = tmp_path / "headline_metrics.json"
    path.write_text(
        json.dumps(
            {
                "promotion_status": "provisionally_promoted",
                "selected_model": "calibrated_gradient_boosting_sigmoid",
                "action_row_count": 100,
                "player_name_coverage": 1.0,
                "team_name_coverage": 1.0,
                "name_source_used": "sqlite_database",
            }
        ),
        encoding="utf-8",
    )

    headline = load_portfolio_scorecard(path)

    assert headline["promotion_status"] == "provisionally_promoted"
    assert headline["selected_model"] == "calibrated_gradient_boosting_sigmoid"
    assert headline["name_source_used"] == "sqlite_database"


def test_streamlit_image_helper_falls_back_to_legacy_width_argument():
    calls = []

    def image_func(path: str, **kwargs):
        calls.append((path, kwargs))
        if "use_container_width" in kwargs:
            raise TypeError("unsupported keyword")

    render_streamlit_image("chart.png", "Chart", image_func=image_func)

    assert calls == [
        ("chart.png", {"caption": "Chart", "use_container_width": True}),
        ("chart.png", {"caption": "Chart", "use_column_width": True}),
    ]


def test_readme_links_promoted_cxg_portfolio_outputs_and_commands():
    readme = README_PATH.read_text(encoding="utf-8")

    assert "Promoted Diagnostic CxG Portfolio" in readme
    assert "outputs/portfolio/cxg/cxg_portfolio_summary.md" in readme
    assert "outputs/portfolio/cxg/cxg_model_scorecard.json" in readme
    assert "outputs/portfolio/cxg/charts/" in readme
    for command in (
        "make build-features",
        "make run-cxg-end-to-end",
        "make run-cxg-diagnostic-training",
        "make validate-cxg-diagnostic",
        "make generate-cxg-diagnostic-results",
        "make analyze-cxg-feature-impact",
        "make build-cxg-portfolio-summary",
    ):
        assert command in readme


def test_readme_documents_current_project_status_and_demonstration_scope():
    readme = README_PATH.read_text(encoding="utf-8")

    assert "## Current Project Status" in readme
    assert "| CxG | Promoted | Portfolio/dashboard-ready |" in readme
    assert "| CxA | Provisionally promoted | Portfolio/dashboard-ready |" in readme
    assert "| CxT | Analysis completed | Modelling pending |" in readme
    assert "| CxA+ | Pending | Not implemented |" in readme
    assert "| Advanced CxA | Pending | Not implemented |" in readme
    assert "## What This Project Demonstrates" in readme
    assert "Leakage-aware feature contracts" in readme
    assert "Model promotion gates" in readme
    assert "StatsBomb xG leakage" in readme
    assert "baseline comparison is reference-only/in-sample" in readme


def test_dashboard_contract_declares_provisionally_promoted_cxa_portfolio_outputs():
    contract = _contract()
    cxa = contract["metric_sections"]["cxa"]
    inputs = cxa["inputs"]

    assert "provisionally_promoted_cxa_portfolio" in contract["dashboard_pages"]
    assert cxa["portfolio_summary"]["path"] == "outputs/portfolio/cxa/portfolio_summary.md"
    assert inputs["portfolio_headline_metrics"]["path"] == (
        "outputs/portfolio/cxa/headline_metrics.json"
    )
    assert inputs["portfolio_player_rankings"]["path"] == (
        "outputs/portfolio/cxa/top_players_by_cxa.csv"
    )
    assert inputs["portfolio_team_rankings"]["path"] == (
        "outputs/portfolio/cxa/top_teams_by_cxa.csv"
    )
    assert inputs["portfolio_sequence_rankings"]["path"] == (
        "outputs/portfolio/cxa/top_sequences_by_cxa.csv"
    )
    assert inputs["portfolio_feature_drivers"]["path"] == (
        "outputs/portfolio/cxa/feature_driver_summary.csv"
    )
    assert "player_name" in inputs["portfolio_player_rankings"]["required_columns"]
    assert "team_name" in inputs["portfolio_team_rankings"]["required_columns"]
    assert "team_name" in inputs["portfolio_sequence_rankings"]["required_columns"]


def test_cxa_portfolio_paths_use_name_enriched_static_outputs():
    assert CXA_PORTFOLIO_PLAYERS_PATH.as_posix() == "outputs/portfolio/cxa/top_players_by_cxa.csv"
    assert CXA_PORTFOLIO_TEAMS_PATH.as_posix() == "outputs/portfolio/cxa/top_teams_by_cxa.csv"
    assert (
        CXA_PORTFOLIO_SEQUENCES_PATH.as_posix() == "outputs/portfolio/cxa/top_sequences_by_cxa.csv"
    )


def test_readme_links_cxa_portfolio_outputs_and_commands():
    readme = README_PATH.read_text(encoding="utf-8")

    assert "Provisionally Promoted Diagnostic CxA Portfolio" in readme
    assert "outputs/portfolio/cxa/portfolio_summary.md" in readme
    assert "outputs/portfolio/cxa/headline_metrics.json" in readme
    assert "outputs/portfolio/cxa/charts/" in readme
    for command in (
        "make build-cxa-action-features",
        "make run-cxa-end-to-end",
        "make prepare-cxa-diagnostic-contract",
        "make run-cxa-diagnostic-training",
        "make validate-cxa-diagnostic",
        "make generate-cxa-diagnostic-results",
        "make analyze-cxa-feature-impact",
        "make build-cxa-portfolio-summary",
    ):
        assert command in readme


def test_active_dashboard_command_and_tab_are_documented():
    makefile = MAKEFILE_PATH.read_text(encoding="utf-8")
    readme = README_PATH.read_text(encoding="utf-8")
    app_source = APP_PATH.read_text(encoding="utf-8")

    assert "dashboard:" in makefile
    assert "streamlit run app/streamlit_app.py" in makefile
    assert "poetry run streamlit run app/streamlit_app.py" in readme
    assert "Promoted CxG portfolio" in app_source
    assert "Provisionally Promoted CxA portfolio" in app_source
    assert "for the governed model stories" in app_source
    assert "dashboard/app.py" not in readme


def test_dashboard_project_status_navigation_declares_metric_readiness():
    statuses = {row["metric"]: row for row in PROJECT_STATUS}
    app_source = APP_PATH.read_text(encoding="utf-8")

    assert statuses["CxG"]["status"] == "promoted"
    assert statuses["CxG"]["dashboard"] == "portfolio/dashboard-ready"
    assert statuses["CxA"]["status"] == "provisionally promoted"
    assert statuses["CxA"]["dashboard"] == "portfolio/dashboard-ready"
    assert statuses["CxT"]["status"] == "analysis completed"
    assert statuses["CxT"]["dashboard"] == "modelling pending"
    assert statuses["CxA+"]["status"] == "pending"
    assert statuses["Advanced CxA"]["status"] == "pending"
    assert "render_project_status_navigation()" in app_source


def test_promoted_cxg_missing_output_guidance_lists_full_generation_chain():
    app_source = APP_PATH.read_text(encoding="utf-8")

    for command in (
        "make build-features",
        "make run-cxg-end-to-end",
        "make run-cxg-diagnostic-training",
        "make validate-cxg-diagnostic",
        "make generate-cxg-diagnostic-results",
        "make analyze-cxg-feature-impact",
        "make build-cxg-portfolio-summary",
        "make dashboard",
    ):
        assert command in app_source


def test_provisionally_promoted_cxa_dashboard_is_static_display_only():
    app_source = APP_PATH.read_text(encoding="utf-8")

    for command in (
        "make build-cxa-action-features",
        "make run-cxa-end-to-end",
        "make prepare-cxa-diagnostic-contract",
        "make run-cxa-diagnostic-training",
        "make validate-cxa-diagnostic",
        "make generate-cxa-diagnostic-results",
        "make analyze-cxa-feature-impact",
        "make build-cxa-portfolio-summary",
    ):
        assert command in app_source

    assert "make build-cxa-portfolio-summary" in app_source
    assert "provisionally promoted" in app_source.lower()
    assert "diagnostic pipeline, validation, governed results, and feature-impact artifacts" in (
        app_source
    )
    assert "baseline comparison is reference-only/in-sample" in app_source
    assert "CxA+ and Advanced CxA are later work" in app_source
    assert "run_cxa_diagnostic_training" not in app_source
    assert "validate_cxa_diagnostic_model" not in app_source
    assert "generate_cxa_diagnostic_results" not in app_source
