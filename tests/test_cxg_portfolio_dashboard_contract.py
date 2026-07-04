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

from app.streamlit_app import load_portfolio_table, load_text_file, render_streamlit_image


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
    assert "make build-cxg-portfolio-summary" in readme


def test_active_dashboard_command_and_tab_are_documented():
    makefile = MAKEFILE_PATH.read_text(encoding="utf-8")
    readme = README_PATH.read_text(encoding="utf-8")
    app_source = APP_PATH.read_text(encoding="utf-8")

    assert "dashboard:" in makefile
    assert "streamlit run app/streamlit_app.py" in makefile
    assert "poetry run streamlit run app/streamlit_app.py" in readme
    assert "Promoted CxG portfolio" in app_source
    assert "for the governed model story" in app_source
    assert "dashboard/app.py" not in readme


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
