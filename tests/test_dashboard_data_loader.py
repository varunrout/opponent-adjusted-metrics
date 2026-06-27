from pathlib import Path

import pandas as pd

from opponent_adjusted.dashboard.data_loader import (
    expected_columns,
    iter_resource_specs,
    load_dashboard_contract,
    load_resource,
    status_table,
)


CONTRACT_PATH = Path("configs/dashboard/v1_dashboard_contract.json")
APP_PATH = Path("app/streamlit_app.py")
MAKEFILE_PATH = Path("Makefile")
README_PATH = Path("README.md")
DEMO_WALKTHROUGH_PATH = Path("docs/dashboard/demo_walkthrough.md")


def test_dashboard_contract_can_be_loaded():
    contract = load_dashboard_contract(CONTRACT_PATH)

    assert contract["name"] == "v1_dashboard_contract"
    assert set(contract["metric_sections"]) == {"cxg", "cxa", "cxt"}
    assert {metric for metric, _, _ in iter_resource_specs(contract)} == {"cxg", "cxa", "cxt"}


def test_loader_returns_empty_dataframe_and_status_when_path_missing(tmp_path: Path):
    spec = {
        "path": "missing/player_cxt.parquet",
        "required_columns": ["player_id", "total_cxt"],
        "optional_columns": ["player_name"],
    }

    resource = load_resource("cxt", "player_aggregates", spec, project_root=tmp_path)

    assert resource.status.found is False
    assert resource.status.missing is True
    assert resource.status.row_count == 0
    assert resource.status.column_count == 3
    assert list(resource.dataframe.columns) == ["player_id", "total_cxt", "player_name"]


def test_loader_can_load_csv_fixture(tmp_path: Path):
    data_path = tmp_path / "outputs" / "modeling" / "cxt" / "reports" / "top_actions.csv"
    data_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [{"rank": 1, "direction": "top_positive", "match_id": 1, "cxt_value": 0.12}]
    ).to_csv(data_path, index=False)
    spec = {
        "path": "outputs/modeling/cxt/reports/top_actions.csv",
        "required_columns": ["rank", "direction", "match_id", "cxt_value"],
    }

    resource = load_resource("cxt", "top_actions", spec, project_root=tmp_path)

    assert resource.status.found is True
    assert resource.status.row_count == 1
    assert resource.status.column_count == 4
    assert resource.dataframe.loc[0, "direction"] == "top_positive"


def test_loader_can_load_parquet_fixture(tmp_path: Path):
    data_path = tmp_path / "outputs" / "modeling" / "cxt" / "aggregates" / "player_cxt.parquet"
    data_path.parent.mkdir(parents=True)
    pd.DataFrame([{"player_id": 1, "total_cxt": 0.2}]).to_parquet(data_path, index=False)
    spec = {
        "path": "outputs/modeling/cxt/aggregates/player_cxt.parquet",
        "required_columns": ["player_id", "total_cxt"],
    }

    resource = load_resource("cxt", "player_aggregates", spec, project_root=tmp_path)

    assert resource.status.found is True
    assert resource.status.row_count == 1
    assert resource.dataframe.loc[0, "total_cxt"] == 0.2


def test_loader_can_load_json_report(tmp_path: Path):
    data_path = tmp_path / "outputs" / "modeling" / "cxt" / "reports" / "metrics.json"
    data_path.parent.mkdir(parents=True)
    data_path.write_text('{"number_of_actions": 3, "total_cxt": 0.4}', encoding="utf-8")
    spec = {
        "path": "outputs/modeling/cxt/reports/metrics.json",
        "required_fields": ["number_of_actions", "total_cxt"],
    }

    resource = load_resource("cxt", "metrics", spec, project_root=tmp_path)

    assert resource.status.found is True
    assert resource.status.row_count == 1
    assert resource.json_data["number_of_actions"] == 3


def test_status_table_is_produced_for_loaded_resources(tmp_path: Path):
    missing = load_resource(
        "cxt",
        "player_aggregates",
        {
            "path": "missing.parquet",
            "required_columns": ["player_id", "total_cxt"],
        },
        project_root=tmp_path,
    )

    table = status_table({"cxt": {"player_aggregates": missing}})

    assert list(table.columns) == [
        "metric",
        "name",
        "found",
        "missing",
        "path",
        "row_count",
        "column_count",
        "error",
    ]
    assert table.loc[0, "metric"] == "cxt"
    assert bool(table.loc[0, "missing"]) is True


def test_expected_columns_combines_required_and_optional_without_duplicates():
    spec = {
        "required_columns": ["player_id", "total_cxt"],
        "optional_columns": ["player_name", "total_cxt"],
    }

    assert expected_columns(spec) == ["player_id", "total_cxt", "player_name"]


def test_streamlit_app_makefile_and_readme_are_documented():
    assert APP_PATH.exists()

    makefile = MAKEFILE_PATH.read_text(encoding="utf-8")
    readme = README_PATH.read_text(encoding="utf-8")
    app_source = APP_PATH.read_text(encoding="utf-8")

    assert "dashboard:" in makefile
    assert "streamlit run app/streamlit_app.py" in makefile
    assert "make dashboard" in readme
    assert "streamlit run app/streamlit_app.py" in readme
    assert "load_all_resources" in app_source
    assert "Some generated outputs are missing" in app_source


def test_streamlit_app_contains_guided_storytelling_helpers_and_scope():
    app_source = APP_PATH.read_text(encoding="utf-8")

    for helper in (
        "def render_insight_card",
        "def render_metric_explanation",
        "def render_page_intro",
        "def render_missing_guidance",
        "def render_v1_scope_banner",
        "def render_reviewer_walkthrough",
    ):
        assert helper in app_source

    for implemented in (
        "CxG",
        "CxA",
        "baseline CxT",
        "dashboard shell",
        "aggregate/report views",
    ):
        assert implemented in app_source

    for deferred in ("CxT+", "Contextual CxT", "Advanced CxT", "OD-CxT / OD-CxT+"):
        assert deferred in app_source


def test_streamlit_app_contains_metric_explanations_and_missing_output_guidance():
    app_source = APP_PATH.read_text(encoding="utf-8")

    expected_copy = (
        "What problem does this project solve",
        "How to read it",
        "CxG evaluates shot quality, not whether the shot became a goal.",
        "A high CxA player contributes actions that move possessions closer to chance creation.",
        "A high CxT player repeatedly moves the ball into more dangerous areas.",
        "Baseline CxT is location-threat movement, not full possession-state value.",
        "Empty tables here are expected in a clean checkout.",
    )

    for text in expected_copy:
        assert text in app_source


def test_demo_walkthrough_doc_and_readme_reference_dashboard_demo_flow():
    assert DEMO_WALKTHROUGH_PATH.exists()

    walkthrough = DEMO_WALKTHROUGH_PATH.read_text(encoding="utf-8")
    readme = README_PATH.read_text(encoding="utf-8")

    assert "Suggested Reviewer Walkthrough" in walkthrough
    assert "Screenshot / GIF Targets" in walkthrough
    assert "make dashboard" in walkthrough
    assert "Suggested demo flow" in readme
    assert "docs/dashboard/demo_walkthrough.md" in readme
