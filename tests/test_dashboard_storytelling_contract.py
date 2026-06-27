import json
from pathlib import Path


DASHBOARD_DESIGN_PATH = Path("docs/dashboard/v1_dashboard_design.md")
STORY_PATH = Path("docs/storytelling/v1_project_story.md")
V1_SCOPE_PATH = Path("docs/releases/v1_scope.md")
DASHBOARD_CONTRACT_PATH = Path("configs/dashboard/v1_dashboard_contract.json")


def _load_contract() -> dict:
    return json.loads(DASHBOARD_CONTRACT_PATH.read_text(encoding="utf-8"))


def test_dashboard_storytelling_and_release_docs_exist():
    assert DASHBOARD_DESIGN_PATH.exists()
    assert STORY_PATH.exists()
    assert V1_SCOPE_PATH.exists()
    assert DASHBOARD_CONTRACT_PATH.exists()


def test_dashboard_design_doc_defines_required_product_shape():
    text = DASHBOARD_DESIGN_PATH.read_text(encoding="utf-8")

    for section in (
        "## Product Goal",
        "## Intended Users",
        "## Dashboard Pages",
        "## Metric Explanations",
        "## Storytelling Flow",
        "## Example Insights",
    ):
        assert section in text

    for page in (
        "Project Overview",
        "Player Analysis",
        "Team Analysis",
        "CxG Analysis",
        "CxA Analysis",
        "CxT Analysis",
        "Action-Level Explorer",
        "Model and Report Diagnostics",
    ):
        assert page in text

    assert "This document defines the product shape and data contract" in text
    assert "it does not implement the UI" in text


def test_storytelling_doc_explains_metric_narrative():
    text = STORY_PATH.read_text(encoding="utf-8")

    assert "why contextual and opponent-adjusted metrics matter" in text.lower()
    assert "CxG is the shot-quality layer" in text
    assert "CxA is the chance-creation layer" in text
    assert "CxT is the territorial progression layer" in text
    assert "raw events become reproducible metrics" in text
    assert "football insight" in text


def test_v1_scope_lists_included_and_excluded_work():
    text = V1_SCOPE_PATH.read_text(encoding="utf-8")

    for included in (
        "CxG implemented",
        "CxA implemented",
        "Baseline CxT implemented",
        "CxT aggregate and interpretation reports implemented",
        "Generated outputs under `feature_store/` and `outputs/` remain ignored by Git",
        "Tests and CI quality gates",
    ):
        assert included in text

    for excluded in (
        "CxT+ is not implemented in v1",
        "Contextual CxT is not implemented in v1",
        "Advanced CxT is not implemented in v1",
        "OD-CxT is not implemented in v1",
        "Production deployment is not included in v1",
        "Live data ingestion is not included in v1",
        "Tracking data is not required or included in v1",
    ):
        assert excluded in text


def test_dashboard_contract_includes_metric_sections_and_pages():
    contract = _load_contract()

    assert contract["name"] == "v1_dashboard_contract"
    assert contract["status"] == "planned_dashboard_contract"
    assert set(contract["metric_sections"]) == {"cxg", "cxa", "cxt"}

    for page in (
        "project_overview",
        "player_analysis",
        "team_analysis",
        "cxg_analysis",
        "cxa_analysis",
        "cxt_analysis",
        "action_level_explorer",
        "model_report_diagnostics",
        "example_insights",
    ):
        assert page in contract["dashboard_pages"]


def test_dashboard_contract_references_expected_generated_paths():
    contract = _load_contract()
    sections = contract["metric_sections"]

    expected_paths = {
        "outputs/modeling/cxg/predictions/shot_predictions.parquet",
        "outputs/modeling/cxg/reports/metrics.json",
        "outputs/modeling/cxa/predictions/action_predictions.parquet",
        "outputs/modeling/cxa/aggregates/player_cxa.parquet",
        "outputs/modeling/cxa/aggregates/team_cxa.parquet",
        "outputs/modeling/cxa/aggregates/sequence_cxa.parquet",
        "outputs/modeling/cxt/predictions/action_threat.parquet",
        "outputs/modeling/cxt/aggregates/player_cxt.parquet",
        "outputs/modeling/cxt/aggregates/team_cxt.parquet",
        "outputs/modeling/cxt/aggregates/sequence_cxt.parquet",
        "outputs/modeling/cxt/reports/zone_transition_summary.csv",
        "outputs/modeling/cxt/reports/top_actions.csv",
        "outputs/modeling/cxt/reports/interpretation_summary.json",
        "outputs/modeling/cxt/reports/metrics.json",
    }

    actual_paths = {
        spec["path"]
        for section in sections.values()
        for spec in section["inputs"].values()
        if "path" in spec
    }

    assert expected_paths <= actual_paths


def test_dashboard_contract_defines_required_columns_and_page_ownership():
    contract = _load_contract()

    for metric_name, section in contract["metric_sections"].items():
        assert section["page_ownership"], metric_name
        assert section["inputs"], metric_name
        for input_name, input_spec in section["inputs"].items():
            assert input_spec["path"], f"{metric_name}:{input_name}"
            assert input_spec.get("required_columns") or input_spec.get(
                "required_fields"
            ), f"{metric_name}:{input_name}"

    cxt = contract["metric_sections"]["cxt"]
    assert "CxT+" in cxt["roadmap_note"]
    assert "deferred until after v1" in cxt["roadmap_note"]


def test_docs_do_not_claim_advanced_cxt_variants_are_implemented():
    docs = [
        DASHBOARD_DESIGN_PATH,
        STORY_PATH,
        V1_SCOPE_PATH,
        Path("README.md"),
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in docs).lower()

    prohibited_claims = (
        "cxt+ implemented",
        "contextual cxt implemented",
        "advanced cxt implemented",
        "od-cxt implemented",
        "od-cxt+ implemented",
        "cxt+ is implemented",
        "advanced cxt is implemented",
        "od-cxt is implemented",
    )

    for claim in prohibited_claims:
        assert claim not in text
