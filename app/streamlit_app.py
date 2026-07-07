"""Streamlit dashboard v1 for opponent-adjusted football metrics."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from opponent_adjusted.dashboard.data_loader import (  # noqa: E402
    DashboardResource,
    load_all_resources,
    load_dashboard_contract,
    status_table,
)


st.set_page_config(
    page_title="Opponent-Adjusted Football Metrics",
    page_icon="football",
    layout="wide",
    initial_sidebar_state="expanded",
)

V1_IMPLEMENTED = [
    "CxG",
    "CxA",
    "baseline CxT",
    "dashboard shell",
    "aggregate/report views",
]
V1_DEFERRED = [
    "CxT+",
    "Contextual CxT",
    "Advanced CxT",
    "OD-CxT / OD-CxT+",
]
EXAMPLE_INTERPRETATIONS = [
    "A high CxT player repeatedly moves the ball into more dangerous areas.",
    "A high CxA player contributes actions that move possessions closer to chance creation.",
    "CxG evaluates shot quality, not whether the shot became a goal.",
    "Baseline CxT is location-threat movement, not full possession-state value.",
]
PORTFOLIO_ROOT = Path("outputs/portfolio/cxg")
PORTFOLIO_CHARTS = PORTFOLIO_ROOT / "charts"
PORTFOLIO_SCORECARD_PATH = PORTFOLIO_ROOT / "cxg_model_scorecard.json"
PORTFOLIO_SUMMARY_PATH = PORTFOLIO_ROOT / "cxg_portfolio_summary.md"
PORTFOLIO_TEAM_RANKINGS_PATH = PORTFOLIO_ROOT / "cxg_team_rankings.csv"
PORTFOLIO_PLAYER_RANKINGS_PATH = PORTFOLIO_ROOT / "cxg_player_rankings.csv"
PORTFOLIO_FEATURE_DRIVERS_PATH = PORTFOLIO_ROOT / "cxg_feature_driver_summary.csv"
PORTFOLIO_CATEGORY_INSIGHTS_PATH = PORTFOLIO_ROOT / "cxg_category_insights.csv"
PORTFOLIO_CHART_FILES = [
    "model_metric_comparison.png",
    "feature_group_impact.png",
    "top_feature_importance.png",
    "team_cxg_ranking.png",
    "player_cxg_ranking.png",
    "goals_minus_cxg_teams.png",
]
CXA_PORTFOLIO_ROOT = Path("outputs/portfolio/cxa")
CXA_PORTFOLIO_CHARTS = CXA_PORTFOLIO_ROOT / "charts"
CXA_PORTFOLIO_SUMMARY_PATH = CXA_PORTFOLIO_ROOT / "portfolio_summary.md"
CXA_PORTFOLIO_HEADLINE_PATH = CXA_PORTFOLIO_ROOT / "headline_metrics.json"
CXA_PORTFOLIO_PLAYERS_PATH = CXA_PORTFOLIO_ROOT / "top_players_by_cxa.csv"
CXA_PORTFOLIO_TEAMS_PATH = CXA_PORTFOLIO_ROOT / "top_teams_by_cxa.csv"
CXA_PORTFOLIO_SEQUENCES_PATH = CXA_PORTFOLIO_ROOT / "top_sequences_by_cxa.csv"
CXA_PORTFOLIO_FEATURE_DRIVERS_PATH = CXA_PORTFOLIO_ROOT / "feature_driver_summary.csv"
CXA_PORTFOLIO_CHART_FILES = [
    "baseline_vs_diagnostic_metrics.png",
    "feature_group_impact.png",
    "prediction_distribution.png",
    "top_players_by_cxa.png",
    "top_teams_by_cxa.png",
]
CXG_PORTFOLIO_REGENERATION_STEPS = """Promoted CxG portfolio outputs are missing. Run:

```bash
make build-features
make run-cxg-end-to-end
make run-cxg-diagnostic-training
make validate-cxg-diagnostic
make generate-cxg-diagnostic-results
make analyze-cxg-feature-impact
make build-cxg-portfolio-summary
make dashboard
```"""
CXA_PORTFOLIO_REGENERATION_STEPS = """CxA portfolio outputs are missing. Run:

```bash
make build-cxa-portfolio-summary
make dashboard
```"""


@st.cache_data(show_spinner=False)
def _load_contract() -> dict[str, Any]:
    return load_dashboard_contract()


@st.cache_data(show_spinner=False)
def _load_resources() -> dict[str, dict[str, DashboardResource]]:
    return load_all_resources()


def _resource(
    resources: dict[str, dict[str, DashboardResource]], metric: str, name: str
) -> DashboardResource | None:
    return resources.get(metric, {}).get(name)


def _dataframe(
    resources: dict[str, dict[str, DashboardResource]], metric: str, name: str
) -> pd.DataFrame:
    resource = _resource(resources, metric, name)
    if resource is None:
        return pd.DataFrame()
    return resource.dataframe


def _json_report(
    resources: dict[str, dict[str, DashboardResource]], metric: str, name: str
) -> dict[str, Any]:
    resource = _resource(resources, metric, name)
    if resource is None:
        return {}
    return resource.json_data


def _resolve_project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_text_file(path: Path) -> str:
    """Load optional text output, returning an empty string when missing."""

    resolved = _resolve_project_path(path)
    if not resolved.exists():
        return ""
    return resolved.read_text(encoding="utf-8")


def load_portfolio_markdown(path: Path = PORTFOLIO_SUMMARY_PATH) -> str:
    """Load the static CxG portfolio Markdown summary."""

    return load_text_file(path)


def load_portfolio_scorecard(path: Path = PORTFOLIO_SCORECARD_PATH) -> dict[str, Any]:
    """Load the static CxG portfolio scorecard if it exists."""

    resolved = _resolve_project_path(path)
    if not resolved.exists():
        return {}
    return json.loads(resolved.read_text(encoding="utf-8"))


def load_portfolio_table(path: Path) -> pd.DataFrame:
    """Load an optional CxG portfolio table, returning an empty frame when missing."""

    resolved = _resolve_project_path(path)
    if not resolved.exists():
        return pd.DataFrame()
    if resolved.suffix.lower() == ".parquet":
        return pd.read_parquet(resolved)
    return pd.read_csv(resolved)


def _format_metric_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_scorecard_metrics(scorecard: dict[str, Any]) -> None:
    """Render promoted CxG status cards from the static scorecard."""

    if not scorecard:
        st.info(CXG_PORTFOLIO_REGENERATION_STEPS)
        return

    cards = [
        ("Promotion status", scorecard.get("promotion_status")),
        ("Promotion gate", scorecard.get("promotion_gate_passed")),
        ("Governance", scorecard.get("governance_status")),
        ("Baseline join rate", scorecard.get("baseline_join_rate")),
        ("Selected features", scorecard.get("selected_feature_count")),
    ]
    cols = st.columns(len(cards))
    for col, (label, value) in zip(cols, cards, strict=False):
        col.metric(label, _format_metric_value(value))


def render_metric_comparison_table(scorecard: dict[str, Any]) -> None:
    """Render baseline-vs-diagnostic model metrics and deltas."""

    comparison = scorecard.get("metric_comparison", {}) if scorecard else {}
    baseline = comparison.get("baseline", {})
    diagnostic = comparison.get("diagnostic", {})
    deltas = comparison.get("diagnostic_minus_baseline", {})
    metric_order = ["log_loss", "brier", "roc_auc", "expected_calibration_error"]
    directions = {
        "log_loss": "Lower is better",
        "brier": "Lower is better",
        "roc_auc": "Higher is better",
        "expected_calibration_error": "Lower is better",
    }
    rows = [
        {
            "metric": metric,
            "baseline": baseline.get(metric),
            "diagnostic": diagnostic.get(metric),
            "diagnostic_minus_baseline": deltas.get(metric),
            "direction": directions[metric],
        }
        for metric in metric_order
        if metric in baseline or metric in diagnostic or metric in deltas
    ]
    st.subheader("Model Scorecard")
    if not rows:
        st.info("Model scorecard metrics are not available yet.")
        return
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_streamlit_image(path: str, caption: str, image_func: Any | None = None) -> None:
    """Render an image across Streamlit versions with renamed width arguments."""

    image = image_func or st.image
    try:
        image(path, caption=caption, use_container_width=True)
    except TypeError:
        image(path, caption=caption, use_column_width=True)


def render_portfolio_chart(
    chart_name: str,
    title: str,
    charts_dir: Path = PORTFOLIO_CHARTS,
) -> None:
    """Render a static portfolio chart if available."""

    chart_path = _resolve_project_path(charts_dir / chart_name)
    st.subheader(title)
    if not chart_path.exists():
        st.info(f"Chart not available yet: `{charts_dir / chart_name}`")
        return
    render_streamlit_image(str(chart_path), title)


def _search_filter(df: pd.DataFrame, column: str, label: str, key: str) -> pd.DataFrame:
    if df.empty or column not in df.columns:
        return df
    query = st.text_input(label, key=key).strip().lower()
    if not query:
        return df
    return df[df[column].fillna("").astype(str).str.lower().str.contains(query)]


def render_portfolio_table(
    df: pd.DataFrame,
    title: str,
    *,
    max_rows: int = 50,
    sort_column: str | None = None,
) -> None:
    """Render a static portfolio table with graceful empty handling."""

    st.subheader(title)
    if df.empty:
        st.info(f"No rows available for {title}.")
        return
    table = df.copy()
    if sort_column and sort_column in table.columns:
        table = table.sort_values(sort_column, ascending=False)
    st.dataframe(table.head(max_rows), use_container_width=True, hide_index=True)


def _select_existing_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return df[[column for column in columns if column in df.columns]].copy()


def _availability_card(resources: dict[str, dict[str, DashboardResource]]) -> None:
    status = status_table(resources)
    found = int(status["found"].sum()) if not status.empty else 0
    missing = int(status["missing"].sum()) if not status.empty else 0
    rows = int(status["row_count"].sum()) if not status.empty else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Available outputs", found)
    col2.metric("Missing outputs", missing)
    col3.metric("Loaded rows", rows)

    if missing:
        st.info(
            "Some generated outputs are missing. Run the CxG, CxA, and CxT commands in "
            "the README to populate the dashboard; the app will still run without them."
        )


def render_insight_card(title: str, body: str, status: str = "Guide") -> None:
    """Render a compact narrative card for dashboard interpretation."""

    st.markdown(
        f"""
        <div style="border: 1px solid #d9e2ec; border-radius: 8px; padding: 0.9rem;
                    margin: 0.35rem 0; background: #f8fafc;">
            <div style="font-size: 0.78rem; color: #52616b; font-weight: 700;">{status}</div>
            <div style="font-size: 1rem; font-weight: 700; color: #102a43;">{title}</div>
            <div style="font-size: 0.92rem; color: #334e68;">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_explanation(metric_name: str, question: str, interpretation: str) -> None:
    """Explain what a metric answers and how to read it."""

    render_insight_card(
        metric_name,
        f"Football question: {question}<br>How to read it: {interpretation}",
        status="Metric",
    )


def render_page_intro(title: str, purpose: str, question: str, interpretation: str) -> None:
    """Render standard page-level storytelling copy."""

    st.header(title)
    st.write(purpose)
    render_insight_card("Football question", question, status="Question")
    render_insight_card("How to interpret this page", interpretation, status="Read this first")


def render_missing_guidance() -> None:
    """Explain missing generated-output behavior."""

    st.info(
        "If this section is empty, generated outputs are missing locally. "
        "Run `make cxg-smoke`, `make cxa-smoke`, and `make cxt-baseline` to populate "
        "the dashboard. Empty tables here are expected in a clean checkout."
    )


def render_v1_scope_banner() -> None:
    """Show implemented and deferred v1 scope."""

    st.subheader("V1 Status")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Implemented in v1**")
        for item in V1_IMPLEMENTED:
            st.markdown(f"- {item}")
    with col2:
        st.markdown("**Deferred after v1**")
        for item in V1_DEFERRED:
            st.markdown(f"- {item}")


def render_reviewer_walkthrough() -> None:
    """Give reviewers a suggested first path through the dashboard."""

    render_insight_card(
        "What to look at first",
        (
            "Start with output availability, then compare player/team aggregate tables. "
            "Use CxG for shot quality, CxA for chance-creation actions, and baseline CxT "
            "for progression into dangerous zones."
        ),
        status="Reviewer path",
    )


def _show_table(df: pd.DataFrame, label: str, *, max_rows: int = 25) -> None:
    st.subheader(label)
    if df.empty:
        st.info(f"No rows available for {label}.")
        return
    st.dataframe(df.head(max_rows), use_container_width=True, hide_index=True)


def _bar_chart(df: pd.DataFrame, label_col: str, value_col: str, title: str) -> None:
    if df.empty or label_col not in df.columns or value_col not in df.columns:
        st.info(f"{title} needs `{label_col}` and `{value_col}` columns.")
        return
    chart_df = df[[label_col, value_col]].dropna().sort_values(value_col, ascending=False).head(10)
    if chart_df.empty:
        st.info(f"No chartable rows for {title}.")
        return
    st.subheader(title)
    st.bar_chart(chart_df.set_index(label_col))


def _report_cards(report: dict[str, Any], keys: list[str]) -> None:
    if not report:
        st.info("Report JSON is not available yet.")
        return
    cols = st.columns(min(len(keys), 4))
    for index, key in enumerate(keys):
        value = report.get(key)
        if value is None:
            continue
        cols[index % len(cols)].metric(key.replace("_", " ").title(), value)


def overview_tab(
    resources: dict[str, dict[str, DashboardResource]], contract: dict[str, Any]
) -> None:
    render_page_intro(
        "Project Overview",
        "This dashboard turns modelling outputs into a guided football analytics review.",
        "What problem does this project solve, and which outputs are available?",
        (
            "Use this page to understand the product scope before jumping into tables. "
            "Availability cards show whether generated outputs exist locally."
        ),
    )
    st.write(
        "Opponent-adjusted football analytics built from StatsBomb-style event data. "
        "CxG explains shot quality, CxA explains chance creation, and baseline CxT "
        "explains ball progression into more threatening pitch zones."
    )
    render_v1_scope_banner()
    render_reviewer_walkthrough()
    _availability_card(resources)

    st.subheader("Metric Families")
    render_metric_explanation(
        "CxG",
        "How good was the shot?",
        "CxG evaluates shot quality, not whether the shot became a goal.",
    )
    render_metric_explanation(
        "CxA",
        "Which actions moved a possession toward chance creation?",
        "A high CxA player contributes actions that move possessions closer to chance creation.",
    )
    render_metric_explanation(
        "Baseline CxT",
        "How much threat did ball progression add?",
        "A high CxT player repeatedly moves the ball into more dangerous areas.",
    )
    render_metric_explanation(
        "Opponent-adjusted metrics",
        "How should opponent context change interpretation?",
        (
            "CxG has API-backed inference and opponent-aware context. OD-CxT and richer "
            "opponent-adjusted progression metrics are future extensions, not v1 claims."
        ),
    )
    with st.expander("Dashboard contract metric sections"):
        for metric, section in contract["metric_sections"].items():
            st.markdown(f"**{metric.upper()}**: {section['metric_explanation']}")
            if roadmap_note := section.get("roadmap_note"):
                st.caption(roadmap_note)


def player_tab(resources: dict[str, dict[str, DashboardResource]]) -> None:
    render_page_intro(
        "Player Analysis",
        "Compare player contribution across shot quality, chance creation, and progression.",
        "Which players add value, and through which attacking channel?",
        (
            "High CxG points to shot quality, high CxA to chance creation, and high "
            "baseline CxT to repeated progression into dangerous zones."
        ),
    )
    render_missing_guidance()
    cxg_players = _dataframe(resources, "cxg", "player_aggregates")
    cxa_players = _dataframe(resources, "cxa", "player_aggregates")
    cxt_players = _dataframe(resources, "cxt", "player_aggregates")

    _bar_chart(cxt_players, "player_name", "total_cxt", "Top Players By Baseline CxT")
    _bar_chart(cxa_players, "player_name", "total_cxa", "Top Players By CxA")
    _bar_chart(cxg_players, "player_name", "total_cxg", "Top Players By CxG")
    _show_table(cxt_players, "CxT Player Aggregates")
    _show_table(cxa_players, "CxA Player Aggregates")
    _show_table(cxg_players, "CxG Player Aggregates")


def team_tab(resources: dict[str, dict[str, DashboardResource]]) -> None:
    render_page_intro(
        "Team Analysis",
        "Compare how teams create value through shots, chances, and territory.",
        "Which teams create value through finishing chances versus progressing the ball?",
        (
            "A team with strong CxT but weaker CxG may progress well without converting "
            "territory into high-quality shots."
        ),
    )
    render_missing_guidance()
    cxg_teams = _dataframe(resources, "cxg", "team_aggregates")
    cxa_teams = _dataframe(resources, "cxa", "team_aggregates")
    cxt_teams = _dataframe(resources, "cxt", "team_aggregates")

    _bar_chart(cxt_teams, "team_name", "total_cxt", "Top Teams By Baseline CxT")
    _bar_chart(cxa_teams, "team_name", "total_cxa", "Top Teams By CxA")
    _bar_chart(cxg_teams, "team_name", "total_cxg", "Top Teams By CxG")
    _show_table(cxt_teams, "CxT Team Aggregates")
    _show_table(cxa_teams, "CxA Team Aggregates")
    _show_table(cxg_teams, "CxG Team Aggregates")


def promoted_cxg_portfolio_tab() -> None:
    render_page_intro(
        "Promoted CxG Portfolio",
        "Review the governed diagnostic CxG model through static portfolio outputs.",
        "Why did the promoted diagnostic CxG model improve on the fair baseline?",
        (
            "This page reads the generated portfolio pack. It does not train, validate, "
            "or promote models inside Streamlit."
        ),
    )
    st.caption(
        "Fair baseline excludes StatsBomb xG as a training feature. Calibration remains "
        "monitored alongside log loss, Brier score, and ROC AUC."
    )

    scorecard = load_portfolio_scorecard()
    team_rankings = load_portfolio_table(PORTFOLIO_TEAM_RANKINGS_PATH)
    player_rankings = load_portfolio_table(PORTFOLIO_PLAYER_RANKINGS_PATH)
    feature_drivers = load_portfolio_table(PORTFOLIO_FEATURE_DRIVERS_PATH)
    category_insights = load_portfolio_table(PORTFOLIO_CATEGORY_INSIGHTS_PATH)
    summary_markdown = load_portfolio_markdown()

    if (
        not scorecard
        and team_rankings.empty
        and player_rankings.empty
        and feature_drivers.empty
        and category_insights.empty
        and not summary_markdown
    ):
        st.info(CXG_PORTFOLIO_REGENERATION_STEPS)
        return

    render_scorecard_metrics(scorecard)
    render_metric_comparison_table(scorecard)

    st.subheader("Static Portfolio Charts")
    chart_pairs = [
        ("model_metric_comparison.png", "Baseline vs Diagnostic Metrics"),
        ("feature_group_impact.png", "Feature Group Impact"),
        ("top_feature_importance.png", "Top Feature Importance"),
        ("team_cxg_ranking.png", "Team CxG Ranking"),
        ("player_cxg_ranking.png", "Player CxG Ranking"),
        ("goals_minus_cxg_teams.png", "Team Goals Minus CxG"),
    ]
    for left, right in zip(chart_pairs[0::2], chart_pairs[1::2], strict=False):
        col1, col2 = st.columns(2)
        with col1:
            render_portfolio_chart(left[0], left[1])
        with col2:
            render_portfolio_chart(right[0], right[1])
    if len(chart_pairs) % 2:
        render_portfolio_chart(chart_pairs[-1][0], chart_pairs[-1][1])

    st.subheader("Interactive Portfolio Tables")
    team_limit = st.slider("Top teams to show", min_value=5, max_value=50, value=15, step=5)
    filtered_teams = _search_filter(
        team_rankings, "team_name", "Filter teams by name", "portfolio_team_search"
    )
    render_portfolio_table(
        filtered_teams,
        "Team Rankings",
        max_rows=team_limit,
        sort_column="total_cxg",
    )

    player_limit = st.slider("Top players to show", min_value=5, max_value=100, value=20, step=5)
    filtered_players = _search_filter(
        player_rankings, "team_name", "Filter players by team", "portfolio_player_team_search"
    )
    filtered_players = _search_filter(
        filtered_players, "player_name", "Filter players by name", "portfolio_player_search"
    )
    if "player_id" in player_rankings.columns and int(player_rankings["player_id"].isna().sum()):
        st.warning("Player rankings contain missing `player_id` values.")
    render_portfolio_table(
        filtered_players,
        "Player Rankings",
        max_rows=player_limit,
        sort_column="total_cxg",
    )

    render_portfolio_table(
        feature_drivers,
        "Feature Driver Summary",
        max_rows=50,
        sort_column="log_loss_delta",
    )
    render_portfolio_table(
        category_insights,
        "Category Insights",
        max_rows=100,
        sort_column="total_predicted_cxg",
    )

    st.subheader("Narrative Summary")
    if summary_markdown:
        with st.expander("Read portfolio summary", expanded=False):
            st.markdown(summary_markdown)
    else:
        st.info("Run `make build-cxg-portfolio-summary` to generate the Markdown summary.")


def render_cxa_headline_metrics(headline: dict[str, Any]) -> None:
    """Render provisionally promoted CxA status cards from static headline metrics."""

    if not headline:
        st.info(CXA_PORTFOLIO_REGENERATION_STEPS)
        return

    cards = [
        ("Promotion status", headline.get("promotion_status")),
        ("Selected model", headline.get("selected_model")),
        ("Actions", headline.get("action_row_count")),
        ("Total diagnostic CxA", headline.get("total_diagnostic_cxa")),
        ("Mean probability", headline.get("mean_predicted_probability")),
        ("Selected features", headline.get("selected_feature_count")),
        ("Player names", headline.get("player_name_coverage")),
        ("Team names", headline.get("team_name_coverage")),
        ("Name source", headline.get("name_source_used")),
    ]
    cols = st.columns(3)
    for index, (label, value) in enumerate(cards):
        cols[index % len(cols)].metric(label, _format_metric_value(value))


def provisionally_promoted_cxa_portfolio_tab() -> None:
    render_page_intro(
        "Provisionally Promoted CxA Portfolio",
        "Review the provisionally promoted diagnostic CxA model through static portfolio outputs.",
        "Which actions and contributors create the most shot-creation probability?",
        (
            "This page reads the generated CxA portfolio pack only. It does not train, "
            "validate, regenerate results, or implement CxA+ inside Streamlit."
        ),
    )
    st.caption(
        "CxA is provisionally promoted because the diagnostic model improves the current "
        "baseline reference metrics, but that baseline comparison is full-data/in-sample. "
        "CxA+ and Advanced CxA are later work."
    )

    headline = load_portfolio_scorecard(CXA_PORTFOLIO_HEADLINE_PATH)
    players = load_portfolio_table(CXA_PORTFOLIO_PLAYERS_PATH)
    teams = load_portfolio_table(CXA_PORTFOLIO_TEAMS_PATH)
    sequences = load_portfolio_table(CXA_PORTFOLIO_SEQUENCES_PATH)
    feature_drivers = load_portfolio_table(CXA_PORTFOLIO_FEATURE_DRIVERS_PATH)
    summary_markdown = load_portfolio_markdown(CXA_PORTFOLIO_SUMMARY_PATH)

    if (
        not headline
        and players.empty
        and teams.empty
        and sequences.empty
        and feature_drivers.empty
        and not summary_markdown
    ):
        st.info(CXA_PORTFOLIO_REGENERATION_STEPS)
        return

    render_cxa_headline_metrics(headline)
    st.warning(
        "Promotion status is provisional: the baseline comparison is reference-only/in-sample, "
        "so this page should be read as a governed portfolio overview, not a final CxA+ claim."
    )

    top_feature = headline.get("top_feature_driver") or {}
    top_group = headline.get("top_feature_group_driver") or {}
    col1, col2 = st.columns(2)
    col1.metric("Top feature driver", _format_metric_value(top_feature.get("name")))
    col2.metric("Top feature group", _format_metric_value(top_group.get("name")))

    st.subheader("Static Portfolio Charts")
    chart_pairs = [
        ("baseline_vs_diagnostic_metrics.png", "Baseline vs Diagnostic Metrics"),
        ("feature_group_impact.png", "Feature Group Impact"),
        ("prediction_distribution.png", "Prediction Distribution"),
        ("top_players_by_cxa.png", "Top Players By CxA"),
        ("top_teams_by_cxa.png", "Top Teams By CxA"),
    ]
    for left, right in zip(chart_pairs[0::2], chart_pairs[1::2], strict=False):
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            render_portfolio_chart(left[0], left[1], CXA_PORTFOLIO_CHARTS)
        with chart_col2:
            render_portfolio_chart(right[0], right[1], CXA_PORTFOLIO_CHARTS)
    if len(chart_pairs) % 2:
        render_portfolio_chart(chart_pairs[-1][0], chart_pairs[-1][1], CXA_PORTFOLIO_CHARTS)

    st.subheader("Top CxA Contributors")
    player_limit = st.slider(
        "Top CxA players to show", min_value=5, max_value=100, value=20, step=5
    )
    filtered_players = _search_filter(
        players, "team_name", "Filter CxA players by team", "cxa_portfolio_player_team_search"
    )
    filtered_players = _search_filter(
        filtered_players,
        "player_name",
        "Filter CxA players by name",
        "cxa_portfolio_player_search",
    )
    render_portfolio_table(
        _select_existing_columns(
            filtered_players,
            [
                "player_name",
                "team_name",
                "total_diagnostic_cxa",
                "mean_diagnostic_cxa",
                "shot_creating_actions",
                "actions",
                "rank",
            ],
        ),
        "Top Players",
        max_rows=player_limit,
        sort_column="total_diagnostic_cxa",
    )
    with st.expander("Player IDs and full player table", expanded=False):
        render_portfolio_table(
            filtered_players,
            "Full Player Ranking",
            max_rows=player_limit,
            sort_column="total_diagnostic_cxa",
        )

    team_limit = st.slider("Top CxA teams to show", min_value=5, max_value=50, value=15, step=5)
    filtered_teams = _search_filter(
        teams, "team_name", "Filter CxA teams by name", "cxa_portfolio_team_search"
    )
    render_portfolio_table(
        _select_existing_columns(
            filtered_teams,
            [
                "team_name",
                "total_diagnostic_cxa",
                "mean_diagnostic_cxa",
                "shot_creating_actions",
                "actions",
                "rank",
            ],
        ),
        "Top Teams",
        max_rows=team_limit,
        sort_column="total_diagnostic_cxa",
    )

    sequence_limit = st.slider(
        "Top CxA sequences to show", min_value=5, max_value=100, value=20, step=5
    )
    filtered_sequences = _search_filter(
        sequences,
        "team_name",
        "Filter CxA sequences by team",
        "cxa_portfolio_sequence_team_search",
    )
    render_portfolio_table(
        _select_existing_columns(
            filtered_sequences,
            [
                "sequence_id",
                "team_name",
                "match_id",
                "total_diagnostic_cxa",
                "mean_diagnostic_cxa",
                "sequence_led_to_shot",
                "rank",
            ],
        ),
        "Top Sequences",
        max_rows=sequence_limit,
        sort_column="total_diagnostic_cxa",
    )

    render_portfolio_table(
        feature_drivers,
        "Feature Driver Summary",
        max_rows=50,
        sort_column="impact",
    )

    st.subheader("Narrative Summary")
    if summary_markdown:
        with st.expander("Read CxA portfolio summary", expanded=False):
            st.markdown(summary_markdown)
    else:
        st.info("Run `make build-cxa-portfolio-summary` to generate the Markdown summary.")


def cxg_tab(resources: dict[str, dict[str, DashboardResource]]) -> None:
    render_page_intro(
        "CxG",
        "CxG estimates contextual shot quality.",
        "How good was each shot before we know the outcome?",
        "CxG evaluates shot quality, not whether the shot became a goal.",
    )
    render_missing_guidance()
    _report_cards(
        _json_report(resources, "cxg", "metrics"),
        ["row_count", "brier_score", "log_loss", "roc_auc"],
    )
    _show_table(_dataframe(resources, "cxg", "predictions"), "Shot Predictions")
    validation = _json_report(resources, "cxg", "validation_summary")
    if validation:
        st.subheader("Validation Summary")
        st.json(validation)


def cxa_tab(resources: dict[str, dict[str, DashboardResource]]) -> None:
    render_page_intro(
        "CxA",
        "CxA estimates chance-creation value from eligible attacking actions.",
        "Which actions moved possessions closer to chance creation?",
        (
            "A high CxA player contributes actions that create or progress toward shots, "
            "even when they do not take the final shot."
        ),
    )
    render_missing_guidance()
    _report_cards(
        _json_report(resources, "cxa", "metrics"),
        ["row_count", "brier_score", "log_loss", "roc_auc"],
    )
    attribution = _json_report(resources, "cxa", "attribution_summary")
    if attribution:
        st.subheader("Attribution Summary")
        st.json(attribution)
    _show_table(_dataframe(resources, "cxa", "predictions"), "CxA Action Predictions")
    _show_table(_dataframe(resources, "cxa", "sequence_aggregates"), "CxA Sequence Aggregates")


def cxt_tab(resources: dict[str, dict[str, DashboardResource]]) -> None:
    render_page_intro(
        "CxT",
        "Baseline CxT measures the threat gained by moving from one pitch zone to another.",
        "Which actions and possessions added territorial threat?",
        (
            "Baseline CxT is location-threat movement, not full possession-state value. "
            "CxT+ and opponent-adjusted CxT are future roadmap items."
        ),
    )
    render_missing_guidance()
    _report_cards(
        _json_report(resources, "cxt", "interpretation_summary"),
        ["total_cxt", "pass_cxt", "carry_cxt", "final_third_entry_cxt", "box_entry_cxt"],
    )
    _show_table(_dataframe(resources, "cxt", "sequence_aggregates"), "CxT Sequence Aggregates")
    _show_table(_dataframe(resources, "cxt", "zone_transition_summary"), "Zone Transitions")
    _show_table(_dataframe(resources, "cxt", "top_actions"), "Top Positive / Negative Actions")


def action_explorer_tab(resources: dict[str, dict[str, DashboardResource]]) -> None:
    render_page_intro(
        "Action Explorer",
        "Inspect individual CxA and CxT action rows.",
        "Which specific actions explain aggregate player or team value?",
        (
            "Use this table after reading the leaderboards. Strong aggregate numbers "
            "should be traceable back to repeated valuable actions."
        ),
    )
    render_missing_guidance()
    cxa_actions = _dataframe(resources, "cxa", "predictions")
    cxt_actions = _dataframe(resources, "cxt", "action_threat")

    metric = st.radio(
        "Action source", ["CxT action threat", "CxA action predictions"], horizontal=True
    )
    df = cxt_actions if metric == "CxT action threat" else cxa_actions
    if df.empty:
        st.info(f"No rows available for {metric}.")
        return

    action_types = (
        sorted(df["action_type"].dropna().unique()) if "action_type" in df.columns else []
    )
    if action_types:
        selected = st.multiselect("Action type", action_types, default=action_types)
        df = df[df["action_type"].isin(selected)]
    st.dataframe(df.head(200), use_container_width=True, hide_index=True)


def diagnostics_tab(resources: dict[str, dict[str, DashboardResource]]) -> None:
    render_page_intro(
        "Reports / Diagnostics",
        "Inspect generated-output availability and report metadata.",
        "Which local outputs exist, and which reports are missing?",
        (
            "Use this page to verify the dashboard is reading generated files rather "
            "than committed outputs. Missing files mean the local run has not produced them yet."
        ),
    )
    status = status_table(resources)
    st.dataframe(status, use_container_width=True, hide_index=True)

    for metric in ("cxg", "cxa", "cxt"):
        with st.expander(f"{metric.upper()} JSON reports"):
            for name, resource in resources.get(metric, {}).items():
                if resource.json_data:
                    st.subheader(name)
                    st.json(resource.json_data)


def methodology_tab() -> None:
    render_page_intro(
        "About Methodology",
        "Read the modelling story and v1 limitations.",
        "What is implemented in v1, and what is intentionally deferred?",
        "This project is a demo and portfolio surface, not a production deployment claim.",
    )
    render_v1_scope_banner()
    st.markdown(
        """
        **CxG** asks how good a shot was.

        **CxA** asks which attacking actions created or progressed toward chances.

        **Baseline CxT** asks how much threat was added by moving the ball between
        pitch zones.

        This v1 dashboard is a demo and portfolio surface. It reads generated
        outputs from local ignored directories and degrades gracefully when those
        outputs have not been regenerated yet. It is not a production deployment,
        live-data service, or final calibration claim.
        """
    )
    st.subheader("Example interpretations")
    for interpretation in EXAMPLE_INTERPRETATIONS:
        render_insight_card("Example interpretation", interpretation, status="Example")


def main() -> None:
    contract = _load_contract()
    resources = _load_resources()

    st.title("Opponent-Adjusted Football Metrics")
    st.caption("CxG, CxA, and baseline CxT dashboard v1")
    st.write(
        "Guided demo flow: start with Overview, then open Promoted CxG portfolio "
        "and Provisionally Promoted CxA portfolio for the governed model stories. "
        "Use the legacy CxG, CxA, CxT, Player, Team, and Action explorer tabs for "
        "supporting generated outputs."
    )

    tabs = st.tabs(
        [
            "Overview",
            "Player analysis",
            "Team analysis",
            "Promoted CxG portfolio",
            "Provisionally Promoted CxA portfolio",
            "CxG",
            "CxA",
            "CxT",
            "Action explorer",
            "Reports / diagnostics",
            "About methodology",
        ]
    )

    with tabs[0]:
        overview_tab(resources, contract)
    with tabs[1]:
        player_tab(resources)
    with tabs[2]:
        team_tab(resources)
    with tabs[3]:
        promoted_cxg_portfolio_tab()
    with tabs[4]:
        provisionally_promoted_cxa_portfolio_tab()
    with tabs[5]:
        cxg_tab(resources)
    with tabs[6]:
        cxa_tab(resources)
    with tabs[7]:
        cxt_tab(resources)
    with tabs[8]:
        action_explorer_tab(resources)
    with tabs[9]:
        diagnostics_tab(resources)
    with tabs[10]:
        methodology_tab()


if __name__ == "__main__":
    main()
