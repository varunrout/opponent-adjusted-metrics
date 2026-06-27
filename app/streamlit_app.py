"""Streamlit dashboard v1 for opponent-adjusted football metrics."""

from __future__ import annotations

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
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)


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
    st.header("Project Overview")
    st.write(
        "Opponent-adjusted football analytics built from StatsBomb-style event data. "
        "CxG explains shot quality, CxA explains chance creation, and baseline CxT "
        "explains ball progression into more threatening pitch zones."
    )
    _availability_card(resources)

    st.subheader("Metric Families")
    for metric, section in contract["metric_sections"].items():
        st.markdown(f"**{metric.upper()}**: {section['metric_explanation']}")
        if roadmap_note := section.get("roadmap_note"):
            st.caption(roadmap_note)


def player_tab(resources: dict[str, dict[str, DashboardResource]]) -> None:
    st.header("Player Analysis")
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
    st.header("Team Analysis")
    cxg_teams = _dataframe(resources, "cxg", "team_aggregates")
    cxa_teams = _dataframe(resources, "cxa", "team_aggregates")
    cxt_teams = _dataframe(resources, "cxt", "team_aggregates")

    _bar_chart(cxt_teams, "team_name", "total_cxt", "Top Teams By Baseline CxT")
    _bar_chart(cxa_teams, "team_name", "total_cxa", "Top Teams By CxA")
    _bar_chart(cxg_teams, "team_name", "total_cxg", "Top Teams By CxG")
    _show_table(cxt_teams, "CxT Team Aggregates")
    _show_table(cxa_teams, "CxA Team Aggregates")
    _show_table(cxg_teams, "CxG Team Aggregates")


def cxg_tab(resources: dict[str, dict[str, DashboardResource]]) -> None:
    st.header("CxG")
    st.write("CxG estimates contextual shot quality.")
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
    st.header("CxA")
    st.write("CxA estimates chance-creation value from eligible attacking actions.")
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
    st.header("CxT")
    st.write(
        "Baseline CxT measures the threat gained by moving from one pitch zone to another. "
        "CxT+ and opponent-adjusted CxT are future roadmap items."
    )
    _report_cards(
        _json_report(resources, "cxt", "interpretation_summary"),
        ["total_cxt", "pass_cxt", "carry_cxt", "final_third_entry_cxt", "box_entry_cxt"],
    )
    _show_table(_dataframe(resources, "cxt", "sequence_aggregates"), "CxT Sequence Aggregates")
    _show_table(_dataframe(resources, "cxt", "zone_transition_summary"), "Zone Transitions")
    _show_table(_dataframe(resources, "cxt", "top_actions"), "Top Positive / Negative Actions")


def action_explorer_tab(resources: dict[str, dict[str, DashboardResource]]) -> None:
    st.header("Action Explorer")
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
    st.header("Reports / Diagnostics")
    st.write("Generated-output availability and lightweight report metadata.")
    status = status_table(resources)
    st.dataframe(status, use_container_width=True, hide_index=True)

    for metric in ("cxg", "cxa", "cxt"):
        with st.expander(f"{metric.upper()} JSON reports"):
            for name, resource in resources.get(metric, {}).items():
                if resource.json_data:
                    st.subheader(name)
                    st.json(resource.json_data)


def methodology_tab() -> None:
    st.header("About Methodology")
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


def main() -> None:
    contract = _load_contract()
    resources = _load_resources()

    st.title("Opponent-Adjusted Football Metrics")
    st.caption("CxG, CxA, and baseline CxT dashboard v1")

    tabs = st.tabs(
        [
            "Overview",
            "Player analysis",
            "Team analysis",
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
        cxg_tab(resources)
    with tabs[4]:
        cxa_tab(resources)
    with tabs[5]:
        cxt_tab(resources)
    with tabs[6]:
        action_explorer_tab(resources)
    with tabs[7]:
        diagnostics_tab(resources)
    with tabs[8]:
        methodology_tab()


if __name__ == "__main__":
    main()
