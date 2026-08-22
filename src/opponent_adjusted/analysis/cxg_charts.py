"""Render CxG analysis chart artifacts from BigQuery analysis tables."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mplsoccer import Pitch
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from google.cloud import bigquery, storage

from opponent_adjusted.analysis.cxg import ANALYSIS_DATASET, LOCATION, PROJECT_ID


DEFAULT_ARTIFACT_BUCKET = "oam-varun-260819-artifacts"
DEFAULT_ARTIFACT_PREFIX = "analysis/cxg"

# Explicit feature_family -> raw-value analysis surface mapping. Deliberately not a
# `_360`-suffix heuristic: `opponent_adjusted` is a 360-population family (joins to the
# 360-eligible cohort) but doesn't end in `_360`, and a suffix check would silently
# mis-route it (or any future family whose name doesn't happen to match the pattern)
# to `cxg_analysis_event_v1`, which has none of its columns.
FAMILY_SURFACE = {
    "base_identity_target": "cxg_analysis_event_v1",
    "shot_geometry": "cxg_analysis_event_v1",
    "event_context": "cxg_analysis_event_v1",
    "buildup": "cxg_analysis_event_v1",
    "defensive_360": "cxg_analysis_360_v1",
    "goalkeeper_360": "cxg_analysis_360_v1",
    "line_shape_360": "cxg_analysis_360_v1",
    "opponent_adjusted": "cxg_analysis_opponent_adjusted_v1",
}

# `defensive_profile_cluster` is stored INT64 (`cluster_label`) with only 4 distinct
# values -- the generic `data_type != 'STRING' and distinct_count > 2` numeric-detection
# heuristic used below would misclassify it as numeric (distinct_count=4 > 2) and try to
# histogram a categorical archetype code. Force it down the categorical path explicitly
# rather than changing the shared heuristic (which every other family also relies on).
CATEGORICAL_OVERRIDE_COLUMNS = {"defensive_profile_cluster"}

RENDERED_CHART_REGISTRY_SCHEMA = [
    bigquery.SchemaField("run_id", "STRING"),
    bigquery.SchemaField("chart_name", "STRING"),
    bigquery.SchemaField("html_uri", "STRING"),
    bigquery.SchemaField("png_uri", "STRING"),
    bigquery.SchemaField("rendered_at", "TIMESTAMP"),
]


@dataclass(frozen=True)
class ChartRow:
    run_id: str
    feature_family: str
    chart_name: str
    chart_type: str
    backing_table: str


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value)


class CxGChartRenderer:
    def __init__(
        self,
        project_id: str = PROJECT_ID,
        location: str = LOCATION,
        analysis_dataset: str = ANALYSIS_DATASET,
        artifact_bucket: str = DEFAULT_ARTIFACT_BUCKET,
        artifact_prefix: str = DEFAULT_ARTIFACT_PREFIX,
        run_id: str | None = None,
        local_output_dir: Path | None = None,
        limit: int | None = None,
        skip_upload: bool = False,
    ) -> None:
        self.project_id = project_id
        self.location = location
        self.analysis_dataset = analysis_dataset
        self.artifact_bucket = artifact_bucket
        self.artifact_prefix = artifact_prefix.strip("/")
        self.run_id = run_id
        self.local_output_dir = local_output_dir
        self.limit = limit
        self.skip_upload = skip_upload
        self.bq = bigquery.Client(project=project_id)
        self.gcs = storage.Client(project=project_id)

    def run(self) -> dict[str, object]:
        charts = self._load_chart_registry()
        if self.limit:
            charts = charts[: self.limit]
        run_id = self.run_id or (charts[0].run_id if charts else self._latest_run_id())
        render_dir = self._render_dir(run_id)
        render_dir.mkdir(parents=True, exist_ok=True)

        rendered = []
        for chart in charts:
            fig = self._render_chart(chart)
            html_path = render_dir / f"{chart.chart_name}.html"
            png_path = render_dir / f"{chart.chart_name}.png"
            fig.write_html(html_path, include_plotlyjs="cdn", full_html=True)
            self._write_png(chart, fig, png_path)
            if not self.skip_upload:
                self._upload(run_id, html_path)
                self._upload(run_id, png_path)
            rendered.append(
                {
                    "chart_name": chart.chart_name,
                    "html_uri": self._artifact_uri(run_id, html_path.name),
                    "png_uri": self._artifact_uri(run_id, png_path.name),
                }
            )

        manifest = {
            "run_id": run_id,
            "rendered_at": datetime.now(UTC).isoformat(),
            "chart_count": len(rendered),
            "html_count": len(rendered),
            "png_count": len(rendered),
            "artifact_prefix": f"gs://{self.artifact_bucket}/{self.artifact_prefix}/{run_id}/rendered_charts/",
            "local_prefix": str(render_dir),
            "upload_skipped": self.skip_upload,
            "charts": rendered,
        }
        manifest_path = render_dir / "rendered_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        if not self.skip_upload:
            self._upload(run_id, manifest_path)
            self._materialize_render_registry(manifest)
        return manifest

    def _render_dir(self, run_id: str) -> Path:
        base = self.local_output_dir or Path("audit_outputs/cxg_analysis")
        return base / run_id / "rendered_charts"

    def _latest_run_id(self) -> str:
        sql = f"""
        SELECT run_id
        FROM `{self.project_id}.{self.analysis_dataset}.cxg_chart_registry_v1`
        GROUP BY run_id
        ORDER BY MAX(materialized_at) DESC
        LIMIT 1
        """
        rows = list(self.bq.query(sql, location=self.location).result())
        if not rows:
            raise ValueError("No chart registry rows found.")
        return rows[0].run_id

    def _load_chart_registry(self) -> list[ChartRow]:
        run_filter = f"WHERE run_id = '{self.run_id}'" if self.run_id else ""
        sql = f"""
        SELECT run_id, feature_family, chart_name, chart_type, backing_table
        FROM `{self.project_id}.{self.analysis_dataset}.cxg_chart_registry_v1`
        {run_filter}
        ORDER BY feature_family, chart_name
        """
        return [
            ChartRow(row.run_id, row.feature_family, row.chart_name, row.chart_type, row.backing_table)
            for row in self.bq.query(sql, location=self.location).result()
        ]

    def _query_df(self, sql: str) -> pd.DataFrame:
        rows = list(self.bq.query(sql, location=self.location).result())
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([dict(row.items()) for row in rows])

    def _surface_for_family(self, family: str) -> str:
        try:
            return FAMILY_SURFACE[family]
        except KeyError:
            raise ValueError(f"No analysis surface mapping registered for feature_family={family!r}")

    def _render_chart(self, chart: ChartRow) -> go.Figure:
        if "null_profile" in chart.chart_name:
            return self._null_profile(chart)
        if "summary_box" in chart.chart_name:
            return self._summary_ranges(chart)
        if "eda_histogram" in chart.chart_name:
            return self._eda_bins(chart)
        if "target_lift" in chart.chart_name:
            return self._target_lift(chart)
        if chart.chart_type == "pitch_heatmap":
            return self._shot_pitch_heatmap(chart)
        if chart.chart_type == "pitch_scatter":
            return self._pitch_scatter(chart)
        if chart.chart_type == "heatmap":
            return self._correlation_heatmap(chart)
        if chart.chart_type == "feature_correlation_heatmap":
            return self._feature_correlation_heatmap(chart)
        if chart.chart_type == "pca_scree":
            return self._pca_scree(chart)
        if chart.chart_type == "bivariate_significance_grid":
            return self._bivariate_significance_grid(chart)
        if chart.chart_type == "bivariate_stratified_bar":
            return self._bivariate_stratified_bar(chart)
        if chart.chart_type == "baseline_calibration":
            return self._baseline_calibration(chart)
        if chart.chart_type == "baseline_divergence_bar":
            return self._baseline_divergence_bar(chart)
        if chart.chart_type == "archetype_goal_rate_bar":
            return self._archetype_goal_rate_bar(chart)
        if chart.chart_type == "archetype_role_heatmap":
            return self._archetype_role_heatmap(chart)
        if chart.chart_type == "role_displacement_bar":
            return self._role_displacement_bar(chart)
        return self._family_summary(chart)

    def _layout(self, fig: go.Figure, title: str) -> go.Figure:
        fig.update_layout(
            title=title,
            template="plotly_white",
            font={"family": "Arial", "size": 13},
            margin={"l": 56, "r": 28, "t": 72, "b": 70},
            legend_title_text="Feature",
        )
        return fig

    def _null_profile(self, chart: ChartRow) -> go.Figure:
        sql = f"""
        SELECT column_name, null_pct * 100 AS null_pct
        FROM `{self.project_id}.{self.analysis_dataset}.cxg_null_profile_v1`
        WHERE feature_family = '{chart.feature_family}'
        ORDER BY null_pct DESC, column_name
        LIMIT 40
        """
        df = self._query_df(sql)
        if df.empty:
            return self._layout(go.Figure(), f"{chart.feature_family}: No Null Profile")
        fig = px.bar(df, x="null_pct", y="column_name", orientation="h", labels={"null_pct": "Null %"})
        return self._layout(fig, f"{chart.feature_family}: Null Profile")

    def _summary_ranges(self, chart: ChartRow) -> go.Figure:
        sql = f"""
        SELECT column_name, min_value, p25_value, median_value, p75_value, max_value, distinct_count
        FROM `{self.project_id}.{self.analysis_dataset}.cxg_summary_stats_v1`
        WHERE feature_family = '{chart.feature_family}'
          AND median_value IS NOT NULL
          AND max_value IS NOT NULL
          AND min_value IS NOT NULL
        ORDER BY distinct_count DESC, column_name
        LIMIT 30
        """
        df = self._query_df(sql)
        if df.empty:
            return self._layout(go.Figure(), f"{chart.feature_family}: Summary Ranges")
        span = (df["max_value"] - df["min_value"]).replace(0, np.nan)
        for metric in ["p25_value", "median_value", "p75_value"]:
            df[f"{metric}_norm"] = ((df[metric] - df["min_value"]) / span).fillna(0.5)
        fig = go.Figure()
        for _, row in df.iterrows():
            fig.add_trace(go.Scatter(x=[0, 1], y=[row["column_name"], row["column_name"]], mode="lines", line={"color": "#d1d5db", "width": 2}, showlegend=False, hoverinfo="skip"))
        for metric, label, color, symbol in [
            ("p25_value_norm", "p25", "#60a5fa", "circle"),
            ("median_value_norm", "median", "#111827", "diamond"),
            ("p75_value_norm", "p75", "#34d399", "circle"),
        ]:
            fig.add_trace(go.Scatter(x=df[metric], y=df["column_name"], mode="markers", name=label, marker={"color": color, "symbol": symbol, "size": 9}, customdata=df[["min_value", "p25_value", "median_value", "p75_value", "max_value"]], hovertemplate="%{y}<br>normalized=%{x:.2f}<br>min=%{customdata[0]:.3f}<br>p25=%{customdata[1]:.3f}<br>median=%{customdata[2]:.3f}<br>p75=%{customdata[3]:.3f}<br>max=%{customdata[4]:.3f}<extra></extra>"))
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(range=[-0.03, 1.03], title="Within-feature normalized range: min=0, max=1")
        return self._layout(fig, f"{chart.feature_family}: Normalized Summary Ranges")

    def _eda_bins(self, chart: ChartRow) -> go.Figure:
        feature_sql = f"""
        SELECT column_name, data_type, distinct_count
        FROM `{self.project_id}.{self.analysis_dataset}.cxg_summary_stats_v1`
        WHERE feature_family = '{chart.feature_family}' AND non_null_count > 0
        ORDER BY
          CASE WHEN data_type IN ('FLOAT64', 'INT64', 'NUMERIC', 'BIGNUMERIC') AND distinct_count > 2 THEN 0 ELSE 1 END,
          distinct_count DESC,
          column_name
        LIMIT 6
        """
        features = self._query_df(feature_sql)
        numeric_features = [
            row
            for _, row in features.iterrows()
            if row["data_type"] != "STRING"
            and row["distinct_count"] > 2
            and row["column_name"] not in CATEGORICAL_OVERRIDE_COLUMNS
        ]
        if numeric_features:
            surface = self._surface_for_family(chart.feature_family)
            selects = []
            for row in numeric_features[:6]:
                col = row["column_name"]
                selects.append(f"""
                SELECT '{col}' AS column_name, CAST(`{col}` AS FLOAT64) AS value
                FROM `{self.project_id}.{self.analysis_dataset}.{surface}`
                WHERE `{col}` IS NOT NULL
                """)
            df = self._query_df("\nUNION ALL\n".join(selects))
            fig = px.histogram(df, x="value", color="column_name", facet_col="column_name", facet_col_wrap=2, nbins=30, histnorm="percent", labels={"value": "Feature value", "percent": "% of available rows", "column_name": "Feature"})
            fig.update_yaxes(matches=None)
            fig.update_xaxes(matches=None)
            return self._layout(fig, f"{chart.feature_family}: Value Distributions")
        sql = f"""
        SELECT column_name, bin_label, row_count
        FROM `{self.project_id}.{self.analysis_dataset}.cxg_eda_distribution_bins_v1`
        WHERE feature_family = '{chart.feature_family}'
        QUALIFY ROW_NUMBER() OVER (PARTITION BY column_name ORDER BY row_count DESC) <= 20
        """
        df = self._query_df(sql)
        if df.empty:
            return self._layout(go.Figure(), f"{chart.feature_family}: No Feature Distributions")
        fig = px.bar(df, x="bin_label", y="row_count", color="column_name", barmode="group", labels={"bin_label": "Bin / category", "row_count": "Rows", "column_name": "Feature"})
        return self._layout(fig, f"{chart.feature_family}: Category Distributions")

    def _target_lift(self, chart: ChartRow) -> go.Figure:
        sql = f"""
        SELECT column_name, point_biserial_corr
        FROM `{self.project_id}.{self.analysis_dataset}.cxg_univariate_target_v1`
        WHERE feature_family = '{chart.feature_family}' AND point_biserial_corr IS NOT NULL
        ORDER BY ABS(point_biserial_corr) DESC
        LIMIT 30
        """
        df = self._query_df(sql)
        if df.empty:
            return self._layout(go.Figure(), f"{chart.feature_family}: No Univariate Target Signal")
        fig = px.bar(df, x="point_biserial_corr", y="column_name", orientation="h")
        return self._layout(fig, f"{chart.feature_family}: Univariate Target Signal")

    def _add_plotly_pitch(self, fig: go.Figure) -> go.Figure:
        line = {"color": "#111827", "width": 1.2}
        shapes = [
            {"type": "rect", "x0": 0, "y0": 0, "x1": 120, "y1": 80, "line": line},
            {"type": "line", "x0": 60, "y0": 0, "x1": 60, "y1": 80, "line": line},
            {"type": "circle", "x0": 50, "y0": 30, "x1": 70, "y1": 50, "line": line},
            {"type": "rect", "x0": 0, "y0": 18, "x1": 18, "y1": 62, "line": line},
            {"type": "rect", "x0": 102, "y0": 18, "x1": 120, "y1": 62, "line": line},
            {"type": "rect", "x0": 0, "y0": 30, "x1": 6, "y1": 50, "line": line},
            {"type": "rect", "x0": 114, "y0": 30, "x1": 120, "y1": 50, "line": line},
        ]
        fig.update_layout(shapes=shapes, plot_bgcolor="#f8fafc")
        fig.update_xaxes(range=[0, 120], showgrid=False, zeroline=False, title="StatsBomb x")
        fig.update_yaxes(range=[0, 80], showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1, title="StatsBomb y")
        return fig

    def _shot_pitch_heatmap(self, chart: ChartRow) -> go.Figure:
        sql = f"""
        SELECT shot_x_sb, shot_y_sb
        FROM `{self.project_id}.{self.analysis_dataset}.cxg_analysis_event_v1`
        WHERE shot_x_sb IS NOT NULL AND shot_y_sb IS NOT NULL
        """
        df = self._query_df(sql)
        fig = px.density_heatmap(df, x="shot_x_sb", y="shot_y_sb", nbinsx=24, nbinsy=16, color_continuous_scale="YlOrRd", labels={"count": "Shot count"})
        fig = self._add_plotly_pitch(fig)
        return self._layout(fig, "Shot Geometry: Shot Density On Pitch")

    def _pitch_scatter(self, chart: ChartRow) -> go.Figure:
        if chart.feature_family == "goalkeeper_360":
            x_col, y_col, color_col = "gk_x", "gk_y", "gk_depth"
        else:
            x_col, y_col, color_col = "shot_x_sb", "shot_y_sb", "nearest_defender_distance"
        sql = f"""
        SELECT {x_col} AS x, {y_col} AS y, {color_col} AS color_value, is_goal
        FROM `{self.project_id}.{self.analysis_dataset}.cxg_analysis_360_v1`
        WHERE {x_col} IS NOT NULL AND {y_col} IS NOT NULL
        LIMIT 5000
        """
        df = self._query_df(sql)
        fig = px.scatter(df, x="x", y="y", color="color_value", symbol="is_goal", opacity=0.72, labels={"color_value": color_col, "is_goal": "Goal"})
        fig = self._add_plotly_pitch(fig)
        return self._layout(fig, f"{chart.feature_family}: Pitch Scatter")

    def _correlation_heatmap(self, chart: ChartRow) -> go.Figure:
        sql = f"""
        SELECT left_column, right_column, pearson_corr
        FROM `{self.project_id}.{self.analysis_dataset}.cxg_correlation_v1`
        WHERE left_family = '{chart.feature_family}' AND pearson_corr IS NOT NULL
        ORDER BY ABS(pearson_corr) DESC
        LIMIT 80
        """
        df = self._query_df(sql)
        if df.empty:
            return self._layout(go.Figure(), f"{chart.feature_family}: Correlation Heatmap")
        matrix = df.pivot(index="right_column", columns="left_column", values="pearson_corr")
        fig = px.imshow(matrix, color_continuous_scale="RdBu", zmin=-1, zmax=1, aspect="auto", labels={"color": "Pearson r", "x": "Feature", "y": "Feature"})
        return self._layout(fig, f"{chart.feature_family}: Correlation Heatmap")

    def _correlation_screen_track(self, chart: ChartRow) -> str:
        """Track is encoded as a synthetic feature_family (cxg_event_correlation /
        cxg_plus_correlation / cxg_event_pca / cxg_plus_pca) to avoid adding a new column
        to the existing chart-registry schema -- consistent with how every other chart
        type in this registry is keyed by feature_family."""
        return chart.feature_family.rsplit("_", 1)[0]

    def _feature_correlation_heatmap(self, chart: ChartRow) -> go.Figure:
        track = self._correlation_screen_track(chart)
        df = self._query_df(f"""
        SELECT feature_a, feature_b, r_train
        FROM `{self.project_id}.{self.analysis_dataset}.cxg_feature_correlation_v1`
        WHERE track = '{track}' AND r_train IS NOT NULL
        """)
        if df.empty:
            return self._layout(go.Figure(), f"{track}: Feature Correlation Heatmap")
        matrix = self._symmetric_corr_matrix(df)
        fig = px.imshow(matrix, color_continuous_scale="RdBu", zmin=-1, zmax=1, aspect="auto", labels={"color": "Pearson r (train)"})
        return self._layout(fig, f"{track}: Feature Correlation Heatmap (train)")

    @staticmethod
    def _symmetric_corr_matrix(df: pd.DataFrame) -> pd.DataFrame:
        features = sorted(set(df["feature_a"]) | set(df["feature_b"]))
        matrix = pd.DataFrame(np.eye(len(features)), index=features, columns=features)
        for _, row in df.iterrows():
            matrix.loc[row["feature_a"], row["feature_b"]] = row["r_train"]
            matrix.loc[row["feature_b"], row["feature_a"]] = row["r_train"]
        return matrix

    def _pca_scree(self, chart: ChartRow) -> go.Figure:
        track = self._correlation_screen_track(chart)
        df = self._query_df(f"""
        SELECT component_number, explained_variance_ratio, cumulative_variance_ratio
        FROM `{self.project_id}.{self.analysis_dataset}.cxg_pca_components_v1`
        WHERE track = '{track}'
        ORDER BY component_number
        """)
        if df.empty:
            return self._layout(go.Figure(), f"{track}: PCA Scree")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df["component_number"], y=df["explained_variance_ratio"], name="Explained variance ratio"))
        fig.add_trace(go.Scatter(x=df["component_number"], y=df["cumulative_variance_ratio"], name="Cumulative variance ratio", mode="lines+markers"))
        fig.update_xaxes(title="Component", dtick=1)
        fig.update_yaxes(title="Variance ratio", range=[0, 1.05])
        return self._layout(fig, f"{track}: PCA Scree (train)")

    # Phase A/B candidates -- any confirmed Tier 1 pair touching one of these is "new this
    # round" (only exists in the pool because of Phase A/B); a confirmed pair touching none
    # of them is "reconfirmed" (also held in the pre-Phase-A/B pool). Derived from the pool
    # membership, not a hardcoded pair list -- stays correct if the confirmed set changes.
    _PHASE_AB_FEATURES = frozenset({
        "nearest_defender_role", "second_nearest_defender_role",
        "nearest_defender_zone_displacement", "nearest_defender_gap",
        "nearest_defender_style_archetype",
    })

    def _bivariate_significance_grid(self, chart: ChartRow) -> go.Figure:
        """Tier 1 exhaustive pairwise grid: FDR-adjusted p-value per pair, one grid per track
        (feature_family encoded as `<track>_bivariate`, same synthetic-name pattern as the
        correlation-screen/PCA charts above). Confirmed pairs (FDR<0.10 AND validated on the
        held-out split) are annotated: a gold star for a pair newly confirmed because of a
        Phase A/B feature, a white check for a pair also confirmed before the enlarged pool
        (reconfirmed) -- so the grid distinguishes "new this round" from "still holds"."""
        track = self._correlation_screen_track(chart)
        df = self._query_df(f"""
        SELECT feature_a, feature_b, interaction_p_fdr
        FROM `{self.project_id}.{self.analysis_dataset}.cxg_bivariate_interaction_v1`
        WHERE track = '{track}' AND tier = 1 AND interaction_p_fdr IS NOT NULL
        """)
        if df.empty:
            return self._layout(go.Figure(), f"{track}: Tier 1 Interaction Significance Grid")
        confirmed = self._query_df(f"""
        SELECT feature_a, feature_b
        FROM `{self.project_id}.{self.analysis_dataset}.cxg_bivariate_interaction_v1`
        WHERE track = '{track}' AND tier = 1 AND validated_on_val_split = TRUE
        """)
        df = df.rename(columns={"interaction_p_fdr": "r_train"})
        matrix = self._symmetric_corr_matrix(df).clip(0, 1)
        neg_log10 = matrix.apply(lambda col: -np.log10(col.clip(lower=1e-300)))
        fig = px.imshow(neg_log10, color_continuous_scale="Viridis", aspect="auto", labels={"color": "-log10(FDR p)"})
        for _, row in confirmed.iterrows():
            a, b = row["feature_a"], row["feature_b"]
            if a not in matrix.index or b not in matrix.columns:
                continue
            is_new = a in self._PHASE_AB_FEATURES or b in self._PHASE_AB_FEATURES
            symbol, color = ("★ NEW", "#facc15") if is_new else ("✓ reconfirmed", "#f8fafc")
            for x, y in ((b, a), (a, b)):
                fig.add_annotation(x=x, y=y, text=symbol, showarrow=False, font={"color": color, "size": 10, "family": "Arial Black"})
        return self._layout(fig, f"{track}: Tier 1 Interaction Significance Grid (higher = more significant; ★ = newly confirmed, ✓ = reconfirmed)")

    @staticmethod
    def _bivariate_tier_from_chart_name(chart_name: str) -> int:
        for tier in (2, 3):
            if f"tier{tier}" in chart_name:
                return tier
        raise ValueError(f"Cannot infer bivariate tier from chart_name={chart_name!r}")

    def _bivariate_stratified_bar(self, chart: ChartRow) -> go.Figure:
        track = self._correlation_screen_track(chart)
        tier = self._bivariate_tier_from_chart_name(chart.chart_name)
        df = self._query_df(f"""
        SELECT feature_a, feature_b, stratum_a, stratum_b, n, goal_count, goal_rate
        FROM `{self.project_id}.{self.analysis_dataset}.cxg_bivariate_stratified_v1`
        WHERE track = '{track}' AND tier = {tier}
        ORDER BY stratum_a, stratum_b
        """)
        if df.empty:
            return self._layout(go.Figure(), f"{track}: Tier {tier} Stratified Goal Rate")
        feature_a, feature_b = df["feature_a"].iloc[0], df["feature_b"].iloc[0]
        fig = px.bar(
            df, x="stratum_a", y="goal_rate", color="stratum_b", barmode="group",
            hover_data=["n", "goal_count"],
            labels={"stratum_a": feature_a, "goal_rate": "Goal rate", "stratum_b": feature_b},
        )
        return self._layout(fig, f"{track}: Tier {tier} Stratified Goal Rate ({feature_a} x {feature_b})")

    def _baseline_calibration(self, chart: ChartRow) -> go.Figure:
        """Dumb-baseline vs v1 (vs v2, when the chart's family is `<track>_v2`) calibration,
        test split, one grid per track (feature_family encoded as `<track>_baseline` /
        `<track>_v2`, same synthetic-name pattern as the correlation-screen/PCA/bivariate
        charts). Decile assignment via NTILE(10) -- the SQL equivalent of the rank-based
        binning `modeling.calibration_table` uses, so it stays well-defined even for the
        dumb baseline's tied constant predictions. For a v2 chart, v1 is overlaid alongside
        v2 (both already at hand via separate tables) so the calibration comparison the v2
        report needs is visible directly in the chart, not just the metrics table."""
        track = self._correlation_screen_track(chart)
        is_v2 = chart.feature_family.endswith("_v2")
        rows = []
        for model_col, label in (("dumb_baseline_prob", "dumb_baseline"), ("v1_predicted_prob", "v1")):
            df = self._query_df(f"""
            WITH ranked AS (
              SELECT `{model_col}` AS p, CAST(is_goal AS INT64) AS y,
                     NTILE(10) OVER (ORDER BY `{model_col}`, event_id) AS decile
              FROM `{self.project_id}.oam_ml.cxg_baseline_v1_predictions`
              WHERE track = '{track}' AND split = 'test'
            )
            SELECT decile, AVG(p) AS mean_predicted, AVG(y) AS actual_rate, COUNT(*) AS n
            FROM ranked GROUP BY decile ORDER BY decile
            """)
            df["model"] = label
            rows.append(df)
        if is_v2:
            df_v2 = self._query_df(f"""
            WITH ranked AS (
              SELECT v2_predicted_prob AS p, CAST(is_goal AS INT64) AS y,
                     NTILE(10) OVER (ORDER BY v2_predicted_prob, event_id) AS decile
              FROM `{self.project_id}.oam_ml.cxg_plus_v2_predictions`
              WHERE track = '{track}' AND split = 'test'
            )
            SELECT decile, AVG(p) AS mean_predicted, AVG(y) AS actual_rate, COUNT(*) AS n
            FROM ranked GROUP BY decile ORDER BY decile
            """)
            df_v2["model"] = "v2"
            rows.append(df_v2)
        df_all = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        if df_all.empty:
            return self._layout(go.Figure(), f"{track}: Calibration (test)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line={"dash": "dot", "color": "#9ca3af"}, name="Perfect calibration", hoverinfo="skip"))
        palette = {"dumb_baseline": "#f59e0b", "v1": "#2563eb", "v2": "#16a34a"}
        for label, color in palette.items():
            d = df_all[df_all.model == label]
            if d.empty:
                continue
            fig.add_trace(go.Scatter(x=d["mean_predicted"], y=d["actual_rate"], mode="lines+markers", name=label, marker={"color": color, "size": 9}, line={"color": color}))
        fig.update_xaxes(title="Mean predicted probability (decile)")
        fig.update_yaxes(title="Actual goal rate")
        title_models = "dumb baseline vs v1 vs v2" if is_v2 else "dumb baseline vs v1"
        return self._layout(fig, f"{track}: Calibration, {title_models} (test split)")

    def _baseline_divergence_stratum_type(self, chart: ChartRow) -> str:
        if "archetype" in chart.chart_name:
            return "nearest_defender_style_archetype"
        if "odi_tercile" in chart.chart_name:
            return "odi_tercile_nearest_defender_odi"
        return "defensive_profile_cluster"

    def _baseline_divergence_bar(self, chart: ChartRow) -> go.Figure:
        is_v2 = chart.feature_family.endswith("_v2")
        stratum_type = self._baseline_divergence_stratum_type(chart)
        table = "cxg_plus_v2_divergence" if is_v2 else "cxg_baseline_v1_divergence"
        prob_col = "mean_v2_prob" if is_v2 else "mean_v1_prob"
        model_label = "v2_predicted_prob" if is_v2 else "v1_predicted_prob"
        df = self._query_df(f"""
        SELECT stratum_value, mean_divergence, {prob_col}, mean_statsbomb_xg, n
        FROM `{self.project_id}.oam_ml.{table}`
        WHERE stratum_type = '{stratum_type}'
        ORDER BY stratum_value
        """)
        if df.empty:
            return self._layout(go.Figure(), f"cxg_plus: Divergence by {stratum_type}")
        fig = px.bar(df, x="stratum_value", y="mean_divergence", hover_data=["n", prob_col, "mean_statsbomb_xg"], labels={"stratum_value": stratum_type, "mean_divergence": f"mean({model_label} - statsbomb_xg)"})
        fig.add_hline(y=0, line_dash="dot", line_color="#9ca3af")
        model_name = "v2" if is_v2 else "v1"
        return self._layout(fig, f"CxG+: {model_name} vs StatsBomb xG Divergence by {stratum_type} (test split)")

    _ROLE_ORDER = ("GK", "CB", "Fullback_WingBack", "Midfield", "Attack")

    def _archetype_goal_rate_bar(self, chart: ChartRow) -> go.Figure:
        """Goal rate by nearest_defender_style_archetype level, CxG+ only, train split (same
        population/discipline as every other univariate goal-rate check in this pipeline).
        `cxg_univariate_target_v1` only carries one aggregate row per feature (matching
        defensive_profile_cluster's own precedent), not a per-level breakdown -- checked
        before assuming it was reusable, and it isn't for this chart, so this queries the
        already-materialized feature + split tables directly instead (a single batched
        GROUP BY, not a loop). The muddy 4th cluster (`unresolved_5050_annotation_density`)
        and the null/below-threshold case are both included, not hidden."""
        df = self._query_df(f"""
        SELECT COALESCE(d.nearest_defender_style_archetype, 'null (no defender / below threshold)') AS archetype,
               COUNT(*) AS n, AVG(IF(m.is_goal, 1.0, 0.0)) AS goal_rate
        FROM `{self.project_id}.{self.analysis_dataset}.cxg_plus_360_model_matrix_v1` m
        JOIN `{self.project_id}.oam_features.cxg_defensive_360_features` d ON m.event_id = d.event_id
        WHERE m.split = 'train'
        GROUP BY archetype
        ORDER BY goal_rate DESC
        """)
        if df.empty:
            return self._layout(go.Figure(), "CxG+: Goal Rate by Defender Style Archetype")
        fig = px.bar(df, x="archetype", y="goal_rate", hover_data=["n"], labels={"archetype": "nearest_defender_style_archetype", "goal_rate": "Goal rate (train split)"})
        return self._layout(fig, "CxG+: Goal Rate by Defender Style Archetype (train split)")

    def _archetype_role_heatmap(self, chart: ChartRow) -> go.Figure:
        """Cross-tab of nearest_defender_style_archetype x nearest_defender_role, cell = %
        share within each archetype row. Neither cxg_defender_style_cluster_profile_v1 nor
        any other materialized table already has this cross-tab (checked before writing this
        query) -- computed here as a single batched GROUP BY over the already-materialized
        cxg_defensive_360_features, not a re-derivation of anything from raw events."""
        df = self._query_df(f"""
        SELECT COALESCE(nearest_defender_style_archetype, 'null') AS archetype,
               COALESCE(nearest_defender_role, 'null') AS role, COUNT(*) AS n
        FROM `{self.project_id}.oam_features.cxg_defensive_360_features`
        GROUP BY archetype, role
        """)
        if df.empty:
            return self._layout(go.Figure(), "CxG+: Defender Style Archetype x Role")
        pivot = df.pivot(index="archetype", columns="role", values="n").fillna(0)
        pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
        fig = px.imshow(pct, color_continuous_scale="Blues", aspect="auto", text_auto=".0f", labels={"color": "% of archetype's shots"})
        return self._layout(fig, "CxG+: Defender Style Archetype x Nearest Defender Role (row %)")

    def _role_displacement_bar(self, chart: ChartRow) -> go.Figure:
        """Mean nearest_defender_zone_displacement by nearest_defender_role, ordered
        GK -> CB -> Fullback_WingBack -> Midfield -> Attack (Phase A's own coordinate-fix
        verification order) -- single batched GROUP BY, not a per-role loop."""
        df = self._query_df(f"""
        SELECT nearest_defender_role AS role, COUNT(*) AS n,
               AVG(nearest_defender_zone_displacement) AS mean_displacement
        FROM `{self.project_id}.oam_features.cxg_defensive_360_features`
        WHERE nearest_defender_role IS NOT NULL AND nearest_defender_zone_displacement IS NOT NULL
        GROUP BY role
        """)
        if df.empty:
            return self._layout(go.Figure(), "CxG+: Zone Displacement by Nearest Defender Role")
        df["role"] = pd.Categorical(df["role"], categories=self._ROLE_ORDER, ordered=True)
        df = df.sort_values("role")
        fig = px.bar(df, x="role", y="mean_displacement", hover_data=["n"], labels={"role": "nearest_defender_role", "mean_displacement": "Mean zone displacement (m)"})
        return self._layout(fig, "CxG+: Mean Zone Displacement by Nearest Defender Role")

    def _family_summary(self, chart: ChartRow) -> go.Figure:
        sql = f"SELECT * FROM `{self.project_id}.{self.analysis_dataset}.cxg_family_summary_v1`"
        df = self._query_df(sql)
        fig = px.bar(df, x="feature_family", y="feature_count", color="avg_null_pct")
        return self._layout(fig, "CxG Family Summary")

    def _write_png(self, chart: ChartRow, fig: go.Figure, path: Path) -> None:
        if chart.chart_type in {"pitch_heatmap", "pitch_scatter"}:
            self._write_pitch_png(chart, path)
            return
        if chart.chart_type == "heatmap":
            self._write_correlation_png(chart, path)
            return
        if chart.chart_type == "feature_correlation_heatmap":
            self._write_feature_correlation_png(chart, path)
            return
        if chart.chart_type == "bivariate_significance_grid":
            self._write_bivariate_significance_grid_png(chart, path)
            return
        if chart.chart_type == "archetype_role_heatmap":
            self._write_archetype_role_heatmap_png(chart, path)
            return
        if "summary_box" in chart.chart_name:
            self._write_summary_png(chart, path)
            return
        if "eda_histogram" in chart.chart_name:
            self._write_distribution_png(chart, path)
            return
        self._write_generic_png(fig, path)

    def _write_generic_png(self, fig: go.Figure, path: Path) -> None:
        title = fig.layout.title.text if fig.layout.title and fig.layout.title.text else path.stem
        plt_fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=150)
        drew = False
        for trace in fig.data:
            trace_type = getattr(trace, "type", "")
            if trace_type == "bar":
                x = list(trace.x) if trace.x is not None else []
                y = list(trace.y) if trace.y is not None else []
                name = getattr(trace, "name", None)
                if getattr(trace, "orientation", None) == "h":
                    ax.barh(y, x, label=name if name else None, alpha=0.82)
                    ax.tick_params(axis="y", labelsize=8)
                else:
                    ax.bar(range(len(x)), y, label=name if name else None, alpha=0.82)
                    labels = [str(item)[:18] + ("..." if len(str(item)) > 18 else "") for item in x]
                    ax.set_xticks(range(len(labels)))
                    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
                drew = True
            elif trace_type == "scatter":
                x = list(trace.x) if trace.x is not None else []
                y = list(trace.y) if trace.y is not None else []
                name = getattr(trace, "name", None)
                ax.scatter(x, y, label=name if name else None, alpha=0.72, s=18, edgecolors="none")
                drew = True
        if not drew:
            ax.text(0.5, 0.5, "No plottable data", ha="center", va="center", fontsize=14, transform=ax.transAxes)
            ax.set_axis_off()
        else:
            ax.grid(True, alpha=0.18)
            if len(fig.data) <= 12 and any(getattr(trace, "name", None) for trace in fig.data):
                ax.legend(loc="best", fontsize=8)
        ax.set_title(title, fontsize=14, pad=14)
        plt_fig.tight_layout()
        plt_fig.savefig(path, format="png")
        plt.close(plt_fig)

    def _write_pitch_png(self, chart: ChartRow, path: Path) -> None:
        pitch = Pitch(pitch_type="statsbomb", pitch_color="#f8fafc", line_color="#111827", linewidth=1.2)
        plt_fig, ax = pitch.draw(figsize=(12.8, 7.2))
        if chart.chart_type == "pitch_heatmap":
            df = self._query_df(f"""
            SELECT shot_x_sb AS x, shot_y_sb AS y
            FROM `{self.project_id}.{self.analysis_dataset}.cxg_analysis_event_v1`
            WHERE shot_x_sb IS NOT NULL AND shot_y_sb IS NOT NULL
            """)
            if not df.empty:
                hb = pitch.hexbin(df["x"], df["y"], ax=ax, gridsize=(24, 16), cmap="YlOrRd", edgecolors="#ffffff", linewidths=0.25)
                plt_fig.colorbar(hb, ax=ax, fraction=0.035, pad=0.02, label="Shot count")
            ax.set_title("Shot Geometry: Shot Density On Pitch", fontsize=14, pad=14)
        else:
            if chart.feature_family == "goalkeeper_360":
                x_col, y_col, color_col, title = "gk_x", "gk_y", "gk_depth", "Goalkeeper 360: GK Positions"
            else:
                x_col, y_col, color_col, title = "shot_x_sb", "shot_y_sb", "nearest_defender_distance", "Defensive 360: Shot Locations By Nearest Defender"
            df = self._query_df(f"""
            SELECT {x_col} AS x, {y_col} AS y, {color_col} AS color_value
            FROM `{self.project_id}.{self.analysis_dataset}.cxg_analysis_360_v1`
            WHERE {x_col} IS NOT NULL AND {y_col} IS NOT NULL
            LIMIT 5000
            """)
            if not df.empty:
                sc = pitch.scatter(df["x"], df["y"], c=df["color_value"], cmap="viridis", s=16, alpha=0.72, ax=ax)
                plt_fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02, label=color_col)
            ax.set_title(title, fontsize=14, pad=14)
        plt_fig.tight_layout()
        plt_fig.savefig(path, format="png", dpi=150)
        plt.close(plt_fig)

    def _write_correlation_png(self, chart: ChartRow, path: Path) -> None:
        df = self._query_df(f"""
        SELECT left_column, right_column, pearson_corr
        FROM `{self.project_id}.{self.analysis_dataset}.cxg_correlation_v1`
        WHERE left_family = '{chart.feature_family}' AND pearson_corr IS NOT NULL
        ORDER BY ABS(pearson_corr) DESC
        LIMIT 80
        """)
        plt_fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=150)
        if df.empty:
            ax.text(0.5, 0.5, "No valid correlation pairs", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
        else:
            matrix = df.pivot(index="right_column", columns="left_column", values="pearson_corr")
            im = ax.imshow(matrix.fillna(0), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
            ax.set_xticks(range(len(matrix.columns)))
            ax.set_xticklabels([str(v)[:22] for v in matrix.columns], rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(len(matrix.index)))
            ax.set_yticklabels([str(v)[:28] for v in matrix.index], fontsize=7)
            plt_fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="Pearson r")
        ax.set_title(f"{chart.feature_family}: Correlation Heatmap", fontsize=14, pad=14)
        plt_fig.tight_layout()
        plt_fig.savefig(path, format="png")
        plt.close(plt_fig)

    def _write_feature_correlation_png(self, chart: ChartRow, path: Path) -> None:
        track = self._correlation_screen_track(chart)
        df = self._query_df(f"""
        SELECT feature_a, feature_b, r_train
        FROM `{self.project_id}.{self.analysis_dataset}.cxg_feature_correlation_v1`
        WHERE track = '{track}' AND r_train IS NOT NULL
        """)
        plt_fig, ax = plt.subplots(figsize=(12.8, 10.0), dpi=150)
        if df.empty:
            ax.text(0.5, 0.5, "No valid correlation pairs", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
        else:
            matrix = self._symmetric_corr_matrix(df)
            im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
            ax.set_xticks(range(len(matrix.columns)))
            ax.set_xticklabels([str(v)[:22] for v in matrix.columns], rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(len(matrix.index)))
            ax.set_yticklabels([str(v)[:28] for v in matrix.index], fontsize=7)
            plt_fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="Pearson r (train)")
        ax.set_title(f"{track}: Feature Correlation Heatmap (train)", fontsize=14, pad=14)
        plt_fig.tight_layout()
        plt_fig.savefig(path, format="png")
        plt.close(plt_fig)

    def _write_bivariate_significance_grid_png(self, chart: ChartRow, path: Path) -> None:
        track = self._correlation_screen_track(chart)
        df = self._query_df(f"""
        SELECT feature_a, feature_b, interaction_p_fdr
        FROM `{self.project_id}.{self.analysis_dataset}.cxg_bivariate_interaction_v1`
        WHERE track = '{track}' AND tier = 1 AND interaction_p_fdr IS NOT NULL
        """)
        plt_fig, ax = plt.subplots(figsize=(12.8, 10.0), dpi=150)
        if df.empty:
            ax.text(0.5, 0.5, "No valid interaction pairs", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
        else:
            df = df.rename(columns={"interaction_p_fdr": "r_train"})
            matrix = self._symmetric_corr_matrix(df).clip(0, 1)
            neg_log10 = matrix.apply(lambda col: -np.log10(col.clip(lower=1e-300)))
            im = ax.imshow(neg_log10, cmap="viridis", aspect="auto")
            ax.set_xticks(range(len(matrix.columns)))
            ax.set_xticklabels([str(v)[:22] for v in matrix.columns], rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(len(matrix.index)))
            ax.set_yticklabels([str(v)[:28] for v in matrix.index], fontsize=7)
            plt_fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="-log10(FDR p)")
        ax.set_title(f"{track}: Tier 1 Interaction Significance Grid", fontsize=14, pad=14)
        plt_fig.tight_layout()
        plt_fig.savefig(path, format="png")
        plt.close(plt_fig)

    def _write_archetype_role_heatmap_png(self, chart: ChartRow, path: Path) -> None:
        df = self._query_df(f"""
        SELECT COALESCE(nearest_defender_style_archetype, 'null') AS archetype,
               COALESCE(nearest_defender_role, 'null') AS role, COUNT(*) AS n
        FROM `{self.project_id}.oam_features.cxg_defensive_360_features`
        GROUP BY archetype, role
        """)
        plt_fig, ax = plt.subplots(figsize=(10.0, 7.0), dpi=150)
        if df.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
        else:
            pivot = df.pivot(index="archetype", columns="role", values="n").fillna(0)
            pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
            im = ax.imshow(pct, cmap="Blues", aspect="auto")
            ax.set_xticks(range(len(pct.columns)))
            ax.set_xticklabels(pct.columns, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(len(pct.index)))
            ax.set_yticklabels(pct.index, fontsize=8)
            for i in range(pct.shape[0]):
                for j in range(pct.shape[1]):
                    ax.text(j, i, f"{pct.iat[i, j]:.0f}", ha="center", va="center", fontsize=7, color="black")
            plt_fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="% of archetype's shots")
        ax.set_title("CxG+: Defender Style Archetype x Nearest Defender Role (row %)", fontsize=13, pad=14)
        plt_fig.tight_layout()
        plt_fig.savefig(path, format="png")
        plt.close(plt_fig)

    def _write_summary_png(self, chart: ChartRow, path: Path) -> None:
        df = self._query_df(f"""
        SELECT column_name, min_value, p25_value, median_value, p75_value, max_value, distinct_count
        FROM `{self.project_id}.{self.analysis_dataset}.cxg_summary_stats_v1`
        WHERE feature_family = '{chart.feature_family}'
          AND median_value IS NOT NULL
          AND max_value IS NOT NULL
          AND min_value IS NOT NULL
        ORDER BY distinct_count DESC, column_name
        LIMIT 24
        """)
        plt_fig, ax = plt.subplots(figsize=(12.8, 8.8), dpi=150)
        if df.empty:
            ax.text(0.5, 0.5, "No numeric summary data", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
        else:
            span = (df["max_value"] - df["min_value"]).replace(0, np.nan)
            y = np.arange(len(df))
            p25 = ((df["p25_value"] - df["min_value"]) / span).fillna(0.5)
            med = ((df["median_value"] - df["min_value"]) / span).fillna(0.5)
            p75 = ((df["p75_value"] - df["min_value"]) / span).fillna(0.5)
            ax.hlines(y, 0, 1, color="#d1d5db", linewidth=1.5)
            ax.scatter(p25, y, color="#60a5fa", label="p25", s=28)
            ax.scatter(med, y, color="#111827", label="median", marker="D", s=30)
            ax.scatter(p75, y, color="#34d399", label="p75", s=28)
            ax.set_yticks(y)
            ax.set_yticklabels([str(v)[:34] for v in df["column_name"]], fontsize=8)
            ax.invert_yaxis()
            ax.set_xlim(-0.03, 1.03)
            ax.set_xlabel("Within-feature normalized range: min=0, max=1")
            ax.grid(axis="x", alpha=0.18)
            ax.legend(loc="lower right", fontsize=8)
        ax.set_title(f"{chart.feature_family}: Normalized Summary Ranges", fontsize=14, pad=14)
        plt_fig.tight_layout()
        plt_fig.savefig(path, format="png")
        plt.close(plt_fig)

    def _write_distribution_png(self, chart: ChartRow, path: Path) -> None:
        features = self._query_df(f"""
        SELECT column_name, data_type, distinct_count
        FROM `{self.project_id}.{self.analysis_dataset}.cxg_summary_stats_v1`
        WHERE feature_family = '{chart.feature_family}' AND non_null_count > 0
        ORDER BY
          CASE WHEN data_type IN ('FLOAT64', 'INT64', 'NUMERIC', 'BIGNUMERIC') AND distinct_count > 2 THEN 0 ELSE 1 END,
          distinct_count DESC,
          column_name
        LIMIT 6
        """)
        numeric = [
            row
            for _, row in features.iterrows()
            if row["data_type"] != "STRING"
            and row["distinct_count"] > 2
            and row["column_name"] not in CATEGORICAL_OVERRIDE_COLUMNS
        ]
        if numeric:
            cols = [row["column_name"] for row in numeric[:6]]
            surface = self._surface_for_family(chart.feature_family)
            cols_per_row = 2
            rows = math.ceil(len(cols) / cols_per_row)
            plt_fig, axes = plt.subplots(rows, cols_per_row, figsize=(12.8, max(4.2, rows * 3.2)), dpi=150)
            axes_arr = np.atleast_1d(axes).flatten()
            for ax, col in zip(axes_arr, cols, strict=False):
                df = self._query_df(f"""
                SELECT CAST(`{col}` AS FLOAT64) AS value
                FROM `{self.project_id}.{self.analysis_dataset}.{surface}`
                WHERE `{col}` IS NOT NULL
                """)
                ax.hist(df["value"], bins=30, color="#2563eb", alpha=0.82)
                ax.set_title(str(col)[:42], fontsize=10)
                ax.tick_params(axis="x", labelrotation=25, labelsize=7)
                ax.tick_params(axis="y", labelsize=7)
                ax.grid(axis="y", alpha=0.16)
            for ax in axes_arr[len(cols):]:
                ax.set_axis_off()
            plt_fig.suptitle(f"{chart.feature_family}: Value Distributions", fontsize=14)
        else:
            df = self._query_df(f"""
            SELECT column_name, bin_label, row_count
            FROM `{self.project_id}.{self.analysis_dataset}.cxg_eda_distribution_bins_v1`
            WHERE feature_family = '{chart.feature_family}'
            QUALIFY ROW_NUMBER() OVER (PARTITION BY column_name ORDER BY row_count DESC) <= 8
            """)
            plt_fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=150)
            if df.empty:
                ax.text(0.5, 0.5, "No distribution data", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
            else:
                for name, group in df.groupby("column_name"):
                    ax.bar(group["bin_label"], group["row_count"], alpha=0.75, label=str(name)[:28])
                ax.tick_params(axis="x", labelrotation=35, labelsize=8)
                ax.legend(title="Feature", fontsize=8)
            ax.set_title(f"{chart.feature_family}: Category Distributions", fontsize=14, pad=14)
        plt_fig.tight_layout()
        plt_fig.savefig(path, format="png")
        plt.close(plt_fig)

    def _artifact_uri(self, run_id: str, file_name: str) -> str:
        return f"gs://{self.artifact_bucket}/{self.artifact_prefix}/{run_id}/rendered_charts/{file_name}"

    def _upload(self, run_id: str, path: Path) -> None:
        blob_name = f"{self.artifact_prefix}/{run_id}/rendered_charts/{path.name}"
        content_type = "text/html" if path.suffix == ".html" else "image/png" if path.suffix == ".png" else "application/json"
        self.gcs.bucket(self.artifact_bucket).blob(blob_name).upload_from_filename(path, content_type=content_type)

    def _materialize_render_registry(self, manifest: dict[str, object]) -> None:
        """Additive per run_id: delete this run_id's rows (if any, e.g. a re-render),
        then insert its current rows. Never touches any other run_id's rows -- the
        previous `CREATE OR REPLACE TABLE ... AS SELECT <this run only>` implementation
        silently destroyed every other run's chart-registry history on every upload."""
        table_ref = f"{self.project_id}.{self.analysis_dataset}.cxg_rendered_chart_registry_v1"
        self.bq.create_table(bigquery.Table(table_ref, schema=RENDERED_CHART_REGISTRY_SCHEMA), exists_ok=True)

        run_id = manifest["run_id"]
        self.bq.query(
            f"DELETE FROM `{table_ref}` WHERE run_id = @run_id",
            job_config=bigquery.QueryJobConfig(
                query_parameters=[bigquery.ScalarQueryParameter("run_id", "STRING", run_id)]
            ),
            location=self.location,
        ).result()

        rows = [
            {
                "run_id": run_id,
                "chart_name": chart["chart_name"],
                "html_uri": chart["html_uri"],
                "png_uri": chart["png_uri"],
                "rendered_at": datetime.now(UTC).isoformat(),
            }
            for chart in manifest["charts"]
        ]
        if not rows:
            return
        job_config = bigquery.LoadJobConfig(schema=RENDERED_CHART_REGISTRY_SCHEMA, write_disposition="WRITE_APPEND")
        self.bq.load_table_from_json(rows, table_ref, job_config=job_config, location=self.location).result()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render CxG analysis charts as HTML and PNG.")
    parser.add_argument("--project-id", default=PROJECT_ID)
    parser.add_argument("--location", default=LOCATION)
    parser.add_argument("--analysis-dataset", default=ANALYSIS_DATASET)
    parser.add_argument("--artifact-bucket", default=DEFAULT_ARTIFACT_BUCKET)
    parser.add_argument("--artifact-prefix", default=DEFAULT_ARTIFACT_PREFIX)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--local-output-dir", default="audit_outputs/cxg_analysis")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-upload", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    renderer = CxGChartRenderer(
        project_id=args.project_id,
        location=args.location,
        analysis_dataset=args.analysis_dataset,
        artifact_bucket=args.artifact_bucket,
        artifact_prefix=args.artifact_prefix,
        run_id=args.run_id,
        local_output_dir=Path(args.local_output_dir) if args.local_output_dir else None,
        limit=args.limit,
        skip_upload=args.skip_upload,
    )
    print(json.dumps(renderer.run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
