"""Render curated CxG findings charts from findings-grade BigQuery tables."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from google.cloud import bigquery, storage

PROJECT_ID = "oam-varun-260819"
LOCATION = "europe-west2"
ANALYSIS_DATASET = "oam_analysis"
ARTIFACT_BUCKET = "oam-varun-260819-artifacts"
ARTIFACT_PREFIX = "analysis/cxg"

CHARTS = [
    ("feature_decision_counts", "Feature decision mix", "decision_manifest"),
    ("candidate_signal_leaderboard", "Strongest candidate signals", "decision_manifest"),
    ("family_signal_summary", "Family signal summary", "decision_manifest"),
    ("top_target_response_bins", "Top one-feature target response bins", "binned_target_response"),
    ("top_pair_interactions", "Top two-feature interactions", "pair_interaction_target"),
    ("cross_family_correlation_edges", "Strong cross-family correlations", "cross_family_correlation"),
    ("redundancy_pairs", "Redundancy pairs to review", "redundancy_pairs"),
]


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value)[:180]


class FindingsChartRenderer:
    def __init__(self, run_id: str, local_output_dir: Path, skip_upload: bool = False) -> None:
        self.run_id = run_id
        self.skip_upload = skip_upload
        self.out_dir = local_output_dir / run_id / "findings_charts"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.bq = bigquery.Client(project=PROJECT_ID)
        self.gcs = storage.Client(project=PROJECT_ID)

    def query_df(self, sql: str) -> pd.DataFrame:
        rows = list(self.bq.query(sql, location=LOCATION).result())
        return pd.DataFrame([dict(row.items()) for row in rows]) if rows else pd.DataFrame()

    def render(self) -> dict[str, object]:
        rendered = []
        for chart_name, title, backing_table in CHARTS:
            df = getattr(self, f"data_{chart_name}")()
            fig = getattr(self, f"plotly_{chart_name}")(df, title)
            html_path = self.out_dir / f"{chart_name}.html"
            png_path = self.out_dir / f"{chart_name}.png"
            fig.write_html(html_path, include_plotlyjs="cdn", full_html=True)
            getattr(self, f"png_{chart_name}")(df, title, png_path)
            if not self.skip_upload:
                self.upload(html_path)
                self.upload(png_path)
            rendered.append({
                "chart_name": chart_name,
                "chart_title": title,
                "backing_table": backing_table,
                "html_uri": self.uri(html_path.name),
                "png_uri": self.uri(png_path.name),
            })
        manifest = {
            "run_id": self.run_id,
            "rendered_at": datetime.now(UTC).isoformat(),
            "chart_count": len(rendered),
            "html_count": len(rendered),
            "png_count": len(rendered),
            "artifact_prefix": f"gs://{ARTIFACT_BUCKET}/{ARTIFACT_PREFIX}/{self.run_id}/findings_charts/",
            "local_prefix": str(self.out_dir),
            "upload_skipped": self.skip_upload,
            "charts": rendered,
        }
        manifest_path = self.out_dir / "findings_charts_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        if not self.skip_upload:
            self.upload(manifest_path)
            self.materialize_registry(rendered)
        return manifest

    def data_feature_decision_counts(self) -> pd.DataFrame:
        return self.query_df(f"""
        SELECT feature_family, preliminary_decision, COUNT(*) AS feature_count
        FROM `{PROJECT_ID}.{ANALYSIS_DATASET}.cxg_feature_decision_manifest_v1`
        GROUP BY feature_family, preliminary_decision
        ORDER BY feature_family, preliminary_decision
        """)

    def data_candidate_signal_leaderboard(self) -> pd.DataFrame:
        return self.query_df(f"""
        SELECT feature_family, column_name, preliminary_decision, non_null_count,
               null_pct, point_biserial_corr, abs_signal AS abs_corr
        FROM `{PROJECT_ID}.{ANALYSIS_DATASET}.cxg_feature_decision_manifest_v1`
        WHERE point_biserial_corr IS NOT NULL
        ORDER BY abs_signal DESC, non_null_count DESC
        LIMIT 30
        """)

    def data_family_signal_summary(self) -> pd.DataFrame:
        return self.query_df(f"""
        SELECT feature_family,
               COUNT(*) AS feature_count,
               AVG(null_pct) AS avg_null_pct,
               AVG(abs_signal) AS avg_abs_point_biserial_corr,
               MAX(abs_signal) AS max_abs_point_biserial_corr,
               COUNTIF(preliminary_decision = 'candidate_keep') AS keep_count,
               COUNTIF(preliminary_decision = 'candidate_review') AS review_count,
               COUNTIF(preliminary_decision = 'watchlist_sparse') AS sparse_count,
               COUNTIF(preliminary_decision = 'weak_signal_review') AS weak_count,
               COUNTIF(preliminary_decision = 'redundant_review') AS redundant_count
        FROM `{PROJECT_ID}.{ANALYSIS_DATASET}.cxg_feature_decision_manifest_v1`
        GROUP BY feature_family
        ORDER BY max_abs_point_biserial_corr DESC
        """)

    def data_top_target_response_bins(self) -> pd.DataFrame:
        return self.query_df(f"""
        SELECT feature_family, column_name, bin_label, row_count, goal_count, goal_rate, lift_vs_baseline
        FROM `{PROJECT_ID}.{ANALYSIS_DATASET}.cxg_binned_target_response_v1`
        WHERE row_count >= 50 AND lift_vs_baseline IS NOT NULL
        ORDER BY lift_vs_baseline DESC, row_count DESC
        LIMIT 30
        """)

    def data_top_pair_interactions(self) -> pd.DataFrame:
        return self.query_df(f"""
        SELECT feature_family, left_column, right_column, left_bin, right_bin, row_count, goal_count,
               goal_rate, lift_vs_baseline
        FROM `{PROJECT_ID}.{ANALYSIS_DATASET}.cxg_pair_interaction_target_v1`
        WHERE row_count >= 100 AND lift_vs_baseline IS NOT NULL
        ORDER BY lift_vs_baseline DESC, row_count DESC
        LIMIT 30
        """)

    def data_cross_family_correlation_edges(self) -> pd.DataFrame:
        return self.query_df(f"""
        SELECT left_family, left_column, right_family, right_column, pearson_corr, pair_non_null_count
        FROM `{PROJECT_ID}.{ANALYSIS_DATASET}.cxg_cross_family_correlation_v1`
        WHERE pearson_corr IS NOT NULL AND pair_non_null_count >= 500
        ORDER BY ABS(pearson_corr) DESC, pair_non_null_count DESC
        LIMIT 30
        """)

    def data_redundancy_pairs(self) -> pd.DataFrame:
        return self.query_df(f"""
        SELECT left_family, right_family, left_column, right_column, pearson_corr, pair_non_null_count, redundancy_band
        FROM `{PROJECT_ID}.{ANALYSIS_DATASET}.cxg_redundancy_pairs_v1`
        ORDER BY ABS(pearson_corr) DESC, pair_non_null_count DESC
        LIMIT 30
        """)

    def layout(self, fig: go.Figure, title: str) -> go.Figure:
        fig.update_layout(
            title=title,
            template="plotly_white",
            font={"family": "Arial", "size": 13},
            margin={"l": 64, "r": 28, "t": 72, "b": 72},
            legend_title_text="Group",
        )
        return fig

    def plotly_feature_decision_counts(self, df: pd.DataFrame, title: str) -> go.Figure:
        fig = px.bar(df, x="feature_family", y="feature_count", color="preliminary_decision", barmode="stack")
        return self.layout(fig, title)

    def plotly_candidate_signal_leaderboard(self, df: pd.DataFrame, title: str) -> go.Figure:
        df = df.sort_values("abs_corr")
        fig = px.bar(df, x="point_biserial_corr", y="column_name", color="feature_family", orientation="h", hover_data=["preliminary_decision", "non_null_count", "null_pct"])
        return self.layout(fig, title)

    def plotly_family_signal_summary(self, df: pd.DataFrame, title: str) -> go.Figure:
        fig = px.scatter(df, x="avg_null_pct", y="max_abs_point_biserial_corr", size="feature_count", color="feature_family", hover_data=["keep_count", "review_count", "sparse_count", "weak_count", "redundant_count"])
        fig.update_xaxes(title="Average null %")
        fig.update_yaxes(title="Max absolute target correlation")
        return self.layout(fig, title)

    def plotly_top_target_response_bins(self, df: pd.DataFrame, title: str) -> go.Figure:
        df = df.assign(label=df["feature_family"] + ": " + df["column_name"] + " | " + df["bin_label"].astype(str)).sort_values("lift_vs_baseline")
        fig = px.bar(df, x="lift_vs_baseline", y="label", color="feature_family", orientation="h", hover_data=["row_count", "goal_count", "goal_rate"])
        return self.layout(fig, title)

    def plotly_top_pair_interactions(self, df: pd.DataFrame, title: str) -> go.Figure:
        df = df.assign(label=df["left_column"] + " + " + df["right_column"] + " | " + df["left_bin"].astype(str) + " / " + df["right_bin"].astype(str)).sort_values("lift_vs_baseline")
        fig = px.bar(df, x="lift_vs_baseline", y="label", color="feature_family", orientation="h", hover_data=["row_count", "goal_count", "goal_rate"])
        return self.layout(fig, title)

    def plotly_cross_family_correlation_edges(self, df: pd.DataFrame, title: str) -> go.Figure:
        df = df.assign(label=df["left_family"] + ":" + df["left_column"] + " <> " + df["right_family"] + ":" + df["right_column"]).sort_values("pearson_corr")
        fig = px.bar(df, x="pearson_corr", y="label", color="right_family", orientation="h", hover_data=["pair_non_null_count"])
        return self.layout(fig, title)

    def plotly_redundancy_pairs(self, df: pd.DataFrame, title: str) -> go.Figure:
        df = df.assign(label=df["left_column"] + " <> " + df["right_column"]).sort_values("pearson_corr")
        fig = px.bar(df, x="pearson_corr", y="label", color="redundancy_band", orientation="h", hover_data=["left_family", "right_family", "pair_non_null_count"])
        return self.layout(fig, title)

    def png_feature_decision_counts(self, df: pd.DataFrame, title: str, path: Path) -> None:
        pivot = df.pivot_table(index="feature_family", columns="preliminary_decision", values="feature_count", aggfunc="sum", fill_value=0)
        self.write_stacked_bar(pivot, title, path, "Features")

    def png_candidate_signal_leaderboard(self, df: pd.DataFrame, title: str, path: Path) -> None:
        df = df.sort_values("abs_corr", ascending=True).tail(24)
        self.write_hbar(df, "point_biserial_corr", "column_name", "feature_family", title, path, "Target correlation")

    def png_family_signal_summary(self, df: pd.DataFrame, title: str, path: Path) -> None:
        fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=150)
        if df.empty:
            self.no_data(ax)
        else:
            sizes = (df["feature_count"].astype(float).clip(lower=1) * 18).to_numpy()
            scatter = ax.scatter(df["avg_null_pct"], df["max_abs_point_biserial_corr"], s=sizes, alpha=0.72, c=range(len(df)), cmap="tab10")
            for _, row in df.iterrows():
                ax.annotate(row["feature_family"], (row["avg_null_pct"], row["max_abs_point_biserial_corr"]), fontsize=8, xytext=(5, 4), textcoords="offset points")
            ax.set_xlabel("Average null %")
            ax.set_ylabel("Max absolute target correlation")
            ax.grid(True, alpha=0.18)
        ax.set_title(title, fontsize=14, pad=14)
        fig.tight_layout()
        fig.savefig(path, format="png")
        plt.close(fig)

    def png_top_target_response_bins(self, df: pd.DataFrame, title: str, path: Path) -> None:
        df = df.assign(label=df["column_name"] + " | " + df["bin_label"].astype(str)).sort_values("lift_vs_baseline", ascending=True).tail(24)
        self.write_hbar(df, "lift_vs_baseline", "label", "feature_family", title, path, "Lift vs family target")

    def png_top_pair_interactions(self, df: pd.DataFrame, title: str, path: Path) -> None:
        df = df.assign(label=df["left_column"] + " + " + df["right_column"]).sort_values("lift_vs_baseline", ascending=True).tail(24)
        self.write_hbar(df, "lift_vs_baseline", "label", "feature_family", title, path, "Lift vs family target")

    def png_cross_family_correlation_edges(self, df: pd.DataFrame, title: str, path: Path) -> None:
        df = df.assign(label=df["left_column"] + " <> " + df["right_column"]).sort_values("pearson_corr", ascending=True).tail(24)
        self.write_hbar(df, "pearson_corr", "label", "right_family", title, path, "Pearson r")

    def png_redundancy_pairs(self, df: pd.DataFrame, title: str, path: Path) -> None:
        df = df.assign(label=df["left_column"] + " <> " + df["right_column"]).sort_values("pearson_corr", ascending=True).tail(24)
        self.write_hbar(df, "pearson_corr", "label", "redundancy_band", title, path, "Pearson r")

    def write_stacked_bar(self, pivot: pd.DataFrame, title: str, path: Path, ylabel: str) -> None:
        fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=150)
        if pivot.empty:
            self.no_data(ax)
        else:
            pivot.plot(kind="bar", stacked=True, ax=ax, width=0.72)
            ax.set_xlabel("Feature family")
            ax.set_ylabel(ylabel)
            ax.tick_params(axis="x", labelrotation=25, labelsize=8)
            ax.legend(title="Decision", fontsize=8)
            ax.grid(axis="y", alpha=0.18)
        ax.set_title(title, fontsize=14, pad=14)
        fig.tight_layout()
        fig.savefig(path, format="png")
        plt.close(fig)

    def write_hbar(self, df: pd.DataFrame, x_col: str, y_col: str, color_col: str, title: str, path: Path, xlabel: str) -> None:
        fig, ax = plt.subplots(figsize=(13.6, 9.2), dpi=150)
        if df.empty:
            self.no_data(ax)
        else:
            groups = list(dict.fromkeys(df[color_col].astype(str)))
            cmap = plt.get_cmap("tab10")
            colors = {group: cmap(i % 10) for i, group in enumerate(groups)}
            y = range(len(df))
            ax.barh(list(y), df[x_col], color=[colors[str(v)] for v in df[color_col]], alpha=0.86)
            labels = [str(v)[:56] + ("..." if len(str(v)) > 56 else "") for v in df[y_col]]
            ax.set_yticks(list(y))
            ax.set_yticklabels(labels, fontsize=7)
            ax.set_xlabel(xlabel)
            ax.grid(axis="x", alpha=0.18)
            handles = [plt.Rectangle((0, 0), 1, 1, color=colors[group]) for group in groups[:10]]
            ax.legend(handles, groups[:10], title=color_col, fontsize=8, loc="best")
        ax.set_title(title, fontsize=14, pad=14)
        fig.tight_layout()
        fig.savefig(path, format="png")
        plt.close(fig)

    def no_data(self, ax) -> None:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()

    def uri(self, name: str) -> str:
        return f"gs://{ARTIFACT_BUCKET}/{ARTIFACT_PREFIX}/{self.run_id}/findings_charts/{name}"

    def upload(self, path: Path) -> None:
        content_type = "text/html" if path.suffix == ".html" else "image/png" if path.suffix == ".png" else "application/json"
        blob_name = f"{ARTIFACT_PREFIX}/{self.run_id}/findings_charts/{path.name}"
        self.gcs.bucket(ARTIFACT_BUCKET).blob(blob_name).upload_from_filename(path, content_type=content_type)

    def materialize_registry(self, rendered: list[dict[str, object]]) -> None:
        rows = []
        for chart in rendered:
            rows.append(
                "SELECT "
                f"'{self.run_id}' AS run_id, "
                f"'{chart['chart_name']}' AS chart_name, "
                f"'{chart['chart_title']}' AS chart_title, "
                f"'{chart['backing_table']}' AS backing_table, "
                f"'{chart['html_uri']}' AS html_uri, "
                f"'{chart['png_uri']}' AS png_uri, "
                "CURRENT_TIMESTAMP() AS rendered_at"
            )
        sql = f"CREATE OR REPLACE TABLE `{PROJECT_ID}.{ANALYSIS_DATASET}.cxg_findings_chart_registry_v1` AS\n" + "\nUNION ALL\n".join(rows)
        self.bq.query(sql, location=LOCATION).result()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--local-output-dir", default="audit_outputs/cxg_analysis")
    parser.add_argument("--skip-upload", action="store_true")
    args = parser.parse_args()
    renderer = FindingsChartRenderer(args.run_id, Path(args.local_output_dir), skip_upload=args.skip_upload)
    print(json.dumps(renderer.render(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()



