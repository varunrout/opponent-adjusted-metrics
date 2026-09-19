"""Render split-aware CxG/CxG+ analysis charts from BigQuery tables."""

from __future__ import annotations

import argparse
import json
import shutil
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
SPLIT_CHART_DIR = "split_analysis_charts"
REGISTRY_TABLE = "cxg_split_chart_registry_v1"


def qtable(name: str) -> str:
    return f"`{PROJECT_ID}.{ANALYSIS_DATASET}.{name}`"


def sql_quote(value: object) -> str:
    return str(value).replace("\\", "\\\\").replace("'", "\\'")


class SplitChartRenderer:
    def __init__(self, run_id: str, local_output_dir: Path, skip_upload: bool) -> None:
        self.run_id = run_id
        self.skip_upload = skip_upload
        self.out_dir = local_output_dir / run_id / SPLIT_CHART_DIR
        self.bq = bigquery.Client(project=PROJECT_ID)
        self.gcs = storage.Client(project=PROJECT_ID)

    def query_df(self, sql: str) -> pd.DataFrame:
        rows = list(self.bq.query(sql, location=LOCATION).result())
        return pd.DataFrame([dict(row.items()) for row in rows]) if rows else pd.DataFrame()

    def cleanup(self) -> dict[str, int]:
        local_removed = 0
        if self.out_dir.exists():
            local_removed = sum(1 for p in self.out_dir.rglob("*") if p.is_file())
            shutil.rmtree(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        gcs_removed = 0
        if not self.skip_upload:
            bucket = self.gcs.bucket(ARTIFACT_BUCKET)
            prefix = f"{ARTIFACT_PREFIX}/{self.run_id}/{SPLIT_CHART_DIR}/"
            blobs = list(bucket.list_blobs(prefix=prefix))
            gcs_removed = len(blobs)
            for blob in blobs:
                blob.delete()
        return {"local_files_removed": local_removed, "gcs_objects_removed": gcs_removed}

    def render(self) -> dict[str, object]:
        cleanup = self.cleanup()
        charts = self.chart_specs()
        rendered: list[dict[str, object]] = []
        for spec in charts:
            df = spec["data"]()
            fig = spec["plotly"](df, spec["title"])
            html_path = self.out_dir / f"{spec['name']}.html"
            png_path = self.out_dir / f"{spec['name']}.png"
            fig.write_html(html_path, include_plotlyjs="cdn", full_html=True)
            spec["png"](df, spec["title"], png_path)
            if not self.skip_upload:
                self.upload(html_path)
                self.upload(png_path)
            rendered.append({
                "run_id": self.run_id,
                "chart_name": spec["name"],
                "chart_title": spec["title"],
                "chart_group": spec["group"],
                "backing_table": spec["table"],
                "html_uri": self.uri(html_path.name),
                "png_uri": self.uri(png_path.name),
            })

        manifest = {
            "run_id": self.run_id,
            "rendered_at": datetime.now(UTC).isoformat(),
            "chart_count": len(rendered),
            "html_count": len(list(self.out_dir.glob("*.html"))),
            "png_count": len(list(self.out_dir.glob("*.png"))),
            "artifact_prefix": f"gs://{ARTIFACT_BUCKET}/{ARTIFACT_PREFIX}/{self.run_id}/{SPLIT_CHART_DIR}/",
            "local_prefix": str(self.out_dir),
            "upload_skipped": self.skip_upload,
            "cleanup": cleanup,
            "charts": rendered,
        }
        manifest_path = self.out_dir / "split_analysis_charts_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        if not self.skip_upload:
            self.upload(manifest_path)
            self.materialize_registry(rendered)

        validation = self.validate(expected_charts=len(rendered), expected_gcs=(len(rendered) * 2) + 1)
        checkpoint = self.write_checkpoint(manifest, validation)
        return {"manifest": manifest, "validation": validation, "checkpoint": str(checkpoint)}

    def chart_specs(self) -> list[dict[str, object]]:
        return [
            self.spec("univariate_event_validation_signal", "univariate", "cxg_split_univariate_v1", "CxG event validation univariate signal", self.data_uni_event_validation, self.plotly_uni_signal, self.png_uni_signal),
            self.spec("univariate_plus_validation_signal", "univariate", "cxg_split_univariate_v1", "CxG+ validation univariate signal", self.data_uni_plus_validation, self.plotly_uni_signal, self.png_uni_signal),
            self.spec("univariate_event_train_validation_stability", "univariate", "cxg_split_univariate_v1", "CxG event train-validation univariate stability", self.data_uni_event_stability, self.plotly_uni_stability, self.png_uni_stability),
            self.spec("univariate_plus_train_validation_stability", "univariate", "cxg_split_univariate_v1", "CxG+ train-validation univariate stability", self.data_uni_plus_stability, self.plotly_uni_stability, self.png_uni_stability),
            self.spec("bivariate_event_validation_lift", "bivariate", "cxg_split_bivariate_v1", "CxG event validation bivariate lift", self.data_bi_event_validation, self.plotly_bi_lift, self.png_bi_lift),
            self.spec("bivariate_plus_validation_lift", "bivariate", "cxg_split_bivariate_v1", "CxG+ validation bivariate lift", self.data_bi_plus_validation, self.plotly_bi_lift, self.png_bi_lift),
            self.spec("bivariate_event_train_validation_stability", "bivariate", "cxg_split_bivariate_v1", "CxG event train-validation bivariate stability", self.data_bi_event_stability, self.plotly_bi_stability, self.png_bi_stability),
            self.spec("bivariate_plus_train_validation_stability", "bivariate", "cxg_split_bivariate_v1", "CxG+ train-validation bivariate stability", self.data_bi_plus_stability, self.plotly_bi_stability, self.png_bi_stability),
            self.spec("multivariate_validation_auc", "multivariate", "cxg_split_multivariate_results_v1", "Validation AUC by split-aware model", self.data_multi_validation_auc, self.plotly_multi_metric, self.png_multi_metric),
            self.spec("multivariate_test_auc_audit", "multivariate", "cxg_split_multivariate_results_v1", "Test AUC audit by split-aware model", self.data_multi_test_auc, self.plotly_multi_metric, self.png_multi_metric),
            self.spec("multivariate_validation_log_loss", "multivariate", "cxg_split_multivariate_results_v1", "Validation log loss by split-aware model", self.data_multi_validation_log_loss, self.plotly_multi_metric, self.png_multi_metric),
            self.spec("multivariate_auc_by_split", "multivariate", "cxg_split_multivariate_results_v1", "AUC movement across train validation test", self.data_multi_auc_by_split, self.plotly_multi_split_line, self.png_multi_split_line),
            self.spec("multivariate_cxg_event_calibration", "multivariate", "cxg_split_model_predictions_v1", "CxG event validation prediction calibration", self.data_cal_event, self.plotly_calibration, self.png_calibration),
            self.spec("multivariate_cxg_plus_calibration", "multivariate", "cxg_split_model_predictions_v1", "CxG+ validation prediction calibration", self.data_cal_plus, self.plotly_calibration, self.png_calibration),
        ]

    def spec(self, name, group, table, title, data, plotly, png) -> dict[str, object]:
        return {"name": name, "group": group, "table": table, "title": title, "data": data, "plotly": plotly, "png": png}

    def data_uni_event_validation(self) -> pd.DataFrame:
        return self.query_df(f"""
        SELECT feature_family, column_name, point_biserial_corr, abs_signal, non_null_count, null_pct, goal_rate
        FROM {qtable('cxg_split_univariate_v1')}
        WHERE run_id = '{sql_quote(self.run_id)}' AND split = 'validation' AND track = 'cxg_event'
        ORDER BY abs_signal DESC, non_null_count DESC
        LIMIT 30
        """)

    def data_uni_plus_validation(self) -> pd.DataFrame:
        return self.query_df(f"""
        SELECT feature_family, column_name, point_biserial_corr, abs_signal, non_null_count, null_pct, goal_rate
        FROM {qtable('cxg_split_univariate_v1')}
        WHERE run_id = '{sql_quote(self.run_id)}' AND split = 'validation' AND track = 'cxg_plus'
        ORDER BY abs_signal DESC, non_null_count DESC
        LIMIT 30
        """)

    def data_uni_event_stability(self) -> pd.DataFrame:
        return self.univariate_stability("cxg_event")

    def data_uni_plus_stability(self) -> pd.DataFrame:
        return self.univariate_stability("cxg_plus")

    def univariate_stability(self, track: str) -> pd.DataFrame:
        return self.query_df(f"""
        SELECT
          train.feature_family,
          train.column_name,
          train.point_biserial_corr AS train_corr,
          val.point_biserial_corr AS validation_corr,
          train.non_null_count AS train_non_null_count,
          val.non_null_count AS validation_non_null_count
        FROM {qtable('cxg_split_univariate_v1')} train
        JOIN {qtable('cxg_split_univariate_v1')} val
          USING (run_id, track, feature_family, column_name)
        WHERE train.run_id = '{sql_quote(self.run_id)}'
          AND train.track = '{track}'
          AND train.split = 'train'
          AND val.split = 'validation'
          AND train.point_biserial_corr IS NOT NULL
          AND val.point_biserial_corr IS NOT NULL
        ORDER BY ABS(val.point_biserial_corr) DESC
        LIMIT 80
        """)

    def data_bi_event_validation(self) -> pd.DataFrame:
        return self.bivariate_validation("cxg_event")

    def data_bi_plus_validation(self) -> pd.DataFrame:
        return self.bivariate_validation("cxg_plus")

    def bivariate_validation(self, track: str) -> pd.DataFrame:
        return self.query_df(f"""
        SELECT feature_family, left_column, right_column, left_bin, right_bin, row_count, goal_count, goal_rate, lift_vs_split_baseline
        FROM {qtable('cxg_split_bivariate_v1')}
        WHERE run_id = '{sql_quote(self.run_id)}' AND split = 'validation' AND track = '{track}'
          AND lift_vs_split_baseline IS NOT NULL
        ORDER BY lift_vs_split_baseline DESC, row_count DESC
        LIMIT 30
        """)

    def data_bi_event_stability(self) -> pd.DataFrame:
        return self.bivariate_stability("cxg_event")

    def data_bi_plus_stability(self) -> pd.DataFrame:
        return self.bivariate_stability("cxg_plus")

    def bivariate_stability(self, track: str) -> pd.DataFrame:
        return self.query_df(f"""
        SELECT
          train.feature_family,
          train.left_column,
          train.right_column,
          train.left_bin,
          train.right_bin,
          train.lift_vs_split_baseline AS train_lift,
          val.lift_vs_split_baseline AS validation_lift,
          train.row_count AS train_row_count,
          val.row_count AS validation_row_count
        FROM {qtable('cxg_split_bivariate_v1')} train
        JOIN {qtable('cxg_split_bivariate_v1')} val
          USING (run_id, track, feature_family, left_column, right_column, left_bin, right_bin)
        WHERE train.run_id = '{sql_quote(self.run_id)}'
          AND train.track = '{track}'
          AND train.split = 'train'
          AND val.split = 'validation'
          AND train.lift_vs_split_baseline IS NOT NULL
          AND val.lift_vs_split_baseline IS NOT NULL
        ORDER BY val.lift_vs_split_baseline DESC, val.row_count DESC
        LIMIT 80
        """)

    def data_multi_validation_auc(self) -> pd.DataFrame:
        return self.multivariate_metric("validation", "auc", ascending=False)

    def data_multi_test_auc(self) -> pd.DataFrame:
        return self.multivariate_metric("test", "auc", ascending=False)

    def data_multi_validation_log_loss(self) -> pd.DataFrame:
        return self.multivariate_metric("validation", "log_loss", ascending=True)

    def multivariate_metric(self, split: str, metric: str, ascending: bool) -> pd.DataFrame:
        order = "ASC" if ascending else "DESC"
        return self.query_df(f"""
        SELECT model_name, model_stage, track, split, feature_count, row_count, goal_count, {metric} AS metric_value
        FROM {qtable('cxg_split_multivariate_results_v1')}
        WHERE run_id = '{sql_quote(self.run_id)}' AND split = '{split}' AND {metric} IS NOT NULL
        ORDER BY {metric} {order}
        LIMIT 30
        """)

    def data_multi_auc_by_split(self) -> pd.DataFrame:
        return self.query_df(f"""
        SELECT model_name, model_stage, track, split, feature_count, auc
        FROM {qtable('cxg_split_multivariate_results_v1')}
        WHERE run_id = '{sql_quote(self.run_id)}' AND auc IS NOT NULL
          AND model_stage IN ('true_baseline','candidate_model')
        ORDER BY track, model_name, CASE split WHEN 'train' THEN 1 WHEN 'validation' THEN 2 ELSE 3 END
        """)

    def data_cal_event(self) -> pd.DataFrame:
        return self.calibration("cxg_event", "cxg_event_base_plus_event_context")

    def data_cal_plus(self) -> pd.DataFrame:
        return self.calibration("cxg_plus", "cxg_plus_base_plus_selected_core")

    def calibration(self, track: str, model_name: str) -> pd.DataFrame:
        return self.query_df(f"""
        SELECT
          score_bin,
          COUNT(*) AS row_count,
          AVG(predicted_goal_probability) AS avg_prediction,
          AVG(IF(is_goal, 1.0, 0.0)) AS observed_goal_rate
        FROM (
          SELECT
            predicted_goal_probability,
            is_goal,
            NTILE(10) OVER (ORDER BY predicted_goal_probability) AS score_bin
          FROM {qtable('cxg_split_model_predictions_v1')}
          WHERE run_id = '{sql_quote(self.run_id)}'
            AND track = '{track}'
            AND model_name = '{model_name}'
            AND split = 'validation'
        )
        GROUP BY score_bin
        ORDER BY score_bin
        """)

    def layout(self, fig: go.Figure, title: str) -> go.Figure:
        fig.update_layout(
            title=title,
            template="plotly_white",
            font={"family": "Arial", "size": 13},
            margin={"l": 72, "r": 32, "t": 72, "b": 72},
            legend_title_text="Group",
        )
        return fig

    def plotly_uni_signal(self, df: pd.DataFrame, title: str) -> go.Figure:
        if df.empty:
            return self.empty_fig(title)
        plot = df.assign(label=df["feature_family"] + ": " + df["column_name"]).sort_values("abs_signal")
        fig = px.bar(plot, x="point_biserial_corr", y="label", color="feature_family", orientation="h", hover_data=["non_null_count", "null_pct", "goal_rate"])
        fig.update_xaxes(title="Point-biserial correlation")
        return self.layout(fig, title)

    def plotly_uni_stability(self, df: pd.DataFrame, title: str) -> go.Figure:
        if df.empty:
            return self.empty_fig(title)
        fig = px.scatter(df, x="train_corr", y="validation_corr", color="feature_family", hover_data=["column_name", "train_non_null_count", "validation_non_null_count"])
        fig.add_trace(go.Scatter(x=[-0.25, 0.25], y=[-0.25, 0.25], mode="lines", name="Parity", line={"dash": "dash", "color": "#444"}))
        fig.update_xaxes(title="Train target correlation")
        fig.update_yaxes(title="Validation target correlation")
        return self.layout(fig, title)

    def plotly_bi_lift(self, df: pd.DataFrame, title: str) -> go.Figure:
        if df.empty:
            return self.empty_fig(title)
        plot = df.assign(label=df["left_column"] + " + " + df["right_column"] + " | " + df["left_bin"].astype(str) + "/" + df["right_bin"].astype(str)).sort_values("lift_vs_split_baseline")
        fig = px.bar(plot, x="lift_vs_split_baseline", y="label", color="feature_family", orientation="h", hover_data=["row_count", "goal_count", "goal_rate"])
        fig.update_xaxes(title="Lift vs split baseline")
        return self.layout(fig, title)

    def plotly_bi_stability(self, df: pd.DataFrame, title: str) -> go.Figure:
        if df.empty:
            return self.empty_fig(title)
        fig = px.scatter(df, x="train_lift", y="validation_lift", color="feature_family", hover_data=["left_column", "right_column", "left_bin", "right_bin", "train_row_count", "validation_row_count"])
        fig.add_trace(go.Scatter(x=[0, 4], y=[0, 4], mode="lines", name="Parity", line={"dash": "dash", "color": "#444"}))
        fig.update_xaxes(title="Train lift")
        fig.update_yaxes(title="Validation lift")
        return self.layout(fig, title)

    def plotly_multi_metric(self, df: pd.DataFrame, title: str) -> go.Figure:
        if df.empty:
            return self.empty_fig(title)
        plot = df.assign(label=df["track"] + ": " + df["model_name"]).sort_values("metric_value")
        fig = px.bar(plot, x="metric_value", y="label", color="model_stage", orientation="h", hover_data=["feature_count", "row_count", "goal_count"])
        fig.update_xaxes(title="Metric value")
        return self.layout(fig, title)

    def plotly_multi_split_line(self, df: pd.DataFrame, title: str) -> go.Figure:
        if df.empty:
            return self.empty_fig(title)
        fig = px.line(df, x="split", y="auc", color="model_name", markers=True, line_dash="track", hover_data=["model_stage", "feature_count"])
        return self.layout(fig, title)

    def plotly_calibration(self, df: pd.DataFrame, title: str) -> go.Figure:
        if df.empty:
            return self.empty_fig(title)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["avg_prediction"], y=df["observed_goal_rate"], mode="lines+markers", name="Observed"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect", line={"dash": "dash", "color": "#444"}))
        fig.update_xaxes(title="Average predicted goal probability")
        fig.update_yaxes(title="Observed goal rate")
        return self.layout(fig, title)

    def png_uni_signal(self, df: pd.DataFrame, title: str, path: Path) -> None:
        plot = df.assign(label=df.get("column_name", pd.Series(dtype=str))).sort_values("abs_signal", ascending=True).tail(24) if not df.empty else df
        self.write_hbar(plot, "point_biserial_corr", "label", "feature_family", title, path, "Target correlation")

    def png_uni_stability(self, df: pd.DataFrame, title: str, path: Path) -> None:
        self.write_scatter(df, "train_corr", "validation_corr", "feature_family", title, path, "Train correlation", "Validation correlation", parity=(-0.25, 0.25))

    def png_bi_lift(self, df: pd.DataFrame, title: str, path: Path) -> None:
        plot = df.assign(label=df["left_column"] + " + " + df["right_column"]).sort_values("lift_vs_split_baseline", ascending=True).tail(24) if not df.empty else df
        self.write_hbar(plot, "lift_vs_split_baseline", "label", "feature_family", title, path, "Lift vs split baseline")

    def png_bi_stability(self, df: pd.DataFrame, title: str, path: Path) -> None:
        self.write_scatter(df, "train_lift", "validation_lift", "feature_family", title, path, "Train lift", "Validation lift", parity=(0.0, 4.0))

    def png_multi_metric(self, df: pd.DataFrame, title: str, path: Path) -> None:
        plot = df.assign(label=df["model_name"]).sort_values("metric_value", ascending=True).tail(24) if not df.empty else df
        self.write_hbar(plot, "metric_value", "label", "model_stage", title, path, "Metric value")

    def png_multi_split_line(self, df: pd.DataFrame, title: str, path: Path) -> None:
        fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=150)
        if df.empty:
            self.no_data(ax)
        else:
            order = ["train", "validation", "test"]
            for model_name, g in df.groupby("model_name"):
                g = g.set_index("split").reindex(order).reset_index()
                ax.plot(order, g["auc"], marker="o", label=str(model_name)[:42])
            ax.set_xlabel("Split")
            ax.set_ylabel("AUC")
            ax.grid(True, alpha=0.18)
            ax.legend(fontsize=7, loc="best")
        ax.set_title(title, fontsize=14, pad=14)
        fig.tight_layout()
        fig.savefig(path, format="png")
        plt.close(fig)

    def png_calibration(self, df: pd.DataFrame, title: str, path: Path) -> None:
        fig, ax = plt.subplots(figsize=(8.5, 7.2), dpi=150)
        if df.empty:
            self.no_data(ax)
        else:
            ax.plot(df["avg_prediction"], df["observed_goal_rate"], marker="o", label="Observed")
            ax.plot([0, 1], [0, 1], linestyle="--", color="#444", label="Perfect")
            ax.set_xlabel("Average predicted goal probability")
            ax.set_ylabel("Observed goal rate")
            ax.grid(True, alpha=0.18)
            ax.legend(fontsize=8)
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
            labels = [str(v)[:64] + ("..." if len(str(v)) > 64 else "") for v in df[y_col]]
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

    def write_scatter(self, df: pd.DataFrame, x_col: str, y_col: str, color_col: str, title: str, path: Path, xlabel: str, ylabel: str, parity: tuple[float, float]) -> None:
        fig, ax = plt.subplots(figsize=(9.6, 7.2), dpi=150)
        if df.empty:
            self.no_data(ax)
        else:
            groups = list(dict.fromkeys(df[color_col].astype(str)))
            cmap = plt.get_cmap("tab10")
            for i, group in enumerate(groups):
                g = df[df[color_col].astype(str) == group]
                ax.scatter(g[x_col], g[y_col], s=38, alpha=0.72, color=cmap(i % 10), label=group)
            ax.plot([parity[0], parity[1]], [parity[0], parity[1]], linestyle="--", color="#444", linewidth=1)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.18)
            ax.legend(fontsize=8, loc="best")
        ax.set_title(title, fontsize=14, pad=14)
        fig.tight_layout()
        fig.savefig(path, format="png")
        plt.close(fig)

    def empty_fig(self, title: str) -> go.Figure:
        fig = go.Figure()
        fig.add_annotation(text="No data available", x=0.5, y=0.5, showarrow=False)
        return self.layout(fig, title)

    def no_data(self, ax) -> None:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()

    def uri(self, name: str) -> str:
        return f"gs://{ARTIFACT_BUCKET}/{ARTIFACT_PREFIX}/{self.run_id}/{SPLIT_CHART_DIR}/{name}"

    def upload(self, path: Path) -> None:
        content_type = "text/html" if path.suffix == ".html" else "image/png" if path.suffix == ".png" else "application/json"
        blob_name = f"{ARTIFACT_PREFIX}/{self.run_id}/{SPLIT_CHART_DIR}/{path.name}"
        self.gcs.bucket(ARTIFACT_BUCKET).blob(blob_name).upload_from_filename(path, content_type=content_type)

    def materialize_registry(self, rendered: list[dict[str, object]]) -> None:
        rows = []
        for chart in rendered:
            rows.append(
                "SELECT "
                f"'{sql_quote(chart['run_id'])}' AS run_id, "
                f"'{sql_quote(chart['chart_name'])}' AS chart_name, "
                f"'{sql_quote(chart['chart_title'])}' AS chart_title, "
                f"'{sql_quote(chart['chart_group'])}' AS chart_group, "
                f"'{sql_quote(chart['backing_table'])}' AS backing_table, "
                f"'{sql_quote(chart['html_uri'])}' AS html_uri, "
                f"'{sql_quote(chart['png_uri'])}' AS png_uri, "
                "CURRENT_TIMESTAMP() AS rendered_at"
            )
        sql = f"CREATE OR REPLACE TABLE `{PROJECT_ID}.{ANALYSIS_DATASET}.{REGISTRY_TABLE}` AS\n" + "\nUNION ALL\n".join(rows)
        self.bq.query(sql, location=LOCATION).result()

    def validate(self, expected_charts: int, expected_gcs: int) -> dict[str, object]:
        html_count = len(list(self.out_dir.glob("*.html")))
        png_count = len(list(self.out_dir.glob("*.png")))
        manifest_exists = (self.out_dir / "split_analysis_charts_manifest.json").exists()
        gcs_count = None
        registry_count = None
        if not self.skip_upload:
            prefix = f"{ARTIFACT_PREFIX}/{self.run_id}/{SPLIT_CHART_DIR}/"
            gcs_count = sum(1 for _ in self.gcs.bucket(ARTIFACT_BUCKET).list_blobs(prefix=prefix))
            row = self.query_df(f"SELECT COUNT(*) AS n FROM `{PROJECT_ID}.{ANALYSIS_DATASET}.{REGISTRY_TABLE}` WHERE run_id = '{sql_quote(self.run_id)}'")
            registry_count = int(row.iloc[0]["n"]) if not row.empty else 0
        checks = {
            "local_html_count": html_count,
            "local_png_count": png_count,
            "manifest_exists": manifest_exists,
            "gcs_object_count": gcs_count,
            "bigquery_registry_count": registry_count,
            "expected_chart_count": expected_charts,
            "passed": html_count == expected_charts
            and png_count == expected_charts
            and manifest_exists
            and (self.skip_upload or (gcs_count == expected_gcs and registry_count == expected_charts)),
        }
        if not checks["passed"]:
            raise RuntimeError(f"Split chart validation failed: {checks}")
        return checks

    def write_checkpoint(self, manifest: dict[str, object], validation: dict[str, object]) -> Path:
        path = self.out_dir / "split_analysis_charts_checkpoint.md"
        lines = [
            "# CxG / CxG+ Split Analysis Charts Checkpoint",
            "",
            f"Run: `{self.run_id}`",
            f"Rendered at: `{manifest['rendered_at']}`",
            "",
            "## Outputs",
            "",
            f"- Chart count: {manifest['chart_count']}",
            f"- Local HTML/PNG: {validation['local_html_count']} / {validation['local_png_count']}",
            f"- GCS objects: {validation['gcs_object_count']}",
            f"- BigQuery registry rows: {validation['bigquery_registry_count']}",
            f"- Artifact prefix: `{manifest['artifact_prefix']}`",
            "",
            "## Groups",
            "",
        ]
        groups = pd.DataFrame(manifest["charts"]).groupby("chart_group").size().to_dict()
        for group, count in sorted(groups.items()):
            lines.append(f"- {group}: {count}")
        path.write_text("\n".join(lines), encoding="utf-8")
        return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--local-output-dir", default="audit_outputs/cxg_analysis")
    parser.add_argument("--skip-upload", action="store_true")
    args = parser.parse_args()
    renderer = SplitChartRenderer(args.run_id, Path(args.local_output_dir), skip_upload=args.skip_upload)
    print(json.dumps(renderer.render(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
