"""Pre-model CxG target and feature analysis.

This module sits between shot feature engineering and model training. It reads
shot-level feature rows joined to the base shots table, studies the goal target
and candidate feature behaviour, and writes modelling recommendations. It does
not read CxG predictions, model registry rows, or aggregate model outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib
import numpy as np
import pandas as pd
from sqlalchemy import Select, select
from sqlalchemy.orm import Session

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from opponent_adjusted.db.models import Shot, ShotFeature

TARGET_COLUMN = "is_goal"
BENCHMARK_COLUMNS = {"statsbomb_xg", "provider_xg"}
DEFAULT_OUTPUT_DIR = Path("outputs/analysis/cxg")
REQUIRED_SECTIONS = (
    "Question",
    "Calculation",
    "Visual/Table",
    "Interpretation",
    "Modelling implication",
    "Limitation",
)
ID_COLUMNS = {
    "shot_id",
    "id",
    "event_id",
    "match_id",
    "team_id",
    "player_id",
    "opponent_team_id",
    "competition_id",
}
OUTCOME_COLUMNS = {"outcome", "shot_outcome"}
TARGET_COLUMNS = {TARGET_COLUMN}
POST_MODEL_PATTERNS = (
    "cxg",
    "prediction",
    "probability",
    "model",
    "registry",
    "aggregate",
    "neutral",
    "adjusted",
    "leaderboard",
)
CONTEXT_COLUMNS = (
    "body_part",
    "technique",
    "shot_type",
    "play_pattern",
    "first_time",
    "is_blocked",
)
POSSESSION_COLUMNS = (
    "possession_sequence_length",
    "possession_duration",
    "previous_action_gap",
    "recent_def_actions_count",
    "pressure_proxy_score",
)
REDUNDANCY_PAIRS = (
    ("shot_distance", "distance_to_goal_line"),
    ("shot_distance", "shot_angle"),
    ("shot_angle", "centrality"),
    ("possession_duration", "possession_sequence_length"),
    ("pressure_proxy_score", "recent_def_actions_count"),
)


@dataclass(frozen=True)
class CxGAnalysisResult:
    """Paths and summary metrics emitted by the pre-model analysis."""

    output_dir: Path
    report_path: Path
    row_count: int
    goal_rate: float
    feature_count: int
    leakage_risk_count: int


def load_shot_feature_dataset(
    session: Session,
    *,
    version_tag: str | None = None,
) -> pd.DataFrame:
    """Load pre-model CxG analysis rows from `shot_features` joined to `shots`."""

    stmt: Select[tuple[ShotFeature, Shot]] = select(ShotFeature, Shot).join(
        Shot, Shot.id == ShotFeature.shot_id
    )
    if version_tag is not None:
        stmt = stmt.where(ShotFeature.version_tag == version_tag)

    rows = session.execute(stmt).all()
    records: list[dict[str, object]] = []
    for feature, shot in rows:
        records.append(
            {
                "shot_id": shot.id,
                "match_id": shot.match_id,
                "team_id": shot.team_id,
                "player_id": shot.player_id,
                "opponent_team_id": shot.opponent_team_id,
                "outcome": shot.outcome,
                "statsbomb_xg": shot.statsbomb_xg,
                "body_part": shot.body_part,
                "technique": shot.technique,
                "shot_type": shot.shot_type,
                "first_time": shot.first_time,
                "is_blocked": shot.is_blocked,
                "version_tag": feature.version_tag,
                "shot_distance": feature.shot_distance,
                "shot_angle": feature.shot_angle,
                "centrality": feature.centrality,
                "distance_to_goal_line": feature.distance_to_goal_line,
                "score_diff_at_shot": feature.score_diff_at_shot,
                "is_leading": feature.is_leading,
                "is_trailing": feature.is_trailing,
                "is_drawing": feature.is_drawing,
                "minute_bucket": feature.minute_bucket,
                "possession_sequence_length": feature.possession_sequence_length,
                "possession_duration": feature.possession_duration,
                "previous_action_gap": feature.previous_action_gap,
                "recent_def_actions_count": feature.recent_def_actions_count,
                "pressure_proxy_score": feature.pressure_proxy_score,
            }
        )
    return pd.DataFrame.from_records(records)


def build_pre_model_cxg_analysis(
    shot_features: pd.DataFrame,
    shots: pd.DataFrame | None = None,
    *,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    min_slice_size: int = 30,
) -> CxGAnalysisResult:
    """Generate the full pre-model CxG target and feature study."""

    output_path = Path(output_dir)
    folders = _create_output_folders(output_path)
    frame = _prepare_dataset(shot_features, shots)
    _apply_matplotlib_style()

    target = _target_summary(frame, folders["00_target"])
    distributions = _feature_distributions(frame, folders["01_feature_distributions"])
    relationships = _feature_target_relationships(
        frame,
        folders["02_feature_target_relationships"],
        min_slice_size=min_slice_size,
    )
    correlations = _feature_correlations(frame, folders["03_feature_correlations"])
    stability = _slice_stability(
        frame,
        folders["04_slice_stability"],
        min_slice_size=min_slice_size,
    )
    quality = _data_quality(frame, folders["05_data_quality"], min_slice_size=min_slice_size)
    leakage = _leakage_checks(frame, folders["06_leakage_checks"])

    report_path = output_path / "report.md"
    report_path.write_text(
        _render_report(
            target=target,
            distributions=distributions,
            relationships=relationships,
            correlations=correlations,
            stability=stability,
            quality=quality,
            leakage=leakage,
            min_slice_size=min_slice_size,
        ),
        encoding="utf-8",
    )

    return CxGAnalysisResult(
        output_dir=output_path,
        report_path=report_path,
        row_count=int(target["row_count"]),
        goal_rate=float(target["goal_rate"]),
        feature_count=int(distributions["feature_count"]),
        leakage_risk_count=int(leakage["risk_count"]),
    )


def run_pre_model_cxg_analysis(
    session: Session,
    *,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    version_tag: str | None = None,
    min_slice_size: int = 30,
) -> CxGAnalysisResult:
    """Load DB shot features and write the pre-model CxG analysis report."""

    frame = load_shot_feature_dataset(session, version_tag=version_tag)
    return build_pre_model_cxg_analysis(
        frame,
        output_dir=output_dir,
        min_slice_size=min_slice_size,
    )


def _create_output_folders(output_dir: Path) -> dict[str, Path]:
    names = (
        "00_target",
        "01_feature_distributions",
        "02_feature_target_relationships",
        "03_feature_correlations",
        "04_slice_stability",
        "05_data_quality",
        "06_leakage_checks",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    folders = {name: output_dir / name for name in names}
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "plots").mkdir(exist_ok=True)
        (folder / "tables").mkdir(exist_ok=True)
    return folders


def _prepare_dataset(shot_features: pd.DataFrame, shots: pd.DataFrame | None) -> pd.DataFrame:
    if shots is None:
        frame = shot_features.copy()
    else:
        features = shot_features.copy()
        shots_frame = shots.copy()
        if "id" in shots_frame.columns and "shot_id" not in shots_frame.columns:
            shots_frame = shots_frame.rename(columns={"id": "shot_id"})
        frame = features.merge(shots_frame, on="shot_id", how="left", suffixes=("", "_shot"))

    if TARGET_COLUMN not in frame.columns:
        if "outcome" not in frame.columns:
            raise ValueError("CxG analysis requires either `is_goal` or `outcome`.")
        frame[TARGET_COLUMN] = frame["outcome"].astype(str).str.lower().eq("goal").astype(int)

    frame[TARGET_COLUMN] = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
    frame = frame.dropna(subset=[TARGET_COLUMN]).copy()
    frame[TARGET_COLUMN] = frame[TARGET_COLUMN].astype(int)
    return frame


def _apply_matplotlib_style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (11, 7),
            "figure.dpi": 120,
            "axes.facecolor": "#f7f7f4",
            "figure.facecolor": "#ffffff",
            "axes.edgecolor": "#444444",
            "axes.grid": True,
            "grid.color": "#dddddd",
            "grid.linewidth": 0.7,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelcolor": "#222222",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "font.size": 10,
            "savefig.bbox": "tight",
        }
    )


def _save_plot(
    fig: plt.Figure,
    ax: plt.Axes,
    path: Path,
    *,
    title: str,
    subtitle: str,
    caption: str,
) -> None:
    ax.set_title(title, loc="left", pad=26, fontsize=13, fontweight="bold")
    ax.text(0.0, 1.025, subtitle, transform=ax.transAxes, fontsize=10, color="#374151")
    fig.text(0.01, 0.01, caption, ha="left", va="bottom", fontsize=8.5, color="#4b5563")
    fig.subplots_adjust(top=0.82, bottom=0.18, left=0.12, right=0.96)
    fig.savefig(path)
    plt.close(fig)


def _target_summary(frame: pd.DataFrame, folder: Path) -> dict[str, object]:
    row_count = len(frame)
    goal_count = int(frame[TARGET_COLUMN].sum())
    goal_rate = float(frame[TARGET_COLUMN].mean()) if row_count else 0.0
    status = _target_status(row_count, goal_rate)
    table = pd.DataFrame(
        [
            {
                "rows": row_count,
                "goals": goal_count,
                "non_goals": row_count - goal_count,
                "goal_rate": goal_rate,
                "status": status,
            }
        ]
    )
    table.to_csv(folder / "tables" / "target_summary.csv", index=False)
    table.to_csv(folder / "target_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 6))
    labels = ["Non-goal", "Goal"]
    values = [row_count - goal_count, goal_count]
    ax.bar(labels, values, color=["#6b7280", "#0f766e"])
    ax.axhline(max(row_count * 0.05, 1), color="#b45309", linestyle="--", label="5% reference")
    ax.set_ylabel("Shot count")
    ax.legend(loc="upper right")
    _save_plot(
        fig,
        ax,
        folder / "plots" / "target_balance.png",
        title="Is the goal target usable and balanced enough for modelling?",
        subtitle=f"{goal_count:,} goals from {row_count:,} shots; goal rate {goal_rate:.1%}.",
        caption="Source: shot_features joined to shots. Caveat: target usability is not model calibration.",
    )
    return {
        "row_count": row_count,
        "goal_count": goal_count,
        "goal_rate": goal_rate,
        "status": status,
        "visual": "00_target/plots/target_balance.png",
        "table": "00_target/tables/target_summary.csv",
    }


def _feature_distributions(frame: pd.DataFrame, folder: Path) -> dict[str, object]:
    feature_columns = _candidate_feature_columns(frame)
    numeric_columns = _numeric_features(frame, feature_columns)
    profile_rows = []
    for column in feature_columns:
        profile_rows.append(_feature_profile_row(frame[column], column))

    profile = pd.DataFrame(profile_rows).sort_values(
        ["missing_rate", "column"], ascending=[False, True]
    )
    profile.to_csv(folder / "tables" / "feature_missingness.csv", index=False)
    profile.to_csv(folder / "feature_missingness.csv", index=False)

    numeric_profile = pd.DataFrame(
        [_numeric_profile_row(frame[column], column) for column in numeric_columns]
    )
    numeric_profile.to_csv(folder / "tables" / "numeric_feature_profiles.csv", index=False)

    plot_paths = []
    for column in numeric_columns:
        plot_path = folder / "plots" / f"{_slug(column)}_distribution.png"
        _plot_numeric_distribution(frame, column, plot_path)
        plot_paths.append(f"01_feature_distributions/plots/{plot_path.name}")

    categorical_rows = []
    for column in _categorical_features(frame, feature_columns):
        counts = frame[column].fillna("missing").astype(str).value_counts().head(20)
        for value, count in counts.items():
            categorical_rows.append({"column": column, "value": value, "rows": int(count)})
    pd.DataFrame(categorical_rows).to_csv(
        folder / "tables" / "categorical_top_levels.csv", index=False
    )
    return {
        "feature_count": len(feature_columns),
        "numeric_feature_count": len(numeric_columns),
        "high_missing": profile.loc[profile["missing_rate"] >= 0.25, "column"].tolist(),
        "table": "01_feature_distributions/tables/feature_missingness.csv",
        "numeric_profile": "01_feature_distributions/tables/numeric_feature_profiles.csv",
        "visuals": plot_paths,
    }


def _plot_numeric_distribution(frame: pd.DataFrame, column: str, path: Path) -> None:
    series = pd.to_numeric(frame[column], errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(10, 6.5))
    if series.empty:
        ax.text(0.5, 0.5, "No non-missing numeric values", ha="center", va="center")
    else:
        ax.hist(
            series, bins=min(30, max(8, int(np.sqrt(len(series))))), color="#2563eb", alpha=0.82
        )
        median = float(series.median())
        ax.axvline(median, color="#b45309", linestyle="--", label=f"Median: {median:.2f}")
        ax.legend(loc="upper right")
        ax.set_xlabel(column)
        ax.set_ylabel("Shot count")
    _save_plot(
        fig,
        ax,
        path,
        title=f"Does {column} need scaling, transformation, or binning?",
        subtitle=_distribution_takeaway(series, column),
        caption=(
            f"Source: shot_features joined to shots. n={len(series):,} non-missing values. "
            "Caveat: distribution shape alone does not prove target signal."
            + (_outlier_note(series) if not series.empty else "")
        ),
    )


def _feature_target_relationships(
    frame: pd.DataFrame,
    folder: Path,
    *,
    min_slice_size: int,
) -> dict[str, object]:
    feature_columns = _candidate_feature_columns(frame)
    numeric_table = _generic_numeric_relationships(frame, feature_columns)
    numeric_table.to_csv(folder / "tables" / "numeric_target_relationships.csv", index=False)
    numeric_table.to_csv(folder / "numeric_target_relationships.csv", index=False)

    categorical_table = _generic_categorical_relationships(
        frame, _categorical_features(frame, feature_columns), min_slice_size
    )
    categorical_table.to_csv(
        folder / "tables" / "categorical_target_relationships.csv", index=False
    )

    outputs: dict[str, object] = {
        "top_numeric_signal": (
            numeric_table.head(5)["column"].tolist() if not numeric_table.empty else []
        ),
        "table": "02_feature_target_relationships/tables/numeric_target_relationships.csv",
        "shot_distance": _empty_summary("shot_distance"),
        "shot_angle": _empty_summary("shot_angle"),
        "pressure": _empty_summary("pressure"),
        "context": {},
        "possession": {},
    }
    if "shot_distance" in frame.columns:
        outputs["shot_distance"] = _binned_goal_rate_output(
            frame,
            "shot_distance",
            folder,
            table_name="shot_distance_bins.csv",
            plot_name="shot_distance_vs_goal_rate.png",
            question="Does goal rate increase as shot distance decreases?",
            xlabel="Shot distance bin",
            ascending=True,
        )
    if "shot_angle" in frame.columns:
        outputs["shot_angle"] = _binned_goal_rate_output(
            frame,
            "shot_angle",
            folder,
            table_name="shot_angle_bins.csv",
            plot_name="shot_angle_vs_goal_rate.png",
            question="Does goal rate improve with better shot angle?",
            xlabel="Shot angle bin",
            ascending=True,
        )

    pressure_columns = [column for column in _pressure_columns(frame) if column in frame.columns]
    if pressure_columns:
        outputs["pressure"] = _pressure_relationship(
            frame, folder, pressure_columns, min_slice_size
        )

    context_outputs = {}
    for column in CONTEXT_COLUMNS:
        if column in frame.columns:
            context_outputs[column] = _categorical_goal_rate_output(
                frame,
                column,
                folder,
                question=f"Does {column} separate goal probability?",
                min_slice_size=min_slice_size,
            )
    outputs["context"] = context_outputs

    possession_outputs = {}
    for column in POSSESSION_COLUMNS:
        if column in frame.columns:
            possession_outputs[column] = _binned_goal_rate_output(
                frame,
                column,
                folder,
                table_name=f"{_slug(column)}_bins.csv",
                plot_name=f"{_slug(column)}_vs_goal_rate.png",
                question=f"Does {column} contain target signal?",
                xlabel=f"{column} bin",
                ascending=True,
            )
    outputs["possession"] = possession_outputs
    return outputs


def _generic_numeric_relationships(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    rows = []
    for column in _numeric_features(frame, feature_columns):
        valid = frame[[column, TARGET_COLUMN]].dropna()
        if valid[column].nunique() < 2:
            continue
        corr = valid[column].corr(valid[TARGET_COLUMN])
        rates = _binned_goal_rate(valid, column, bins=min(4, valid[column].nunique()))
        rows.append(
            {
                "column": column,
                "rows": len(valid),
                "pearson_with_goal": float(corr) if pd.notna(corr) else np.nan,
                "min_bin_goal_rate": float(rates["goal_rate"].min()),
                "max_bin_goal_rate": float(rates["goal_rate"].max()),
                "goal_rate_spread": float(rates["goal_rate"].max() - rates["goal_rate"].min()),
            }
        )
    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.sort_values("goal_rate_spread", ascending=False)
    return table


def _generic_categorical_relationships(
    frame: pd.DataFrame,
    columns: Iterable[str],
    min_slice_size: int,
) -> pd.DataFrame:
    rows = []
    for column in columns:
        table = _categorical_goal_rate(frame, column, min_slice_size=min_slice_size)
        for _, row in table.iterrows():
            rows.append(
                {
                    "column": column,
                    "value": row["value"],
                    "rows": int(row["rows"]),
                    "goals": int(row["goals"]),
                    "goal_rate": float(row["goal_rate"]),
                }
            )
    return pd.DataFrame(rows, columns=["column", "value", "rows", "goals", "goal_rate"])


def _binned_goal_rate_output(
    frame: pd.DataFrame,
    column: str,
    folder: Path,
    *,
    table_name: str,
    plot_name: str,
    question: str,
    xlabel: str,
    ascending: bool,
) -> dict[str, object]:
    table = _binned_goal_rate(frame, column)
    table.to_csv(folder / "tables" / table_name, index=False)
    if not table.empty:
        plot_table = table.sort_values("bin_midpoint", ascending=ascending)
        fig, ax = plt.subplots(figsize=(10.5, 6.5))
        ax.plot(plot_table["bin_label"], plot_table["goal_rate"], marker="o", color="#0f766e")
        ax.axhline(
            frame[TARGET_COLUMN].mean(),
            color="#b45309",
            linestyle="--",
            label="Global goal rate",
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Goal rate")
        ax.tick_params(axis="x", labelrotation=25)
        ax.legend(loc="upper right")
        _save_plot(
            fig,
            ax,
            folder / "plots" / plot_name,
            title=question,
            subtitle=_goal_rate_takeaway(plot_table, column),
            caption=(
                f"Source: shot_features joined to shots. n={int(plot_table['rows'].sum()):,}. "
                "Caveat: binned rates are descriptive and not adjusted for other features."
            ),
        )
    return {
        "table": f"02_feature_target_relationships/tables/{table_name}",
        "visual": f"02_feature_target_relationships/plots/{plot_name}",
        "interpretation": _relationship_interpretation(table, column),
    }


def _categorical_goal_rate_output(
    frame: pd.DataFrame,
    column: str,
    folder: Path,
    *,
    question: str,
    min_slice_size: int,
) -> dict[str, object]:
    table_name = f"{_slug(column)}_goal_rate.csv"
    plot_name = f"{_slug(column)}_goal_rate.png"
    table = _categorical_goal_rate(frame, column, min_slice_size=min_slice_size)
    table.to_csv(folder / "tables" / table_name, index=False)
    if not table.empty:
        plot_table = table.sort_values(["goal_rate", "rows"], ascending=[True, False]).tail(15)
        fig, ax = plt.subplots(figsize=(10.5, max(6, len(plot_table) * 0.45)))
        ax.barh(plot_table["value"].astype(str), plot_table["goal_rate"], color="#2563eb")
        ax.axvline(frame[TARGET_COLUMN].mean(), color="#b45309", linestyle="--", label="Global")
        ax.set_xlabel("Goal rate")
        ax.set_ylabel(column)
        ax.legend(loc="lower right")
        _save_plot(
            fig,
            ax,
            folder / "plots" / plot_name,
            title=question,
            subtitle=_categorical_takeaway(table, column),
            caption=(
                f"Source: shot_features joined to shots. Minimum slice size {min_slice_size}. "
                "Caveat: rare levels are excluded from the plot and should be encoded carefully."
            ),
        )
    return {
        "table": f"02_feature_target_relationships/tables/{table_name}",
        "visual": f"02_feature_target_relationships/plots/{plot_name}",
        "interpretation": _categorical_takeaway(table, column),
    }


def _pressure_relationship(
    frame: pd.DataFrame,
    folder: Path,
    pressure_columns: list[str],
    min_slice_size: int,
) -> dict[str, object]:
    rows = []
    for column in pressure_columns:
        if pd.api.types.is_bool_dtype(frame[column]) or frame[column].nunique(dropna=True) <= 6:
            table = _categorical_goal_rate(frame, column, min_slice_size=min_slice_size)
            for _, row in table.iterrows():
                rows.append(
                    {
                        "pressure_column": column,
                        "pressure_value": row["value"],
                        "rows": int(row["rows"]),
                        "goals": int(row["goals"]),
                        "goal_rate": float(row["goal_rate"]),
                    }
                )
        else:
            table = _binned_goal_rate(frame, column)
            for _, row in table.iterrows():
                rows.append(
                    {
                        "pressure_column": column,
                        "pressure_value": row["bin_label"],
                        "rows": int(row["rows"]),
                        "goals": int(row["goals"]),
                        "goal_rate": float(row["goal_rate"]),
                    }
                )
    table = pd.DataFrame(
        rows, columns=["pressure_column", "pressure_value", "rows", "goals", "goal_rate"]
    )
    table.to_csv(folder / "tables" / "pressure_goal_rate.csv", index=False)
    if not table.empty:
        plot_table = table.sort_values(["pressure_column", "goal_rate"])
        labels = plot_table["pressure_column"] + "=" + plot_table["pressure_value"].astype(str)
        fig, ax = plt.subplots(figsize=(11, max(6, len(plot_table) * 0.4)))
        ax.barh(labels, plot_table["goal_rate"], color="#7c3aed")
        ax.axvline(frame[TARGET_COLUMN].mean(), color="#b45309", linestyle="--", label="Global")
        ax.set_xlabel("Goal rate")
        ax.set_ylabel("Pressure state/bin")
        ax.legend(loc="lower right")
        _save_plot(
            fig,
            ax,
            folder / "plots" / "pressure_vs_goal_rate.png",
            title="Are pressured shots converted less often?",
            subtitle=_pressure_takeaway(table),
            caption=(
                f"Source: shot_features joined to shots. Pressure fields: {', '.join(pressure_columns)}. "
                "Caveat: pressure proxies may encode defensive context imperfectly."
            ),
        )
    return {
        "table": "02_feature_target_relationships/tables/pressure_goal_rate.csv",
        "visual": "02_feature_target_relationships/plots/pressure_vs_goal_rate.png",
        "interpretation": _pressure_takeaway(table),
    }


def _feature_correlations(frame: pd.DataFrame, folder: Path) -> dict[str, object]:
    numeric = _numeric_features(frame, _candidate_feature_columns(frame))
    corr = frame[numeric].corr(numeric_only=True) if len(numeric) >= 2 else pd.DataFrame()
    corr.to_csv(folder / "tables" / "numeric_correlations.csv")
    corr.to_csv(folder / "numeric_correlations.csv")

    high_rows = []
    for i, left in enumerate(corr.columns):
        for right in corr.columns[i + 1 :]:
            value = corr.loc[left, right]
            if pd.notna(value) and abs(value) >= 0.8:
                high_rows.append(
                    {"feature_a": left, "feature_b": right, "correlation": float(value)}
                )
    high = pd.DataFrame(high_rows, columns=["feature_a", "feature_b", "correlation"])
    if not high.empty:
        high = high.sort_values("correlation", key=lambda s: s.abs(), ascending=False)
    high.to_csv(folder / "tables" / "high_correlations.csv", index=False)
    high.to_csv(folder / "high_correlations.csv", index=False)

    targeted_rows = []
    for left, right in REDUNDANCY_PAIRS:
        if left in frame.columns and right in frame.columns:
            valid = frame[[left, right]].dropna()
            corr_value = valid[left].corr(valid[right]) if valid[left].nunique() > 1 else np.nan
            targeted_rows.append(
                {
                    "feature_a": left,
                    "feature_b": right,
                    "rows": len(valid),
                    "correlation": float(corr_value) if pd.notna(corr_value) else np.nan,
                    "redundancy_flag": bool(pd.notna(corr_value) and abs(float(corr_value)) >= 0.8),
                    "modelling_implication": _redundancy_implication(corr_value, left, right),
                }
            )
    targeted = pd.DataFrame(
        targeted_rows,
        columns=[
            "feature_a",
            "feature_b",
            "rows",
            "correlation",
            "redundancy_flag",
            "modelling_implication",
        ],
    )
    targeted.to_csv(folder / "tables" / "targeted_redundancy_pairs.csv", index=False)

    if not corr.empty:
        fig, ax = plt.subplots(figsize=(max(9, len(corr) * 0.72), max(7, len(corr) * 0.58)))
        image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(corr.index)), corr.index)
        fig.colorbar(image, ax=ax, label="Pearson correlation")
        _save_plot(
            fig,
            ax,
            folder / "plots" / "correlation_heatmap.png",
            title="Which candidate features may be redundant before model training?",
            subtitle=f"{len(high):,} feature pairs exceed the high-correlation threshold.",
            caption=(
                "Source: shot_features joined to shots. Reference: absolute Pearson correlation >= 0.80. "
                "Caveat: correlation only captures linear numeric relationships."
            ),
        )

    return {
        "high_pairs": len(high),
        "targeted_pairs": len(targeted),
        "table": "03_feature_correlations/tables/high_correlations.csv",
        "targeted_table": "03_feature_correlations/tables/targeted_redundancy_pairs.csv",
        "visual": "03_feature_correlations/plots/correlation_heatmap.png",
    }


def _slice_stability(
    frame: pd.DataFrame,
    folder: Path,
    *,
    min_slice_size: int,
) -> dict[str, object]:
    slice_columns = [
        column
        for column in (
            "competition_id",
            "competition_name",
            "team_id",
            "period",
            "body_part",
            "technique",
            "shot_type",
            "play_pattern",
            "minute_bucket",
            "under_pressure",
            "pressure_state",
            "is_leading",
            "is_trailing",
            "is_drawing",
            "score_state",
            "simple_state",
        )
        if column in frame.columns
    ]
    rows = []
    global_rate = float(frame[TARGET_COLUMN].mean()) if len(frame) else 0.0
    for column in slice_columns:
        grouped = (
            frame.assign(_value=frame[column].fillna("missing").astype(str))
            .groupby("_value", observed=True)[TARGET_COLUMN]
            .agg(rows="count", goals="sum", goal_rate="mean")
            .reset_index()
        )
        grouped = grouped[grouped["rows"] >= min_slice_size]
        for _, row in grouped.iterrows():
            delta = float(row["goal_rate"] - global_rate)
            rows.append(
                {
                    "slice_column": column,
                    "slice_value": row["_value"],
                    "rows": int(row["rows"]),
                    "goals": int(row["goals"]),
                    "goal_rate": float(row["goal_rate"]),
                    "global_goal_rate": global_rate,
                    "goal_rate_delta": delta,
                    "sample_size_warning": int(row["rows"]) < max(min_slice_size * 2, 100),
                    "modelling_implication": _slice_implication(delta),
                }
            )
    table = pd.DataFrame(
        rows,
        columns=[
            "slice_column",
            "slice_value",
            "rows",
            "goals",
            "goal_rate",
            "global_goal_rate",
            "goal_rate_delta",
            "sample_size_warning",
            "modelling_implication",
        ],
    )
    table.to_csv(folder / "tables" / "slice_stability.csv", index=False)
    table.to_csv(folder / "slice_stability.csv", index=False)

    if not table.empty:
        plot = table.reindex(
            table["goal_rate_delta"].abs().sort_values(ascending=False).index
        ).head(16)
        fig, ax = plt.subplots(figsize=(11, max(7, len(plot) * 0.42)))
        labels = plot["slice_column"] + "=" + plot["slice_value"]
        ax.barh(labels, plot["goal_rate_delta"], color="#7c3aed")
        ax.axvline(0, color="#111827", linewidth=1)
        ax.axvline(0.03, color="#b45309", linestyle="--")
        ax.axvline(-0.03, color="#b45309", linestyle="--", label="3pp reference")
        ax.set_xlabel("Goal-rate delta from global rate")
        ax.set_ylabel("Slice")
        ax.legend(loc="lower right")
        _save_plot(
            fig,
            ax,
            folder / "plots" / "slice_stability.png",
            title="Do feature-target relationships hold across modelling slices?",
            subtitle=f"{int((table['goal_rate_delta'].abs() >= 0.03).sum())} slices exceed a 3pp delta.",
            caption=(
                f"Source: shot_features joined to shots. Minimum slice size {min_slice_size}. "
                "Caveat: slice rates are descriptive and may reflect schedule or team mix."
            ),
        )

    return {
        "slice_columns": slice_columns,
        "unstable_slices": (
            int((table["goal_rate_delta"].abs() >= 0.03).sum()) if not table.empty else 0
        ),
        "table": "04_slice_stability/tables/slice_stability.csv",
        "visual": "04_slice_stability/plots/slice_stability.png",
    }


def _data_quality(frame: pd.DataFrame, folder: Path, *, min_slice_size: int) -> dict[str, object]:
    rows = []
    for column in _candidate_feature_columns(frame):
        series = frame[column]
        recommendation = _quality_recommendation(series)
        rows.append(
            {
                "column": column,
                "missing_rate": float(series.isna().mean()),
                "unique_values": int(series.nunique(dropna=True)),
                "recommendation": recommendation,
            }
        )
    table = pd.DataFrame(rows).sort_values(["recommendation", "column"])
    table.to_csv(folder / "tables" / "data_quality.csv", index=False)
    table.to_csv(folder / "data_quality.csv", index=False)

    value_checks = _football_value_checks(frame, min_slice_size=min_slice_size)
    value_checks.to_csv(folder / "tables" / "football_value_checks.csv", index=False)

    recommendations = _cleaning_recommendations(table, value_checks)
    recommendations.to_csv(folder / "tables" / "cleaning_recommendations.csv", index=False)
    recommendations.to_csv(folder / "cleaning_recommendations.csv", index=False)
    return {
        "action_count": int(len(recommendations)),
        "table": "05_data_quality/tables/data_quality.csv",
        "football_checks": "05_data_quality/tables/football_value_checks.csv",
        "recommendations": "05_data_quality/tables/cleaning_recommendations.csv",
    }


def _football_value_checks(frame: pd.DataFrame, *, min_slice_size: int) -> pd.DataFrame:
    rows = []
    checks = {
        "shot_distance": lambda s: s < 0,
        "shot_angle": lambda s: (s < 0) | (s > np.pi),
        "centrality": lambda s: (s < 0) | (s > 1),
        "possession_duration": lambda s: s < 0,
        "possession_sequence_length": lambda s: s < 0,
        "previous_action_gap": lambda s: s < 0,
        "recent_def_actions_count": lambda s: s < 0,
        "pressure_proxy_score": lambda s: (s < 0) | (s > 1),
    }
    for column, predicate in checks.items():
        if column not in frame.columns:
            continue
        series = pd.to_numeric(frame[column], errors="coerce")
        invalid = predicate(series.dropna())
        rows.append(
            {
                "column": column,
                "check": "football_value_range",
                "issue_count": int(invalid.sum()),
                "issue_rate": float(invalid.mean()) if len(invalid) else 0.0,
                "modelling_implication": _value_check_implication(int(invalid.sum())),
            }
        )

    for column in _candidate_feature_columns(frame):
        series = frame[column]
        if series.isna().mean() >= 0.25:
            rows.append(
                {
                    "column": column,
                    "check": "high_missingness",
                    "issue_count": int(series.isna().sum()),
                    "issue_rate": float(series.isna().mean()),
                    "modelling_implication": "Impute with monitoring or exclude if structurally missing.",
                }
            )
        unique = series.nunique(dropna=True)
        if unique <= 1:
            rows.append(
                {
                    "column": column,
                    "check": "constant_or_near_constant",
                    "issue_count": len(series),
                    "issue_rate": 1.0,
                    "modelling_implication": "Exclude because the feature cannot separate outcomes.",
                }
            )
        if pd.api.types.is_numeric_dtype(series) and unique > 2:
            numeric = pd.to_numeric(series, errors="coerce").dropna()
            if not numeric.empty:
                q1, q3 = numeric.quantile([0.25, 0.75])
                iqr = q3 - q1
                if iqr > 0:
                    outliers = (numeric < q1 - 3 * iqr) | (numeric > q3 + 3 * iqr)
                    if outliers.any():
                        rows.append(
                            {
                                "column": column,
                                "check": "extreme_outliers",
                                "issue_count": int(outliers.sum()),
                                "issue_rate": float(outliers.mean()),
                                "modelling_implication": "Review transformations or winsorisation.",
                            }
                        )
        if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            counts = series.fillna("missing").astype(str).value_counts()
            rare_count = int((counts < min_slice_size).sum())
            if rare_count:
                rows.append(
                    {
                        "column": column,
                        "check": "rare_categorical_levels",
                        "issue_count": rare_count,
                        "issue_rate": float(rare_count / len(counts)),
                        "modelling_implication": "Pool rare levels before one-hot or target encoding.",
                    }
                )
    return pd.DataFrame(
        rows, columns=["column", "check", "issue_count", "issue_rate", "modelling_implication"]
    )


def _cleaning_recommendations(
    quality_table: pd.DataFrame, value_checks: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for _, row in quality_table.iterrows():
        if row["recommendation"] != "keep":
            rows.append(
                {
                    "column": row["column"],
                    "reason": row["recommendation"],
                    "recommendation": row["recommendation"],
                }
            )
    for _, row in value_checks.iterrows():
        if int(row["issue_count"]) > 0:
            rows.append(
                {
                    "column": row["column"],
                    "reason": row["check"],
                    "recommendation": row["modelling_implication"],
                }
            )
    return pd.DataFrame(rows, columns=["column", "reason", "recommendation"]).drop_duplicates()


def _leakage_checks(frame: pd.DataFrame, folder: Path) -> dict[str, object]:
    rows = []
    for column in frame.columns:
        group, recommendation = _training_eligibility(column)
        rows.append(
            {
                "column": column,
                "eligibility_group": group,
                "training_eligibility": recommendation,
                "risk": "none" if group == "safe candidate features" else group,
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(folder / "tables" / "feature_training_eligibility.csv", index=False)
    table.to_csv(folder / "tables" / "leakage_checks.csv", index=False)
    table.to_csv(folder / "leakage_checks.csv", index=False)
    risk_count = int((table["training_eligibility"] != "eligible_candidate_feature").sum())
    return {
        "risk_count": risk_count,
        "risky_columns": table.loc[
            table["training_eligibility"] != "eligible_candidate_feature", "column"
        ].tolist(),
        "table": "06_leakage_checks/tables/feature_training_eligibility.csv",
    }


def _training_eligibility(column: str) -> tuple[str, str]:
    lower = column.lower()
    if column in TARGET_COLUMNS:
        return "target columns", "exclude_target"
    if column in OUTCOME_COLUMNS or "outcome" in lower:
        return "outcome columns", "exclude_target_proxy"
    if column in BENCHMARK_COLUMNS:
        return "provider/reference benchmark columns", "reference_only_not_training_feature"
    if any(pattern in lower for pattern in POST_MODEL_PATTERNS):
        return "post-model columns", "exclude_post_model_output"
    if "aggregate" in lower or "prediction" in lower or "model" in lower:
        return "aggregate/model/prediction columns", "exclude_post_model_output"
    if column in ID_COLUMNS or lower.endswith("_id"):
        return "ID columns", "exclude_identifier"
    return "safe candidate features", "eligible_candidate_feature"


def _candidate_feature_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if _training_eligibility(column)[1] == "eligible_candidate_feature"
    ]


def _numeric_features(frame: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    return [
        column
        for column in columns
        if pd.api.types.is_numeric_dtype(frame[column])
        and not pd.api.types.is_bool_dtype(frame[column])
    ]


def _categorical_features(frame: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    return [
        column
        for column in columns
        if not pd.api.types.is_numeric_dtype(frame[column])
        or pd.api.types.is_bool_dtype(frame[column])
    ]


def _binned_goal_rate(frame: pd.DataFrame, column: str, bins: int = 5) -> pd.DataFrame:
    valid = frame[[column, TARGET_COLUMN]].dropna().copy()
    if valid.empty or valid[column].nunique() < 2:
        return pd.DataFrame(
            columns=[
                "bin_label",
                "bin_min",
                "bin_max",
                "bin_midpoint",
                "rows",
                "goals",
                "goal_rate",
            ]
        )
    q = min(bins, valid[column].nunique())
    valid["_bin"] = pd.qcut(valid[column], q=q, duplicates="drop")
    grouped = (
        valid.groupby("_bin", observed=True)
        .agg(
            rows=(TARGET_COLUMN, "count"),
            goals=(TARGET_COLUMN, "sum"),
            goal_rate=(TARGET_COLUMN, "mean"),
            bin_min=(column, "min"),
            bin_max=(column, "max"),
        )
        .reset_index()
    )
    grouped["bin_label"] = grouped["_bin"].astype(str)
    grouped["bin_midpoint"] = (grouped["bin_min"] + grouped["bin_max"]) / 2
    return grouped[
        ["bin_label", "bin_min", "bin_max", "bin_midpoint", "rows", "goals", "goal_rate"]
    ]


def _categorical_goal_rate(
    frame: pd.DataFrame, column: str, *, min_slice_size: int
) -> pd.DataFrame:
    table = (
        frame.assign(_value=frame[column].fillna("missing").astype(str))
        .groupby("_value", observed=True)[TARGET_COLUMN]
        .agg(rows="count", goals="sum", goal_rate="mean")
        .reset_index()
        .rename(columns={"_value": "value"})
    )
    return table[table["rows"] >= min_slice_size].sort_values(
        ["goal_rate", "rows"], ascending=[False, False]
    )


def _feature_profile_row(series: pd.Series, column: str) -> dict[str, object]:
    return {
        "column": column,
        "dtype": str(series.dtype),
        "missing_rate": float(series.isna().mean()),
        "unique_values": int(series.nunique(dropna=True)),
        "recommendation": _quality_recommendation(series),
    }


def _numeric_profile_row(series: pd.Series, column: str) -> dict[str, object]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    return {
        "column": column,
        "rows": int(len(numeric)),
        "missing_rate": float(series.isna().mean()),
        "mean": float(numeric.mean()) if not numeric.empty else np.nan,
        "median": float(numeric.median()) if not numeric.empty else np.nan,
        "std": float(numeric.std()) if len(numeric) > 1 else np.nan,
        "p01": float(numeric.quantile(0.01)) if not numeric.empty else np.nan,
        "p99": float(numeric.quantile(0.99)) if not numeric.empty else np.nan,
        "skew": float(numeric.skew()) if len(numeric) > 2 else np.nan,
        "recommendation": _quality_recommendation(series),
    }


def _quality_recommendation(series: pd.Series) -> str:
    missing_rate = float(series.isna().mean())
    unique_values = int(series.nunique(dropna=True))
    if missing_rate >= 0.5:
        return "exclude_or_backfill"
    if missing_rate >= 0.1:
        return "impute_and_monitor"
    if pd.api.types.is_numeric_dtype(series) and unique_values > 20:
        skew = series.dropna().skew()
        if pd.notna(skew) and abs(float(skew)) >= 1.5:
            return "transform_or_bin"
    if not pd.api.types.is_numeric_dtype(series) and unique_values > 12:
        return "encode_with_rare_level_handling"
    return "keep"


def _target_status(row_count: int, goal_rate: float) -> str:
    if row_count < 100:
        return "sample_size_warning"
    if goal_rate < 0.04 or goal_rate > 0.35:
        return "imbalance_warning"
    return "usable"


def _pressure_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if "pressure" in column.lower() or column in {"under_pressure", "pressure_state"}
    ]


def _distribution_takeaway(series: pd.Series, column: str) -> str:
    if series.empty:
        return f"{column} has no non-missing numeric values."
    return (
        f"Median {series.median():.2f}; p1-p99 range "
        f"{series.quantile(0.01):.2f} to {series.quantile(0.99):.2f}."
    )


def _outlier_note(series: pd.Series) -> str:
    if series.empty:
        return ""
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    if iqr <= 0:
        return ""
    outliers = (series < q1 - 3 * iqr) | (series > q3 + 3 * iqr)
    if outliers.any():
        return f" Outlier note: {int(outliers.sum()):,} values exceed 3*IQR; review winsorisation."
    return " Outlier note: no extreme 3*IQR outliers flagged."


def _goal_rate_takeaway(table: pd.DataFrame, column: str) -> str:
    if table.empty:
        return f"{column} has too little variation for a binned target relationship."
    low = table.iloc[0]
    high = table.iloc[-1]
    return (
        f"Goal rate moves from {low['goal_rate']:.1%} in the first bin to "
        f"{high['goal_rate']:.1%} in the last bin."
    )


def _relationship_interpretation(table: pd.DataFrame, column: str) -> str:
    if table.empty:
        return f"{column} could not be binned with the available data."
    spread = table["goal_rate"].max() - table["goal_rate"].min()
    return f"{column} shows a {spread:.1%} goal-rate spread across bins."


def _categorical_takeaway(table: pd.DataFrame, column: str) -> str:
    if table.empty:
        return f"No {column} levels meet the minimum sample size."
    top = table.iloc[0]
    bottom = table.iloc[-1]
    return (
        f"{column} ranges from {bottom['goal_rate']:.1%} to {top['goal_rate']:.1%} "
        "among reported levels."
    )


def _pressure_takeaway(table: pd.DataFrame) -> str:
    if table.empty:
        return "No pressure relationship could be computed from available columns."
    return f"Pressure bins/states span {table['goal_rate'].min():.1%} to {table['goal_rate'].max():.1%} goal rate."


def _redundancy_implication(corr_value: float | np.floating | None, left: str, right: str) -> str:
    if pd.notna(corr_value) and abs(float(corr_value)) >= 0.8:
        return f"Review whether both {left} and {right} are needed or regularise strongly."
    return f"Keep both {left} and {right} available unless model diagnostics say otherwise."


def _slice_implication(delta: float) -> str:
    if abs(delta) >= 0.05:
        return "Use this slice for validation, monitoring, and possible interaction checks."
    if abs(delta) >= 0.03:
        return "Monitor this slice during validation."
    return "No immediate slice-specific modelling action."


def _value_check_implication(issue_count: int) -> str:
    if issue_count:
        return "Clean or exclude invalid values before model training."
    return "No football-specific invalid values found."


def _slug(value: str) -> str:
    return value.lower().replace(" ", "_").replace("/", "_").replace("-", "_").replace("__", "_")


def _empty_summary(name: str) -> dict[str, object]:
    return {"table": "", "visual": "", "interpretation": f"{name} was not available."}


def _render_report(
    *,
    target: dict[str, object],
    distributions: dict[str, object],
    relationships: dict[str, object],
    correlations: dict[str, object],
    stability: dict[str, object],
    quality: dict[str, object],
    leakage: dict[str, object],
    min_slice_size: int,
) -> str:
    context = relationships["context"]
    possession = relationships["possession"]
    lines = [
        "# CxG Pre-Model Target and Feature Analysis",
        "",
        "This analysis sits between feature engineering and model training. It uses "
        "`shot_features` joined to `shots`; provider xG is treated only as an external "
        "benchmark/reference variable when present.",
        "",
        _section(
            "1. Target usability",
            {
                "Question": "Is the binary goal target usable for CxG training?",
                "Calculation": (
                    f"{target['row_count']} shots, {target['goal_count']} goals, "
                    f"goal rate {float(target['goal_rate']):.3f}."
                ),
                "Visual/Table": f"{target['table']} and {target['visual']}.",
                "Interpretation": f"Target status is {target['status']}.",
                "Modelling implication": "Proceed with binary target modelling if validation remains stratified.",
                "Limitation": "This does not measure out-of-sample calibration.",
            },
        ),
        _section(
            "2. Target imbalance",
            {
                "Question": "Is the positive goal class imbalanced enough to affect training?",
                "Calculation": "The goal rate is compared with simple positive-class reference thresholds.",
                "Visual/Table": f"{target['visual']}.",
                "Interpretation": f"The observed positive class rate is {float(target['goal_rate']):.1%}.",
                "Modelling implication": "Use stratified splits and report calibration plus threshold-free metrics.",
                "Limitation": "Class-balance thresholds are heuristics, not performance guarantees.",
            },
        ),
        _section(
            "3. Feature distribution findings",
            {
                "Question": "Which features need scaling, transformation, binning, encoding, or monitoring?",
                "Calculation": (
                    f"{distributions['feature_count']} candidate features were profiled; "
                    f"{distributions['numeric_feature_count']} numeric features have individual plots."
                ),
                "Visual/Table": (
                    f"{distributions['numeric_profile']} plus per-feature plots under "
                    "`01_feature_distributions/plots/`."
                ),
                "Interpretation": "Distribution plots identify skew, sparse fields, and outlier candidates.",
                "Modelling implication": "Apply preprocessing decisions before fitting CxG models.",
                "Limitation": "Distribution shape does not prove target signal.",
            },
        ),
        _section(
            "4. Shot geometry signal",
            {
                "Question": (
                    "Does shot geometry show pre-model target signal? "
                    "Subquestions: shot distance, shot angle, centrality, and distance to goal line."
                ),
                "Calculation": "Geometry features are binned and compared by observed goal rate.",
                "Visual/Table": (
                    f"{relationships['shot_distance']['table']}; "
                    f"{relationships['shot_angle']['table']}; "
                    f"{relationships['table']}."
                ),
                "Interpretation": (
                    f"Distance: {relationships['shot_distance']['interpretation']} "
                    f"Angle: {relationships['shot_angle']['interpretation']}"
                ),
                "Modelling implication": "Keep geometry features as core CxG candidates and test redundancy.",
                "Limitation": "Binned rates are univariate and do not control for shot context.",
            },
        ),
        _section(
            "5. Shot context signal",
            {
                "Question": (
                    "Do body part, technique, shot type, first time, blocked, and play pattern "
                    "separate goal probability?"
                ),
                "Calculation": "Each available context field receives its own goal-rate table and chart.",
                "Visual/Table": _report_paths(context),
                "Interpretation": _report_interpretations(context),
                "Modelling implication": "Encode stable context categories and pool rare levels.",
                "Limitation": f"Levels below {min_slice_size} rows are excluded from context plots.",
            },
        ),
        _section(
            "6. Possession context signal",
            {
                "Question": (
                    "Do possession length, duration, previous action gap, recent defensive actions, "
                    "and pressure proxy contain target signal?"
                ),
                "Calculation": "Available possession-context features are binned and compared by goal rate.",
                "Visual/Table": _report_paths(possession),
                "Interpretation": _report_interpretations(possession),
                "Modelling implication": "Keep possession-context fields that show stable signal after validation.",
                "Limitation": "Possession features may be missing or proxy tactical state imperfectly.",
            },
        ),
        _section(
            "7. Feature redundancy",
            {
                "Question": "Which candidate features are redundant before modelling?",
                "Calculation": "Pearson correlations plus targeted football feature-pair checks.",
                "Visual/Table": f"{correlations['table']}; {correlations['targeted_table']}; {correlations['visual']}.",
                "Interpretation": f"{correlations['high_pairs']} high-correlation pairs were flagged.",
                "Modelling implication": "Drop, combine, or regularise redundant features during model design.",
                "Limitation": "Correlation is linear and does not capture nonlinear redundancy.",
            },
        ),
        _section(
            "8. Slice stability",
            {
                "Question": "Do feature-target relationships hold across slices?",
                "Calculation": (
                    f"Available slices with at least {min_slice_size} rows are compared with global goal rate."
                ),
                "Visual/Table": f"{stability['table']} and {stability['visual']}.",
                "Interpretation": f"{stability['unstable_slices']} slices exceed a 3pp reference delta.",
                "Modelling implication": "Use unstable slices for validation, monitoring, and interaction checks.",
                "Limitation": "Slice differences can reflect team, schedule, or competition mix.",
            },
        ),
        _section(
            "9. Data quality and cleaning recommendations",
            {
                "Question": "Which features need cleaning before modelling?",
                "Calculation": "Missingness, rare levels, constants, outliers, and football value checks are applied.",
                "Visual/Table": f"{quality['football_checks']} and {quality['recommendations']}.",
                "Interpretation": f"{quality['action_count']} cleaning recommendations were produced.",
                "Modelling implication": "Resolve invalid values and encode cleaning rules before training.",
                "Limitation": "Automated checks need domain review before permanent exclusions.",
            },
        ),
        _section(
            "10. Leakage risks and training eligibility",
            {
                "Question": "Which columns must be excluded before training?",
                "Calculation": "Columns are grouped as target, outcome, benchmark, post-model, ID, or safe candidate.",
                "Visual/Table": f"{leakage['table']}.",
                "Interpretation": "Non-candidate columns: "
                + ", ".join(leakage["risky_columns"][:20]),
                "Modelling implication": "Train only on safe candidate features; use provider xG as benchmark only.",
                "Limitation": "Name-based checks cannot prove semantic safety for every future column.",
            },
        ),
        _section(
            "11. Modelling recommendations",
            {
                "Question": "What should the modelling layer do next?",
                "Calculation": "Combine target, feature signal, redundancy, stability, quality, and leakage evidence.",
                "Visual/Table": "Use all numbered artifact folders under `outputs/analysis/cxg/`.",
                "Interpretation": (
                    "The analysis supports a pre-model feature contract before any CxG predictions are produced."
                ),
                "Modelling implication": (
                    "Build training inputs from safe candidates, encode cleaning rules, stratify validation, "
                    "benchmark against provider xG, and monitor unstable slices."
                ),
                "Limitation": "This report does not train, score, aggregate, or publish CxG predictions.",
            },
        ),
    ]
    return "\n".join(lines) + "\n"


def _section(title: str, values: dict[str, str]) -> str:
    lines = [f"## {title}", ""]
    for key in REQUIRED_SECTIONS:
        lines.append(f"**{key}:** {values[key]}")
        lines.append("")
    return "\n".join(lines)


def _report_paths(outputs: object) -> str:
    if not isinstance(outputs, dict) or not outputs:
        return "No optional columns were available for this section."
    return "; ".join(
        str(value["table"])
        for value in outputs.values()
        if isinstance(value, dict) and value.get("table")
    )


def _report_interpretations(outputs: object) -> str:
    if not isinstance(outputs, dict) or not outputs:
        return "No optional columns were available for this section."
    return " ".join(
        str(value["interpretation"])
        for value in outputs.values()
        if isinstance(value, dict) and value.get("interpretation")
    )
