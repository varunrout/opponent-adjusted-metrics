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

LEAKAGE_PATTERNS = (
    "goal",
    "outcome",
    "cxg",
    "prediction",
    "probability",
    "model",
    "registry",
    "aggregate",
    "neutral",
    "adjusted",
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
    quality = _data_quality(frame, folders["05_data_quality"])
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
            "figure.figsize": (10, 6),
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
    table.to_csv(folder / "target_summary.csv", index=False)

    fig, ax = plt.subplots()
    labels = ["Non-goal", "Goal"]
    values = [row_count - goal_count, goal_count]
    ax.bar(labels, values, color=["#6b7280", "#0f766e"])
    ax.axhline(max(row_count * 0.05, 1), color="#b45309", linestyle="--", label="5% reference")
    ax.set_title("Is the goal target usable and balanced enough for modelling?")
    ax.set_ylabel("Shot count")
    ax.legend()
    fig.savefig(folder / "target_balance.png")
    plt.close(fig)

    return {
        "row_count": row_count,
        "goal_count": goal_count,
        "goal_rate": goal_rate,
        "status": status,
        "visual": "00_target/target_balance.png",
        "table": "00_target/target_summary.csv",
    }


def _feature_distributions(frame: pd.DataFrame, folder: Path) -> dict[str, object]:
    feature_columns = _candidate_feature_columns(frame)
    profile_rows = []
    for column in feature_columns:
        series = frame[column]
        profile_rows.append(
            {
                "column": column,
                "dtype": str(series.dtype),
                "missing_rate": float(series.isna().mean()),
                "unique_values": int(series.nunique(dropna=True)),
                "recommendation": _quality_recommendation(series),
            }
        )
    profile = pd.DataFrame(profile_rows).sort_values(
        ["missing_rate", "column"], ascending=[False, True]
    )
    profile.to_csv(folder / "feature_missingness.csv", index=False)

    numeric = _numeric_features(frame, feature_columns)[:8]
    if numeric:
        cols = min(4, len(numeric))
        rows = int(np.ceil(len(numeric) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
        axes_array = np.atleast_1d(axes).ravel()
        for ax, column in zip(axes_array, numeric):
            frame[column].dropna().hist(ax=ax, bins=20, color="#2563eb", alpha=0.8)
            ax.axvline(frame[column].median(), color="#b45309", linestyle="--", label="Median")
            ax.set_title(f"Does {column} need scaling or binning?")
            ax.legend()
        for ax in axes_array[len(numeric) :]:
            ax.axis("off")
        fig.savefig(folder / "numeric_distributions.png")
        plt.close(fig)

    categorical_rows = []
    for column in _categorical_features(frame, feature_columns):
        counts = frame[column].fillna("missing").astype(str).value_counts().head(10)
        for value, count in counts.items():
            categorical_rows.append({"column": column, "value": value, "rows": int(count)})
    pd.DataFrame(categorical_rows).to_csv(folder / "categorical_top_levels.csv", index=False)
    return {
        "feature_count": len(feature_columns),
        "high_missing": profile.loc[profile["missing_rate"] >= 0.25, "column"].tolist(),
        "table": "01_feature_distributions/feature_missingness.csv",
        "visual": "01_feature_distributions/numeric_distributions.png",
    }


def _feature_target_relationships(
    frame: pd.DataFrame,
    folder: Path,
    *,
    min_slice_size: int,
) -> dict[str, object]:
    feature_columns = _candidate_feature_columns(frame)
    numeric_rows = []
    for column in _numeric_features(frame, feature_columns):
        valid = frame[[column, TARGET_COLUMN]].dropna()
        if valid[column].nunique() < 2:
            continue
        corr = valid[column].corr(valid[TARGET_COLUMN])
        quartiles = pd.qcut(valid[column], q=min(4, valid[column].nunique()), duplicates="drop")
        rates = valid.groupby(quartiles, observed=True)[TARGET_COLUMN].agg(["count", "mean"])
        numeric_rows.append(
            {
                "column": column,
                "rows": len(valid),
                "pearson_with_goal": float(corr) if pd.notna(corr) else np.nan,
                "min_bin_goal_rate": float(rates["mean"].min()),
                "max_bin_goal_rate": float(rates["mean"].max()),
                "goal_rate_spread": float(rates["mean"].max() - rates["mean"].min()),
            }
        )
    numeric_table = pd.DataFrame(numeric_rows).sort_values("goal_rate_spread", ascending=False)
    numeric_table.to_csv(folder / "numeric_target_relationships.csv", index=False)

    cat_rows = []
    for column in _categorical_features(frame, feature_columns):
        grouped = (
            frame.assign(_value=frame[column].fillna("missing").astype(str))
            .groupby("_value", observed=True)[TARGET_COLUMN]
            .agg(["count", "mean"])
            .reset_index()
        )
        grouped = grouped[grouped["count"] >= min_slice_size]
        for _, row in grouped.iterrows():
            cat_rows.append(
                {
                    "column": column,
                    "value": row["_value"],
                    "rows": int(row["count"]),
                    "goal_rate": float(row["mean"]),
                }
            )
    pd.DataFrame(cat_rows).to_csv(folder / "categorical_target_relationships.csv", index=False)

    top = numeric_table.head(8)
    if not top.empty:
        fig, ax = plt.subplots()
        ax.barh(top["column"], top["goal_rate_spread"], color="#0f766e")
        ax.axvline(0.03, color="#b45309", linestyle="--", label="3pp reference")
        ax.invert_yaxis()
        ax.set_title("Which engineered features separate goal outcomes before modelling?")
        ax.set_xlabel("Goal-rate spread across quartile bins")
        ax.legend()
        fig.savefig(folder / "numeric_target_relationships.png")
        plt.close(fig)

    return {
        "top_numeric_signal": top["column"].tolist(),
        "table": "02_feature_target_relationships/numeric_target_relationships.csv",
        "visual": "02_feature_target_relationships/numeric_target_relationships.png",
    }


def _feature_correlations(frame: pd.DataFrame, folder: Path) -> dict[str, object]:
    numeric = _numeric_features(frame, _candidate_feature_columns(frame))
    corr = frame[numeric].corr(numeric_only=True) if len(numeric) >= 2 else pd.DataFrame()
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
    high.to_csv(folder / "high_correlations.csv", index=False)

    if not corr.empty:
        fig, ax = plt.subplots(figsize=(max(8, len(corr) * 0.65), max(6, len(corr) * 0.55)))
        image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(corr.index)), corr.index)
        ax.set_title("Which candidate features may be redundant before model training?")
        fig.colorbar(image, ax=ax, label="Pearson correlation")
        fig.savefig(folder / "correlation_heatmap.png")
        plt.close(fig)

    return {
        "high_pairs": len(high),
        "table": "03_feature_correlations/high_correlations.csv",
        "visual": "03_feature_correlations/correlation_heatmap.png",
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
            "shot_type",
            "body_part",
            "minute_bucket",
            "is_leading",
            "is_trailing",
            "is_drawing",
        )
        if column in frame.columns
    ]
    rows = []
    global_rate = float(frame[TARGET_COLUMN].mean()) if len(frame) else 0.0
    for column in slice_columns:
        grouped = (
            frame.assign(_value=frame[column].fillna("missing").astype(str))
            .groupby("_value", observed=True)[TARGET_COLUMN]
            .agg(["count", "mean"])
            .reset_index()
        )
        grouped = grouped[grouped["count"] >= min_slice_size]
        if grouped.empty:
            continue
        for _, row in grouped.iterrows():
            rows.append(
                {
                    "slice_column": column,
                    "slice_value": row["_value"],
                    "rows": int(row["count"]),
                    "goal_rate": float(row["mean"]),
                    "goal_rate_delta": float(row["mean"] - global_rate),
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(folder / "slice_stability.csv", index=False)

    if not table.empty:
        plot = table.reindex(
            table["goal_rate_delta"].abs().sort_values(ascending=False).index
        ).head(12)
        fig, ax = plt.subplots()
        labels = plot["slice_column"] + "=" + plot["slice_value"]
        ax.barh(labels, plot["goal_rate_delta"], color="#7c3aed")
        ax.axvline(0, color="#111827", linewidth=1)
        ax.axvline(0.03, color="#b45309", linestyle="--")
        ax.axvline(-0.03, color="#b45309", linestyle="--", label="3pp reference")
        ax.invert_yaxis()
        ax.set_title("Where does target behaviour shift across modelling slices?")
        ax.set_xlabel("Goal-rate delta from global rate")
        ax.legend()
        fig.savefig(folder / "slice_stability.png")
        plt.close(fig)

    return {
        "slice_columns": slice_columns,
        "unstable_slices": (
            int((table["goal_rate_delta"].abs() >= 0.03).sum()) if not table.empty else 0
        ),
        "table": "04_slice_stability/slice_stability.csv",
        "visual": "04_slice_stability/slice_stability.png",
    }


def _data_quality(frame: pd.DataFrame, folder: Path) -> dict[str, object]:
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
    table.to_csv(folder / "data_quality.csv", index=False)
    table[table["recommendation"] != "keep"].to_csv(
        folder / "cleaning_recommendations.csv",
        index=False,
    )
    return {
        "action_count": int((table["recommendation"] != "keep").sum()),
        "table": "05_data_quality/data_quality.csv",
    }


def _leakage_checks(frame: pd.DataFrame, folder: Path) -> dict[str, object]:
    rows = []
    for column in frame.columns:
        lower = column.lower()
        if column in {"outcome", TARGET_COLUMN}:
            risk = "target_definition"
            recommendation = "exclude_from_features"
        elif column in BENCHMARK_COLUMNS:
            risk = "external_benchmark"
            recommendation = "use_only_for_reference"
        elif any(pattern in lower for pattern in LEAKAGE_PATTERNS):
            risk = "possible_post_model_or_target_leakage"
            recommendation = "exclude_or_review_before_training"
        else:
            risk = "none"
            recommendation = "eligible_if_quality_checks_pass"
        rows.append({"column": column, "risk": risk, "recommendation": recommendation})
    table = pd.DataFrame(rows)
    table.to_csv(folder / "leakage_checks.csv", index=False)
    risk_count = int((table["risk"] != "none").sum())
    return {
        "risk_count": risk_count,
        "risky_columns": table.loc[table["risk"] != "none", "column"].tolist(),
        "table": "06_leakage_checks/leakage_checks.csv",
    }


def _candidate_feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {
        TARGET_COLUMN,
        "shot_id",
        "id",
        "event_id",
        "match_id",
        "team_id",
        "player_id",
        "opponent_team_id",
        "outcome",
    }
    excluded.update(BENCHMARK_COLUMNS)
    return [column for column in frame.columns if column not in excluded]


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
    lines = [
        "# Pre-model CxG target and feature study",
        "",
        "This analysis sits between feature engineering and model training. It uses "
        "`shot_features` joined to `shots`; provider xG is treated only as an external "
        "benchmark/reference variable when present.",
        "",
        _section(
            "1. Is the target usable?",
            {
                "Question": "Does the goal target have enough observations and positive cases?",
                "Calculation": (
                    f"{target['row_count']} shots, {target['goal_count']} goals, "
                    f"goal rate {float(target['goal_rate']):.3f}."
                ),
                "Visual/Table": f"{target['table']} and {target['visual']}.",
                "Interpretation": f"Target status: {target['status']}.",
                "Modelling implication": "Use binary goal as the CxG target if sample size is acceptable.",
                "Limitation": "This does not estimate out-of-sample model calibration.",
            },
        ),
        _section(
            "2. Is the target imbalanced?",
            {
                "Question": "Is the positive goal class rare enough to need special handling?",
                "Calculation": "Goal share is compared with simple reference thresholds.",
                "Visual/Table": f"{target['visual']}.",
                "Interpretation": (
                    "Imbalance handling is recommended when the goal rate is very low or very high."
                ),
                "Modelling implication": "Track stratified splits, calibration, and class-sensitive metrics.",
                "Limitation": "Thresholds are heuristics and should be revisited on larger samples.",
            },
        ),
        _section(
            "3. Which features show signal against the goal target?",
            {
                "Question": "Which engineered features separate goal and non-goal outcomes?",
                "Calculation": "Numeric features are ranked by goal-rate spread across quantile bins.",
                "Visual/Table": f"{relationships['table']} and {relationships['visual']}.",
                "Interpretation": (
                    "Top numeric signals: " + ", ".join(relationships["top_numeric_signal"][:5])
                    if relationships["top_numeric_signal"]
                    else "No numeric signal table was produced."
                ),
                "Modelling implication": "Prioritise stable signal features and benchmark them against provider xG.",
                "Limitation": "Univariate signal can disappear after controlling for correlated features.",
            },
        ),
        _section(
            "4. Which features are redundant or highly correlated?",
            {
                "Question": "Which candidate inputs duplicate each other before modelling?",
                "Calculation": "Pairwise Pearson correlations are flagged at absolute correlation >= 0.80.",
                "Visual/Table": f"{correlations['table']} and {correlations['visual']}.",
                "Interpretation": f"{correlations['high_pairs']} high-correlation pairs were flagged.",
                "Modelling implication": "Consider dropping, combining, or regularising redundant features.",
                "Limitation": "Correlation only captures linear numeric relationships.",
            },
        ),
        _section(
            "5. Which relationships are stable or unstable across slices?",
            {
                "Question": "Where does the target rate shift across common modelling slices?",
                "Calculation": (
                    f"Slices with at least {min_slice_size} rows are compared with the global goal rate."
                ),
                "Visual/Table": f"{stability['table']} and {stability['visual']}.",
                "Interpretation": f"{stability['unstable_slices']} slices exceed the reference delta.",
                "Modelling implication": "Use unstable slices for validation reporting and monitoring.",
                "Limitation": "Small or missing optional slice columns are skipped.",
            },
        ),
        _section(
            "6. Which features need cleaning, transformation, binning, encoding, exclusion, or monitoring?",
            {
                "Question": "Which candidate columns need preprocessing decisions before training?",
                "Calculation": "Missingness, cardinality, and numeric skewness drive recommendations.",
                "Visual/Table": f"{quality['table']} and {distributions['table']}.",
                "Interpretation": f"{quality['action_count']} features need a non-keep action.",
                "Modelling implication": "Encode categoricals, impute monitored fields, and transform skewed inputs.",
                "Limitation": "Automated recommendations need domain review before exclusions.",
            },
        ),
        _section(
            "7. Which columns are leakage risks?",
            {
                "Question": "Which columns should not enter training as ordinary features?",
                "Calculation": "Target, outcome, post-model naming patterns, and benchmarks are flagged.",
                "Visual/Table": f"{leakage['table']}.",
                "Interpretation": "Risk columns: " + ", ".join(leakage["risky_columns"]),
                "Modelling implication": (
                    "Exclude target/outcome/post-model columns; keep provider xG only as a benchmark."
                ),
                "Limitation": "Name-based leakage checks cannot prove semantic safety.",
            },
        ),
        _section(
            "8. What should the modelling layer do next?",
            {
                "Question": "What decisions should CxG training consume from this analysis?",
                "Calculation": "Combine target usability, signal, redundancy, stability, quality, and leakage results.",
                "Visual/Table": "See the numbered artifact folders under `outputs/analysis/cxg/`.",
                "Interpretation": (
                    "Train only after feature cleaning and leakage exclusions are encoded in the modelling layer."
                ),
                "Modelling implication": (
                    "Build a pre-model feature contract, stratified validation plan, benchmark comparison, "
                    "and slice monitoring before producing CxG predictions."
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
