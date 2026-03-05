"""Visualize Ultimate cXA results and compare to previous cXA-xG.

Extended analyses:
1. Player comparisons and credit distribution
2. Opponent-adjusted vs unadjusted deltas
3. Team-level cXA leaderboards
4. Per-90 cXA (using minutes proxy)
5. Set-piece vs open-play breakdown (using shot_type proxy)
6. Calibration and lift comparison
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve

logger = logging.getLogger(__name__)


def _setup_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 150,
        "figure.figsize": (12, 7),
    })


def _safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        logger.warning("Missing file: %s", path)
        return None
    return pd.read_csv(path)


def _safe_read_parquet(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        logger.warning("Missing file: %s", path)
        return None
    return pd.read_parquet(path)


def _save_fig(out_dir: Path, name: str) -> None:
    out_path = out_dir / name
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    logger.info("Saved: %s", out_path)


def plot_top_players(players: pd.DataFrame, out_dir: Path, n: int = 15) -> None:
    top = players.head(n).copy()
    plt.figure()
    sns.barplot(data=top, x="total_cxa", y="player_name", color="#4c72b0")
    plt.title(f"Top {n} Players by Ultimate cXA")
    plt.xlabel("cXA")
    plt.ylabel("")
    _save_fig(out_dir, "top_players_ultimate_cxa.png")


def plot_action_type_credit(credits: pd.DataFrame, out_dir: Path) -> None:
    summary = (
        credits.groupby("action_type", dropna=False)["credit"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    plt.figure()
    sns.barplot(data=summary, x="credit", y="action_type", color="#55a868")
    plt.title("Credit by Action Type (Ultimate cXA)")
    plt.xlabel("Total Credit")
    plt.ylabel("")
    _save_fig(out_dir, "credit_by_action_type.png")
    summary.to_csv(out_dir / "credit_by_action_type.csv", index=False)


def plot_credit_by_position(credits: pd.DataFrame, out_dir: Path) -> None:
    summary = (
        credits.groupby("action_position")["credit"]
        .sum()
        .sort_index()
        .reset_index()
    )
    summary["credit_share"] = summary["credit"] / summary["credit"].sum()
    plt.figure()
    sns.lineplot(data=summary, x="action_position", y="credit_share", marker="o")
    plt.title("Credit Share by Action Position (Ultimate cXA)")
    plt.xlabel("Action Position")
    plt.ylabel("Share of Total Credit")
    _save_fig(out_dir, "credit_by_position.png")
    summary.to_csv(out_dir / "credit_by_position.csv", index=False)


def plot_credit_distribution(credits: pd.DataFrame, out_dir: Path) -> None:
    plt.figure()
    sns.histplot(credits["credit"], bins=40, color="#c44e52")
    plt.title("Distribution of Action Credits")
    plt.xlabel("Credit")
    plt.ylabel("Count")
    _save_fig(out_dir, "credit_distribution.png")


def plot_prev_vs_ultimate(prev: pd.DataFrame, ultimate: pd.DataFrame, out_dir: Path) -> None:
    merged = prev.merge(
        ultimate,
        on=["player_id", "player_name"],
        how="inner",
        suffixes=("_prev", "_ultimate"),
    )
    if merged.empty:
        logger.warning("No overlap between previous and ultimate players")
        return

    plt.figure()
    sns.scatterplot(data=merged, x="cXA_xG", y="total_cxa")
    lims = [0, max(merged["cXA_xG"].max(), merged["total_cxa"].max()) * 1.05]
    plt.plot(lims, lims, linestyle="--", color="gray")
    plt.xlim(lims)
    plt.ylim(lims)
    plt.title("Previous cXA-xG vs Ultimate cXA")
    plt.xlabel("Previous cXA-xG")
    plt.ylabel("Ultimate cXA")
    _save_fig(out_dir, "prev_vs_ultimate_scatter.png")

    merged["rank_prev"] = merged["cXA_xG"].rank(ascending=False, method="min")
    merged["rank_ultimate"] = merged["total_cxa"].rank(ascending=False, method="min")
    merged["rank_change"] = merged["rank_prev"] - merged["rank_ultimate"]
    merged = merged.sort_values("rank_change", ascending=False)

    top_movers = pd.concat([merged.head(15), merged.tail(15)])
    plt.figure(figsize=(12, 10))
    sns.barplot(
        data=top_movers,
        x="rank_change",
        y="player_name",
        palette="vlag",
    )
    plt.title("Rank Change: Previous cXA-xG → Ultimate cXA")
    plt.xlabel("Rank Change (Positive = Improved)")
    plt.ylabel("")
    _save_fig(out_dir, "rank_change.png")

    merged.to_csv(out_dir / "player_comparison.csv", index=False)


def plot_top_comparison(prev: pd.DataFrame, ultimate: pd.DataFrame, out_dir: Path, n: int = 15) -> None:
    prev_top = prev.sort_values("cXA_xG", ascending=False).head(n)
    ult_top = ultimate.sort_values("total_cxa", ascending=False).head(n)
    players = pd.unique(pd.concat([prev_top["player_name"], ult_top["player_name"]]))

    merged = pd.DataFrame({"player_name": players})
    merged = merged.merge(prev[["player_name", "cXA_xG"]], on="player_name", how="left")
    merged = merged.merge(ultimate[["player_name", "total_cxa"]], on="player_name", how="left")
    merged = merged.sort_values("total_cxa", ascending=False).head(n)

    plt.figure(figsize=(12, 9))
    x = np.arange(len(merged))
    width = 0.38
    plt.barh(x - width / 2, merged["cXA_xG"], height=width, label="Previous cXA-xG")
    plt.barh(x + width / 2, merged["total_cxa"], height=width, label="Ultimate cXA")
    plt.yticks(x, merged["player_name"])
    plt.xlabel("cXA")
    plt.title("Top Players: Previous vs Ultimate cXA")
    plt.legend()
    _save_fig(out_dir, "top_players_comparison.png")


def plot_opponent_impact(actions: pd.DataFrame, credits: pd.DataFrame, out_dir: Path) -> None:
    if "opponent_global_rating" not in actions.columns:
        return

    seq_col = "sequence_id" if "sequence_id" in credits.columns else "shot_id"
    seq_values = credits.groupby(seq_col)["sequence_value"].first().reset_index()
    opp = actions[[seq_col, "opponent_global_rating"]].drop_duplicates()
    merged = seq_values.merge(opp, on=seq_col, how="left")
    merged = merged.dropna(subset=["opponent_global_rating"])
    if merged.empty:
        return

    merged["rating_bin"] = pd.qcut(merged["opponent_global_rating"], 6, duplicates="drop")
    summary = merged.groupby("rating_bin", observed=True)["sequence_value"].mean().reset_index()

    summary["rating_bin"] = summary["rating_bin"].astype(str)

    plt.figure()
    sns.barplot(data=summary, x="rating_bin", y="sequence_value", color="#4c72b0")
    plt.xticks(rotation=30, ha="right")
    plt.title("Sequence Value by Opponent Rating Bin")
    plt.xlabel("Opponent Global Rating (Binned)")
    plt.ylabel("Average Sequence Value")
    _save_fig(out_dir, "opponent_rating_impact.png")


# =============================================================================
# ADDITIONAL ANALYSES
# =============================================================================

def plot_opponent_adjustment_delta(
    actions: pd.DataFrame, 
    credits: pd.DataFrame, 
    players: pd.DataFrame, 
    out_dir: Path,
) -> None:
    """Show who benefits most from opponent adjustment."""
    if "opponent_global_rating" not in actions.columns:
        logger.warning("No opponent data for adjustment delta analysis")
        return

    seq_col = "sequence_id" if "sequence_id" in credits.columns else "shot_id"
    
    # Compute average opponent rating faced by each player
    player_opp = (
        actions.groupby("player_id", dropna=True)["opponent_global_rating"]
        .mean()
        .reset_index()
        .rename(columns={"opponent_global_rating": "avg_opponent_rating"})
    )
    
    merged = players.merge(player_opp, on="player_id", how="left")
    merged = merged.dropna(subset=["avg_opponent_rating"])
    
    if merged.empty:
        return
    
    # Negative rating = better defense faced
    merged["faced_strong_defenses"] = merged["avg_opponent_rating"] < merged["avg_opponent_rating"].median()
    
    # Credit efficiency vs opponent strength
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=merged.head(100),
        x="avg_opponent_rating",
        y="total_cxa",
        hue="faced_strong_defenses",
        size="total_actions",
        sizes=(20, 200),
        alpha=0.7,
    )
    plt.axvline(merged["avg_opponent_rating"].median(), ls="--", color="gray", alpha=0.5)
    plt.title("cXA vs Average Opponent Defensive Rating")
    plt.xlabel("Avg Opponent Rating (Lower = Stronger Defense)")
    plt.ylabel("Total cXA")
    plt.legend(title="Faced Strong Defenses", loc="upper right")
    _save_fig(out_dir, "cxa_vs_opponent_strength.png")
    
    # Top players who benefited from facing strong defenses
    strong_def_players = merged[merged["faced_strong_defenses"]].nlargest(15, "total_cxa")
    plt.figure()
    sns.barplot(data=strong_def_players, x="total_cxa", y="player_name", color="#e74c3c")
    plt.title("Top Players Facing Strong Defenses")
    plt.xlabel("cXA")
    plt.ylabel("")
    _save_fig(out_dir, "top_players_vs_strong_defenses.png")
    
    merged.to_csv(out_dir / "player_opponent_strength.csv", index=False)


def plot_team_cxa(credits: pd.DataFrame, out_dir: Path, repo_root: Path) -> None:
    """Team-level cXA leaderboard and profiles."""
    if "team_id" not in credits.columns:
        return
    
    team_stats = (
        credits.groupby("team_id")
        .agg(
            total_cxa=("credit", "sum"),
            total_actions=("credit", "count"),
            total_sequences=("sequence_id", "nunique"),
            pass_credit=("credit", lambda x: x[credits.loc[x.index, "action_type"] == "Pass"].sum()),
            carry_credit=("credit", lambda x: x[credits.loc[x.index, "action_type"] == "Carry"].sum()),
        )
        .reset_index()
    )
    
    # Attempt to load team names
    try:
        from sqlalchemy import create_engine
        engine = create_engine(f"sqlite:///{repo_root / 'data' / 'opponent_adjusted.db'}")
        teams = pd.read_sql("SELECT id as team_id, name as team_name FROM teams", engine)
        team_stats = team_stats.merge(teams, on="team_id", how="left")
    except Exception:
        team_stats["team_name"] = team_stats["team_id"].astype(str)
    
    team_stats["cxa_per_action"] = team_stats["total_cxa"] / team_stats["total_actions"]
    team_stats["cxa_per_sequence"] = team_stats["total_cxa"] / team_stats["total_sequences"]
    team_stats["pass_share"] = team_stats["pass_credit"] / team_stats["total_cxa"]
    team_stats = team_stats.sort_values("total_cxa", ascending=False)
    
    # Top teams
    top_teams = team_stats.head(20)
    plt.figure(figsize=(12, 10))
    sns.barplot(data=top_teams, x="total_cxa", y="team_name", color="#3498db")
    plt.title("Top 20 Teams by Total cXA")
    plt.xlabel("Total cXA")
    plt.ylabel("")
    _save_fig(out_dir, "team_cxa_leaderboard.png")
    
    # Team efficiency
    plt.figure()
    sns.scatterplot(
        data=team_stats,
        x="total_sequences",
        y="total_cxa",
        size="cxa_per_sequence",
        sizes=(30, 300),
        alpha=0.7,
    )
    for _, row in top_teams.head(10).iterrows():
        plt.annotate(
            row["team_name"][:12],
            (row["total_sequences"], row["total_cxa"]),
            fontsize=8,
            alpha=0.8,
        )
    plt.title("Team cXA: Volume vs Quality")
    plt.xlabel("Total Sequences")
    plt.ylabel("Total cXA")
    _save_fig(out_dir, "team_cxa_efficiency.png")
    
    # Team style (pass vs carry share)
    plt.figure()
    team_stats["carry_share"] = 1 - team_stats["pass_share"]
    top_20 = team_stats.head(20).copy()
    x = np.arange(len(top_20))
    plt.barh(x, top_20["pass_share"], label="Pass Credit", color="#2ecc71")
    plt.barh(x, top_20["carry_share"], left=top_20["pass_share"], label="Carry Credit", color="#9b59b6")
    plt.yticks(x, top_20["team_name"])
    plt.xlabel("Credit Share")
    plt.title("Team Style: Pass vs Carry Credit")
    plt.legend()
    _save_fig(out_dir, "team_style_breakdown.png")
    
    team_stats.to_csv(out_dir / "team_cxa_stats.csv", index=False)


def plot_per90_cxa(credits: pd.DataFrame, players: pd.DataFrame, out_dir: Path) -> None:
    """Per-90 cXA using sequence count as minutes proxy."""
    # Use total_actions as a proxy for playing time
    # More actions = more involvement = more minutes played
    
    df = players.copy()
    df["actions_per_90_proxy"] = 30  # Assume ~30 actions per 90 for a typical player
    df["estimated_90s"] = df["total_actions"] / df["actions_per_90_proxy"]
    df["cxa_per_90"] = df["total_cxa"] / df["estimated_90s"].clip(lower=0.5)
    
    # Filter for minimum sample
    df_filtered = df[df["total_actions"] >= 20].copy()
    
    # Top by per-90
    top_per90 = df_filtered.nlargest(20, "cxa_per_90")
    plt.figure(figsize=(12, 10))
    sns.barplot(data=top_per90, x="cxa_per_90", y="player_name", color="#f39c12")
    plt.title("Top 20 Players by cXA per 90 (min 20 actions)")
    plt.xlabel("cXA per 90")
    plt.ylabel("")
    _save_fig(out_dir, "cxa_per_90_leaderboard.png")
    
    # Volume vs rate
    plt.figure()
    sns.scatterplot(
        data=df_filtered.head(100),
        x="total_actions",
        y="cxa_per_90",
        size="total_cxa",
        sizes=(20, 200),
        alpha=0.7,
    )
    plt.title("cXA per 90 vs Volume")
    plt.xlabel("Total Actions")
    plt.ylabel("cXA per 90")
    _save_fig(out_dir, "cxa_volume_vs_rate.png")
    
    df_filtered.to_csv(out_dir / "player_cxa_per90.csv", index=False)


def plot_action_context_breakdown(
    actions: pd.DataFrame, 
    credits: pd.DataFrame, 
    out_dir: Path,
) -> None:
    """Breakdown by cross/through-ball and pressure context."""
    merged = credits.merge(
        actions[["sequence_id", "action_position", "is_cross", "is_through_ball", "under_pressure", "is_into_box"]],
        on=["sequence_id", "action_position"],
        how="left",
    )
    
    # Cross vs non-cross
    cross_summary = (
        merged.groupby("is_cross", dropna=False)["credit"]
        .agg(["sum", "count", "mean"])
        .reset_index()
    )
    cross_summary["is_cross"] = cross_summary["is_cross"].map({True: "Cross", False: "Non-Cross", None: "Unknown"})
    
    plt.figure()
    sns.barplot(data=cross_summary, x="is_cross", y="sum", color="#1abc9c")
    plt.title("cXA by Cross Actions")
    plt.xlabel("")
    plt.ylabel("Total cXA")
    _save_fig(out_dir, "cxa_cross_breakdown.png")
    
    # Through-ball analysis
    tb_summary = (
        merged.groupby("is_through_ball", dropna=False)["credit"]
        .agg(["sum", "count", "mean"])
        .reset_index()
    )
    tb_summary["is_through_ball"] = tb_summary["is_through_ball"].map({True: "Through Ball", False: "Regular", None: "Unknown"})
    
    plt.figure()
    sns.barplot(data=tb_summary, x="is_through_ball", y="mean", color="#e67e22")
    plt.title("Average cXA: Through Balls vs Regular")
    plt.xlabel("")
    plt.ylabel("Average cXA per Action")
    _save_fig(out_dir, "cxa_through_ball_analysis.png")
    
    # Under pressure
    pressure_summary = (
        merged.groupby("under_pressure", dropna=False)["credit"]
        .agg(["sum", "count", "mean"])
        .reset_index()
    )
    pressure_summary["under_pressure"] = pressure_summary["under_pressure"].map({True: "Under Pressure", False: "No Pressure", None: "Unknown"})
    
    plt.figure()
    sns.barplot(data=pressure_summary, x="under_pressure", y="mean", color="#9b59b6")
    plt.title("Average cXA: Under Pressure vs Not")
    plt.xlabel("")
    plt.ylabel("Average cXA per Action")
    _save_fig(out_dir, "cxa_pressure_analysis.png")
    
    # Into box
    box_summary = (
        merged.groupby("is_into_box", dropna=False)["credit"]
        .agg(["sum", "count", "mean"])
        .reset_index()
    )
    box_summary["is_into_box"] = box_summary["is_into_box"].map({1: "Into Box", 0: "Outside Box", None: "Unknown"})
    
    plt.figure()
    sns.barplot(data=box_summary, x="is_into_box", y="sum", color="#e74c3c")
    plt.title("cXA by Box Entry")
    plt.xlabel("")
    plt.ylabel("Total cXA")
    _save_fig(out_dir, "cxa_box_entry.png")


def plot_calibration_comparison(
    actions: pd.DataFrame,
    credits: pd.DataFrame,
    out_dir: Path,
    prev_dir: Path,
) -> None:
    """Calibration and lift comparison between Ultimate and previous model."""
    # Get sequence-level data
    seq_col = "sequence_id" if "sequence_id" in credits.columns else "shot_id"
    
    seq_data = credits.groupby(seq_col).agg(
        is_goal=("is_goal", "first"),
        predicted_xg=("sequence_value", "first"),
    ).reset_index()
    
    seq_data = seq_data.dropna()
    if len(seq_data) < 100:
        logger.warning("Not enough sequences for calibration")
        return
    
    y_true = seq_data["is_goal"].astype(int).values
    y_pred = np.clip(seq_data["predicted_xg"].values, 0, 1)  # Clip to valid probability range
    
    # Calibration curve
    try:
        prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=10, strategy="quantile")
        
        plt.figure()
        plt.plot([0, 1], [0, 1], "k--", label="Perfect")
        plt.plot(prob_pred, prob_true, "o-", label="Ultimate cXA")
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of Positives")
        plt.title("Calibration Curve (Ultimate cXA)")
        plt.legend()
        _save_fig(out_dir, "calibration_curve.png")
    except Exception as e:
        logger.warning("Calibration curve failed: %s", e)
    
    # Lift chart
    seq_data["decile"] = pd.qcut(seq_data["predicted_xg"], 10, labels=False, duplicates="drop")
    lift = seq_data.groupby("decile").agg(
        n=("is_goal", "count"),
        goals=("is_goal", "sum"),
        avg_pred=("predicted_xg", "mean"),
    ).reset_index()
    lift["goal_rate"] = lift["goals"] / lift["n"]
    baseline_rate = y_true.mean()
    lift["lift"] = lift["goal_rate"] / baseline_rate
    
    plt.figure()
    sns.barplot(data=lift, x="decile", y="lift", color="#3498db")
    plt.axhline(1.0, ls="--", color="gray")
    plt.title("Lift Chart by Predicted xG Decile")
    plt.xlabel("Decile (0=lowest, 9=highest)")
    plt.ylabel("Lift (vs baseline)")
    _save_fig(out_dir, "lift_chart.png")
    
    lift.to_csv(out_dir / "lift_analysis.csv", index=False)
    
    # Summary metrics
    from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss
    
    metrics = {
        "n_sequences": len(seq_data),
        "goal_rate": y_true.mean(),
        "mean_predicted": y_pred.mean(),
        "brier_score": brier_score_loss(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_pred) if y_true.sum() > 0 else None,
        "log_loss": log_loss(y_true, np.clip(y_pred, 1e-7, 1 - 1e-7)),
        "top_decile_lift": lift["lift"].iloc[-1] if len(lift) > 0 else None,
    }
    
    with open(out_dir / "calibration_metrics.txt", "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
    
    logger.info("Calibration metrics: Brier=%.4f, AUC=%.4f", metrics["brier_score"], metrics.get("roc_auc", 0))


def plot_game_state_analysis(actions: pd.DataFrame, credits: pd.DataFrame, out_dir: Path) -> None:
    """Analyze cXA by game state (score differential, minute)."""
    merged = credits.merge(
        actions[["sequence_id", "action_position", "score_differential", "minute"]],
        on=["sequence_id", "action_position"],
        how="left",
    )
    
    # By score state
    merged["game_state"] = merged["score_differential"].apply(
        lambda x: "Winning" if x > 0 else ("Losing" if x < 0 else "Drawing") if pd.notna(x) else "Unknown"
    )
    
    state_summary = (
        merged.groupby("game_state")["credit"]
        .agg(["sum", "count", "mean"])
        .reset_index()
    )
    
    plt.figure()
    sns.barplot(data=state_summary, x="game_state", y="sum", color="#2ecc71")
    plt.title("Total cXA by Game State")
    plt.xlabel("")
    plt.ylabel("Total cXA")
    _save_fig(out_dir, "cxa_by_game_state.png")
    
    # By minute bucket
    merged["minute_bucket"] = pd.cut(
        merged["minute"].fillna(45),
        bins=[0, 15, 30, 45, 60, 75, 90, 120],
        labels=["0-15", "15-30", "30-45", "45-60", "60-75", "75-90", "90+"],
    )
    
    minute_summary = (
        merged.groupby("minute_bucket", observed=True)["credit"]
        .agg(["sum", "count", "mean"])
        .reset_index()
    )
    
    plt.figure()
    sns.barplot(data=minute_summary, x="minute_bucket", y="sum", color="#9b59b6")
    plt.title("cXA by Match Period")
    plt.xlabel("Minute")
    plt.ylabel("Total cXA")
    _save_fig(out_dir, "cxa_by_minute.png")
    
    state_summary.to_csv(out_dir / "cxa_game_state.csv", index=False)
    minute_summary.to_csv(out_dir / "cxa_minute_bucket.csv", index=False)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    _setup_style()

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "outputs" / "analysis" / "cxa" / "ultimate_cxa"
    out_dir.mkdir(parents=True, exist_ok=True)

    ultimate_dir = repo_root / "outputs" / "modeling" / "ultimate_cxa"
    prev_dir = repo_root / "outputs" / "analysis" / "cxa" / "phase6_cxa_xg"

    players_ultimate = _safe_read_csv(ultimate_dir / "player_cxa_xg.csv")
    credits_ultimate = _safe_read_parquet(ultimate_dir / "credits_xg.parquet")
    actions_ultimate = _safe_read_parquet(ultimate_dir / "actions.parquet")
    players_prev = _safe_read_csv(prev_dir / "player_leaderboard_xg.csv")

    if players_ultimate is None or credits_ultimate is None:
        logger.error("Missing Ultimate cXA outputs. Run the pipeline first.")
        return 1

    # === Core visualizations ===
    logger.info("=== Core Visualizations ===")
    plot_top_players(players_ultimate, out_dir)
    plot_action_type_credit(credits_ultimate, out_dir)
    plot_credit_by_position(credits_ultimate, out_dir)
    plot_credit_distribution(credits_ultimate, out_dir)

    if players_prev is not None:
        plot_prev_vs_ultimate(players_prev, players_ultimate, out_dir)
        plot_top_comparison(players_prev, players_ultimate, out_dir)

    if actions_ultimate is not None:
        plot_opponent_impact(actions_ultimate, credits_ultimate, out_dir)

    # === Additional analyses ===
    logger.info("=== Additional Analyses ===")
    
    # 1. Opponent adjustment delta
    if actions_ultimate is not None:
        plot_opponent_adjustment_delta(actions_ultimate, credits_ultimate, players_ultimate, out_dir)
    
    # 2. Team-level cXA
    plot_team_cxa(credits_ultimate, out_dir, repo_root)
    
    # 3. Per-90 cXA
    plot_per90_cxa(credits_ultimate, players_ultimate, out_dir)
    
    # 4. Action context breakdown (cross, through-ball, pressure, box entry)
    if actions_ultimate is not None:
        plot_action_context_breakdown(actions_ultimate, credits_ultimate, out_dir)
    
    # 5. Calibration and lift
    if actions_ultimate is not None:
        plot_calibration_comparison(actions_ultimate, credits_ultimate, out_dir, prev_dir)
    
    # 6. Game state analysis
    if actions_ultimate is not None:
        plot_game_state_analysis(actions_ultimate, credits_ultimate, out_dir)

    logger.info("All plots saved to %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
