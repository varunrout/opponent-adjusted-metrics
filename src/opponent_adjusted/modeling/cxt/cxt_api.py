"""
CxT API - Final Integration Module.

Provides clean interface for CxT predictions with:
- Opponent adjustment (built into model)
- Player aggregation
- Team aggregation
- Match-level summaries
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import logging

logger = logging.getLogger(__name__)


@dataclass
class CxTResult:
    """CxT prediction result."""
    
    cxt: float  # Expected xT value (opponent-adjusted)
    p_complete: float  # Probability of completion
    xt_if_complete: float  # Expected xT gain if completed
    raw_xt: float  # Original static xT delta
    opponent_adjustment: float  # Adjustment factor from opponent context
    
    @property
    def above_expectation(self) -> float:
        """How much actual outcome exceeded expectation."""
        return self.raw_xt - self.cxt


@dataclass  
class PlayerCxTSummary:
    """Player-level CxT aggregation."""
    
    player_id: int
    player_name: str | None
    n_actions: int
    total_cxt: float
    mean_cxt: float
    total_actual_xt: float
    xt_vs_expected: float  # positive = overperformer
    completion_rate: float
    progressive_pct: float
    
    
@dataclass
class TeamCxTSummary:
    """Team-level CxT aggregation."""
    
    team_id: int
    team_name: str | None
    n_actions: int
    total_cxt: float
    mean_cxt: float
    cxt_per_90: float | None
    total_actual_xt: float
    xt_vs_expected: float
    top_contributors: list[PlayerCxTSummary]


class CxTPredictor:
    """Main interface for CxT predictions."""
    
    def __init__(self, model_dir: Path | str | None = None):
        """
        Initialize predictor with trained model.
        
        Args:
            model_dir: Path to model directory. If None, uses latest model.
        """
        from opponent_adjusted.modeling.cxt.contextual_model import CxTModel
        
        if model_dir is None:
            # Default to latest model
            self.model_dir = (
                Path(__file__).resolve().parents[3] / 
                "outputs" / "modeling" / "cxt" / "latest"
            )
        else:
            self.model_dir = Path(model_dir)
        
        self.model = CxTModel.load(self.model_dir)
        logger.info(f"Loaded CxT model from {self.model_dir}")
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate CxT predictions for a dataframe of progressions.
        
        Args:
            df: DataFrame with required features (from feature pipeline)
            
        Returns:
            DataFrame with prediction columns added:
            - cxt: Expected threat value
            - p_complete: Completion probability
            - xt_if_complete: Expected xT gain if completed
            - opponent_adj: Opponent adjustment factor
        """
        df = df.copy()
        
        # Get predictions
        df["p_complete"] = self.model.predict_completion_prob(df)
        df["xt_if_complete"] = self.model.predict_xt_gain(df)
        df["cxt"] = self.model.predict_cxt(df)
        
        # Calculate opponent adjustment factor
        # This is implicit in the model but we can estimate it
        if "opponent_zone_rating" in df.columns:
            # Higher rating = weaker opponent = higher expected xT
            baseline = 50  # Neutral opponent rating
            df["opponent_adj"] = (100 - df["opponent_zone_rating"]) / baseline
        else:
            df["opponent_adj"] = 1.0
        
        return df
    
    def predict_single(
        self,
        start_x: float,
        start_y: float,
        end_x: float,
        end_y: float,
        action_type: str,
        under_pressure: bool = False,
        minute: float = 45.0,
        opponent_rating: float = 50.0,
        **kwargs,
    ) -> CxTResult:
        """
        Predict CxT for a single action.
        
        Args:
            start_x, start_y: Start location (0-120, 0-80)
            end_x, end_y: End location (0-120, 0-80)
            action_type: 'pass', 'carry', or 'dribble'
            under_pressure: Whether action is under pressure
            minute: Match minute
            opponent_rating: Opponent zone defensive rating (0-100)
            **kwargs: Additional feature overrides
            
        Returns:
            CxTResult with predictions
        """
        from opponent_adjusted.features.cxt.xt_model import get_xt_value
        
        # Calculate xT values
        start_xt = get_xt_value(start_x, start_y)
        end_xt = get_xt_value(end_x, end_y)
        xt_delta = end_xt - start_xt
        
        # Build feature row
        row = {
            "start_xt": start_xt,
            "xt_delta": xt_delta,
            "minute_normalized": minute / 90,
            "opponent_zone_rating": opponent_rating,
            "opponent_global_rating": opponent_rating,
            "opponent_global_block_rate": 0.1,
            "opponent_zone_block_rate": 0.1,
            "under_pressure": under_pressure,
            "action_type": action_type,
            "is_pass": action_type == "pass",
            "is_carry": action_type == "carry", 
            "is_dribble": action_type == "dribble",
            "start_third": "MID" if start_x < 80 else ("ATT" if start_x >= 80 else "DEF"),
            "macro_zone_start": "4",  # Default central zone
            "is_progressive": end_x > start_x + 10,
            "is_into_final_third": start_x < 80 <= end_x,
            "is_into_penalty_area": start_x < 102 <= end_x and 18 <= end_y <= 62,
            "is_late_game": minute >= 75,
            "is_early_game": minute < 45,
            "is_first_half": minute <= 45,
            "is_second_half": 45 < minute <= 90,
            "is_extra_time": minute > 90,
            "is_very_late": minute >= 85,
            "start_is_central": 20 <= start_y <= 60,
            "moved_to_att_third": False,
            "moved_wide_to_central": False,
            "zone_changed": True,
            "pressure_flag": under_pressure,
            "opponent_is_strong": opponent_rating >= 60,
            "opponent_is_weak": opponent_rating < 40,
        }
        row.update(kwargs)
        
        df = pd.DataFrame([row])
        
        p_complete = self.model.predict_completion_prob(df)[0]
        xt_if_complete = self.model.predict_xt_gain(df)[0]
        cxt = self.model.predict_cxt(df)[0]
        
        opponent_adj = (100 - opponent_rating) / 50
        
        return CxTResult(
            cxt=cxt,
            p_complete=p_complete,
            xt_if_complete=xt_if_complete,
            raw_xt=xt_delta,
            opponent_adjustment=opponent_adj,
        )
    
    def aggregate_by_player(
        self,
        df: pd.DataFrame,
        player_col: str = "player_id",
        player_name_col: str | None = "player_name",
    ) -> list[PlayerCxTSummary]:
        """
        Aggregate CxT predictions by player.
        
        Args:
            df: DataFrame with predictions (call predict() first)
            player_col: Column name for player ID
            player_name_col: Column name for player name (optional)
            
        Returns:
            List of PlayerCxTSummary sorted by total_cxt descending
        """
        if "cxt" not in df.columns:
            df = self.predict(df)
        
        # Ensure success column
        if "success" not in df.columns:
            if "action_success" in df.columns:
                df["success"] = df["action_success"].astype(int)
            else:
                df["success"] = 1
        
        # Calculate actual xT (0 for failures)
        df["actual_xt"] = np.where(df["success"] == 1, df["xt_delta"], 0)
        
        # Progressive flag
        if "is_progressive" not in df.columns:
            df["is_progressive"] = (df.get("xt_delta", 0) > 0.01)
        
        results = []
        for pid, group in df.groupby(player_col):
            name = group[player_name_col].iloc[0] if player_name_col in group.columns else None
            
            summary = PlayerCxTSummary(
                player_id=int(pid),
                player_name=name,
                n_actions=len(group),
                total_cxt=float(group["cxt"].sum()),
                mean_cxt=float(group["cxt"].mean()),
                total_actual_xt=float(group["actual_xt"].sum()),
                xt_vs_expected=float(group["actual_xt"].sum() - group["cxt"].sum()),
                completion_rate=float(group["success"].mean()),
                progressive_pct=float(group["is_progressive"].mean()),
            )
            results.append(summary)
        
        return sorted(results, key=lambda x: x.total_cxt, reverse=True)
    
    def aggregate_by_team(
        self,
        df: pd.DataFrame,
        team_col: str = "team_id",
        team_name_col: str | None = "team_name",
        player_col: str = "player_id",
        minutes_played: dict[int, float] | None = None,
        top_n_players: int = 5,
    ) -> list[TeamCxTSummary]:
        """
        Aggregate CxT predictions by team.
        
        Args:
            df: DataFrame with predictions
            team_col: Column name for team ID
            team_name_col: Column name for team name
            player_col: Column for player aggregation
            minutes_played: Dict of team_id -> total minutes for per-90 calculation
            top_n_players: Number of top contributors to include
            
        Returns:
            List of TeamCxTSummary sorted by total_cxt descending
        """
        if "cxt" not in df.columns:
            df = self.predict(df)
        
        # Ensure success column
        if "success" not in df.columns:
            if "action_success" in df.columns:
                df["success"] = df["action_success"].astype(int)
            else:
                df["success"] = 1
        
        df["actual_xt"] = np.where(df["success"] == 1, df["xt_delta"], 0)
        
        results = []
        for tid, group in df.groupby(team_col):
            name = group[team_name_col].iloc[0] if team_name_col in group.columns else None
            
            # Player aggregation within team
            player_summaries = self.aggregate_by_player(group, player_col)
            top_contributors = player_summaries[:top_n_players]
            
            # Per-90 calculation
            cxt_per_90 = None
            if minutes_played and int(tid) in minutes_played:
                mins = minutes_played[int(tid)]
                if mins > 0:
                    cxt_per_90 = float(group["cxt"].sum()) * 90 / mins
            
            summary = TeamCxTSummary(
                team_id=int(tid),
                team_name=name,
                n_actions=len(group),
                total_cxt=float(group["cxt"].sum()),
                mean_cxt=float(group["cxt"].mean()),
                cxt_per_90=cxt_per_90,
                total_actual_xt=float(group["actual_xt"].sum()),
                xt_vs_expected=float(group["actual_xt"].sum() - group["cxt"].sum()),
                top_contributors=top_contributors,
            )
            results.append(summary)
        
        return sorted(results, key=lambda x: x.total_cxt, reverse=True)


def get_cxt_predictor(model_dir: Path | str | None = None) -> CxTPredictor:
    """
    Factory function to get CxT predictor.
    
    This is the main entry point for external usage.
    """
    return CxTPredictor(model_dir)


# Convenience function for quick predictions
def predict_cxt(df: pd.DataFrame, model_dir: Path | str | None = None) -> pd.DataFrame:
    """
    Quick CxT prediction on a dataframe.
    
    Args:
        df: DataFrame with required features
        model_dir: Optional model directory path
        
    Returns:
        DataFrame with cxt, p_complete, xt_if_complete columns added
    """
    predictor = get_cxt_predictor(model_dir)
    return predictor.predict(df)
