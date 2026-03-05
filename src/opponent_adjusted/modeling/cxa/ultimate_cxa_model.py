"""Ultimate cXA Model - Opponent-Adjusted Contextual Expected Assists.

This is the production-ready cXA model that:
1. Uses True xA (target: is_goal, not shot_xg) to isolate passer contribution
2. Includes all action types (Pass, Carry, Dribble) 
3. Distributes credit via position decay + contribution weighting
4. Adjusts for opponent defensive quality (opponent profiles)
5. Incorporates game state context

Key improvements over baseline xA:
- Target: is_goal (binary) instead of shot_xg (conflates passer/shooter)
- Actions: Includes carries/dribbles (39% of pre-shot actions)
- Credit: Distributed across sequence, not just key pass
- Context: Opponent strength, game state, pressure
"""

from __future__ import annotations

import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
import joblib

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for the Ultimate cXA Model."""
    
    # Sequence settings
    max_actions: int = 5  # Max actions to consider before shot
    max_seconds: float = 15.0  # Max time window before shot
    
    # Credit distribution
    position_decay_rate: float = 0.5  # Exponential decay rate (e^(-rate * (pos-1)))
    contribution_weight: float = 0.3  # Weight for contribution features vs position
    temperature: float = 1.0  # Softmax temperature
    min_credit_share: float = 0.05  # Minimum credit share per action
    
    # Model settings
    model_type: str = "gbm"  # "lr" for logistic regression, "gbm" for gradient boosting
    cv_folds: int = 5
    random_state: int = 42
    
    # Opponent adjustment
    use_opponent_adjustment: bool = True
    opponent_shrinkage: float = 0.3  # How much to shrink toward average opponent
    opponent_profiles_version: Optional[str] = None  # Version tag for opponent profiles
    
    # Output modes
    default_mode: str = "xG"  # "xG" or "goals"


# Features for True xA model (action-only, no shot info)
ACTION_FEATURES = [
    # Location features
    "end_x",
    "end_y", 
    "start_x",
    "start_y",
    "distance_to_goal",
    "angle_to_goal",
    
    # Progression features
    "xt_delta",
    "progression_distance",
    "is_progressive",
    
    # Action type
    "is_pass",
    "is_carry",
    "is_dribble",
    
    # Pass characteristics (0 for non-passes)
    "is_through_ball",
    "is_cross",
    "is_into_box",
    "is_switch",
    
    # Context
    "under_pressure",
    "action_position",  # 1=key, 2=pre-assist, etc.
]

# Additional contribution features
CONTRIBUTION_FEATURES = [
    "breaks_line",
    "enters_zone14",
    "enters_half_space",
    "estimated_defenders_bypassed",
]

# Opponent context features
OPPONENT_FEATURES = [
    "opponent_global_rating",
    "opponent_zone_rating",
    "opponent_block_rate",
    "sequence_pressure_rate",
]

# Game state features
GAME_STATE_FEATURES = [
    "score_differential",
    "minute",
    "is_home",
]


class UltimateCxAModel:
    """The Ultimate cXA Model with opponent adjustment.
    
    This model:
    1. Predicts P(goal | action_features) - True xA
    2. Distributes credit via position decay + contribution
    3. Adjusts for opponent defensive quality
    
    Example usage:
        model = UltimateCxAModel()
        model.fit(action_sequences_df)
        credits = model.predict(action_sequences_df)
        player_cxa = model.aggregate_players(credits)
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.pipeline: Optional[Pipeline] = None
        self.feature_names: List[str] = []
        self.metrics: Dict[str, float] = {}
        self._is_fitted = False
        
    def _get_all_features(self) -> List[str]:
        """Get all features to use in the model."""
        features = ACTION_FEATURES.copy()
        
        if self.config.use_opponent_adjustment:
            features.extend(OPPONENT_FEATURES)
            features.extend(GAME_STATE_FEATURES)
        
        return features
    
    def _build_pipeline(self) -> Pipeline:
        """Build the sklearn pipeline."""
        if self.config.model_type == "lr":
            model = LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=self.config.random_state,
                solver="lbfgs",
            )
        else:  # gbm
            model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                random_state=self.config.random_state,
            )
        
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", model),
        ])
    
    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare feature matrix from action-level dataframe."""
        features = self._get_all_features()
        available_features = [f for f in features if f in df.columns]
        
        X = df[available_features].copy()
        
        # Fill missing with 0
        X = X.fillna(0)
        
        # Store feature names
        self.feature_names = available_features
        
        return X
    
    def fit(
        self, 
        actions_df: pd.DataFrame,
        group_col: str = "match_id",
    ) -> "UltimateCxAModel":
        """Fit the True xA model.
        
        Args:
            actions_df: Action-level DataFrame with is_goal target
            group_col: Column to group by for CV (prevents leakage)
        
        Returns:
            Self for chaining
        """
        logger.info("=" * 60)
        logger.info("Fitting Ultimate cXA Model (True xA)")
        logger.info("=" * 60)
        
        # Validate required columns
        required = ["is_goal", "action_position"]
        missing = [c for c in required if c not in actions_df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Prepare data
        X = self._prepare_features(actions_df)
        y = actions_df["is_goal"].astype(int).values
        groups = actions_df[group_col].values if group_col in actions_df.columns else None
        
        logger.info(f"Training on {len(X):,} actions")
        logger.info(f"Features: {len(self.feature_names)}")
        logger.info(f"Positive rate (goal): {y.mean():.4f}")
        
        # Build and train pipeline
        self.pipeline = self._build_pipeline()
        
        # Cross-validation for metrics
        if groups is not None:
            cv = GroupKFold(n_splits=self.config.cv_folds)
            cv_splits = list(cv.split(X, y, groups))
            
            y_proba_cv = cross_val_predict(
                self.pipeline, X, y, cv=cv_splits, method="predict_proba"
            )[:, 1]
            
            self.metrics["cv_auc"] = roc_auc_score(y, y_proba_cv)
            self.metrics["cv_brier"] = brier_score_loss(y, y_proba_cv)
            self.metrics["cv_logloss"] = log_loss(y, y_proba_cv)
            
            logger.info(f"CV AUC: {self.metrics['cv_auc']:.4f}")
            logger.info(f"CV Brier: {self.metrics['cv_brier']:.4f}")
        
        # Final fit on all data
        self.pipeline.fit(X, y)
        self._is_fitted = True
        
        # Feature importance (for interpretability)
        if self.config.model_type == "lr":
            coefs = self.pipeline.named_steps["model"].coef_[0]
            self.feature_importance = dict(zip(self.feature_names, coefs))
        else:
            importances = self.pipeline.named_steps["model"].feature_importances_
            self.feature_importance = dict(zip(self.feature_names, importances))
        
        logger.info("\nTop 10 features:")
        sorted_fi = sorted(self.feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
        for feat, imp in sorted_fi[:10]:
            logger.info(f"  {feat}: {imp:.4f}")
        
        return self
    
    def predict_proba(self, actions_df: pd.DataFrame) -> np.ndarray:
        """Predict P(goal | action_features) for each action."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        X = self._prepare_features(actions_df)
        return self.pipeline.predict_proba(X)[:, 1]
    
    def _compute_position_weights(self, positions: np.ndarray) -> np.ndarray:
        """Compute position-based decay weights.
        
        Position 1 (key action) gets highest weight, decaying for earlier actions.
        """
        return np.exp(-self.config.position_decay_rate * (positions - 1))
    
    def _compute_contribution_scores(self, actions_df: pd.DataFrame) -> np.ndarray:
        """Compute contribution scores based on action features.
        
        Higher scores for:
        - Line-breaking passes
        - Entries into dangerous zones
        - Progressive actions
        - Bypassing defenders
        """
        scores = np.ones(len(actions_df))
        
        # Line breaking
        if "breaks_line" in actions_df.columns:
            scores += 0.3 * actions_df["breaks_line"].fillna(0).values
        
        # Zone entries
        if "enters_zone14" in actions_df.columns:
            scores += 0.2 * actions_df["enters_zone14"].fillna(0).values
        if "enters_half_space" in actions_df.columns:
            scores += 0.15 * actions_df["enters_half_space"].fillna(0).values
        if "is_into_box" in actions_df.columns:
            scores += 0.25 * actions_df["is_into_box"].fillna(0).values
        
        # Progression
        if "is_progressive" in actions_df.columns:
            scores += 0.1 * actions_df["is_progressive"].fillna(0).values
        if "xt_delta" in actions_df.columns:
            # Normalize xt_delta contribution
            xt = actions_df["xt_delta"].fillna(0).values
            scores += 0.2 * np.clip(xt / 0.1, 0, 1)  # 0.1 xT delta = max bonus
        
        # Through balls
        if "is_through_ball" in actions_df.columns:
            scores += 0.2 * actions_df["is_through_ball"].fillna(0).values
        
        # Defenders bypassed
        if "estimated_defenders_bypassed" in actions_df.columns:
            defenders = actions_df["estimated_defenders_bypassed"].fillna(0).values
            scores += 0.1 * np.clip(defenders, 0, 3)  # Cap at 3 defenders
        
        return scores
    
    def _apply_opponent_adjustment(
        self, 
        base_credit: float,
        opponent_rating: float,
        avg_rating: float = 0.0,
    ) -> float:
        """Adjust credit based on opponent defensive quality.
        
        Creating chances against strong defenses is worth more.
        Creating chances against weak defenses is worth less.
        
        Args:
            base_credit: Unadjusted credit
            opponent_rating: Opponent's defensive rating (negative = better defense)
            avg_rating: Average rating for normalization
        
        Returns:
            Adjusted credit
        """
        # Opponent rating is negative (lower = better defense)
        # We want to boost credit when facing good defenses
        # Rating of -0.15 is ~average, -0.20 is good, -0.10 is poor
        
        # Shrink toward average
        adj_rating = (
            self.config.opponent_shrinkage * avg_rating +
            (1 - self.config.opponent_shrinkage) * opponent_rating
        )
        
        # Convert to multiplier: stronger defense = higher multiplier
        # If avg_rating = -0.15, opponent = -0.20 (better), multiplier > 1
        diff = avg_rating - adj_rating  # Positive if opponent better than avg
        multiplier = 1.0 + diff * 3.0  # Scale factor
        
        # Clip to reasonable range
        multiplier = np.clip(multiplier, 0.7, 1.3)
        
        return base_credit * multiplier
    
    def allocate_credit(
        self,
        actions_df: pd.DataFrame,
        mode: str = None,
    ) -> pd.DataFrame:
        """Allocate creation credit across actions for each sequence.
        
        Credit allocation uses:
        1. True xA probability (how likely is this action to lead to a goal?)
        2. Position decay (key action gets more credit than earlier actions)
        3. Contribution features (line-breaking, progressive, etc.)
        4. Opponent adjustment (optional)
        
        Args:
            actions_df: Action-level DataFrame with sequence_id grouping
            mode: "xG" to weight by shot xG, "goals" to weight by is_goal
        
        Returns:
            DataFrame with credit allocations
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        mode = mode or self.config.default_mode
        
        logger.info(f"Allocating credit (mode={mode})...")
        
        # Predict True xA
        actions_df = actions_df.copy()
        actions_df["true_xa"] = self.predict_proba(actions_df)
        
        # Compute position weights
        positions = actions_df["action_position"].values
        actions_df["position_weight"] = self._compute_position_weights(positions)
        
        # Compute contribution scores
        actions_df["contribution_score"] = self._compute_contribution_scores(actions_df)
        
        # Combined score for softmax
        # Blend position decay with contribution features
        pos_contrib = self.config.contribution_weight
        actions_df["combined_score"] = (
            (1 - pos_contrib) * actions_df["position_weight"] +
            pos_contrib * actions_df["contribution_score"]
        ) * actions_df["true_xa"]  # Weight by probability
        
        # Allocate credit per sequence via softmax
        credit_results = []
        
        sequence_col = "sequence_id" if "sequence_id" in actions_df.columns else "shot_id"
        
        for seq_id, group in actions_df.groupby(sequence_col):
            scores = group["combined_score"].values
            
            # Softmax with temperature
            exp_scores = np.exp((scores - scores.max()) / self.config.temperature)
            shares = exp_scores / exp_scores.sum()
            
            # Apply minimum credit share
            shares = np.maximum(shares, self.config.min_credit_share)
            shares = shares / shares.sum()  # Re-normalize
            
            # Get sequence value based on mode
            if mode == "xG":
                seq_value = group["shot_xg"].iloc[0] if "shot_xg" in group.columns else 0.0
            else:  # goals
                seq_value = 1.0 if group["is_goal"].iloc[0] else 0.0
            
            # Opponent adjustment
            if self.config.use_opponent_adjustment and "opponent_global_rating" in group.columns:
                opp_rating = group["opponent_global_rating"].iloc[0]
                if pd.notna(opp_rating):
                    seq_value = self._apply_opponent_adjustment(seq_value, opp_rating)
            
            # Assign credits
            for i, (idx, row) in enumerate(group.iterrows()):
                credit_results.append({
                    sequence_col: seq_id,
                    "action_position": row["action_position"],
                    "player_id": row.get("player_id"),
                    "player_name": row.get("player_name"),
                    "action_type": row.get("action_type"),
                    "team_id": row.get("team_id"),
                    "match_id": row.get("match_id"),
                    "is_goal": row.get("is_goal"),
                    "true_xa": float(row["true_xa"]),
                    "position_weight": float(row["position_weight"]),
                    "contribution_score": float(row["contribution_score"]),
                    "credit_share": float(shares[i]),
                    "sequence_value": float(seq_value),
                    "credit": float(shares[i] * seq_value),
                })
        
        credits_df = pd.DataFrame(credit_results)
        
        logger.info(f"  Total credit allocated: {credits_df['credit'].sum():.2f}")
        
        return credits_df
    
    def aggregate_players(
        self,
        credits_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Aggregate credits to player level.
        
        Args:
            credits_df: Output from allocate_credit()
        
        Returns:
            Player-level aggregation with:
            - total_cxa: Sum of all credits
            - key_actions: Count of position=1 actions
            - pre_assists: Count of position=2 actions
            - passes, carries, dribbles: Action type breakdown
        """
        logger.info("Aggregating to player level...")
        
        player_stats = []
        
        for player_id, group in credits_df.groupby("player_id"):
            if pd.isna(player_id):
                continue
            
            player_name = group["player_name"].iloc[0] if "player_name" in group.columns else None
            team_id = group["team_id"].iloc[0] if "team_id" in group.columns else None
            
            stats = {
                "player_id": player_id,
                "player_name": player_name,
                "team_id": team_id,
                
                # Total credit
                "total_cxa": group["credit"].sum(),
                "total_true_xa": group["true_xa"].sum(),
                
                # Action counts
                "total_actions": len(group),
                "key_actions": (group["action_position"] == 1).sum(),
                "pre_assists": (group["action_position"] == 2).sum(),
                
                # Action types
                "passes": (group["action_type"] == "Pass").sum(),
                "carries": (group["action_type"] == "Carry").sum(),
                "dribbles": (group["action_type"] == "Dribble").sum(),
                
                # Credit by type
                "pass_credit": group.loc[group["action_type"] == "Pass", "credit"].sum(),
                "carry_credit": group.loc[group["action_type"] == "Carry", "credit"].sum(),
                "dribble_credit": group.loc[group["action_type"] == "Dribble", "credit"].sum(),
                
                # Average metrics
                "avg_credit_per_action": group["credit"].mean(),
                "avg_true_xa": group["true_xa"].mean(),
            }
            
            player_stats.append(stats)
        
        players_df = pd.DataFrame(player_stats)
        players_df = players_df.sort_values("total_cxa", ascending=False)
        
        logger.info(f"  Aggregated {len(players_df):,} players")
        logger.info(f"  Top player: {players_df.iloc[0]['player_name']} ({players_df.iloc[0]['total_cxa']:.2f} cXA)")
        
        return players_df
    
    def save(self, path: Path) -> None:
        """Save model to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save pipeline
        joblib.dump(self.pipeline, path / "pipeline.joblib")
        
        # Save config and metadata
        metadata = {
            "config": self.config.__dict__,
            "feature_names": self.feature_names,
            "metrics": self.metrics,
            "feature_importance": self.feature_importance if hasattr(self, "feature_importance") else {},
        }
        joblib.dump(metadata, path / "metadata.joblib")
        
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: Path) -> "UltimateCxAModel":
        """Load model from disk."""
        path = Path(path)
        
        model = cls()
        model.pipeline = joblib.load(path / "pipeline.joblib")
        
        metadata = joblib.load(path / "metadata.joblib")
        model.config = ModelConfig(**metadata["config"])
        model.feature_names = metadata["feature_names"]
        model.metrics = metadata["metrics"]
        model.feature_importance = metadata.get("feature_importance", {})
        model._is_fitted = True
        
        logger.info(f"Model loaded from {path}")
        
        return model
    
    def get_metrics_summary(self) -> str:
        """Get a formatted summary of model metrics."""
        lines = [
            "=" * 50,
            "Ultimate cXA Model Metrics",
            "=" * 50,
            f"Model type: {self.config.model_type.upper()}",
            f"Features: {len(self.feature_names)}",
            "",
        ]
        
        if self.metrics:
            lines.append("Cross-Validation Metrics:")
            for k, v in self.metrics.items():
                lines.append(f"  {k}: {v:.4f}")
        
        if hasattr(self, "feature_importance") and self.feature_importance:
            lines.append("\nTop 5 Features:")
            sorted_fi = sorted(self.feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
            for feat, imp in sorted_fi[:5]:
                lines.append(f"  {feat}: {imp:.4f}")
        
        return "\n".join(lines)


def _get_repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _load_opponent_profiles(version_tag: str) -> pd.DataFrame:
    """Load opponent profiles from the database for a given version tag."""
    from opponent_adjusted.db.session import session_scope
    from opponent_adjusted.db.models import OpponentDefProfile

    with session_scope() as session:
        query = session.query(OpponentDefProfile).filter_by(version_tag=version_tag)
        profiles_df = pd.read_sql(query.statement, session.bind)

    return profiles_df


def build_action_level_dataset(
    sequences_df: pd.DataFrame,
    max_actions: int = 5,
) -> pd.DataFrame:
    """Convert wide-format sequences to action-level dataset.
    
    Args:
        sequences_df: Wide-format sequences from feature_store
        max_actions: Maximum actions per sequence to include
    
    Returns:
        Long-format DataFrame with one row per action
    """
    logger.info(f"Building action-level dataset from {len(sequences_df):,} sequences...")
    
    action_rows = []
    
    for _, seq in sequences_df.iterrows():
        num_actions = int(seq.get("num_actions", 0))
        if num_actions == 0:
            continue
        
        shot_xg = seq.get("shot_xg", 0.0)
        is_goal = seq.get("is_goal", False)
        
        for pos in range(1, min(num_actions + 1, max_actions + 1)):
            prefix = f"action{pos}_"
            
            # Check if this action exists
            action_type = seq.get(f"{prefix}type")
            if pd.isna(action_type):
                continue
            
            action_row = {
                "sequence_id": seq.get("sequence_id"),
                "shot_id": seq.get("shot_id"),
                "match_id": seq.get("match_id"),
                "team_id": seq.get("team_id"),
                "action_position": pos,
                "is_goal": is_goal,
                "shot_xg": shot_xg,
                
                # Action info
                "action_type": action_type,
                "player_id": seq.get(f"{prefix}player_id"),
                "player_name": seq.get(f"{prefix}player_name"),
                
                # Location features
                "start_x": seq.get(f"{prefix}start_x"),
                "start_y": seq.get(f"{prefix}start_y"),
                "end_x": seq.get(f"{prefix}end_x"),
                "end_y": seq.get(f"{prefix}end_y"),

                # Additional action features
                "xt_delta": seq.get(f"{prefix}xt_delta"),
                "is_switch": seq.get(f"{prefix}is_switch"),
                "breaks_line": seq.get(f"{prefix}breaks_line"),
                "estimated_defenders_bypassed": seq.get(f"{prefix}estimated_defenders_bypassed"),
                
                # Derived features
                "under_pressure": seq.get(f"{prefix}under_pressure"),
                "is_cross": seq.get(f"{prefix}is_cross"),
                "is_through_ball": seq.get(f"{prefix}is_through_ball"),
                "minute": seq.get("sequence_minute", seq.get("shot_minute", seq.get(f"{prefix}minute"))),
                "is_home": seq.get("is_home"),

                # Opponent context (sequence-level)
                "opponent_global_rating": seq.get("opponent_global_rating"),
                "opponent_zone_rating": seq.get("opponent_zone_rating"),
                "opponent_block_rate": seq.get("opponent_block_rate"),
                "sequence_pressure_rate": seq.get("sequence_pressure_rate"),
                "score_differential": seq.get("score_differential"),
            }
            
            # Type indicators
            action_row["is_pass"] = 1 if action_type == "Pass" else 0
            action_row["is_carry"] = 1 if action_type == "Carry" else 0
            action_row["is_dribble"] = 1 if action_type == "Dribble" else 0
            
            # Compute additional features
            if pd.notna(action_row["end_x"]) and pd.notna(action_row["end_y"]):
                # Distance to goal (goal at x=120, y=40)
                goal_x, goal_y = 120.0, 40.0
                dx = goal_x - action_row["end_x"]
                dy = goal_y - action_row["end_y"]
                action_row["distance_to_goal"] = np.sqrt(dx**2 + dy**2)
                action_row["angle_to_goal"] = np.abs(np.arctan2(dy, dx))
                
                # Is into box (x > 102, 18 < y < 62)
                action_row["is_into_box"] = int(
                    action_row["end_x"] > 102 and 18 < action_row["end_y"] < 62
                )
                
                # Zone entries
                action_row["enters_zone14"] = int(
                    88 < action_row["end_x"] < 105 and 25 < action_row["end_y"] < 55
                )
                action_row["enters_half_space"] = int(
                    88 < action_row["end_x"] < 120 and
                    (18 < action_row["end_y"] < 30 or 50 < action_row["end_y"] < 62)
                )
            
            # Progression
            if (pd.notna(action_row.get("start_x")) and 
                pd.notna(action_row.get("end_x"))):
                progression = action_row["end_x"] - action_row["start_x"]
                action_row["progression_distance"] = max(0, progression)
                action_row["is_progressive"] = int(progression > 10)
            
            action_rows.append(action_row)
    
    actions_df = pd.DataFrame(action_rows)
    
    logger.info(f"  Created {len(actions_df):,} action rows")
    
    return actions_df


def run_ultimate_cxa_pipeline(
    output_dir: Optional[Path] = None,
    config: Optional[ModelConfig] = None,
) -> Dict[str, Any]:
    """Run the complete Ultimate cXA pipeline.
    
    Steps:
    1. Load action sequences from feature store
    2. Build action-level dataset
    3. Add contribution and opponent features
    4. Train True xA model
    5. Allocate credit
    6. Aggregate to players
    7. Save outputs
    
    Returns:
        Dictionary with results and paths
    """
    repo_root = _get_repo_root()
    if output_dir is None:
        output_dir = repo_root / "outputs" / "modeling" / "ultimate_cxa"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config = config or ModelConfig()
    
    logger.info("=" * 70)
    logger.info("Running Ultimate cXA Pipeline")
    logger.info("=" * 70)
    
    # Load sequences
    sequences_path = repo_root / "feature_store" / "cxa" / "action_sequences.parquet"
    sequences_df = pd.read_parquet(sequences_path)
    logger.info(f"Loaded {len(sequences_df):,} action sequences")
    
    # Add opponent context if enabled
    if config.use_opponent_adjustment:
        try:
            from opponent_adjusted.config import settings
            from opponent_adjusted.db.session import session_scope
            from opponent_adjusted.features.cxa.opposition_context import build_opposition_context

            version_tag = config.opponent_profiles_version or settings.default_feature_version
            profiles_df = _load_opponent_profiles(version_tag)
            if profiles_df.empty:
                logger.warning("No opponent profiles found for version %s", version_tag)
            else:
                with session_scope() as session:
                    sequences_df = build_opposition_context(sequences_df, profiles_df, session)
        except Exception as exc:
            logger.warning("Opponent profiles not integrated: %s", exc)

    # Build action-level dataset
    actions_df = build_action_level_dataset(sequences_df, max_actions=config.max_actions)
    
    # Initialize and train model
    model = UltimateCxAModel(config=config)
    model.fit(actions_df, group_col="match_id")
    
    # Allocate credit (xG mode)
    credits_df = model.allocate_credit(actions_df, mode="xG")
    
    # Aggregate to players
    players_df = model.aggregate_players(credits_df)
    
    # Also compute goals-only version
    credits_goals_df = model.allocate_credit(actions_df, mode="goals")
    players_goals_df = model.aggregate_players(credits_goals_df)
    players_goals_df = players_goals_df.rename(columns={"total_cxa": "total_cxa_goals"})
    
    # Save outputs
    logger.info("\nSaving outputs...")
    
    actions_df.to_parquet(output_dir / "actions.parquet", index=False)
    credits_df.to_parquet(output_dir / "credits_xg.parquet", index=False)
    credits_goals_df.to_parquet(output_dir / "credits_goals.parquet", index=False)
    players_df.to_csv(output_dir / "player_cxa_xg.csv", index=False)
    players_goals_df.to_csv(output_dir / "player_cxa_goals.csv", index=False)
    model.save(output_dir / "model")
    
    # Save metrics
    with open(output_dir / "metrics.txt", "w") as f:
        f.write(model.get_metrics_summary())
    
    logger.info(f"\nOutputs saved to {output_dir}")
    logger.info("\n" + model.get_metrics_summary())
    
    return {
        "model": model,
        "actions_df": actions_df,
        "credits_df": credits_df,
        "players_df": players_df,
        "output_dir": output_dir,
    }


def main() -> int:
    """Command-line entry point."""
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
    )
    
    parser = argparse.ArgumentParser(description="Run Ultimate cXA Pipeline")
    parser.add_argument("--output-dir", type=Path, help="Output directory")
    parser.add_argument("--model-type", choices=["lr", "gbm"], default="gbm")
    parser.add_argument("--max-actions", type=int, default=5)
    parser.add_argument("--no-opponent-adj", action="store_true")
    parser.add_argument("--opponent-version", type=str, help="Opponent profile version tag")
    
    args = parser.parse_args()
    
    config = ModelConfig(
        model_type=args.model_type,
        max_actions=args.max_actions,
        use_opponent_adjustment=not args.no_opponent_adj,
        opponent_profiles_version=args.opponent_version,
    )
    
    run_ultimate_cxa_pipeline(output_dir=args.output_dir, config=config)
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
