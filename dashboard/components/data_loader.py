"""
Data Loading Utilities
Cached functions to load project data efficiently
"""

import pandas as pd
import json
from pathlib import Path
import streamlit as st

# Define data paths relative to project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "outputs" / "modeling" / "cxg"


@st.cache_data(ttl=3600)
def load_shot_data(enriched: bool = True) -> pd.DataFrame:
    """
    Load the main CxG shot dataset.
    
    Parameters:
    -----------
    enriched : bool
        If True, load the enriched dataset with neutral priors
    
    Returns:
    --------
    pd.DataFrame
    """
    filename = "cxg_dataset_enriched.parquet" if enriched else "cxg_dataset.parquet"
    filepath = DATA_DIR / filename
    
    if filepath.exists():
        df = pd.read_parquet(filepath)
        return df
    
    # Fallback to CSV
    csv_path = DATA_DIR / filename.replace('.parquet', '.csv')
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        return df
    
    st.error(f"Data file not found: {filepath}")
    return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_team_aggregates(run_name: str = "pl_2015_16_club") -> pd.DataFrame:
    """
    Load team aggregate data from a prediction run.
    
    Parameters:
    -----------
    run_name : str
        Name of the prediction run folder
    
    Returns:
    --------
    pd.DataFrame
    """
    filepath = DATA_DIR / "prediction_runs" / run_name / "team_aggregates.csv"
    
    if filepath.exists():
        return pd.read_csv(filepath)
    
    st.warning(f"Team aggregates not found: {filepath}")
    return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_match_aggregates(run_name: str = "pl_2015_16_club") -> pd.DataFrame:
    """
    Load match aggregate data from a prediction run.
    
    Parameters:
    -----------
    run_name : str
        Name of the prediction run folder
    
    Returns:
    --------
    pd.DataFrame
    """
    filepath = DATA_DIR / "prediction_runs" / run_name / "match_aggregates.csv"
    
    if filepath.exists():
        return pd.read_csv(filepath)
    
    st.warning(f"Match aggregates not found: {filepath}")
    return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_model_metrics() -> tuple[dict, dict]:
    """
    Load baseline and contextual model metrics.
    
    Returns:
    --------
    tuple[dict, dict]
        (baseline_metrics, contextual_metrics)
    """
    baseline_path = DATA_DIR / "baseline_metrics.json"
    contextual_path = DATA_DIR / "contextual_metrics_enriched.json"
    
    baseline = {}
    contextual = {}
    
    if baseline_path.exists():
        with open(baseline_path, 'r') as f:
            baseline = json.load(f)
    
    if contextual_path.exists():
        with open(contextual_path, 'r') as f:
            contextual = json.load(f)
    
    return baseline, contextual


@st.cache_data(ttl=3600)
def load_feature_effects(version: str = "enriched") -> pd.DataFrame:
    """
    Load feature effects/coefficients.
    
    Parameters:
    -----------
    version : str
        Version of feature effects ('enriched', 'filtered', 'raw')
    
    Returns:
    --------
    pd.DataFrame
    """
    filepath = DATA_DIR / f"contextual_feature_effects_{version}.csv"
    
    if filepath.exists():
        return pd.read_csv(filepath)
    
    st.warning(f"Feature effects not found: {filepath}")
    return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_prediction_run_shots(run_name: str = "pl_2015_16_club") -> pd.DataFrame:
    """
    Load scored shots from a prediction run.
    
    Parameters:
    -----------
    run_name : str
        Name of the prediction run folder
    
    Returns:
    --------
    pd.DataFrame
    """
    filepath = DATA_DIR / "prediction_runs" / run_name / "scored_shots.parquet"
    
    if filepath.exists():
        return pd.read_parquet(filepath)
    
    st.warning(f"Scored shots not found: {filepath}")
    return pd.DataFrame()


def get_available_runs() -> list[str]:
    """
    Get list of available prediction runs.
    
    Returns:
    --------
    list[str]
    """
    runs_dir = DATA_DIR / "prediction_runs"
    
    if runs_dir.exists():
        return [d.name for d in runs_dir.iterdir() 
                if d.is_dir() and (d / "team_aggregates.csv").exists()]
    
    return []


def get_unique_values(df: pd.DataFrame, column: str) -> list:
    """
    Get unique values from a column, excluding nulls.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to query
    column : str
        Column name
    
    Returns:
    --------
    list
    """
    if column in df.columns:
        return sorted(df[column].dropna().unique().tolist())
    return []


# Team name mapping (StatsBomb IDs to names)
# Comprehensive mapping covering World Cup, Euros, and Premier League
TEAM_NAMES = {
    # International Teams (World Cup / Euros)
    2: "Serbia",
    3: "Switzerland",
    4: "Argentina",
    5: "Australia",
    6: "Denmark",
    7: "Brazil",
    8: "Tunisia",
    9: "Ecuador",
    10: "Senegal",
    11: "Netherlands",
    12: "Uruguay",
    13: "South Korea",
    14: "Morocco",
    15: "Portugal",
    16: "France",
    17: "Saudi Arabia",
    18: "Mexico",
    19: "Poland",
    20: "Qatar",
    21: "England",
    22: "Croatia",
    23: "Spain",
    24: "Japan",
    25: "United States",
    26: "Germany",
    27: "Cameroon",
    28: "Costa Rica",
    29: "Belgium",
    30: "Wales",
    31: "Ghana",
    32: "Canada",
    33: "Iran",
    34: "Colombia",
    35: "Sweden",
    36: "Panama",
    37: "Egypt",
    38: "Peru",
    39: "Nigeria",
    40: "Iceland",
    41: "Russia",
    42: "Turkey",
    43: "Austria",
    44: "Ukraine",
    45: "Czech Republic",
    46: "Romania",
    47: "Slovenia",
    48: "Georgia",
    49: "Slovakia",
    50: "Albania",
    51: "Italy",
    52: "Scotland",
    53: "Hungary",
    54: "North Macedonia",
    55: "Finland",
    # Premier League 2015/16
    56: "Sunderland",
    57: "Bournemouth", 
    58: "Watford",
    59: "Norwich City",
    60: "Swansea City",
    61: "Aston Villa",
    62: "Newcastle United",
    63: "Stoke City",
    64: "West Brom",
    65: "Arsenal",
    66: "Manchester City",
    67: "Tottenham",
    68: "Manchester United",
    69: "Southampton",
    70: "West Ham",
    71: "Crystal Palace",
    72: "Leicester City",
    73: "Liverpool",
    74: "Everton",
    75: "Chelsea",
}


def add_team_names(df: pd.DataFrame, team_col: str = 'team_id') -> pd.DataFrame:
    """
    Add team names to a DataFrame based on team IDs.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with team IDs
    team_col : str
        Column containing team IDs
    
    Returns:
    --------
    pd.DataFrame
    """
    df = df.copy()
    df['team_name'] = df[team_col].map(TEAM_NAMES).fillna(df[team_col].astype(str))
    return df


# Team Style Archetypes based on style components
TEAM_ARCHETYPES = {
    'high_press': {'emoji': '🔥', 'name': 'High Press', 'description': 'Aggressive pressing, quick transitions'},
    'possession': {'emoji': '⚽', 'name': 'Possession', 'description': 'Ball retention, patient buildup'},
    'counter': {'emoji': '⚡', 'name': 'Counter-Attack', 'description': 'Fast transitions, direct play'},
    'balanced': {'emoji': '⚖️', 'name': 'Balanced', 'description': 'Adaptable, mixed approach'},
    'defensive': {'emoji': '🛡️', 'name': 'Defensive', 'description': 'Compact defense, set-piece focus'},
    'direct': {'emoji': '💪', 'name': 'Direct', 'description': 'Long balls, aerial duels'},
}


def compute_team_archetypes(shots_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute team archetypes based on style components and shot patterns.
    
    Parameters:
    -----------
    shots_df : pd.DataFrame
        Shot data with style components
    
    Returns:
    --------
    pd.DataFrame with team archetypes
    """
    import numpy as np
    
    # Aggregate by team
    team_stats = shots_df.groupby('team_id').agg({
        'style_att_component': 'mean',
        'style_def_component': 'mean',
        'finishing_bias_logit': 'mean',
        'shot_distance': 'mean',
        'is_goal': ['sum', 'count', 'mean'],
        'statsbomb_xg': ['sum', 'mean'],
    }).round(4)
    
    team_stats.columns = [
        'style_attack', 'style_defense', 'finishing_bias',
        'avg_distance', 'goals', 'shots', 'goal_rate', 'total_xg', 'avg_xg'
    ]
    team_stats = team_stats.reset_index()
    
    # Add chain type preferences
    if 'chain_label' in shots_df.columns:
        chain_pivot = shots_df.groupby(['team_id', 'chain_label']).size().unstack(fill_value=0)
        chain_pivot = chain_pivot.div(chain_pivot.sum(axis=1), axis=0)  # Normalize
        chain_pivot = chain_pivot.reset_index()  # Reset index to avoid ambiguous merge
        
        # Key chain indicators
        through_ball_cols = [c for c in chain_pivot.columns if 'Through Ball' in str(c)]
        cross_cols = [c for c in chain_pivot.columns if 'Cross' in str(c)]
        cutback_cols = [c for c in chain_pivot.columns if 'Cutback' in str(c)]
        
        chain_df = pd.DataFrame({'team_id': chain_pivot['team_id']})
        chain_df['through_ball_rate'] = chain_pivot[through_ball_cols].sum(axis=1) if through_ball_cols else 0
        chain_df['cross_rate'] = chain_pivot[cross_cols].sum(axis=1) if cross_cols else 0
        chain_df['cutback_rate'] = chain_pivot[cutback_cols].sum(axis=1) if cutback_cols else 0
        
        team_stats = team_stats.merge(chain_df, on='team_id', how='left')
    
    # Classify archetypes based on style components
    def classify_archetype(row):
        style_att = row.get('style_attack', 0) or 0
        style_def = row.get('style_defense', 0) or 0
        avg_dist = row.get('avg_distance', 15)
        through_rate = row.get('through_ball_rate', 0) or 0
        cross_rate = row.get('cross_rate', 0) or 0
        
        # Classification logic
        if style_att > 0.05:
            return 'high_press'
        elif style_att < -0.1 and through_rate > 0.05:
            return 'counter'
        elif cross_rate > 0.25 or avg_dist > 18:
            return 'direct'
        elif style_def < -0.1:
            return 'defensive'
        elif abs(style_att) < 0.05 and abs(style_def) < 0.05:
            return 'balanced'
        else:
            return 'possession'
    
    team_stats['archetype'] = team_stats.apply(classify_archetype, axis=1)
    team_stats['archetype_emoji'] = team_stats['archetype'].map(
        lambda x: TEAM_ARCHETYPES.get(x, {}).get('emoji', '❓')
    )
    team_stats['archetype_name'] = team_stats['archetype'].map(
        lambda x: TEAM_ARCHETYPES.get(x, {}).get('name', 'Unknown')
    )
    
    return team_stats


# Shot Profile Archetypes
SHOT_PROFILES = {
    'poacher': {'emoji': '🎯', 'name': 'Poacher', 'description': 'Close range, cutbacks, tap-ins'},
    'long_range': {'emoji': '🚀', 'name': 'Long Range', 'description': 'Distance shots, set pieces'},
    'aerial': {'emoji': '✈️', 'name': 'Aerial', 'description': 'Headers, crosses'},
    'counter_shot': {'emoji': '⚡', 'name': 'Counter', 'description': 'Fast transitions, through balls'},
    'set_piece': {'emoji': '📐', 'name': 'Set Piece', 'description': 'Corners, free kicks'},
    'creator': {'emoji': '🔧', 'name': 'Creator', 'description': 'Cutbacks, wide positions'},
}


def classify_shot_profile(row) -> str:
    """Classify a single shot into a profile archetype."""
    distance = row.get('shot_distance', 15)
    body_part = row.get('shot_body_part', '')
    chain = row.get('chain_label', '')
    set_piece = row.get('set_piece_category', '')
    assist = row.get('assist_category', '')
    
    # Classification logic
    if body_part == 'Head':
        return 'aerial'
    elif set_piece in ['Corner', 'Indirect Free Kick', 'Direct Free Kick', 'Penalty']:
        return 'set_piece'
    elif distance > 22:
        return 'long_range'
    elif distance < 10 and ('Cutback' in str(chain) or 'Cutback' in str(assist)):
        return 'poacher'
    elif 'Through Ball' in str(chain) or 'Through Ball' in str(assist):
        return 'counter_shot'
    elif distance < 12:
        return 'poacher'
    else:
        return 'creator'


def add_shot_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add shot profile classification to each shot.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Shot data
    
    Returns:
    --------
    pd.DataFrame with shot_profile column
    """
    df = df.copy()
    df['shot_profile'] = df.apply(classify_shot_profile, axis=1)
    df['profile_emoji'] = df['shot_profile'].map(
        lambda x: SHOT_PROFILES.get(x, {}).get('emoji', '❓')
    )
    df['profile_name'] = df['shot_profile'].map(
        lambda x: SHOT_PROFILES.get(x, {}).get('name', 'Unknown')
    )
    return df


def get_shot_profile_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get summary statistics for each shot profile.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Shot data with shot_profile column
    
    Returns:
    --------
    pd.DataFrame with profile statistics
    """
    if 'shot_profile' not in df.columns:
        df = add_shot_profiles(df)
    
    summary = df.groupby('shot_profile').agg({
        'is_goal': ['sum', 'count', 'mean'],
        'statsbomb_xg': ['sum', 'mean'],
        'shot_distance': 'mean',
    }).round(3)
    
    summary.columns = ['goals', 'shots', 'goal_rate', 'total_xg', 'avg_xg', 'avg_distance']
    summary = summary.reset_index()
    
    # Add metadata
    summary['emoji'] = summary['shot_profile'].map(
        lambda x: SHOT_PROFILES.get(x, {}).get('emoji', '❓')
    )
    summary['name'] = summary['shot_profile'].map(
        lambda x: SHOT_PROFILES.get(x, {}).get('name', 'Unknown')
    )
    summary['description'] = summary['shot_profile'].map(
        lambda x: SHOT_PROFILES.get(x, {}).get('description', '')
    )
    
    return summary.sort_values('shots', ascending=False)
