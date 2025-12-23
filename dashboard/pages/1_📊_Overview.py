"""
Overview Page - Model Performance & Key Metrics
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add components to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from components.data_loader import (
    load_model_metrics, 
    load_shot_data, 
    load_team_aggregates,
    load_feature_effects,
    add_team_names
)
from components.charts import (
    create_model_comparison_chart,
    create_feature_importance_chart,
    create_reliability_diagram
)

st.set_page_config(page_title="Overview | CxG Dashboard", page_icon="📊", layout="wide")

st.title("📊 Model Overview")
st.markdown("Performance metrics and key insights from the CxG model")

st.markdown("---")

# Load data
baseline, contextual = load_model_metrics()
shots_df = load_shot_data()

# Performance Metrics Section
st.header("🎯 Model Performance")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("ROC AUC")
    baseline_auc = baseline.get('auc_mean', 0)
    contextual_auc = contextual.get('auc_mean', 0)
    improvement = ((contextual_auc - baseline_auc) / baseline_auc * 100) if baseline_auc > 0 else 0
    
    st.metric(
        "CxG Model",
        f"{contextual_auc:.4f}",
        delta=f"+{improvement:.1f}% vs Baseline",
        delta_color="normal"
    )
    st.caption(f"Baseline: {baseline_auc:.4f}")

with col2:
    st.subheader("Brier Score")
    baseline_brier = baseline.get('brier_mean', 0)
    contextual_brier = contextual.get('brier_mean', 0)
    improvement = ((baseline_brier - contextual_brier) / baseline_brier * 100) if baseline_brier > 0 else 0
    
    st.metric(
        "CxG Model",
        f"{contextual_brier:.4f}",
        delta=f"-{improvement:.1f}% vs Baseline",
        delta_color="inverse"
    )
    st.caption(f"Baseline: {baseline_brier:.4f}")

with col3:
    st.subheader("Log Loss")
    baseline_ll = baseline.get('log_loss_mean', 0)
    contextual_ll = contextual.get('log_loss_mean', 0)
    improvement = ((baseline_ll - contextual_ll) / baseline_ll * 100) if baseline_ll > 0 else 0
    
    st.metric(
        "CxG Model",
        f"{contextual_ll:.4f}",
        delta=f"-{improvement:.1f}% vs Baseline",
        delta_color="inverse"
    )
    st.caption(f"Baseline: {baseline_ll:.4f}")

# Dynamic insight box
st.info(f"""
💡 **Key Insight:** The CxG model achieves an AUC of **{contextual_auc:.4f}**, which is 
**{((contextual_auc - baseline_auc) / baseline_auc * 100):.1f}% better** than the geometric baseline ({baseline_auc:.4f}). 
This means CxG is significantly better at distinguishing goals from non-goals by incorporating 
contextual factors like team style, pressure, and game state.
""")

st.markdown("---")

# Model Comparison Chart
st.header("📈 Baseline vs CxG Model")

if baseline and contextual:
    comparison_chart = create_model_comparison_chart(baseline, contextual)
    st.plotly_chart(comparison_chart, use_container_width=True)

st.markdown("---")

# Dataset Statistics
st.header("📋 Dataset Statistics")

if not shots_df.empty:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Shots", f"{len(shots_df):,}")
    
    with col2:
        goals = shots_df['is_goal'].sum()
        goal_rate = goals / len(shots_df) * 100
        st.metric("Goals", f"{goals:,}", delta=f"{goal_rate:.1f}% conversion")
    
    with col3:
        competitions = shots_df['competition_name'].nunique()
        st.metric("Competitions", competitions)
    
    with col4:
        matches = shots_df['match_id'].nunique()
        st.metric("Matches", f"{matches:,}")
    
    # Dataset insight
    avg_shots_per_match = len(shots_df) / matches if matches > 0 else 0
    st.success(f"""
    📊 **Dataset Summary:** Analyzing **{len(shots_df):,} shots** across **{matches:,} matches** 
    ({avg_shots_per_match:.1f} shots/match). Overall conversion rate is **{goal_rate:.1f}%** 
    ({goals:,} goals), which is {'above' if goal_rate > 10 else 'around'} typical professional football averages (~10%).
    """)
    
    st.markdown("---")
    
    # Breakdown by competition
    st.subheader("Shots by Competition")
    
    comp_stats = shots_df.groupby('competition_name').agg({
        'is_goal': ['sum', 'mean', 'count'],
        'statsbomb_xg': 'mean'
    }).round(3)
    comp_stats.columns = ['Goals', 'Goal Rate', 'Shots', 'Avg xG']
    comp_stats = comp_stats.sort_values('Shots', ascending=False)
    
    st.dataframe(comp_stats, use_container_width=True)

st.markdown("---")

# Feature Importance
st.header("🔍 Feature Importance")

feature_effects = load_feature_effects()

if not feature_effects.empty:
    # Display top features
    st.markdown("""
    Model coefficients show how each feature affects goal probability (in log-odds):
    - **Positive** (green): Increases goal probability
    - **Negative** (red): Decreases goal probability
    """)
    
    if 'feature' in feature_effects.columns and 'coefficient' in feature_effects.columns:
        fig = create_feature_importance_chart(feature_effects, top_n=15)
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Display raw table
        st.dataframe(feature_effects.head(20), use_container_width=True)

st.markdown("---")

# Key Insights
st.header("💡 Key Insights")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### What Makes CxG Better?
    
    The Contextual xG model outperforms geometric baselines by incorporating:
    
    1. **Neutral Priors** - Team finishing/concession tendencies from rolling form
    2. **Style Clusters** - K-Means grouping of team playing styles
    3. **Pressure Effects** - Freeze frame analysis for defender proximity
    4. **Game State** - Score differential and minute interactions
    5. **Pass Chain Quality** - Assist type and delivery context
    """)

with col2:
    st.markdown("""
    ### Model Architecture
    
    **Stacked Logistic Regression** with:
    
    - **Submodels**: Finishing bias, concession bias, pressure penalty
    - **Regularization**: L2 (Ridge) with C=0.5
    - **Validation**: GroupKFold by match_id
    - **Calibration**: Well-calibrated across probability bins
    
    The "neutral priors" approach allows generalization across leagues without
    requiring league-specific team IDs.
    """)

st.markdown("---")

st.markdown("""
<div style="text-align: center; color: #888; padding: 1rem;">
    Navigate to other pages using the sidebar to explore shots, teams, and players.
</div>
""", unsafe_allow_html=True)
