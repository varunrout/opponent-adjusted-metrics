"""
Shot Explorer Page - Interactive Pitch Maps
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add components to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from components.data_loader import (
    load_shot_data,
    get_unique_values,
    add_team_names,
    TEAM_NAMES
)
from components.pitch_plots import (
    create_shot_map,
    create_shot_heatmap,
    create_cxg_zones_map,
    create_goal_rate_comparison
)

st.set_page_config(page_title="Shot Explorer | CxG Dashboard", page_icon="⚽", layout="wide")

st.title("⚽ Shot Explorer")
st.markdown("Interactive pitch maps and shot analysis")

st.markdown("---")

# Load data
shots_df = load_shot_data()

if shots_df.empty:
    st.error("No shot data available. Please check data files.")
    st.stop()

# Add team names
shots_df = add_team_names(shots_df, 'team_id')

# Sidebar Filters
st.sidebar.header("🔍 Filters")

# Competition filter
competitions = ['All'] + get_unique_values(shots_df, 'competition_name')
selected_competition = st.sidebar.selectbox("Competition", competitions)

# Team filter
if selected_competition != 'All':
    filtered_teams = shots_df[shots_df['competition_name'] == selected_competition]['team_name'].unique()
else:
    filtered_teams = shots_df['team_name'].unique()
    
teams = ['All'] + sorted(filtered_teams.tolist())
selected_team = st.sidebar.selectbox("Team", teams)

# Outcome filter
outcome_options = ['All Shots', 'Goals Only', 'Non-Goals Only']
selected_outcome = st.sidebar.selectbox("Outcome", outcome_options)

# Pressure filter
pressure_options = ['All', 'Under Pressure', 'Not Under Pressure']
selected_pressure = st.sidebar.selectbox("Pressure State", pressure_options)

# Game state filter
game_states = ['All'] + get_unique_values(shots_df, 'score_state')
selected_game_state = st.sidebar.selectbox("Game State", game_states)

# Body part filter
body_parts = ['All'] + get_unique_values(shots_df, 'shot_body_part')
selected_body_part = st.sidebar.selectbox("Body Part", body_parts)

# Apply filters
filtered_df = shots_df.copy()

if selected_competition != 'All':
    filtered_df = filtered_df[filtered_df['competition_name'] == selected_competition]

if selected_team != 'All':
    filtered_df = filtered_df[filtered_df['team_name'] == selected_team]

if selected_outcome == 'Goals Only':
    filtered_df = filtered_df[filtered_df['is_goal'] == True]
elif selected_outcome == 'Non-Goals Only':
    filtered_df = filtered_df[filtered_df['is_goal'] == False]

if selected_pressure == 'Under Pressure':
    filtered_df = filtered_df[filtered_df['pressure_state'] == 'Under pressure']
elif selected_pressure == 'Not Under Pressure':
    filtered_df = filtered_df[filtered_df['pressure_state'] == 'Not under pressure']

if selected_game_state != 'All':
    filtered_df = filtered_df[filtered_df['score_state'] == selected_game_state]

if selected_body_part != 'All':
    filtered_df = filtered_df[filtered_df['shot_body_part'] == selected_body_part]

# Stats summary
st.subheader(f"📈 Filtered Results: {len(filtered_df):,} shots")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Shots", f"{len(filtered_df):,}")
with col2:
    goals = filtered_df['is_goal'].sum()
    st.metric("Goals", f"{goals:,}")
with col3:
    goal_rate = (goals / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
    st.metric("Goal Rate", f"{goal_rate:.1f}%")
with col4:
    avg_xg = filtered_df['statsbomb_xg'].mean() if len(filtered_df) > 0 else 0
    st.metric("Avg xG", f"{avg_xg:.3f}")
with col5:
    total_xg = filtered_df['statsbomb_xg'].sum()
    st.metric("Total xG", f"{total_xg:.1f}")

# Dynamic insight based on filters
if len(filtered_df) > 0:
    goals_vs_xg = goals - total_xg
    perf_text = "over-performing" if goals_vs_xg > 0 else "under-performing"
    
    insight_parts = []
    if selected_team != 'All':
        insight_parts.append(f"**{selected_team}**")
    if selected_competition != 'All':
        insight_parts.append(f"in **{selected_competition}**")
    
    context = " ".join(insight_parts) if insight_parts else "This selection"
    
    st.info(f"""
    💡 **Quick Insight:** {context} shows **{goals:,} goals** from **{len(filtered_df):,} shots** 
    (conversion: **{goal_rate:.1f}%**). With total xG of **{total_xg:.1f}**, they are 
    **{perf_text}** by **{abs(goals_vs_xg):.1f} goals** ({'+' if goals_vs_xg >= 0 else ''}{goals_vs_xg:.1f} Goals - xG).
    """)

st.markdown("---")

# Main visualization section
st.header("🗺️ Pitch Visualizations")

# Visualization type selector
viz_type = st.radio(
    "Select Visualization",
    ["Shot Map", "Shot Heatmap", "xG Zone Map"],
    horizontal=True
)

# Build title
title_parts = []
if selected_team != 'All':
    title_parts.append(selected_team)
if selected_competition != 'All':
    title_parts.append(selected_competition)
title_parts.append(selected_outcome if selected_outcome != 'All Shots' else 'All Shots')
title = " - ".join(title_parts) if title_parts else "All Shots"

if len(filtered_df) == 0:
    st.warning("No shots match the selected filters.")
else:
    if viz_type == "Shot Map":
        # Color options
        color_option = st.radio(
            "Color by",
            ["Goal/No Goal", "xG Probability"],
            horizontal=True
        )
        color_by = 'is_goal' if color_option == "Goal/No Goal" else 'statsbomb_xg'
        
        fig = create_shot_map(
            filtered_df,
            title=f"Shot Map: {title}",
            color_by=color_by,
            size_by='statsbomb_xg'
        )
        st.pyplot(fig)
        
    elif viz_type == "Shot Heatmap":
        fig = create_shot_heatmap(
            filtered_df,
            title=f"Shot Density: {title}",
            bins=(12, 8)
        )
        st.pyplot(fig)
        
    elif viz_type == "xG Zone Map":
        agg_option = st.radio(
            "Aggregation",
            ["Average xG", "Total xG", "Shot Count"],
            horizontal=True
        )
        
        agg_map = {
            "Average xG": ("statsbomb_xg", "mean", "Average xG by Zone"),
            "Total xG": ("statsbomb_xg", "sum", "Total xG by Zone"),
            "Shot Count": ("statsbomb_xg", "count", "Shots by Zone")
        }
        value_col, agg_func, zone_title = agg_map[agg_option]
        
        fig = create_cxg_zones_map(
            filtered_df,
            value_col=value_col,
            agg_func=agg_func,
            title=f"{zone_title}: {title}"
        )
        st.pyplot(fig)
    
    # Add visualization-specific insights
    avg_distance = filtered_df['shot_distance'].mean()
    close_range = (filtered_df['shot_distance'] < 12).sum()
    long_range = (filtered_df['shot_distance'] > 25).sum()
    st.caption(f"📍 **Shot Distribution:** Average distance: **{avg_distance:.1f}m**. Close-range (<12m): **{close_range:,}** shots ({close_range/len(filtered_df)*100:.1f}%), Long-range (>25m): **{long_range:,}** shots ({long_range/len(filtered_df)*100:.1f}%)")

st.markdown("---")

# Analysis by category
st.header("📊 Goal Rate Analysis")

analysis_col = st.selectbox(
    "Analyze by",
    ['chain_label', 'assist_category', 'pressure_state', 'score_state', 
     'minute_bucket_label', 'shot_body_part', 'set_piece_category']
)

if len(filtered_df) >= 30:
    fig = create_goal_rate_comparison(
        filtered_df,
        group_col=analysis_col,
        title=f"Goal Rate by {analysis_col.replace('_', ' ').title()}"
    )
    st.pyplot(fig)
    
    # Category insight
    if analysis_col in filtered_df.columns:
        cat_stats = filtered_df.groupby(analysis_col)['is_goal'].agg(['sum', 'count', 'mean'])
        best_cat = cat_stats['mean'].idxmax()
        best_rate = cat_stats.loc[best_cat, 'mean']
        best_count = cat_stats.loc[best_cat, 'count']
        st.caption(f"📊 **Best performing category:** '{best_cat}' with **{best_rate:.1%}** conversion rate from {best_count:,.0f} shots.")
else:
    st.warning("Need at least 30 shots for category analysis.")

st.markdown("---")

# Shot details table
st.header("📋 Shot Details")

show_table = st.checkbox("Show shot-level data table")

if show_table:
    display_cols = [
        'team_name', 'competition_name', 'match_id', 'minute',
        'location_x', 'location_y', 'shot_distance', 'shot_angle',
        'shot_body_part', 'shot_outcome', 'statsbomb_xg', 'is_goal',
        'score_state', 'pressure_state', 'chain_label'
    ]
    
    available_cols = [c for c in display_cols if c in filtered_df.columns]
    
    st.dataframe(
        filtered_df[available_cols].head(100),
        use_container_width=True,
        height=400
    )
    
    st.caption(f"Showing first 100 of {len(filtered_df):,} shots")
