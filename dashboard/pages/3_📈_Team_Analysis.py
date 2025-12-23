"""
Team Analysis Page - CxG Rankings, Archetypes, and Performance
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

# Add components to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from components.data_loader import (
    load_team_aggregates,
    load_shot_data,
    get_available_runs,
    add_team_names,
    compute_team_archetypes,
    TEAM_NAMES,
    TEAM_ARCHETYPES
)
from components.charts import (
    create_finishing_delta_chart,
    create_scatter_goals_vs_cxg
)

st.set_page_config(page_title="Team Analysis | CxG Dashboard", page_icon="📈", layout="wide")

st.title("📈 Team Analysis")
st.markdown("League tables, style archetypes, and performance analysis")

st.markdown("---")

# Select prediction run
available_runs = get_available_runs()

if not available_runs:
    st.warning("No prediction runs found. Run the prediction pipeline first.")
    st.stop()

selected_run = st.sidebar.selectbox(
    "📂 Prediction Run",
    available_runs,
    index=0
)

# Load data
team_df = load_team_aggregates(selected_run)
shots_df = load_shot_data()

if team_df.empty:
    st.error("No team data available for selected run.")
    st.stop()

# Add team names
team_df = add_team_names(team_df, 'team_id')

# Compute archetypes if shot data available
has_archetypes = False
if not shots_df.empty and 'style_att_component' in shots_df.columns:
    archetype_df = compute_team_archetypes(shots_df)
    # Merge archetype info with team_df
    team_df = team_df.merge(
        archetype_df[['team_id', 'archetype', 'archetype_emoji', 'archetype_name', 'style_attack', 'style_defense']],
        on='team_id',
        how='left'
    )
    has_archetypes = True

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Data Source:** {selected_run}")
st.sidebar.markdown(f"**Teams:** {len(team_df)}")

# View toggle
view_mode = st.sidebar.radio(
    "🎨 View Mode",
    ["Team Names", "Style Archetypes"] if has_archetypes else ["Team Names"],
    index=0
)

# Summary metrics
st.header("🏆 League Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    total_goals = team_df['goals_for'].sum()
    st.metric("Total Goals", f"{int(total_goals):,}")

with col2:
    total_cxg = team_df['cxg_for'].sum()
    st.metric("Total CxG", f"{total_cxg:.1f}")

with col3:
    delta = total_goals - total_cxg
    st.metric("Goals - CxG", f"{delta:+.1f}")

with col4:
    avg_diff = team_df['cxg_diff'].mean()
    st.metric("Avg CxG Diff", f"{avg_diff:.1f}")

# Dynamic league insight
best_team = team_df.loc[team_df['cxg_diff'].idxmax()]
worst_team = team_df.loc[team_df['cxg_diff'].idxmin()]
best_finisher = team_df.loc[team_df['goals_minus_cxg'].idxmax()]

st.success(f"""
🏆 **League Insight:** **{best_team['team_name']}** leads with a CxG difference of **{best_team['cxg_diff']:+.1f}** 
(creating {best_team['cxg_for']:.1f} CxG vs conceding {best_team['cxg_against']:.1f}). 
**{best_finisher['team_name']}** is the most clinical finisher, scoring **{best_finisher['goals_minus_cxg']:+.1f}** 
more goals than expected. **{worst_team['team_name']}** has the weakest CxG balance at **{worst_team['cxg_diff']:+.1f}**.
""")

st.markdown("---")

# Team Archetypes Section (if available)
if has_archetypes:
    st.header("🎭 Team Style Archetypes")
    
    # Archetype distribution
    archetype_counts = team_df['archetype_name'].value_counts()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Archetype Legend:**")
        for key, info in TEAM_ARCHETYPES.items():
            count = archetype_counts.get(info['name'], 0)
            st.markdown(f"{info['emoji']} **{info['name']}** ({count})")
            st.caption(info['description'])
    
    with col2:
        # Scatter plot by style components
        fig = px.scatter(
            team_df,
            x='style_attack',
            y='style_defense',
            color='archetype_name',
            hover_data=['team_name', 'cxg_for', 'goals_for'],
            text='team_name' if view_mode == "Team Names" else 'archetype_emoji',
            title='Team Positioning by Style Components',
            color_discrete_map={
                'High Press': '#ff6b6b',
                'Possession': '#4ecdc4',
                'Counter-Attack': '#ffe66d',
                'Balanced': '#95e1d3',
                'Defensive': '#786fa6',
                'Direct': '#f8a5c2',
            }
        )
        fig.update_traces(textposition='top center', marker=dict(size=12))
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0.1)',
            height=450,
            xaxis_title='Attack Style (+ = aggressive)',
            yaxis_title='Defense Style (+ = solid)',
        )
        # Add quadrant lines
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Archetype insight
    dominant_archetype = archetype_counts.idxmax()
    dominant_count = archetype_counts.max()
    st.caption(f"📊 **{dominant_archetype}** is the most common style with {dominant_count} teams. The scatter shows team positioning based on their attack aggressiveness (x) and defensive solidity (y).")
    
    st.markdown("---")

# League Table by CxG
st.header("📊 CxG League Table")

# Create sortable table
sort_by = st.selectbox(
    "Sort by",
    ['cxg_diff', 'cxg_for', 'goals_for', 'goals_minus_cxg', 'goal_diff'],
    format_func=lambda x: {
        'cxg_diff': 'CxG Difference (For - Against)',
        'cxg_for': 'CxG For',
        'goals_for': 'Goals For',
        'goals_minus_cxg': 'Finishing Delta (Goals - CxG)',
        'goal_diff': 'Goal Difference'
    }.get(x, x)
)

sorted_df = team_df.sort_values(sort_by, ascending=False).reset_index(drop=True)
sorted_df.index = sorted_df.index + 1  # 1-indexed ranking

# Display columns
display_df = sorted_df[[
    'team_name', 'matches', 'goals_for', 'cxg_for', 'provider_xg_for',
    'goals_against', 'cxg_against', 'cxg_diff', 'goals_minus_cxg', 'goal_diff'
]].copy()

display_df.columns = [
    'Team', 'Matches', 'Goals', 'CxG', 'Provider xG',
    'Goals Against', 'CxG Against', 'CxG Diff', 'Finishing Δ', 'GD'
]

# Style the dataframe
def highlight_positive(val):
    if isinstance(val, (int, float)):
        if val > 0:
            return 'color: #00ff87'
        elif val < 0:
            return 'color: #ff6b6b'
    return ''

st.dataframe(
    display_df.style.map(
        highlight_positive, 
        subset=['CxG Diff', 'Finishing Δ', 'GD']
    ).format({
        'CxG': '{:.1f}',
        'Provider xG': '{:.1f}',
        'CxG Against': '{:.1f}',
        'CxG Diff': '{:+.1f}',
        'Finishing Δ': '{:+.1f}',
    }),
    use_container_width=True,
    height=600
)

st.markdown("---")

# Visual Analysis
st.header("📉 Visual Analysis")

tab1, tab2 = st.tabs(["Finishing Delta", "Goals vs CxG Scatter"])

with tab1:
    st.markdown("""
    **Finishing Delta** = Goals Scored - CxG
    - 🟢 **Positive**: Team scored more than expected (clinical finishing or luck)
    - 🔴 **Negative**: Team scored less than expected (poor finishing or bad luck)
    """)
    
    fig = create_finishing_delta_chart(
        team_df,
        team_col='team_name',
        title=f"Finishing Performance: {selected_run}"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Finishing insight
    over_performers = team_df[team_df['goals_minus_cxg'] > 0]
    under_performers = team_df[team_df['goals_minus_cxg'] < 0]
    st.caption(f"📊 **{len(over_performers)} teams** are over-performing (clinical), **{len(under_performers)} teams** are under-performing. Average finishing delta across all teams: **{team_df['goals_minus_cxg'].mean():+.1f}**")

with tab2:
    st.markdown("""
    **Goals vs CxG Scatter**
    - Points above the line: Over-performed (scored more than CxG)
    - Points below the line: Under-performed (scored less than CxG)
    """)
    
    fig = create_scatter_goals_vs_cxg(
        team_df,
        title=f"Goals vs CxG: {selected_run}",
        label_col='team_name'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Scatter insight
    correlation = team_df['goals_for'].corr(team_df['cxg_for'])
    st.caption(f"📊 Correlation between Goals and CxG: **{correlation:.2f}**. A high correlation indicates CxG is a good predictor of actual goals scored.")

st.markdown("---")

# Team Deep Dive
st.header("🔍 Team Deep Dive")

selected_team = st.selectbox(
    "Select Team",
    sorted(team_df['team_name'].tolist())
)

team_row = team_df[team_df['team_name'] == selected_team].iloc[0]

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("⚔️ Attack")
    st.metric("Goals Scored", int(team_row['goals_for']))
    st.metric("CxG For", f"{team_row['cxg_for']:.1f}")
    st.metric("Provider xG", f"{team_row['provider_xg_for']:.1f}")
    delta = team_row['goals_for'] - team_row['cxg_for']
    st.metric("Finishing Delta", f"{delta:+.1f}", 
              delta="Over-performing" if delta > 0 else "Under-performing",
              delta_color="normal" if delta > 0 else "inverse")

with col2:
    st.subheader("🛡️ Defense")
    st.metric("Goals Conceded", int(team_row['goals_against']))
    st.metric("CxG Against", f"{team_row['cxg_against']:.1f}")
    st.metric("Provider xG Against", f"{team_row['provider_xg_against']:.1f}")
    def_delta = team_row['goals_against'] - team_row['cxg_against']
    st.metric("Defensive Delta", f"{def_delta:+.1f}",
              delta="Leaking goals" if def_delta > 0 else "Solid defense",
              delta_color="inverse" if def_delta > 0 else "normal")

with col3:
    st.subheader("📊 Overall")
    st.metric("CxG Difference", f"{team_row['cxg_diff']:+.1f}")
    st.metric("Goal Difference", f"{int(team_row['goal_diff']):+d}")
    st.metric("Shots For", int(team_row['shots_for']))
    st.metric("Shots Against", int(team_row['shots_against']))
    
    # Show archetype if available
    if has_archetypes and 'archetype_name' in team_row.index:
        st.markdown("---")
        st.markdown(f"**Style:** {team_row.get('archetype_emoji', '🔵')} {team_row.get('archetype_name', 'Unknown')}")

# Comparison table
st.markdown("---")
st.subheader("📋 Team Comparison")

compare_teams = st.multiselect(
    "Select teams to compare",
    team_df['team_name'].tolist(),
    default=[selected_team]
)

if compare_teams:
    cols_to_show = ['team_name', 'goals_for', 'cxg_for', 'goals_against', 'cxg_against',
                    'cxg_diff', 'goals_minus_cxg']
    col_names = ['Team', 'Goals', 'CxG', 'GA', 'CxG Against', 'CxG Diff', 'Finishing Δ']
    
    # Add archetype if available
    if has_archetypes and 'archetype_name' in team_df.columns:
        cols_to_show.insert(1, 'archetype_name')
        col_names.insert(1, 'Archetype')
    
    compare_df = team_df[team_df['team_name'].isin(compare_teams)][cols_to_show]
    compare_df.columns = col_names
    st.dataframe(compare_df, use_container_width=True)

# Archetype Performance Summary
if has_archetypes:
    st.markdown("---")
    st.header("🎯 Archetype Performance Summary")
    
    archetype_perf = team_df.groupby('archetype_name').agg({
        'goals_for': 'mean',
        'cxg_for': 'mean',
        'goals_minus_cxg': 'mean',
        'cxg_diff': 'mean',
        'team_id': 'count'
    }).round(2)
    archetype_perf.columns = ['Avg Goals', 'Avg CxG', 'Avg Finishing Δ', 'Avg CxG Diff', 'Teams']
    archetype_perf = archetype_perf.sort_values('Avg CxG Diff', ascending=False)
    
    st.dataframe(archetype_perf.style.format({
        'Avg Goals': '{:.1f}',
        'Avg CxG': '{:.1f}',
        'Avg Finishing Δ': '{:+.1f}',
        'Avg CxG Diff': '{:+.1f}',
        'Teams': '{:.0f}'
    }), use_container_width=True)
