"""
Shot Profile Analysis Page - Shot Clustering by Characteristics
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
    load_shot_data,
    get_unique_values,
    add_team_names,
    add_shot_profiles,
    get_shot_profile_summary,
    SHOT_PROFILES
)
from components.pitch_plots import create_shot_map

st.set_page_config(page_title="Shot Profiles | CxG Dashboard", page_icon="🎯", layout="wide")

st.title("🎯 Shot Profile Analysis")
st.markdown("Clustering shots by characteristics and location patterns")

st.markdown("---")

# Load data
shots_df = load_shot_data()

if shots_df.empty:
    st.error("No shot data available.")
    st.stop()

# Add team names and shot profiles
shots_df = add_team_names(shots_df, 'team_id')
shots_df = add_shot_profiles(shots_df)

# Sidebar filters
st.sidebar.header("🔍 Filters")

# Competition filter
competitions = ['All'] + get_unique_values(shots_df, 'competition_name')
selected_competition = st.sidebar.selectbox("Competition", competitions)

# Team filter
teams = ['All'] + sorted(get_unique_values(shots_df, 'team_name'))
selected_team = st.sidebar.selectbox("Team", teams)

# Apply filters
filtered_df = shots_df.copy()
if selected_competition != 'All':
    filtered_df = filtered_df[filtered_df['competition_name'] == selected_competition]
if selected_team != 'All':
    filtered_df = filtered_df[filtered_df['team_name'] == selected_team]

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Shots:** {len(filtered_df):,}")

# Profile Legend
st.header("📋 Shot Profile Types")

cols = st.columns(3)
profile_list = list(SHOT_PROFILES.items())

for i, (key, info) in enumerate(profile_list):
    with cols[i % 3]:
        st.markdown(f"### {info['emoji']} {info['name']}")
        st.caption(info['description'])

st.markdown("---")

# Profile Distribution
st.header("📊 Profile Distribution")

profile_summary = get_shot_profile_summary(filtered_df)

col1, col2 = st.columns([1, 2])

with col1:
    # Counts table
    st.subheader("Shot Counts by Profile")
    
    # Add emoji to profile names
    profile_summary_display = profile_summary.copy()
    profile_summary_display['Profile'] = profile_summary_display['shot_profile'].apply(
        lambda x: f"{SHOT_PROFILES.get(x, {}).get('emoji', '⚽')} {SHOT_PROFILES.get(x, {}).get('name', x)}"
    )
    
    st.dataframe(
        profile_summary_display[['Profile', 'shots', 'goals', 'goal_rate', 'avg_xg']].style.format({
            'goal_rate': '{:.1%}',
            'avg_xg': '{:.3f}'
        }),
        use_container_width=True,
        hide_index=True
    )

with col2:
    # Pie chart
    fig = px.pie(
        profile_summary,
        values='shots',
        names='shot_profile',
        title='Shot Profile Distribution',
        color='shot_profile',
        color_discrete_map={
            'poacher': '#ff6b6b',
            'long_range': '#4ecdc4',
            'aerial': '#ffe66d',
            'counter': '#95e1d3',
            'set_piece': '#786fa6',
            'creator': '#f8a5c2'
        }
    )
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(fig, use_container_width=True)

# Dynamic profile insight
most_common = profile_summary.iloc[0]
best_conversion = profile_summary.loc[profile_summary['goal_rate'].idxmax()]
worst_conversion = profile_summary.loc[profile_summary['goal_rate'].idxmin()]

st.info(f"""
💡 **Profile Insight:** The most common shot type is **{SHOT_PROFILES.get(most_common['shot_profile'], {}).get('name', most_common['shot_profile'])}** 
({most_common['shots']:,.0f} shots, {most_common['shots']/len(filtered_df)*100:.1f}% of total). 
**{SHOT_PROFILES.get(best_conversion['shot_profile'], {}).get('name', best_conversion['shot_profile'])}** shots have the highest conversion rate 
at **{best_conversion['goal_rate']:.1%}**, while **{SHOT_PROFILES.get(worst_conversion['shot_profile'], {}).get('name', worst_conversion['shot_profile'])}** 
shots convert at only **{worst_conversion['goal_rate']:.1%}**.
""")

st.markdown("---")

# Conversion Rates by Profile
st.header("🎯 Conversion Rates by Profile")

fig = go.Figure()

# Sort by goal rate for better visualization
profile_summary_sorted = profile_summary.sort_values('goal_rate', ascending=True)

fig.add_trace(go.Bar(
    y=[f"{SHOT_PROFILES.get(p, {}).get('emoji', '')} {SHOT_PROFILES.get(p, {}).get('name', p)}" 
       for p in profile_summary_sorted['shot_profile']],
    x=profile_summary_sorted['goal_rate'] * 100,
    orientation='h',
    marker_color=[
        '#ff6b6b' if p == 'poacher' else
        '#4ecdc4' if p == 'long_range' else
        '#ffe66d' if p == 'aerial' else
        '#95e1d3' if p == 'counter' else
        '#786fa6' if p == 'set_piece' else
        '#f8a5c2'
        for p in profile_summary_sorted['shot_profile']
    ],
    text=[f"{r:.1%}" for r in profile_summary_sorted['goal_rate']],
    textposition='outside'
))

fig.update_layout(
    title='Goal Conversion Rate by Shot Profile',
    xaxis_title='Conversion Rate (%)',
    yaxis_title='',
    template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0.1)',
    height=350,
    showlegend=False
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Shot Maps by Profile
st.header("🗺️ Shot Locations by Profile")

selected_profile = st.selectbox(
    "Select Profile to View",
    list(SHOT_PROFILES.keys()),
    format_func=lambda x: f"{SHOT_PROFILES[x]['emoji']} {SHOT_PROFILES[x]['name']}"
)

profile_shots = filtered_df[filtered_df['shot_profile'] == selected_profile]

if len(profile_shots) > 0:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        profile_info = SHOT_PROFILES[selected_profile]
        st.subheader(f"{profile_info['emoji']} {profile_info['name']} Shots")
        
        # Create shot map
        fig = create_shot_map(
            profile_shots,
            title=f"{profile_info['name']} Shot Locations ({len(profile_shots)} shots)"
        )
        st.pyplot(fig)
    
    with col2:
        st.subheader("📈 Profile Statistics")
        
        goals = profile_shots['is_goal'].sum()
        total = len(profile_shots)
        avg_xg = profile_shots['statsbomb_xg'].mean()
        avg_dist = profile_shots['shot_distance'].mean()
        
        st.metric("Total Shots", total)
        st.metric("Goals", int(goals))
        st.metric("Conversion Rate", f"{goals/total:.1%}")
        st.metric("Avg xG", f"{avg_xg:.3f}")
        st.metric("Avg Distance", f"{avg_dist:.1f}m")
        
        # CxG if available
        if 'cxg_pred' in profile_shots.columns:
            avg_cxg = profile_shots['cxg_pred'].mean()
            st.metric("Avg CxG", f"{avg_cxg:.3f}")
        
        # Profile-specific insight
        actual_vs_expected = (goals/total) - avg_xg
        perf = "over-performing" if actual_vs_expected > 0 else "under-performing"
        st.caption(f"📊 {profile_info['name']} shots are {perf} by {abs(actual_vs_expected)*100:.1f}pp vs xG")
else:
    st.info(f"No shots found for profile: {SHOT_PROFILES[selected_profile]['name']}")

st.markdown("---")

# Team Profile Comparison
st.header("🏆 Team Shot Profile Breakdown")

# Get profile counts by team
team_profiles = filtered_df.groupby(['team_name', 'shot_profile']).size().unstack(fill_value=0)

# Calculate percentages
team_profiles_pct = team_profiles.div(team_profiles.sum(axis=1), axis=0) * 100

# Rename columns with emojis
team_profiles_pct.columns = [
    f"{SHOT_PROFILES.get(col, {}).get('emoji', '')} {SHOT_PROFILES.get(col, {}).get('name', col)}"
    for col in team_profiles_pct.columns
]

# Stacked bar chart
fig = px.bar(
    team_profiles_pct.reset_index(),
    x='team_name',
    y=team_profiles_pct.columns.tolist(),
    title='Shot Profile Distribution by Team',
    labels={'value': 'Percentage', 'team_name': 'Team'},
    color_discrete_sequence=['#ff6b6b', '#4ecdc4', '#ffe66d', '#95e1d3', '#786fa6', '#f8a5c2']
)

fig.update_layout(
    template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0.1)',
    height=500,
    xaxis_tickangle=-45,
    legend_title='Shot Profile',
    barmode='stack'
)

st.plotly_chart(fig, use_container_width=True)

# Profile Performance by Team
st.markdown("---")
st.header("📊 Profile Performance Analysis")

profile_team_stats = filtered_df.groupby(['team_name', 'shot_profile']).agg({
    'is_goal': ['sum', 'count', 'mean'],
    'statsbomb_xg': 'mean'
}).round(3)

profile_team_stats.columns = ['Goals', 'Shots', 'Conv. Rate', 'Avg xG']
profile_team_stats = profile_team_stats.reset_index()

# Filter by profile
profile_filter = st.selectbox(
    "Filter by Profile",
    ['All'] + list(SHOT_PROFILES.keys()),
    format_func=lambda x: 'All Profiles' if x == 'All' else f"{SHOT_PROFILES[x]['emoji']} {SHOT_PROFILES[x]['name']}"
)

if profile_filter != 'All':
    profile_team_stats = profile_team_stats[profile_team_stats['shot_profile'] == profile_filter]

# Sort by shots
profile_team_stats = profile_team_stats.sort_values('Shots', ascending=False)

st.dataframe(
    profile_team_stats.style.format({
        'Conv. Rate': '{:.1%}',
        'Avg xG': '{:.3f}'
    }),
    use_container_width=True,
    height=400
)
