"""
CxG Analytics Dashboard - Main Entry Point
Opponent-Adjusted Football Metrics Visualization
"""

import streamlit as st

st.set_page_config(
    page_title="CxG Analytics Dashboard",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for consistent styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Main page content
st.markdown('<p class="main-header">⚽ CxG Analytics Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Contextual Expected Goals • Opponent-Adjusted Metrics • StatsBomb Data</p>', unsafe_allow_html=True)

st.markdown("---")

# Welcome section
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    ### Welcome to the CxG Analytics Platform
    
    This dashboard provides interactive visualizations for **Contextual Expected Goals (CxG)** - 
    an advanced football analytics model that accounts for:
    
    - 📐 **Shot Geometry** - Distance, angle, and position
    - 🎯 **Game Context** - Score state, minute, possession patterns  
    - 🛡️ **Defensive Pressure** - Freeze frame analysis, defender proximity
    - ⚔️ **Opponent Quality** - Team defensive ratings, style clusters
    
    **Navigate using the sidebar** to explore different analysis views.
    """)

with col2:
    st.markdown("""
    ### Quick Links
    
    📊 **Overview** - Model performance  
    ⚽ **Shot Explorer** - Interactive pitch maps  
    📈 **Team Analysis** - League standings  
    👤 **Player Stats** - Individual metrics  
    """)

st.markdown("---")

# Key metrics preview
st.subheader("📊 Model Performance at a Glance")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="ROC AUC",
        value="0.865",
        delta="+17% vs baseline",
        delta_color="normal"
    )

with col2:
    st.metric(
        label="Brier Score",
        value="0.0631",
        delta="-17% vs baseline",
        delta_color="inverse"
    )

with col3:
    st.metric(
        label="Log Loss",
        value="0.219",
        delta="-20% vs baseline",
        delta_color="inverse"
    )

with col4:
    st.metric(
        label="Shots Analyzed",
        value="15,424",
        delta="World Cup + Euro + PL"
    )

st.markdown("---")

# Footer info
st.markdown("""
<div style="text-align: center; color: #888; padding: 2rem;">
    <p>Built with Streamlit • Data from StatsBomb Open Data</p>
    <p>CxG Model: Stacked Logistic Regression with Neutral Priors</p>
</div>
""", unsafe_allow_html=True)
