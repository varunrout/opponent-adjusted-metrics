"""
Pitch Visualization Components using mplsoccer
Reusable functions for shot maps, heatmaps, and tactical visualizations
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from mplsoccer import Pitch, VerticalPitch
from matplotlib.patches import FancyBboxPatch
import streamlit as st


def create_shot_map(
    df: pd.DataFrame,
    title: str = "Shot Map",
    color_by: str = "is_goal",
    size_by: str = "statsbomb_xg",
    show_goals_only: bool = False,
    pitch_type: str = "statsbomb",
    figsize: tuple = (12, 8),
) -> plt.Figure:
    """
    Create an interactive shot map on a football pitch.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Shot data with location_x, location_y columns
    title : str
        Plot title
    color_by : str
        Column to use for coloring points ('is_goal', 'cxg_pred', etc.)
    size_by : str
        Column to determine point size
    show_goals_only : bool
        If True, only show goals
    pitch_type : str
        Pitch coordinate system ('statsbomb', 'opta', etc.)
    figsize : tuple
        Figure size
    
    Returns:
    --------
    matplotlib.figure.Figure
    """
    # Filter data
    plot_df = df.copy()
    if show_goals_only:
        plot_df = plot_df[plot_df['is_goal'] == True]
    
    if len(plot_df) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No shots to display", ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig
    
    # Create pitch - using half pitch for shots (attacking direction)
    pitch = VerticalPitch(
        pitch_type=pitch_type,
        half=True,
        pitch_color='#22312b',
        line_color='#c7d5cc',
        goal_type='box',
        linewidth=1.5
    )
    
    fig, ax = pitch.draw(figsize=figsize)
    
    # Prepare colors
    if color_by == 'is_goal':
        colors = plot_df[color_by].map({True: '#00ff87', False: '#ff6b6b'})
        alpha = 0.7
    elif color_by in plot_df.columns:
        # Continuous coloring (e.g., CxG probability)
        norm = plt.Normalize(vmin=0, vmax=1)
        cmap = plt.cm.RdYlGn
        colors = cmap(norm(plot_df[color_by].fillna(0)))
        alpha = 0.8
    else:
        colors = '#667eea'
        alpha = 0.6
    
    # Prepare sizes
    if size_by in plot_df.columns:
        sizes = plot_df[size_by].fillna(0.05) * 800 + 50
    else:
        sizes = 100
    
    # Plot shots
    scatter = pitch.scatter(
        plot_df['location_x'],
        plot_df['location_y'],
        s=sizes,
        c=colors,
        alpha=alpha,
        edgecolors='white',
        linewidth=0.8,
        ax=ax,
        zorder=2
    )
    
    # Add title
    ax.set_title(
        title,
        fontsize=16,
        fontweight='bold',
        color='white',
        pad=10
    )
    
    # Add legend for goals/no goals
    if color_by == 'is_goal':
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#00ff87', 
                   markersize=12, label='Goal', linestyle='None'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff6b6b', 
                   markersize=12, label='No Goal', linestyle='None'),
        ]
        ax.legend(
            handles=legend_elements, 
            loc='upper left', 
            fontsize=10,
            facecolor='#22312b',
            edgecolor='white',
            labelcolor='white'
        )
    
    plt.tight_layout()
    return fig


def create_shot_heatmap(
    df: pd.DataFrame,
    title: str = "Shot Density Heatmap",
    pitch_type: str = "statsbomb",
    figsize: tuple = (12, 8),
    cmap: str = "hot",
    bins: tuple = (12, 8),
) -> plt.Figure:
    """
    Create a heatmap showing shot density across the pitch.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Shot data with location_x, location_y columns
    title : str
        Plot title
    pitch_type : str
        Pitch coordinate system
    figsize : tuple
        Figure size
    cmap : str
        Colormap name
    bins : tuple
        Number of bins for x and y dimensions
    
    Returns:
    --------
    matplotlib.figure.Figure
    """
    pitch = VerticalPitch(
        pitch_type=pitch_type,
        half=True,
        pitch_color='#22312b',
        line_color='#c7d5cc',
        linewidth=1.5
    )
    
    fig, ax = pitch.draw(figsize=figsize)
    
    # Create heatmap
    bin_statistic = pitch.bin_statistic(
        df['location_x'], 
        df['location_y'], 
        statistic='count',
        bins=bins
    )
    
    # Plot heatmap
    pitch.heatmap(
        bin_statistic, 
        ax=ax, 
        cmap=cmap,
        edgecolors='#22312b',
        alpha=0.8
    )
    
    # Add labels with count
    pitch.label_heatmap(
        bin_statistic,
        ax=ax,
        color='white',
        fontsize=10,
        str_format='{:.0f}',
        ha='center',
        va='center'
    )
    
    ax.set_title(title, fontsize=16, fontweight='bold', color='white', pad=10)
    
    plt.tight_layout()
    return fig


def create_cxg_zones_map(
    df: pd.DataFrame,
    value_col: str = "statsbomb_xg",
    agg_func: str = "mean",
    title: str = "Average xG by Zone",
    pitch_type: str = "statsbomb",
    figsize: tuple = (12, 8),
) -> plt.Figure:
    """
    Create a zone-based visualization showing average xG/CxG per zone.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Shot data with location and value columns
    value_col : str
        Column to aggregate ('statsbomb_xg', 'cxg_pred', etc.)
    agg_func : str
        Aggregation function ('mean', 'sum', 'count')
    title : str
        Plot title
    pitch_type : str
        Pitch coordinate system
    figsize : tuple
        Figure size
    
    Returns:
    --------
    matplotlib.figure.Figure
    """
    pitch = VerticalPitch(
        pitch_type=pitch_type,
        half=True,
        pitch_color='#22312b',
        line_color='#c7d5cc',
        linewidth=1.5
    )
    
    fig, ax = pitch.draw(figsize=figsize)
    
    # Create bin statistic
    bin_statistic = pitch.bin_statistic(
        df['location_x'],
        df['location_y'],
        values=df[value_col],
        statistic=agg_func,
        bins=(6, 4)
    )
    
    # Plot heatmap
    pitch.heatmap(
        bin_statistic,
        ax=ax,
        cmap='RdYlGn',
        edgecolors='#22312b',
        alpha=0.85
    )
    
    # Add labels
    pitch.label_heatmap(
        bin_statistic,
        ax=ax,
        color='white',
        fontsize=11,
        str_format='{:.2f}',
        ha='center',
        va='center',
        fontweight='bold'
    )
    
    ax.set_title(title, fontsize=16, fontweight='bold', color='white', pad=10)
    
    plt.tight_layout()
    return fig


def create_goal_rate_comparison(
    df: pd.DataFrame,
    group_col: str,
    title: str = "Goal Rate by Category",
    figsize: tuple = (12, 8),
) -> plt.Figure:
    """
    Create a comparison visualization of goal rates and xG across categories.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Shot data with outcome and grouping columns
    group_col : str
        Column to group by (e.g., 'chain_label', 'pressure_state')
    title : str
        Plot title
    figsize : tuple
        Figure size
    
    Returns:
    --------
    matplotlib.figure.Figure
    """
    # Aggregate data
    agg = df.groupby(group_col).agg({
        'is_goal': ['mean', 'sum', 'count'],
        'statsbomb_xg': 'mean'
    }).round(3)
    
    agg.columns = ['goal_rate', 'goals', 'shots', 'avg_xg']
    agg = agg.sort_values('goal_rate', ascending=True)
    agg = agg[agg['shots'] >= 30]  # Filter low sample sizes
    
    if len(agg) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "Insufficient data", ha='center', va='center')
        return fig
    
    fig, ax = plt.subplots(figsize=figsize)
    
    y_pos = np.arange(len(agg))
    
    # Plot bars
    bars_goal = ax.barh(y_pos - 0.2, agg['goal_rate'], 0.4, 
                        label='Actual Goal Rate', color='#00ff87', alpha=0.8)
    bars_xg = ax.barh(y_pos + 0.2, agg['avg_xg'], 0.4, 
                      label='Average xG', color='#667eea', alpha=0.8)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(agg.index)
    ax.set_xlabel('Probability')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_xlim(0, max(agg['goal_rate'].max(), agg['avg_xg'].max()) * 1.1)
    
    # Add sample sizes
    for i, (idx, row) in enumerate(agg.iterrows()):
        ax.text(max(row['goal_rate'], row['avg_xg']) + 0.01, i, 
                f"n={int(row['shots'])}", va='center', fontsize=9, color='gray')
    
    plt.tight_layout()
    return fig


def create_team_finishing_chart(
    df: pd.DataFrame,
    title: str = "Goals vs CxG by Team",
    figsize: tuple = (14, 8),
) -> plt.Figure:
    """
    Create a chart showing goals vs CxG for each team.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Team aggregates with goals_for, cxg_for columns
    title : str
        Plot title
    figsize : tuple
        Figure size
    
    Returns:
    --------
    matplotlib.figure.Figure
    """
    plot_df = df.sort_values('cxg_for', ascending=True).copy()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    y_pos = np.arange(len(plot_df))
    
    # Scatter for CxG and Goals
    ax.scatter(plot_df['cxg_for'], y_pos, s=120, c='#667eea', 
               label='CxG', zorder=3, marker='o')
    ax.scatter(plot_df['goals_for'], y_pos, s=120, c='#00ff87', 
               label='Goals', zorder=3, marker='s')
    
    # Connect with lines
    for i, (_, row) in enumerate(plot_df.iterrows()):
        color = '#00ff87' if row['goals_for'] > row['cxg_for'] else '#ff6b6b'
        ax.plot([row['cxg_for'], row['goals_for']], [i, i], 
                color=color, alpha=0.5, linewidth=2)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df['team_id'].astype(str))
    ax.set_xlabel('Expected / Actual Goals', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig
