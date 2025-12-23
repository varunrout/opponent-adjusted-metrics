"""
Chart Components using Plotly
Interactive charts for non-pitch visualizations
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


def create_model_comparison_chart(
    baseline_metrics: dict,
    contextual_metrics: dict,
    metric_names: list = None,
) -> go.Figure:
    """
    Create a comparison chart between baseline and contextual model metrics.
    
    Parameters:
    -----------
    baseline_metrics : dict
        Dictionary with baseline model metrics
    contextual_metrics : dict
        Dictionary with contextual model metrics
    metric_names : list
        List of metric names to compare
    
    Returns:
    --------
    plotly.graph_objects.Figure
    """
    if metric_names is None:
        metric_names = ['auc_mean', 'brier_mean', 'log_loss_mean']
    
    display_names = {
        'auc_mean': 'ROC AUC ↑',
        'brier_mean': 'Brier Score ↓',
        'log_loss_mean': 'Log Loss ↓'
    }
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[display_names.get(m, m) for m in metric_names],
        horizontal_spacing=0.1
    )
    
    colors = {
        'baseline': '#ff6b6b',
        'contextual': '#00ff87'
    }
    
    for i, metric in enumerate(metric_names, 1):
        baseline_val = baseline_metrics.get(metric, 0)
        contextual_val = contextual_metrics.get(metric, 0)
        
        fig.add_trace(
            go.Bar(
                x=['Baseline'],
                y=[baseline_val],
                name='Baseline' if i == 1 else None,
                marker_color=colors['baseline'],
                showlegend=(i == 1),
                text=[f'{baseline_val:.4f}'],
                textposition='outside'
            ),
            row=1, col=i
        )
        
        fig.add_trace(
            go.Bar(
                x=['CxG Model'],
                y=[contextual_val],
                name='CxG Model' if i == 1 else None,
                marker_color=colors['contextual'],
                showlegend=(i == 1),
                text=[f'{contextual_val:.4f}'],
                textposition='outside'
            ),
            row=1, col=i
        )
    
    fig.update_layout(
        title='Model Performance Comparison',
        height=400,
        barmode='group',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig


def create_reliability_diagram(
    predictions: pd.Series,
    actuals: pd.Series,
    n_bins: int = 10,
    title: str = "Reliability Diagram"
) -> go.Figure:
    """
    Create a calibration/reliability diagram.
    
    Parameters:
    -----------
    predictions : pd.Series
        Predicted probabilities
    actuals : pd.Series
        Actual outcomes (0/1)
    n_bins : int
        Number of bins for calibration
    title : str
        Plot title
    
    Returns:
    --------
    plotly.graph_objects.Figure
    """
    # Bin predictions
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(predictions, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    calibration_data = []
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            mean_pred = predictions[mask].mean()
            mean_actual = actuals[mask].mean()
            count = mask.sum()
            calibration_data.append({
                'bin_center': (bins[i] + bins[i+1]) / 2,
                'mean_predicted': mean_pred,
                'mean_actual': mean_actual,
                'count': count
            })
    
    cal_df = pd.DataFrame(calibration_data)
    
    fig = go.Figure()
    
    # Perfect calibration line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Perfect Calibration',
        line=dict(color='gray', dash='dash', width=2)
    ))
    
    # Actual calibration
    fig.add_trace(go.Scatter(
        x=cal_df['mean_predicted'],
        y=cal_df['mean_actual'],
        mode='lines+markers',
        name='Model Calibration',
        line=dict(color='#00ff87', width=3),
        marker=dict(size=cal_df['count'] / cal_df['count'].max() * 20 + 5)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Mean Predicted Probability',
        yaxis_title='Observed Frequency',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=500,
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1])
    )
    
    return fig


def create_finishing_delta_chart(
    df: pd.DataFrame,
    team_col: str = 'team_id',
    goals_col: str = 'goals_for',
    cxg_col: str = 'cxg_for',
    title: str = "Finishing Performance: Goals - CxG"
) -> go.Figure:
    """
    Create a diverging bar chart showing goals minus CxG.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Team aggregates
    team_col : str
        Column with team identifier
    goals_col : str
        Column with goals scored
    cxg_col : str
        Column with CxG total
    title : str
        Plot title
    
    Returns:
    --------
    plotly.graph_objects.Figure
    """
    plot_df = df.copy()
    plot_df['delta'] = plot_df[goals_col] - plot_df[cxg_col]
    plot_df = plot_df.sort_values('delta', ascending=True)
    
    colors = ['#00ff87' if x > 0 else '#ff6b6b' for x in plot_df['delta']]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=plot_df[team_col].astype(str),
        x=plot_df['delta'],
        orientation='h',
        marker_color=colors,
        text=plot_df['delta'].round(1),
        textposition='outside'
    ))
    
    # Add vertical line at 0
    fig.add_vline(x=0, line_width=2, line_color='white')
    
    fig.update_layout(
        title=title,
        xaxis_title='Goals - CxG (Over/Under Performance)',
        yaxis_title='Team',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=max(400, len(plot_df) * 25),
    )
    
    return fig


def create_scatter_goals_vs_cxg(
    df: pd.DataFrame,
    title: str = "Goals vs CxG",
    x_col: str = 'cxg_for',
    y_col: str = 'goals_for',
    label_col: str = 'team_id'
) -> go.Figure:
    """
    Create a scatter plot of goals vs CxG with team labels.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Team aggregates
    title : str
        Plot title
    x_col : str
        Column for x-axis
    y_col : str
        Column for y-axis
    label_col : str
        Column for point labels
    
    Returns:
    --------
    plotly.graph_objects.Figure
    """
    fig = go.Figure()
    
    # Diagonal line (perfect performance)
    max_val = max(df[x_col].max(), df[y_col].max()) * 1.1
    min_val = min(df[x_col].min(), df[y_col].min()) * 0.9
    
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Match',
        line=dict(color='gray', dash='dash', width=2)
    ))
    
    # Team points
    fig.add_trace(go.Scatter(
        x=df[x_col],
        y=df[y_col],
        mode='markers+text',
        name='Teams',
        marker=dict(
            size=12,
            color=df[y_col] - df[x_col],
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title='Delta')
        ),
        text=df[label_col].astype(str),
        textposition='top center',
        textfont=dict(size=9)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='CxG (Expected)',
        yaxis_title='Goals (Actual)',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=600,
        xaxis=dict(range=[min_val, max_val]),
        yaxis=dict(range=[min_val, max_val])
    )
    
    return fig


def create_feature_importance_chart(
    feature_effects: pd.DataFrame,
    top_n: int = 15,
    title: str = "Feature Importance (Coefficients)"
) -> go.Figure:
    """
    Create a bar chart of feature importance/coefficients.
    
    Parameters:
    -----------
    feature_effects : pd.DataFrame
        DataFrame with feature names and coefficients
    top_n : int
        Number of top features to show
    title : str
        Plot title
    
    Returns:
    --------
    plotly.graph_objects.Figure
    """
    # Assuming columns: 'feature', 'coefficient'
    df = feature_effects.copy()
    
    if 'coefficient' in df.columns:
        df['abs_coef'] = df['coefficient'].abs()
        df = df.nlargest(top_n, 'abs_coef')
        df = df.sort_values('coefficient', ascending=True)
        
        colors = ['#00ff87' if x > 0 else '#ff6b6b' for x in df['coefficient']]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=df['feature'],
            x=df['coefficient'],
            orientation='h',
            marker_color=colors,
            text=df['coefficient'].round(3),
            textposition='outside'
        ))
        
        fig.add_vline(x=0, line_width=2, line_color='white')
        
        fig.update_layout(
            title=title,
            xaxis_title='Coefficient (Log-Odds)',
            yaxis_title='Feature',
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=max(400, top_n * 30),
        )
        
        return fig
    
    # Fallback empty chart
    return go.Figure()


def create_game_state_heatmap(
    df: pd.DataFrame,
    x_col: str = 'minute_bucket_label',
    y_col: str = 'score_state',
    value_col: str = 'is_goal',
    agg: str = 'mean',
    title: str = "Goal Rate by Game State"
) -> go.Figure:
    """
    Create a heatmap of goal rates by game state and minute.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Shot data
    x_col : str
        Column for x-axis (e.g., minute buckets)
    y_col : str
        Column for y-axis (e.g., score state)
    value_col : str
        Column to aggregate
    agg : str
        Aggregation function
    title : str
        Plot title
    
    Returns:
    --------
    plotly.graph_objects.Figure
    """
    pivot = df.pivot_table(
        values=value_col,
        index=y_col,
        columns=x_col,
        aggfunc=agg
    ).round(3)
    
    fig = px.imshow(
        pivot,
        labels=dict(x="Minute", y="Score State", color="Goal Rate"),
        color_continuous_scale='RdYlGn',
        title=title,
        text_auto='.2f'
    )
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=400,
    )
    
    return fig
