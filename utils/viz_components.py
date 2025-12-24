"""
Visualization Components for Anime Analysis
============================================

Reusable Plotly and Matplotlib visualization builders.
"""

import numpy as np
import plotly.graph_objects as go


def create_grouped_bar_chart(
    data,
    x_col,
    y_cols,
    y_labels,
    colors,
    title,
    x_title,
    y_title,
    figsize=(1400, 700),
    sort_by=None,
    ascending=False,
    customdata_cols=None,
    tickangle=0,
):
    """
    Create a grouped bar chart with Plotly.

    Parameters
    ----------
    data : pd.DataFrame
        Data to visualize
    x_col : str
        Column for x-axis
    y_cols : list of str
        Columns for y-axis (one bar per column)
    y_labels : list of str
        Legend labels for each y column
    colors : list of str
        Colors for each trace
    title : str
        Chart title
    x_title : str
        X-axis title
    y_title : str
        Y-axis title
    figsize : tuple, default=(1400, 700)
        Figure size (width, height) in pixels
    sort_by : str, optional
        Column to sort by
    ascending : bool, default=False
        Sort order
    customdata_cols : list of str, optional
        Additional columns for hover data
    tickangle : int, default=0
        X-axis tick label angle

    Returns
    -------
    plotly.graph_objects.Figure
    """
    data = data.copy()

    if sort_by:
        data = data.sort_values(sort_by, ascending=ascending)

    fig = go.Figure()

    for y_col, label, color in zip(y_cols, y_labels, colors):
        hover_template = f"<b>%{{x}}</b><br>{label}: %{{y:.1f}}%<br>"

        if customdata_cols:
            customdata = data[customdata_cols].values
            for i, col in enumerate(customdata_cols):
                hover_template += f"{col}: %{{customdata[{i}]:,}}<br>"
        else:
            customdata = None

        hover_template += "<extra></extra>"

        fig.add_trace(
            go.Bar(
                x=data[x_col],
                y=data[y_col],
                name=label,
                marker=dict(color=color),
                text=data[y_col].apply(lambda x: f"{x:.1f}%"),
                textposition="outside",
                hovertemplate=hover_template,
                customdata=customdata,
            )
        )

    # Calculate y-axis range
    max_val = data[y_cols].max().max()

    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        height=figsize[1],
        width=figsize[0],
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=12),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#CCCCCC",
            borderwidth=1,
        ),
        barmode="group",
        hovermode="x unified",
        font=dict(size=12),
        xaxis=dict(tickangle=tickangle),
        yaxis=dict(range=[0, max_val * 1.15]),
    )

    return fig


def create_dual_line_distribution(
    data1,
    data2,
    bins,
    labels,
    colors,
    title,
    x_title,
    y_title,
    figsize=(1400, 600),
):
    """
    Create interactive line distribution plot with two overlaid metrics.

    Parameters
    ----------
    data1 : pd.Series
        First metric data (e.g., completion_rate)
    data2 : pd.Series
        Second metric data (e.g., drop_rate)
    bins : np.ndarray
        Bin edges for histogram
    labels : list of str
        Labels for each metric [label1, label2]
    colors : list of str
        Colors for each metric [color1, color2]
    title : str
        Chart title
    x_title : str
        X-axis title
    y_title : str
        Y-axis title
    figsize :  tuple, default=(1400, 600)
        Figure size (width, height) in pixels

    Returns
    -------
    plotly.graph_objects.Figure
    """
    # Calculate histograms
    counts1, _ = np.histogram(data1.dropna(), bins=bins)
    counts2, _ = np.histogram(data2.dropna(), bins=bins)

    bin_centers = (bins[:-1] + bins[1:]) / 2

    fig = go.Figure()

    # Add first trace
    fig.add_trace(
        go.Scatter(
            x=bin_centers,
            y=counts1,
            name=labels[0],
            mode="lines+markers",
            line=dict(color="rgb(46, 204, 113)", width=3),
            marker=dict(size=6, color="rgb(46, 204, 113)"),
            fill="tozeroy",
            fillcolor="rgba(46, 204, 113, 0.3)",
            hovertemplate=f"<b>{labels[0]}:  %{{x:.0f}}%</b><br>Anime Count: %{{y: ,}}<extra></extra>",
        )
    )

    # Add second trace
    fig.add_trace(
        go.Scatter(
            x=bin_centers,
            y=counts2,
            name=labels[1],
            mode="lines+markers",
            line=dict(color="rgb(231, 76, 60)", width=3),
            marker=dict(size=6, color="rgb(231, 76, 60)"),
            fill="tozeroy",
            fillcolor="rgba(231, 76, 60, 0.3)",
            hovertemplate=f"<b>{labels[1]}: %{{x:.0f}}%</b><br>Anime Count: %{{y:,}}<extra></extra>",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        template="plotly_white",
        width=figsize[0],
        height=figsize[1],
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#CCCCCC",
            borderwidth=1,
        ),
        hovermode="x unified",
    )

    return fig
