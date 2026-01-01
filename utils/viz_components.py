"""
Visualization Components for Anime Analysis
============================================

Reusable Plotly and Matplotlib visualization builders.
"""

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns


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


def create_horizontal_bar_chart(
    data,
    y_col,
    x_col,
    title,
    x_title,
    y_title,
    figsize=(12, 8),
    colormap="plasma",
    add_labels=True,
    invert_y=True,
    ax=None,
):
    """
    Create a horizontal bar chart with Matplotlib.

    Parameters
    ----------
    data : pd.DataFrame
        Data to visualize (should be pre-sorted)
    y_col : str
        Column for y-axis (categories)
    x_col : str
        Column for x-axis (values)
    title : str
        Chart title
    x_title : str
        X-axis label
    y_title :  str
        Y-axis label
    figsize : tuple, default=(12, 8)
        Figure size (width, height)
    colormap : str, default="plasma"
        Matplotlib colormap name
    add_labels : bool, default=True
        Whether to add value labels on bars
    invert_y : bool, default=True
        Whether to invert y-axis (top = highest)
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.

    Returns
    -------
    matplotlib.figure.Figure, matplotlib.axes.Axes
    """
    # Create figure only if ax not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    cmap = getattr(cm, colormap)
    colors = cmap(np.linspace(0.3, 0.9, len(data)))

    bars = ax.barh(range(len(data)), data[x_col], color=colors)

    ax.set_yticks(range(len(data)))
    ax.set_yticklabels(data[y_col].values)
    ax.set_xlabel(x_title, fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    if add_labels:
        for i, (bar, value) in enumerate(zip(bars, data[x_col])):
            # Format based on value size
            if value >= 1000:
                label = f" {int(value):,}"
            else:
                label = f" {value:. 1f}"

            ax.text(
                value, i, label, va="center", ha="left", fontsize=10, fontweight="bold"
            )

    if invert_y:
        ax.invert_yaxis()

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    # Only call tight_layout if we created the figure
    if ax is None:
        plt.tight_layout()

    return fig, ax


def create_dual_bar_chart(
    data,
    x_col,
    y_cols,
    titles,
    y_titles,
    colors,
    figsize=(14, 5),
    bar_label_formats=None,
    y_limits=None,
):
    """
    Create side-by-side bar charts with Matplotlib.

    Parameters
    ----------
    data : pd.DataFrame
        Data to visualize
    x_col : str
        Column for x-axis (categories)
    y_cols : list of str
        Two columns for y-axis values [left_plot, right_plot]
    titles : list of str
        Two titles [left_title, right_title]
    y_titles : list of str
        Two y-axis labels [left_label, right_label]
    colors : list of str
        Two colors [left_color, right_color]
    figsize : tuple, default=(14, 5)
        Figure size (width, height)
    bar_label_formats : list of str, optional
        Format strings for bar labels [left_format, right_format]
        e.g., ['{: ,.0f}', '{:. 2f}']
    y_limits : list of tuple, optional
        Y-axis limits [(left_min, left_max), (right_min, right_max)]

    Returns
    -------
    matplotlib.figure.Figure, np.ndarray of matplotlib.axes.Axes
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    if bar_label_formats is None:
        bar_label_formats = ["{:,.0f}", "{:. 2f}"]

    for idx, (ax, y_col, title, y_title, color, label_fmt) in enumerate(
        zip(axes, y_cols, titles, y_titles, colors, bar_label_formats)
    ):
        # Create bars
        ax.bar(
            range(len(data)), data[y_col], color=color, edgecolor="none", linewidth=0
        )

        # Set ticks and labels
        ax.set_xticks(range(len(data)))
        ax.set_xticklabels(data[x_col], rotation=45, ha="right")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Episode Range")
        ax.set_ylabel(y_title)
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Add bar labels
        for i, (bar, value) in enumerate(zip(ax.patches, data[y_col])):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                label_fmt.format(value),
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                color="black",
            )

        # Set y-axis limits
        if y_limits and y_limits[idx]:
            ax.set_ylim(y_limits[idx])
        else:
            # Auto-adjust with padding
            max_val = data[y_col].max()
            ax.set_ylim(0, max_val * 1.15)

    plt.tight_layout()

    return fig, axes


def create_hexbin_violin_plot(
    data,
    x_col,
    y_col,
    category_col,
    title_left,
    title_right,
    x_title,
    y_title,
    figsize=(15, 6),
    gridsize=30,
    palette="viridis",
):
    """
    Create hexbin density plot and violin plot side-by-side.

    Parameters
    ----------
    data :  pd.DataFrame
        Data to visualize
    x_col : str
        Column for x-axis (continuous variable)
    y_col : str
        Column for y-axis (continuous variable)
    category_col : str
        Column for categorical grouping (violin plot)
    title_left : str
        Title for hexbin plot (left)
    title_right : str
        Title for violin plot (right)
    x_title : str
        X-axis label
    y_title : str
        Y-axis label
    figsize : tuple, default=(15, 6)
        Figure size (width, height)
    gridsize : int, default=30
        Hexbin grid size
    palette :  str, default="viridis"
        Seaborn color palette

    Returns
    -------
    matplotlib.figure.Figure, np.ndarray of matplotlib.axes.Axes
    correlation_stats : dict
        Dictionary with 'pearson_r', 'interpretation', 'n_samples'
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Calculate correlation
    correlation = data[x_col].corr(data[y_col])
    abs_corr = abs(correlation)

    if abs_corr < 0.3:
        interpretation = "Weak"
    elif abs_corr < 0.7:
        interpretation = "Moderate"
    else:
        interpretation = "Strong"

    # Plot 1: Hexbin Density Heatmap
    hb = axes[0].hexbin(
        data[x_col],
        data[y_col],
        gridsize=gridsize,
        bins="log",
        cmap="YlGnBu",
        xscale="log",
        mincnt=1,
    )

    axes[0].set_title(
        f"{title_left}\n(Pearson r = {correlation:.3f})", fontweight="bold", fontsize=12
    )
    axes[0].set_xlabel(f"{x_title} (Log Scale)")
    axes[0].set_ylabel(y_title)
    axes[0].grid(alpha=0.3)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    # Add colorbar
    cbar = fig.colorbar(hb, ax=axes[0], label="Concentration (Log Scale)")

    # Plot 2: Violin Plot
    sns.violinplot(
        data=data,
        x=category_col,
        y=y_col,
        hue=category_col,
        palette=palette,
        inner="quartile",
        ax=axes[1],
        legend=False,
    )

    axes[1].set_title(title_right, fontweight="bold", fontsize=12)
    axes[1].set_xlabel("Episode Range")
    axes[1].set_ylabel(y_title)
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    plt.tight_layout()

    # Correlation statistics
    correlation_stats = {
        "pearson_r": round(correlation, 4),
        "interpretation": interpretation,
        "n_samples": len(data),
    }

    return fig, axes, correlation_stats


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


def create_dual_horizontal_bar_chart(
    data_left,
    data_right,
    y_col,
    x_cols,
    titles,
    x_titles,
    colors,
    figsize=(16, 8),
    x_limits=None,
    colormap_left="viridis_r",
    colormap_right="plasma_r",
):
    """
    Create side-by-side horizontal bar charts with different metrics.

    Parameters
    ----------
    data_left : pd.DataFrame
        Data for left chart (pre-sorted)
    data_right : pd. DataFrame
        Data for right chart (pre-sorted)
    y_col : str
        Column name for y-axis labels (should exist after reset_index)
    x_cols : list of str
        Column names for x-axis values [left_col, right_col]
    titles : list of str
        Titles for each chart [left_title, right_title]
    x_titles : list of str
        X-axis labels [left_label, right_label]
    colors : list of str
        Base colors [left_color, right_color]
    figsize : tuple, default=(16, 8)
        Figure size (width, height)
    x_limits : list of tuple, optional
        X-axis limits [(left_min, left_max), (right_min, right_max)]
    colormap_left : str, default='viridis_r'
        Colormap for left chart
    colormap_right : str, default='plasma_r'
        Colormap for right chart

    Returns
    -------
    matplotlib.figure.Figure, np.ndarray of matplotlib.axes.Axes
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Prepare data (sort for display)
    data_left_sorted = data_left.sort_values(x_cols[0], ascending=True)
    data_right_sorted = data_right.sort_values(x_cols[1], ascending=True)

    # Get colormaps
    cmap_left = getattr(cm, colormap_left)
    cmap_right = getattr(cm, colormap_right)

    colors_left = cmap_left(np.linspace(0.3, 0.9, len(data_left_sorted)))
    colors_right = cmap_right(np.linspace(0.3, 0.9, len(data_right_sorted)))

    # Left chart
    axes[0].barh(
        range(len(data_left_sorted)),
        data_left_sorted[x_cols[0]],
        color=colors_left,
    )
    axes[0].set_yticks(range(len(data_left_sorted)))
    axes[0].set_yticklabels(data_left_sorted[y_col].values)
    axes[0].set_xlabel(x_titles[0], fontsize=12, fontweight="bold")
    axes[0].set_title(titles[0], fontsize=12, fontweight="bold")
    axes[0].grid(axis="x", linestyle="--", alpha=0.6)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    # Right chart
    axes[1].barh(
        range(len(data_right_sorted)),
        data_right_sorted[x_cols[1]],
        color=colors_right,
    )
    axes[1].set_yticks(range(len(data_right_sorted)))
    axes[1].set_yticklabels(data_right_sorted[y_col].values)
    axes[1].set_xlabel(x_titles[1], fontsize=12, fontweight="bold")
    axes[1].set_title(titles[1], fontsize=12, fontweight="bold")
    axes[1].grid(axis="x", linestyle="--", alpha=0.6)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    # Apply x-axis limits if provided
    if x_limits:
        if x_limits[0]:
            axes[0].set_xlim(x_limits[0])
        if x_limits[1]:
            axes[1].set_xlim(x_limits[1])

    plt.tight_layout()

    return fig, axes


def create_ring_chart(
    data,
    value_col,
    label_col,
    title,
    figsize=(20, 12),
    colormap="tab10",
    center_text="Total",
    percent_threshold=2
):
    """
    Generic function to create a Ring Chart (Donut).
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the data.
    value_col : str
        Name of the column with numerical values.
    label_col : str
        Name of the column with category labels.
    title : str
        Chart title.
    figsize : tuple, default=(20, 12)
        Figure size.
    colormap : str, default='tab10'
        Matplotlib colormap name.
    center_text : str, default='Total'
        Title text to display in the center hole.
    percent_threshold : float, default=3
        Hide percentage text inside slices smaller than this value.

    Returns
    -------
    matplotlib.figure.Figure, matplotlib.axes.Axes
    """
    # Sort by value descending (Visual best practice)
    data = data.sort_values(by=value_col, ascending=False).copy()
    
    values = data[value_col]
    labels = data[label_col]
    
    # Setup Figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Handle Colors
    if colormap == "tab10":
        colors = plt.get_cmap('tab10').colors
    else:
        cmap = getattr(cm, colormap)
        colors = cmap(np.linspace(0.2, 0.8, len(data)))

    # Draw Chart
    wedges, texts, autotexts = ax.pie(
        values,
        labels=None, # We use a side legend instead
        autopct='%1.1f%%',
        startangle=90,
        pctdistance=0.82,
        colors=colors,
        wedgeprops=dict(width=0.4, edgecolor='white'),
        textprops={'fontsize': 14, 'weight': 'bold', 'color': 'white'}
    )

    # Filter Small Percentages (because the text becomes unreadable)
    total_val = values.sum()
    for autotext, val in zip(autotexts, values):
        percentage = val / total_val * 100
        if percentage < percent_threshold:
            autotext.set_text("") 

    # Titles & Center Text
    ax.set_title(title, fontsize=24, fontweight='bold')
    
    ax.text(0, 0, f"{center_text}\n{total_val:,.0f}", 
            ha='center', va='center', fontsize=18, fontweight='bold', color='#333')

    # Legend
    ax.legend(
        wedges, labels,
        title_fontsize=16,
        fontsize=14,
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1),
        frameon=False
    )

    return fig, ax


def create_heatmap(
    data, title, x_label, y_label, cmap="rocket", figsize=(10, 8), annot_format=".1f"
):
    """
    Create annotated heatmap with seaborn.

    Parameters
    ----------
    data : pd.DataFrame
        Pivot table with values to plot
    title : str
        Chart title
    x_label :  str
        X-axis label
    y_label : str
        Y-axis label
    cmap : str, default='rocket'
        Seaborn colormap
    figsize : tuple, default=(10, 8)
        Figure size (width, height)
    annot_format : str, default='.1f'
        Annotation format string

    Returns
    -------
    matplotlib.figure.Figure, matplotlib.axes.Axes
    """

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        data,
        annot=True,
        fmt=annot_format,
        cmap=cmap,
        cbar_kws={"label": "Percentage (%)"},
        linewidths=1.5,
        linecolor="white",
        ax=ax,
    )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.set_xlabel(x_label, fontsize=12, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=12, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()

    return fig, ax

