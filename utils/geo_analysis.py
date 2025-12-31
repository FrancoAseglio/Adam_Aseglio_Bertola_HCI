"""
Geographic Analysis Functions
==============================

User distribution by country and language.
"""

import numpy as np
import math as math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import plotly.express as px

from .data_prep import prepare_country_data
from .dev_utils import add_count_on_bars


def create_animated_map(profiles_df):
    country_year_counts = prepare_country_data(profiles_df)

    mask = country_year_counts['cumulative_users'] > 0
    country_year_counts.loc[mask, 'log_cumulative'] = np.log10(country_year_counts.loc[mask, 'cumulative_users'])
    
    years_sorted = sorted(country_year_counts["year_label"].unique())
    max_users_log = country_year_counts["log_cumulative"].max()
    max_exponent = math.ceil(max_users_log)
    min_exponent = 0 
    
    fig = px.choropleth(
        country_year_counts,
        locations="country_code",
        locationmode="ISO-3",
        color="log_cumulative",
        hover_name="country",
        hover_data={
            "country_code": False,
            "cumulative_users": ":,",
            "user_count": ":,",
            "year_label": True,
            "log_cumulative": False,
        },
        animation_frame="year_label",
        color_continuous_scale="Turbo",
        title="Global Anime Users Distribution by Year (2004-2025) - Cumulative Total (Log Scale)",
        projection="natural earth",
        category_orders={"year_label": years_sorted},
        range_color=[min_exponent, max_exponent], 
    )

    tick_vals = list(range(min_exponent, max_exponent + 1))
    tick_text = [f"10^{i}" for i in tick_vals]

    fig.update_layout(
        height=600,
        geo=dict(showframe=False, showcoastlines=True, projection_type="natural earth"),
        coloraxis_colorbar=dict(
            title="Total Users", 
            tickvals=tick_vals,
            ticktext=tick_text,
            ticks="outside",
            title_side="top",
        ),
    )

    return fig


def language_vs_geographic_distribution(profiles_df, figsize=(10, 6)):
    """
    Create a bar chart showing user distribution by primary language.

    Parameters
    ----------
    profiles_df : pd.DataFrame
        User profiles with location data
    figsize : tuple, default=(10, 6)

    Returns
    -------
    matplotlib.figure.Figure, matplotlib.axes.Axes
    """
    language_map = {
        "United States": "English",
        "Canada": "English",
        "United Kingdom": "English",
        "Australia": "English",
        "New Zealand": "English",
        "Ireland": "English",
        "France": "French",
        "Belgium": "French",
        "Switzerland": "French",
        "Germany": "German",
        "Austria": "German",
        "Spain": "Spanish",
        "Mexico": "Spanish",
        "Argentina": "Spanish",
        "Brazil": "Portuguese",
        "Portugal": "Portuguese",
        "Italy": "Italian",
        "Japan": "Japanese",
        "South Korea": "Korean",
        "China": "Chinese",
        "Thailand": "Thai",
        "Philippines": "English",
        "Poland": "Polish",
        "Russia": "Russian",
    }

    profiles_with_lang = profiles_df.copy()
    profiles_with_lang["language"] = profiles_with_lang["location"].map(language_map)

    language_users = (
        profiles_with_lang.groupby("language").size().reset_index(name="total_users")
    )
    language_users = language_users.dropna().sort_values("total_users", ascending=False)

    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.bar(
        range(len(language_users)),
        language_users["total_users"],
        color=cm.viridis_r(np.linspace(0.2, 0.9, len(language_users))),
    )

    ax.set_xticks(range(len(language_users)))
    ax.set_xticklabels(language_users["language"].values, rotation=45, ha="right")
    ax.set_ylabel("Total Users", fontsize=12, fontweight="bold")
    ax.set_title(
        "Anime User Distribution by Primary Language",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    add_count_on_bars(ax)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()

    return fig, ax

