"""
Geographic Analysis Functions
==============================

User distribution by country and language.
"""

import numpy as np
import math as math
import plotly.express as px

from .data_prep import prepare_country_data


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
