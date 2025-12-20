# ========================================
# IMPORTS AND DEPENDENCIES
# ========================================

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pylab as plt
import plotly.graph_objects as go
import plotly.express as px

from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

sys.path.append(str(Path.cwd().parent))

from utils.dev_utils import add_count_on_bars, parse_list

# ========================================
# DATA LOADING
# ========================================

INPUT_DIR_PATH = Path("~/datasets/cleaned/").expanduser()
if not INPUT_DIR_PATH.exists():
    INPUT_DIR_PATH = Path("C:\\Users\\Lorenzo\\OneDrive\\Desktop\\datasets\\cleaned\\")

INPUT_DIR_PATH = str(INPUT_DIR_PATH) + "/"
FILE_EXTENSION = ".csv"

character_anime_works_df = pd.read_csv(INPUT_DIR_PATH + "character_anime_works_clean" + FILE_EXTENSION)
character_nicknames_df = pd.read_csv(INPUT_DIR_PATH + "character_nicknames_clean" + FILE_EXTENSION)
characters_df = pd.read_csv(INPUT_DIR_PATH + "characters_clean" + FILE_EXTENSION)
details_df = pd.read_csv(INPUT_DIR_PATH + "details_clean" + FILE_EXTENSION)
favs_df = pd.read_csv(INPUT_DIR_PATH + "favs_clean" + FILE_EXTENSION)
person_alternate_names_df = pd.read_csv(INPUT_DIR_PATH + "person_alternate_names_clean" + FILE_EXTENSION)
person_anime_works_df = pd.read_csv(INPUT_DIR_PATH + "person_anime_works_clean" + FILE_EXTENSION)
person_details_df = pd.read_csv(INPUT_DIR_PATH + "person_details_clean" + FILE_EXTENSION)
person_voice_works_df = pd.read_csv(INPUT_DIR_PATH + "person_voice_works_clean" + FILE_EXTENSION)
profiles_df = pd.read_csv(INPUT_DIR_PATH + "profiles_clean" + FILE_EXTENSION)
# ratings_df = pd.read_csv(INPUT_DIR_PATH + "ratings_clean" + FILE_EXTENSION)
recommendations_df = pd.read_csv(INPUT_DIR_PATH + "recommendations_clean" + FILE_EXTENSION)
stats_df = pd.read_csv(INPUT_DIR_PATH + "stats_clean" + FILE_EXTENSION)


# ========================================
# CHARACTER ANALYSIS FUNCTIONS
# ========================================

def top_n_most_favorited_characters(top_n=10, figsize=(12, 8)):
    """
    Plot a bar chart of the most favorited anime characters.

    Parameters
    ----------
    top_n : int, default=10
        Number of characters to display.
    figsize : tuple, default=(12, 8)
        Matplotlib figure size.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
        Visualization of the top-N most favorited characters.
    """

    # Compute user favorites
    char_favs = favs_df[favs_df["fav_type"] == "character"].copy()
    char_fav_counts = char_favs.groupby("id").size().reset_index(name="user_favs")

    # Merge with character dataset
    char_popularity = characters_df.merge(char_fav_counts, left_on="character_mal_id", right_on="id", how="left")
    char_popularity["user_favs"] = char_popularity["user_favs"].fillna(0)

    # Total popularity score
    char_popularity["total_favs"] = (char_popularity["favorites"].fillna(0) + char_popularity["user_favs"])

    # Extract top entries
    top_n_chars = char_popularity.nlargest(top_n, "total_favs").sort_values("total_favs", ascending=False)

    # --- Visualization ---
    fig, ax = plt.subplots(figsize=figsize)
    plot_data = top_n_chars.sort_values("total_favs", ascending=True)

    bars = ax.barh(
        range(len(plot_data)),
        plot_data["total_favs"],
        color=cm.plasma(np.linspace(0.3, 0.9, len(plot_data))),
    )

    ax.set_yticks(range(len(plot_data)))
    ax.set_yticklabels(plot_data["name"].values)
    ax.set_xlabel("Total Favorites", fontsize=12)
    ax.set_title(f"Top {top_n} Most Favorited Characters", fontsize=14, fontweight="bold", pad=20)

    # Value labels
    for i, (bar, value) in enumerate(zip(bars, plot_data["total_favs"])):
        ax.text(
            value,
            i,
            f" {int(value):,}",
            va="center",
            ha="left",
            fontsize=10,
            fontweight="bold",
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.6, linestyle="--")

    plt.tight_layout()
    return fig, ax


def character_popularity_by_role(figsize=(14, 5)):
    """
    Analyze and visualize character popularity based on their role in anime.

    Parameters
    ----------
    figsize : tuple, default=(14, 5)
        Matplotlib figure size for the 1x2 layout.

    Returns
    -------
    fig, axes : matplotlib Figure and Axes array
        Side-by-side bar charts of average favorites and character counts by role.
    """

    # Compute user favorites
    char_favs = favs_df[favs_df["fav_type"] == "character"].copy()
    char_fav_counts = char_favs.groupby("id").size().reset_index(name="user_favs")

    # Merge with character details
    char_popularity = characters_df.merge(char_fav_counts, left_on="character_mal_id", right_on="id", how="left")
    char_popularity["user_favs"] = char_popularity["user_favs"].fillna(0)
    char_popularity["total_favs"] = (char_popularity["favorites"].fillna(0) + char_popularity["user_favs"])

    # Merge with role dataset
    char_roles = character_anime_works_df.merge(
        char_popularity[["character_mal_id", "name", "total_favs"]],
        on="character_mal_id",
        how="inner",
    )

    # Aggregate by role
    role_analysis = (
        char_roles.groupby("role")
        .agg({"total_favs": "mean", "character_mal_id": "count"})
        .round(2)
    )
    role_analysis.columns = ["Avg Favorites", "Count"]
    role_analysis = role_analysis.sort_values("Avg Favorites", ascending=False)

    # --- Visualization ---
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Average favorites
    role_analysis["Avg Favorites"].plot(kind="bar", ax=axes[0], color=["#FF6B6B", "#4ECDC4"])
    axes[0].set_title(
        "Average Favorites by Character Role",
        fontsize=12,
        fontweight="bold",
    )
    axes[0].set_ylabel("Average Favorites")
    axes[0].set_xlabel("Role")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].grid(axis="y", alpha=0.5, linestyle="--")
    add_count_on_bars(axes[0])

    # Count per role
    role_analysis["Count"].plot(kind="bar", ax=axes[1], color=["#FF6B6B", "#4ECDC4"])
    axes[1].set_title(
        "Number of Characters by Role",
        fontsize=12,
        fontweight="bold",
    )
    axes[1].set_ylabel("Character Count")
    axes[1].set_xlabel("Role")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].grid(axis="y", alpha=0.5, linestyle="--")
    add_count_on_bars(axes[1])

    plt.tight_layout()
    return fig, axes


def character_popularity_by_genre_bars(min_char_count=10, top_n=10, figsize=(1200, 700)):
    """
    Interactive horizontal bar chart showing top genres.
    Bar length shows avg favorites, color shows character count (volume).

    Parameters
    ----------
    min_char_count : int, default=10
        Minimum number of characters required for a genre to be included.
    top_n : int, default=10
        Number of top genres to display.
    figsize : tuple, default=(1200, 700)
        Figure size in pixels (width, height).

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Interactive bar chart.
    """

    # Parse and explode genres
    details_df["genres_list"] = details_df["genres"].astype(str).apply(parse_list)
    genre_df = (
        details_df[["mal_id", "title", "genres_list"]]
        .explode("genres_list")
        .query("genres_list != ''")
    )

    # Compute user favorites
    char_favs = favs_df[favs_df["fav_type"] == "character"].copy()
    char_fav_counts = char_favs.groupby("id").size().reset_index(name="user_favs")

    # Merge with character details
    char_popularity = characters_df.merge(char_fav_counts, left_on="character_mal_id", right_on="id", how="left")
    char_popularity["user_favs"] = char_popularity["user_favs"].fillna(0)
    char_popularity["total_favs"] = (char_popularity["favorites"].fillna(0) + char_popularity["user_favs"])

    # Merge character-anime works with genres
    char_genre = character_anime_works_df.merge(
        genre_df, left_on="anime_mal_id", right_on="mal_id", how="inner"
    ).merge(
        char_popularity[["character_mal_id", "name", "total_favs"]],
        on="character_mal_id",
        how="inner",
    )

    # Aggregate by genre
    genre_analysis = (
        char_genre.groupby("genres_list")
        .agg({"total_favs": ["mean", "sum"], "character_mal_id": "count"})
        .round(2)
    )
    genre_analysis.columns = ["Avg Favorites", "Total Favorites", "Character Count"]
    genre_analysis = genre_analysis[genre_analysis["Character Count"] >= min_char_count]

    # Get top N genres
    plot_df = genre_analysis.reset_index().rename(columns={"genres_list": "Genre"})
    plot_df = plot_df.nlargest(top_n, "Avg Favorites")

    # --- Visualization ---
    fig = go.Figure()

    # Add horizontal bar chart colored by character count
    fig.add_trace(
        go.Bar(
            y=plot_df["Genre"],
            x=plot_df["Avg Favorites"],
            name="Avg Favorites",
            orientation="h",
            marker=dict(
                color=plot_df["Character Count"],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Character<br>Count"),
            ),
            text=plot_df["Avg Favorites"].apply(lambda x: f"{x:,.0f}"),
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>"
            + "Avg Favorites: %{x:,.2f}<br>"
            + "Character Count: %{customdata[0]:,}<br>"
            + "Total Favorites: %{customdata[1]:,}<br>"
            + "<extra></extra>",
            customdata=plot_df[["Character Count", "Total Favorites"]].values,
        )
    )

    # Update layout
    fig.update_layout(
        title=f"Top {top_n} Genres by Average Character Popularity",
        xaxis_title="Average Favorites (Efficiency)",
        yaxis_title="Genre",
        yaxis={"categoryorder": "total ascending"},
        height=figsize[1],
        width=figsize[0],
        showlegend=False,
        font=dict(size=12),
        hovermode="closest",
    )

    return fig

def recommendation_genre_cooccurrence(min_recommendations=250, top_n=10, figsize=(1000, 800)):
    """
    Interactive heatmap showing which anime genres get recommended together.

    Parameters
    ----------
    min_recommendations : int, default=250
        Minimum number of recommendations for a genre pair to be included.
    top_n : int, default=10
        Number of top genres to display in the matrix.
    figsize : tuple, default=(1000, 800)
        Figure size in pixels (width, height).

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Interactive heatmap.
    """
    # Parse genres
    details_df["genres_list"] = details_df["genres"].astype(str).apply(parse_list)
    anime_genres = details_df[["mal_id", "title", "genres_list"]].copy()

    # Merge recommendations with source and target genres
    rec_with_genres = recommendations_df.merge(
        anime_genres, left_on="mal_id", right_on="mal_id", how="inner"
    ).merge(
        anime_genres,
        left_on="recommendation_mal_id",
        right_on="mal_id",
        how="inner",
        suffixes=("_source", "_target"),
    )

    # Explode both genre lists to get all genre pairs
    rec_exploded = (
        rec_with_genres.assign(source_genre=rec_with_genres["genres_list_source"])
        .explode("source_genre")
        .assign(target_genre=lambda df: df["genres_list_target"])
        .explode("target_genre")
    )

    # Filter out empty genres
    rec_exploded = rec_exploded[(rec_exploded["source_genre"] != "") & (rec_exploded["target_genre"] != "")]

    # Count co-occurrences
    genre_pairs = (
        rec_exploded.groupby(["source_genre", "target_genre"])
        .size()
        .reset_index(name="recommendation_count")
    )

    # Get top genres by total recommendations
    top_genres = (
        pd.concat(
            [
                genre_pairs.groupby("source_genre")["recommendation_count"].sum(),
                genre_pairs.groupby("target_genre")["recommendation_count"].sum(),
            ]
        )
        .groupby(level=0)
        .sum()
        .nlargest(top_n)
        .index.tolist()
    )

    # Filter to top genres only
    matrix_data = genre_pairs[
        genre_pairs["source_genre"].isin(top_genres)
        & genre_pairs["target_genre"].isin(top_genres)
        & (genre_pairs["recommendation_count"] >= min_recommendations)
    ]

    # Create pivot table for heatmap
    heatmap_matrix = matrix_data.pivot_table(
        index="source_genre",
        columns="target_genre",
        values="recommendation_count",
        fill_value=0,
    )

    # Reorder by similarity (hierarchical clustering)
    if len(heatmap_matrix) > 1:
        # Convert similarity to distance
        dist_matrix = 1 / (heatmap_matrix + 1)

        # Make symmetric
        dist_matrix = (dist_matrix + dist_matrix.T) / 2

        np.fill_diagonal(dist_matrix.values, 0)

        # Cluster
        linkage_matrix = linkage(squareform(dist_matrix), method="average")
        dendro = dendrogram(linkage_matrix, no_plot=True)
        order = dendro["leaves"]

        # Reorder rows/columns
        ordered_genres = heatmap_matrix.index[order].tolist()
        heatmap_matrix = heatmap_matrix.loc[ordered_genres, ordered_genres]

    # Create heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=heatmap_matrix.values,
            x=heatmap_matrix.columns,
            y=heatmap_matrix.index,
            colorscale="YlOrRd",
            text=heatmap_matrix.values,
            texttemplate="%{text:,.0f}",
            textfont={"size": 10},
            hovertemplate="<b>%{y}</b> → <b>%{x}</b><br>"
            + "Recommendations: %{z:,.0f}<br>"
            + "<extra></extra>",
            colorbar=dict(title="Recommendation<br>Count"),
        )
    )

    fig.update_layout(
        title=f"Genre Co-occurrence in Recommendations (Top {top_n} Genres)",
        xaxis_title="Recommended Genre",
        yaxis_title="Source Genre",
        width=figsize[0],
        height=figsize[1],
        font=dict(size=11),
        xaxis=dict(tickangle=-45),
    )

    return fig


def recommendation_type_patterns(min_count=200, figsize=(1200, 600)):
    """
    Interactive bar chart showing what anime types get recommended together.
    Shows the flow: Type A → Type B with recommendation counts.

    Parameters
    ----------
    min_count : int, default=200
        Minimum recommendation count for a type pair to be displayed.
    figsize : tuple, default=(1200, 600)
        Figure size in pixels (width, height).

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Interactive grouped bar chart.
    """

    # Merge recommendations with type info
    rec_types = recommendations_df.merge(
        details_df[["mal_id", "type", "title"]],
        left_on="mal_id",
        right_on="mal_id",
        how="inner",
    ).merge(
        details_df[["mal_id", "type", "title"]],
        left_on="recommendation_mal_id",
        right_on="mal_id",
        how="inner",
        suffixes=("_source", "_target"),
    )

    # Count type pairs
    type_pairs = (
        rec_types.groupby(["type_source", "type_target"])
        .size()
        .reset_index(name="count")
    )

    # Filter by minimum count
    type_pairs = type_pairs[type_pairs["count"] >= min_count]

    # Sort by count
    type_pairs = type_pairs.sort_values("count", ascending=False)

    # Create grouped bar chart
    fig = go.Figure()

    # Get unique source types
    source_types = type_pairs["type_source"].unique()

    for source_type in source_types:
        data = type_pairs[type_pairs["type_source"] == source_type]

        fig.add_trace(
            go.Bar(
                name=f"From {source_type}",
                x=data["type_target"],
                y=data["count"],
                text=data["count"].apply(lambda x: f"{x:,}"),
                textposition="auto",
                hovertemplate=f"<b>{source_type} → %{{x}}</b><br>"
                + "Recommendations: %{y:,}<br>"
                + "<extra></extra>",
            )
        )

    fig.update_layout(
        title="Anime Type Recommendation Patterns",
        xaxis_title="Recommended Type",
        yaxis_title="Number of Recommendations (log scale)",
        barmode="group",
        width=figsize[0],
        height=figsize[1],
        legend_title="Source Type",
        hovermode="x unified",
    )

    # Log scale helps visibility
    fig.update_yaxes(type="log")

    return fig

# ========================================
# GEOGRAPHIC DISTRIBUTION FUNCTIONS
# ========================================

def load_and_prepare_profiles():
    """
    Load profiles dataset and prepare it for analysis by adding the year_label column.

    Returns:
    --------
    pd.DataFrame
        Profiles dataframe with year_label column added
    """
    profiles_df = pd.read_csv(INPUT_DIR_PATH + "profiles_clean" + FILE_EXTENSION)
    profiles_df['year_label'] = pd.to_datetime(profiles_df['joined'], errors='coerce').dt.year

    return profiles_df


def get_country_code(location):

    if pd.isna(location):
        return None, None

    location_str = str(location).strip()

    manual_mapping = {
        'USA': ('United States', 'USA'),
        'US': ('United States', 'USA'),
        'UK': ('United Kingdom', 'GBR'),
        'England': ('United Kingdom', 'GBR'),
        'Scotland': ('United Kingdom', 'GBR'),
        'Wales': ('United Kingdom', 'GBR'),
        'Northern Ireland': ('United Kingdom', 'GBR'),
        'United States': ('United States', 'USA'),
        'United Kingdom': ('United Kingdom', 'GBR'),
        'Korea': ('South Korea', 'KOR'),
        'Republic of Korea': ('South Korea', 'KOR'),
        'South Korea': ('South Korea', 'KOR'),
        'Russia': ('Russian Federation', 'RUS'),
        'Russian Federation': ('Russian Federation', 'RUS'),
        'Vietnam': ('Viet Nam', 'VNM'),
        'Viet Nam': ('Viet Nam', 'VNM'),
        'Czech Republic': ('Czechia', 'CZE'),
        'Czechia': ('Czechia', 'CZE'),
        'Turkey': ('Turkey', 'TUR'),
        'China': ('China', 'CHN'),
        'Argentina': ('Argentina', 'ARG'),
        'Brazil': ('Brazil', 'BRA'),
        'Chile': ('Chile', 'CHL'),
        'Japan': ('Japan', 'JPN'),
        'Canada': ('Canada', 'CAN'),
        'Germany': ('Germany', 'DEU'),
        'France': ('France', 'FRA'),
        'Italy': ('Italy', 'ITA'),
        'Spain': ('Spain', 'ESP'),
        'Mexico': ('Mexico', 'MEX'),
        'Australia': ('Australia', 'AUS'),
        'Netherlands': ('Netherlands', 'NLD'),
        'Poland': ('Poland', 'POL')
    }

    if location_str in manual_mapping:
        return manual_mapping[location_str]

    try:
        country = pycountry.countries.get(name=location_str)
        if country:
            return country.name, country.alpha_3
    except:
        pass

    try:
        countries = pycountry.countries.search_fuzzy(location_str)
        if countries:
            return countries[0].name, countries[0].alpha_3
    except:
        pass

    return None, None


def prepare_country_data(profiles_df):
    profiles_df[['country', 'country_code']] = profiles_df['location'].apply(lambda x: pd.Series(get_country_code(x)))

    profiles_with_country = profiles_df[profiles_df['country_code'].notna()].copy()

    country_year_counts = profiles_with_country.groupby(['country', 'country_code', 'year_label']).size().reset_index(name='user_count')

    country_year_counts = country_year_counts.sort_values(['country', 'year_label'])
    country_year_counts['cumulative_users'] = country_year_counts.groupby('country')['user_count'].cumsum()

    country_year_counts['log_cumulative'] = np.log10(country_year_counts['cumulative_users'] + 1)

    return country_year_counts


def create_animated_map():
    # Carico e preparo il dataset profiles
    profiles_df = load_and_prepare_profiles()

    # Preparo i dati aggregati per paese e anno
    country_year_counts = prepare_country_data(profiles_df)

    # Creo e mostro la mappa animata
    years_sorted = sorted(country_year_counts['year_label'].unique())
    min_users_log = country_year_counts['log_cumulative'].min()
    max_users_log = country_year_counts['log_cumulative'].max()

    fig = px.choropleth(
        country_year_counts,
        locations='country_code',
        locationmode='ISO-3',
        color='log_cumulative',  # Uso scala logaritmica per migliore differenziazione
        hover_name='country',
        hover_data={
            'country_code': False,
            'cumulative_users': ':,',  # Totale cumulativo (valore reale)
            'user_count': ':,',  # Nuovi utenti dell'anno
            'year_label': True,
            'log_cumulative': False
        },
        animation_frame='year_label',
        color_continuous_scale='Turbo',
        labels={
            'cumulative_users': 'Total Users',
            'user_count': 'New Users This Year',
            'year_label': 'Year'
        },
        title='Global Anime Users Distribution by Year (2004-2025) - Cumulative Total (Log Scale)',
        projection='natural earth',
        category_orders={'year_label': years_sorted},
        range_color=[min_users_log, max_users_log]
    )

    fig.update_layout(
        height=600,
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type='natural earth'
        ),
        coloraxis_colorbar=dict(
            title="Total Users",
            tickformat=','
        )
    )

    return fig


def language_vs_geographic_distribution(figsize=(12, 8)):
    """
    Create a bar chart showing user distribution by primary language.
    Maps countries to their primary languages and aggregates user counts.

    Parameters
    ----------
    figsize : tuple, default=(12, 8)
        Figure size (width, height)

    Returns
    -------
    fig, ax : matplotlib figure and axes objects
    """
    # Define country-to-primary-language mapping for anime context
    language_map = {
        'United States': 'English', 'Canada': 'English', 'United Kingdom': 'English',
        'Australia': 'English', 'New Zealand': 'English', 'Ireland': 'English',
        'France': 'French', 'Belgium': 'French', 'Switzerland': 'French',
        'Germany': 'German', 'Austria': 'German',
        'Spain': 'Spanish', 'Mexico': 'Spanish', 'Argentina': 'Spanish',
        'Brazil': 'Portuguese', 'Portugal': 'Portuguese',
        'Italy': 'Italian', 'Japan': 'Japanese',
        'South Korea': 'Korean', 'China': 'Chinese',
        'Thailand': 'Thai', 'Philippines': 'English',
        'Poland': 'Polish', 'Russia': 'Russian'
    }

    # Load profiles data
    profiles_temp = load_and_prepare_profiles()

    # Aggregate users by language
    profiles_with_lang = profiles_temp.copy()
    profiles_with_lang['language'] = profiles_with_lang['location'].map(language_map)

    language_users = profiles_with_lang.groupby('language').size().reset_index(name='total_users')
    language_users = language_users.dropna().sort_values('total_users', ascending=False)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create vertical bar plot
    bars = ax.bar(range(len(language_users)), language_users['total_users'], 
                  color=cm.viridis(np.linspace(0.2, 0.9, len(language_users))))

    # Set labels
    ax.set_xticks(range(len(language_users)))
    ax.set_xticklabels(language_users['language'].values, rotation=45, ha='right')
    ax.set_ylabel('Total Users', fontsize=12, fontweight='bold')
    ax.set_title('Anime User Distribution by Primary Language', fontsize=14, fontweight='bold', pad=20)

    # Add value labels on bars
    add_count_on_bars(ax)

    # Style
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()

    return fig, ax


# ========================================
# LANGUAGE & LOCALIZATION ANALYSIS
# ========================================

def voice_actor_character_popularity(top_n=20, figsize=(16, 10)):
    """
    Analyze the correlation between voice actors and character popularity.
    Shows if certain voice actors consistently voice popular characters.

    Parameters
    ----------
    top_n : int, default=20
        Number of top voice actors to display
    figsize : tuple, default=(16, 10)
        Matplotlib figure size

    Returns
    -------
    fig, axes : matplotlib Figure and Axes
        Two subplots: 
        1) Top N most impactful VAs (by total favorites)
        2) Top N most consistent VAs (by average favorites per character)
    """
    # Load person_voice_works data
    person_voice_works_df = pd.read_csv(INPUT_DIR_PATH + "person_voice_works_clean" + FILE_EXTENSION)
    person_details_df = pd.read_csv(INPUT_DIR_PATH + "person_details_clean" + FILE_EXTENSION)

    # Get character favorites
    char_favs = favs_df[favs_df["fav_type"] == "character"].copy()
    char_fav_counts = char_favs.groupby("id").size().reset_index(name="user_favs")

    # Merge with character dataset to get total favorites
    char_popularity = characters_df.merge(char_fav_counts, left_on="character_mal_id", right_on="id", how="left")
    char_popularity["user_favs"] = char_popularity["user_favs"].fillna(0)
    char_popularity["total_favs"] = (char_popularity["favorites"].fillna(0) + char_popularity["user_favs"])

    # Merge voice actors with characters
    voice_char = person_voice_works_df.merge(
        char_popularity[["character_mal_id", "name", "total_favs"]],
        on="character_mal_id",
        how="inner"
    )

    # Aggregate by voice actor (person_mal_id)
    va_stats = voice_char.groupby("person_mal_id").agg({
        "total_favs": ["sum", "mean", "count"]
    }).reset_index()

    va_stats.columns = ["person_mal_id", "total_character_favs", "avg_character_favs", "num_characters"]

    # Merge with person details to get names
    va_stats = va_stats.merge(
        person_details_df[["person_mal_id", "name"]],
        on="person_mal_id",
        how="left"
    )

    # Filter: only VAs with at least 3 characters voiced
    va_stats = va_stats[va_stats["num_characters"] >= 3]

    # Prepare data for both subplots
    top_by_total = va_stats.sort_values("total_character_favs", ascending=False).head(top_n)
    top_by_avg = va_stats.sort_values("avg_character_favs", ascending=False).head(top_n)

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # --- LEFT SUBPLOT: Most Impactful VAs (Total Favorites) ---
    ax1 = axes[0]
    plot_data_total = top_by_total.sort_values("total_character_favs", ascending=True)
    
    bars1 = ax1.barh(
        range(len(plot_data_total)),
        plot_data_total["total_character_favs"],
        color=cm.plasma(np.linspace(0.3, 0.9, len(plot_data_total)))
    )
    
    ax1.set_yticks(range(len(plot_data_total)))
    ax1.set_yticklabels(plot_data_total["name"].values)
    ax1.set_xlabel("Total Character Favorites", fontsize=12)
    ax1.set_title(f"Top {top_n} Most Impactful VAs\n(Total favorites across all characters)", 
                  fontsize=13, fontweight="bold", pad=15)
    ax1.grid(axis='x', alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Add value labels with character count
    for i, (bar, value, num_chars) in enumerate(zip(bars1, plot_data_total["total_character_favs"], plot_data_total["num_characters"])):
        ax1.text(
            value,
            i,
            f' {int(value):,} ({int(num_chars)} chars)',
            va='center',
            ha='left',
            fontsize=9,
            fontweight='bold'
        )

    # --- RIGHT SUBPLOT: Most Consistent VAs (Average Favorites) ---
    ax2 = axes[1]
    plot_data_avg = top_by_avg.sort_values("avg_character_favs", ascending=True)
    
    bars2 = ax2.barh(
        range(len(plot_data_avg)),
        plot_data_avg["avg_character_favs"],
        color=cm.viridis(np.linspace(0.3, 0.9, len(plot_data_avg)))
    )
    
    ax2.set_yticks(range(len(plot_data_avg)))
    ax2.set_yticklabels(plot_data_avg["name"].values)
    ax2.set_xlabel("Average Favorites per Character", fontsize=12)
    ax2.set_title(f"Top {top_n} Most Consistent VAs\n(Highest average per character)", 
                  fontsize=13, fontweight="bold", pad=15)
    ax2.grid(axis='x', alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Add value labels with character count
    for i, (bar, value, num_chars) in enumerate(zip(bars2, plot_data_avg["avg_character_favs"], plot_data_avg["num_characters"])):
        ax2.text(
            value,
            i,
            f' {int(value):,} ({int(num_chars)} chars)',
            va='center',
            ha='left',
            fontsize=9,
            fontweight='bold'
        )

    plt.tight_layout()
    return fig, axes


def top_n_most_watched_genres(top_n=10, figsize=(12, 6)):
    """
    Plot bar chart of most watched genres with their total viewership counts.
    
    Parameters:
    -----------
    top_n : int, default=10
        Number of top genres to display
    figsize : tuple, default=(12, 6)
        Figure size (width, height)
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
        
    # Merge details with stats
    merged = details_df.merge(stats_df, on='mal_id', how='inner')
    
    # Filter out rows without genres
    merged_with_genres = merged[merged['genres'].notna()].copy()
    
    # Explode genres (split comma-separated genres into separate rows)
    merged_with_genres['genres'] = merged_with_genres['genres'].apply(parse_list)
    exploded = merged_with_genres.explode('genres')
    
    # Calculate total viewers per genre
    genre_stats = exploded.groupby('genres')['watching'].sum().reset_index()
    genre_stats.columns = ['Genre', 'Total Viewers']
    
    # Get top N genres
    top_genres = genre_stats.nlargest(top_n, 'Total Viewers')
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create bar plot
    bars = ax.barh(range(len(top_genres)), top_genres['Total Viewers'], 
                   color=cm.plasma(np.linspace(0.2, 0.9, len(top_genres))))
    
    # Set labels
    ax.set_yticks(range(len(top_genres)))
    ax.set_yticklabels(top_genres['Genre'].values)
    ax.set_xlabel('Total Viewers', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Most Watched Genres', fontsize=14, fontweight='bold', pad=20)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, top_genres['Total Viewers'])):
        ax.text(value, i, f' {value:,.0f}', va='center', fontsize=10, fontweight='bold')
    
    # Invert y-axis so #1 is at top
    ax.invert_yaxis()
    
    # Style
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    return fig, ax


def top_n_most_liked_genres(top_n=10, figsize=(12, 6)):
    """
    Plot bar chart of most liked (favorited) genres by number of favorites.

    Logic:
    - Filter `favs_df` for entries where the favorite type is the anime itself
    - Merge with `details_df` on anime id to obtain genres
    - Explode genres and count how many times each genre appears in the favorites
    - Return a horizontal bar chart of the top-N genres by favorite count

    Returns:
    --------
    fig, ax : matplotlib Figure and Axes
    """

    # Select only anime favorites
    anime_favs = favs_df[favs_df["fav_type"] == "anime"].copy()

    # Merge favorites with anime details using the anime id
    # In `favs_df` the id column refers to the favorited object id (anime id when fav_type=='anime')
    merged = anime_favs.merge(details_df, left_on="id", right_on="mal_id", how="inner")

    # Keep only rows with genre information
    merged = merged[merged["genres"].notna()].copy()
    if merged.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No anime favorites with genre data available.", ha="center", va="center")
        ax.axis("off")
        return fig, ax

    # Explode genres into separate rows
    merged['genres'] = merged['genres'].apply(parse_list)
    exploded = merged.explode("genres")
    exploded = exploded[exploded["genres"] != ""]

    # Count favorites per genre (how many times an anime in that genre was favorited)
    genre_counts = exploded["genres"].value_counts().reset_index()
    genre_counts.columns = ["Genre", "Favorite Count"]

    # Get top N genres
    top_genres = genre_counts.head(top_n)

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(
        range(len(top_genres)),
        top_genres["Favorite Count"],
        color=cm.plasma(np.linspace(0.2, 0.9, len(top_genres)))
    )

    ax.set_yticks(range(len(top_genres)))
    ax.set_yticklabels(top_genres["Genre"].values)
    ax.set_xlabel("Favorite Count", fontsize=12, fontweight="bold")
    ax.set_title(f"Top {top_n} Most Liked Genres", fontsize=14, fontweight="bold", pad=20)

    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, top_genres["Favorite Count"])):
        ax.text(value, i, f" {int(value):,}", va="center", fontsize=10, fontweight="bold")

    # Make highest rank appear on top
    ax.invert_yaxis()

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    plt.tight_layout()
    return fig, ax


def completion_dropout_by_episode_count(min_users=100, figsize=(1400, 700)):
    """
    Analyze completion, dropout, and in-progress rates by episode count categories.
    Shows how anime length affects user commitment and completion behavior.

    Parameters
    ----------
    min_users : int, default=100
        Minimum number of users who started the anime for it to be included.
    figsize :  tuple, default=(1400, 700)
        Figure size in pixels (width, height).

    Returns
    -------
    fig :  plotly.graph_objects.Figure
        Interactive grouped bar chart with 3 metrics.
    """
    # Prepare engagement data
    anime_engagement = details_df[["mal_id", "title", "episodes", "score"]].merge(
        stats_df[["mal_id", "watching", "completed", "on_hold", "dropped"]],
        on="mal_id",
        how="inner",
    )

    anime_engagement["started_users"] = (
        anime_engagement["watching"]
        + anime_engagement["completed"]
        + anime_engagement["on_hold"]
        + anime_engagement["dropped"]
    )

    # Filter by minimum users and valid episodes
    anime_engagement = anime_engagement[
        (anime_engagement["started_users"] >= min_users)
        & (anime_engagement["episodes"]. notna())
        & (anime_engagement["episodes"] > 0)
    ].copy()

    # Calculate rates
    anime_engagement["completion_rate"] = (anime_engagement["completed"] / anime_engagement["started_users"] * 100)
    anime_engagement["dropout_rate"] = (anime_engagement["dropped"] / anime_engagement["started_users"] * 100)
    anime_engagement["watching_rate"] = ( anime_engagement["watching"] / anime_engagement["started_users"] * 100)
    anime_engagement["on_hold_rate"] = (anime_engagement["on_hold"] / anime_engagement["started_users"] * 100)

    # Create episode bins
    bins = [1, 2, 13, 25, 51, 101, 501, 3000]
    labels = [
        "1 ep\n(Movie/Special)",
        "2-12 eps\n(1 cour)",
        "13-24 eps\n(2 cour)",
        "25-50 eps\n(Long)",
        "51-100 eps\n(Very Long)",
        "101-500 eps\n(Ongoing)",
        "501+ eps\n(Mega)",
    ]

    anime_engagement["episode_category"] = pd.cut(
        anime_engagement["episodes"], bins=bins, labels=labels, include_lowest=True
    )

    # Aggregate by episode category
    stats = (
        anime_engagement.groupby("episode_category", observed=True)
        .agg({
            "completion_rate": "mean",
            "dropout_rate": "mean",
            "mal_id": "count",
            "started_users": "sum",
            "watching":  "sum",
            "on_hold": "sum",
        })
        .reset_index()
    )

    stats.columns = [
        "Category",
        "Avg Completion Rate",
        "Avg Dropout Rate",
        "Anime Count",
        "Total Started Users",
        "Total Watching",
        "Total On Hold",
    ]

    # Calculate in-progress rate
    stats["Avg In-Progress Rate"] = ((stats["Total Watching"] + stats["Total On Hold"]) / stats["Total Started Users"] * 100)

    # --- Visualization ---
    fig = go.Figure()

    colors = {
        "Completion Rate": "#4ECDC4",
        "Dropout Rate": "#FF6B6B",
        "In-Progress/On-Hold": "#FFA500",
    }

    for col, label in [
        ("Avg Completion Rate", "Completion Rate"),
        ("Avg Dropout Rate", "Dropout Rate"),
        ("Avg In-Progress Rate", "In-Progress/On-Hold"),
    ]:
        fig. add_trace(
            go. Bar(
                x=stats["Category"],
                y=stats[col],
                name=label,
                marker=dict(color=colors[label]),
                text=stats[col].apply(lambda x: f"{x:.1f}%"),
                textposition="outside",
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    + f"{label}: %{{y:.1f}}%<br>"
                    + "Anime Count: %{customdata[0]:,}<br>"
                    + "Total Started: %{customdata[1]:,}<br>"
                    + "<extra></extra>"
                ),
                customdata=stats[["Anime Count", "Total Started Users"]].values,
            )
        )

    max_val = (
        stats[["Avg Completion Rate", "Avg Dropout Rate", "Avg In-Progress Rate"]]
        .max()
        .max()
    )

    fig.update_layout(
        title="Completion vs Dropout vs In-Progress Rate by Episode Count<br>"
        "<sub>Does anime length affect user commitment?</sub>",
        xaxis_title="Episode Count Category",
        yaxis_title="Rate (%)",
        barmode="group",
        width=figsize[0],
        height=figsize[1],
        hovermode="x unified",
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
        font=dict(size=12),
        yaxis=dict(range=[0, max_val * 1.15]),
    )

    return fig



def completion_dropout_by_type(min_users=100, figsize=(1400, 600)):
    """
    Analyze completion and dropout rates by anime type. 
    Interactive grouped bar chart showing completion vs dropout rates. 

    Parameters
    ----------
    min_users : int, default=100
        Minimum number of users who started the anime for it to be included. 
    figsize : tuple, default=(1400, 600)
        Figure size in pixels (width, height).

    Returns
    -------
    fig : plotly.graph_objects. Figure
        Interactive grouped bar chart.
    """
    # Prepare engagement data
    anime_engagement = details_df[["mal_id", "title", "type", "score"]].merge(
        stats_df[["mal_id", "watching", "completed", "on_hold", "dropped"]],
        on="mal_id",
        how="inner",
    )

    anime_engagement["started_users"] = (
        anime_engagement["watching"]
        + anime_engagement["completed"]
        + anime_engagement["on_hold"]
        + anime_engagement["dropped"]
    )

    # Filter by minimum users
    anime_engagement = anime_engagement[anime_engagement["started_users"] >= min_users].copy()

    # Calculate rates
    anime_engagement["completion_rate"] = (anime_engagement["completed"] / anime_engagement["started_users"] * 100)
    anime_engagement["dropout_rate"] = (anime_engagement["dropped"] / anime_engagement["started_users"] * 100)

    # Aggregate by type
    stats = (
        anime_engagement.groupby("type")
        .agg({
            "completion_rate":  "mean",
            "dropout_rate": "mean",
            "mal_id": "count",
            "started_users": "sum",
        })
        .reset_index()
    )

    stats.columns = [
        "Type",
        "Avg Completion Rate",
        "Avg Dropout Rate",
        "Anime Count",
        "Total Started Users",
    ]

    # Sort by completion rate
    stats = stats.sort_values("Avg Completion Rate", ascending=False)

    # --- Visualization ---
    fig = go.Figure()

    colors = {"Completion Rate": "#4ECDC4", "Dropout Rate": "#FF6B6B"}

    for col, label, color in [
        ("Avg Completion Rate", "Completion Rate", "#4ECDC4"),
        ("Avg Dropout Rate", "Dropout Rate", "#FF6B6B"),
    ]:
        fig.add_trace(
            go.Bar(
                x=stats["Type"],
                y=stats[col],
                name=label,
                marker=dict(color=color),
                text=stats[col].apply(lambda x: f"{x:.1f}%"),
                textposition="outside",
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    + f"{label}: %{{y:.1f}}%<br>"
                    + "Anime Count: %{customdata[0]: ,}<br>"
                    + "Total Started:  %{customdata[1]:,}<br>"
                    + "<extra></extra>"
                ),
                customdata=stats[["Anime Count", "Total Started Users"]]. values,
            )
        )

    max_val = stats[["Avg Completion Rate", "Avg Dropout Rate"]].max().max()

    fig.update_layout(
        title="Completion vs Dropout Rate by Anime Type<br><sub>How does format affect user engagement?</sub>",
        xaxis_title="Anime Type",
        yaxis_title="Rate (%)",
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
        yaxis=dict(range=[0, max_val * 1.15]),
    )

    return fig


def completion_dropout_by_genre(top_n=10, min_users=100, figsize=(1400, 700)):
    """
    Analyze completion and dropout rates by top anime genres.
    Interactive grouped bar chart showing which genres have best/worst engagement.

    Parameters
    ----------
    top_n : int, default=10
        Number of top genres to display.
    min_users : int, default=100
        Minimum number of users who started the anime for it to be included.
    figsize : tuple, default=(1400, 700)
        Figure size in pixels (width, height).

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Interactive grouped bar chart.
    """
    # Prepare engagement data
    anime_engagement = details_df[["mal_id", "title", "genres", "score"]].merge(
        stats_df[["mal_id", "watching", "completed", "on_hold", "dropped"]],
        on="mal_id",
        how="inner",
    )

    anime_engagement["started_users"] = (
        anime_engagement["watching"]
        + anime_engagement["completed"]
        + anime_engagement["on_hold"]
        + anime_engagement["dropped"]
    )

    # Filter by minimum users
    anime_engagement = anime_engagement[anime_engagement["started_users"] >= min_users].copy()

    # Calculate rates
    anime_engagement["completion_rate"] = (anime_engagement["completed"] / anime_engagement["started_users"] * 100)
    anime_engagement["dropout_rate"] = (anime_engagement["dropped"] / anime_engagement["started_users"] * 100)

    # Parse and explode genres
    anime_engagement["genres_list"] = anime_engagement["genres"].astype(str).apply(parse_list)
    genre_df = anime_engagement. explode("genres_list")
    genre_df = genre_df[genre_df["genres_list"] != ""]

    # Aggregate by genre
    stats = (
        genre_df.groupby("genres_list")
        .agg({
            "completion_rate": "mean",
            "dropout_rate":  "mean",
            "mal_id": "count",
            "started_users": "sum",
        })
        .reset_index()
    )

    stats.columns = [
        "Genre",
        "Avg Completion Rate",
        "Avg Dropout Rate",
        "Anime Count",
        "Total Started Users",
    ]

    # Get top N genres by anime count
    stats = stats.nlargest(top_n, "Anime Count")
    stats = stats.sort_values("Avg Completion Rate", ascending=False)

    # --- Visualization ---
    fig = go. Figure()

    for col, label, color in [
        ("Avg Completion Rate", "Completion Rate", "#4ECDC4"),
        ("Avg Dropout Rate", "Dropout Rate", "#FF6B6B"),
    ]:
        fig.add_trace(
            go.Bar(
                x=stats["Genre"],
                y=stats[col],
                name=label,
                marker=dict(color=color),
                text=stats[col].apply(lambda x: f"{x:.1f}%"),
                textposition="outside",
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    + f"{label}: %{{y:.1f}}%<br>"
                    + "Anime Count: %{customdata[0]: ,}<br>"
                    + "Total Started: %{customdata[1]:,}<br>"
                    + "<extra></extra>"
                ),
                customdata=stats[["Anime Count", "Total Started Users"]].values,
            )
        )

    max_val = stats[["Avg Completion Rate", "Avg Dropout Rate"]].max().max()

    fig.update_layout(
        title=f"Completion vs Dropout Rate by Top {top_n} Genres<br><sub>How does genre affect user engagement?</sub>",
        xaxis_title="Genre",
        yaxis_title="Rate (%)",
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
        xaxis=dict(tickangle=-45),
        yaxis=dict(range=[0, max_val * 1.15]),
    )

    return fig
