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

## Comment or uncomment the appropriate data path
# INPUT_DIR_PATH = "C:\\Users\\Lorenzo\\OneDrive\\Desktop\\datasets\\cleaned\\"
INPUT_DIR_PATH = "~/datasets/cleaned/"
FILE_EXTENSION = ".csv"

character_anime_works_df = pd.read_csv(INPUT_DIR_PATH + "character_anime_works_clean" + FILE_EXTENSION)
# character_nicknames_df = pd.read_csv(INPUT_DIR_PATH + "character_nicknames_clean" + FILE_EXTENSION)
characters_df = pd.read_csv(INPUT_DIR_PATH + "characters_clean" + FILE_EXTENSION)
details_df = pd.read_csv(INPUT_DIR_PATH + "details_clean" + FILE_EXTENSION)
favs_df = pd.read_csv(INPUT_DIR_PATH + "favs_clean" + FILE_EXTENSION)
# person_alternate_names_df = pd.read_csv(INPUT_DIR_PATH + "person_alternate_names_clean" + FILE_EXTENSION)
# person_anime_works_df = pd.read_csv(INPUT_DIR_PATH + "person_anime_works_clean" + FILE_EXTENSION)
# person_details_df = pd.read_csv(INPUT_DIR_PATH + "person_details_clean" + FILE_EXTENSION)
# person_voice_works_df = pd.read_csv(INPUT_DIR_PATH + "person_voice_works_clean" + FILE_EXTENSION)
# profiles_df = pd.read_csv(INPUT_DIR_PATH + "profiles_clean" + FILE_EXTENSION)
# ratings_df = pd.read_csv(INPUT_DIR_PATH + "ratings_clean" + FILE_EXTENSION)
recommendations_df = pd.read_csv(INPUT_DIR_PATH + "recommendations_clean" + FILE_EXTENSION)
# stats_df = pd.read_csv(INPUT_DIR_PATH + "stats_clean" + FILE_EXTENSION)


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

    # Statistiche sui dati
    print("Dati aggregati:")
    print(f"Paesi unici prima della normalizzazione: {profiles_df['location'].nunique()}")
    print(f"Paesi unici dopo la normalizzazione: {country_year_counts['country'].nunique()}")
    print(f"\nNumero totale di record: {len(country_year_counts)}")
    print(f"Anni disponibili: {sorted(country_year_counts['year_label'].unique())}")

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


def language_vs_geographic_distribution():
    """
    Create a bar chart showing user distribution by primary language.
    Maps countries to their primary languages and aggregates user counts.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Interactive bar chart of language distribution
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

    # Create visualization
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=language_users['language'],
        y=language_users['total_users'],
        marker=dict(color=language_users['total_users'], colorscale='Viridis'),
        text=language_users['total_users'],
        textposition='auto',
    ))

    fig.update_layout(
        title='Anime User Distribution by Primary Language<br><sub>Does localization drive adoption?</sub>',
        xaxis_title='Language',
        yaxis_title='Total Users',
        height=600,
        showlegend=False
    )

    return fig


# ========================================
# LANGUAGE & LOCALIZATION ANALYSIS
# ========================================

def voice_actor_character_popularity(top_n=20):
    """
    Analyze the correlation between voice actors and character popularity.
    Shows if certain voice actors consistently voice popular characters.

    Parameters
    ----------
    top_n : int, default=20
        Number of top voice actors to display

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Interactive bar chart showing voice actor correlation with character popularity
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

    # Sort by total character favorites
    va_stats = va_stats.sort_values("total_character_favs", ascending=False).head(top_n)

    # Create visualization
    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=va_stats["name"][::-1],  # Reverse for better readability
        x=va_stats["total_character_favs"][::-1],
        orientation='h',
        marker=dict(
            color=va_stats["avg_character_favs"][::-1],
            colorscale='Plasma',
            showscale=True,
            colorbar=dict(title="Avg Favs<br>per Character")
        ),
        text=va_stats["num_characters"][::-1],
        texttemplate='%{text} chars',
        textposition='inside',
        hovertemplate='<b>%{y}</b><br>Total Favorites: %{x:,}<br>Characters Voiced: %{text}<extra></extra>'
    ))

    fig.update_layout(
        title='Top Voice Actors by Character Popularity<br><sub>Do certain VAs consistently voice beloved characters?</sub>',
        xaxis_title='Total Character Favorites',
        yaxis_title='Voice Actor',
        height=700,
        showlegend=False,
        hovermode='closest'
    )

    return fig
