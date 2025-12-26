"""
Geographic Utilities
====================

Helper functions for geographic data processing.
"""

import pandas as pd


def get_country_code(location):
    """
    Map location strings to country names and ISO codes.

    Parameters
    ----------
    location : str
        Location string from user profile

    Returns
    -------
    tuple
        (country_name, country_code) or (None, None)
    """
    if pd.isna(location):
        return None, None

    location_str = str(location).strip()

    manual_mapping = {
        "USA": ("United States", "USA"),
        "US": ("United States", "USA"),
        "UK": ("United Kingdom", "GBR"),
        "England": ("United Kingdom", "GBR"),
        "Scotland": ("United Kingdom", "GBR"),
        "Wales": ("United Kingdom", "GBR"),
        "Northern Ireland": ("United Kingdom", "GBR"),
        "United States": ("United States", "USA"),
        "United Kingdom": ("United Kingdom", "GBR"),
        "Korea": ("South Korea", "KOR"),
        "Republic of Korea": ("South Korea", "KOR"),
        "South Korea": ("South Korea", "KOR"),
        "Russia": ("Russian Federation", "RUS"),
        "Russian Federation": ("Russian Federation", "RUS"),
        "Vietnam": ("Viet Nam", "VNM"),
        "Viet Nam": ("Viet Nam", "VNM"),
        "Czech Republic": ("Czechia", "CZE"),
        "Czechia": ("Czechia", "CZE"),
        "Turkey": ("Turkey", "TUR"),
        "China": ("China", "CHN"),
        "Argentina": ("Argentina", "ARG"),
        "Brazil": ("Brazil", "BRA"),
        "Chile": ("Chile", "CHL"),
        "Japan": ("Japan", "JPN"),
        "Canada": ("Canada", "CAN"),
        "Germany": ("Germany", "DEU"),
        "France": ("France", "FRA"),
        "Italy": ("Italy", "ITA"),
        "Spain": ("Spain", "ESP"),
        "Mexico": ("Mexico", "MEX"),
        "Australia": ("Australia", "AUS"),
        "Netherlands": ("Netherlands", "NLD"),
        "Poland": ("Poland", "POL"),
    }

    return manual_mapping.get(location_str, (None, None))