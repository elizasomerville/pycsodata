"""Constants and configuration for pycsodata.

This module contains all configuration values, URLs, and lookup tables
used throughout the package. Centralising these values makes it easier
to update them and ensures consistency.

Constants:
    CSO_BASE_URL: Base URL for the CSO RESTful API.
    DEFAULT_TIMEOUT: Default timeout for HTTP requests (seconds).
    DEFAULT_RETRIES: Number of retry attempts for failed requests.
    CACHE_TTL_SECONDS: Cache time-to-live (24 hours).
    DEFAULT_CRS: Default coordinate reference system (WGS84).
    ROI_GEOMETRY_URL: URL for Republic of Ireland boundary geometry.
    NATIONAL_AREA_CODE: Code for national-level data ("IE0").
    NATIONAL_AREA_LABELS: Labels for national-level data.
    STATISTIC_LABELS: Dimension names for statistic columns.
    SANITISATION_DICT: Mappings for standardising dimension labels.
    MISENCODED_CHARACTER_MAP: Mappings for fixing incorrectly encoded fadas.
"""

from __future__ import annotations

# Misencoding occurs in e.g. SAP2011T1T1AED, SAP2016T1T1AED
MISENCODED_CHARACTER_MAP: dict[str, str] = {
    "┴": "Á",
    "ß": "á",
    "╔": "É",
    "Θ": "é",
    "φ": "í",
    "╙": "Ó",
    "≤": "ó",
    "·": "ú",
}

# =============================================================================
# API Configuration
# =============================================================================

# Base URL for the CSO RESTful API
CSO_BASE_URL: str = "https://ws.cso.ie/public/api.restful/PxStat.Data.Cube_API"

# Default timeout for HTTP requests (seconds)
DEFAULT_TIMEOUT: int = 30

# Number of retry attempts for failed HTTP requests
DEFAULT_RETRIES: int = 3

# Delay multiplier between retries (seconds)
RETRY_DELAY_MULTIPLIER: float = 0.5

# Cache TTL (time-to-live) in seconds (24 hours)
CACHE_TTL_SECONDS: int = 24 * 60 * 60

# =============================================================================
# Spatial Configuration
# =============================================================================

# WGS84 - standard global CRS, used as fallback
DEFAULT_CRS: str = "EPSG:4326"

# URL for Republic of Ireland boundary geometry (OSi National 250k Map)
ROI_GEOMETRY_URL: str = (
    "https://services-eu1.arcgis.com/FH5XCsx8rYXqnjF5/arcgis/rest/services/"
    "Landmask___OSi_National_250k_Map_Of_Ireland/FeatureServer/0/query"
    "?outFields=*&where=1%3D1&f=geojson"
)

# =============================================================================
# Data Processing
# =============================================================================

# Code used for national-level (all of Ireland) data
NATIONAL_AREA_CODE: str = "IE0"

# Labels used for national-level data in various datasets
NATIONAL_AREA_LABELS: frozenset[str] = frozenset({"Ireland", "State"})

# Column name suffix for ID columns
ID_COLUMN_SUFFIX: str = " ID"

# Statistic dimension names (CSO uses both)
STATISTIC_LABELS: frozenset[str] = frozenset({"Statistic", "STATISTIC"})

# Common sanitisation mappings for dimension labels to ensure greater consistency
# Only the ones I considered most useful/noticed the most often are included here
SANITISATION_DICT: dict[str, str] = {
    "Administrative Counties and Local Government Districts": (
        "Administrative County and Local Government District"
    ),
    "Admin Counties": "Administrative County",
    "Admin County": "Administrative County",
    "Administrative Counties": "Administrative County",
    "Adminstrative Counties": "Administrative County",
    "Administrative Counties 2019": "Administrative County 2019",
    "Catchement": "Catchment Area",  # Typo in source data
    "Catchment": "Catchment Area",
    "CensusYear": "Census Year",
    "Census year": "Census Year",
    "Counties": "County",
    "Counties and Cities": "County and City",
    "County and Cities": "County and City",
    "Counties and HSE Regions": "County and HSE Region",
    "Countries": "Country",
    "Electoral Divisions": "Electoral Division",
    "HalfYear": "Half Year",
    "Licencing Authority": "Licensing Authority",  # Typo in source data
    "Local Electoral Areas": "Local Electoral Area",
    "Martial Status of Mother": "Marital Status of Mother",  # Typo in source data
    "NUTS 2 Regions": "NUTS 2 Region",
    "NUTS 3": "NUTS 3 Region",
    "NUTS 3 region": "NUTS 3 Region",
    "NUTS 3 Regions": "NUTS 3 Region",
    "NUTS3 Regions": "NUTS 3 Region",
    "NUTS3 regions": "NUTS 3 Region",
    "Nuts 2 Region": "NUTS 2 Region",
    "Principle Countries": "Principal Countries",  # Typo in source data
    "Principle Economic Status": "Principal Economic Status",  # Typo in source data
    "Provinces": "Province",
    "Settlements": "Settlement",
    "Small Areas": "Small Area",
}
