"""Spatial data operations for CSO datasets.

This module handles loading and merging spatial boundary data with
CSO statistical datasets to create GeoDataFrames.

Public Functions:
    create_geodataframe: Merge a DataFrame with spatial boundaries.
    create_met_geodataframe: Merge a DataFrame with weather station coordinates.

Examples:
    >>> from pycsodata import CSODataset
    >>> dataset = CSODataset("FY003A")
    >>> gdf = dataset.gdf()  # Uses create_geodataframe internally
    >>> gdf.plot(column="value")
"""

from __future__ import annotations

import io
from typing import Any

import geopandas as gpd
import pandas as pd

from pycsodata.constants import (
    DEFAULT_CRS,
    ID_COLUMN_SUFFIX,
    MET_EIREANN_SPATIAL_KEY,
    WEATHER_STATIONS,
)
from pycsodata.exceptions import SpatialError
from pycsodata.fetchers import fetch_json

# =============================================================================
# Public API
# =============================================================================


def create_geodataframe(
    df: pd.DataFrame,
    spatial_url: str | None,
    spatial_key: str | None,
    *,
    cache: bool = True,
) -> gpd.GeoDataFrame:
    """Create a GeoDataFrame by merging a DataFrame with spatial boundaries.

    This is the main entry point for spatial operations. It fetches the
    spatial boundary data from the specified URL, performs the merge on
    the spatial key column, and returns a GeoDataFrame with geometry.

    The merge uses a left join, so all rows from the input DataFrame are
    preserved. Rows that do not have a matching geometry in the spatial
    data (e.g., aggregate regions like "State" or "Leinster") will have
    null geometries in the resulting GeoDataFrame.

    Args:
        df: The DataFrame to merge with spatial data. Must contain
            either a column matching spatial_key or a corresponding
            ID column (e.g., "County ID").
        spatial_url: URL to the GeoJSON boundary data.
        spatial_key: The dimension label for the spatial join key
            (e.g., "County", "Electoral Division").
        cache: Whether to cache the fetched GeoJSON. Defaults to True.

    Returns:
        A GeoDataFrame with geometry column. Rows without matching
        spatial boundaries will have null geometries.

    Raises:
        SpatialError: If spatial URL or key is missing, if no features
            are found in the GeoJSON, or if the merge fails.
    """
    if not spatial_url or not spatial_key:
        raise SpatialError("Dataset has no spatial information available.")

    try:
        # Fetch and parse the GeoJSON
        geojson = fetch_json(spatial_url, cache=cache)
        features = geojson.get("features", [])

        if not features:
            raise SpatialError("No features found in GeoJSON data.")

        # Create GeoDataFrame from features
        gdf = gpd.GeoDataFrame.from_features(features)

        # Set CRS if not present
        if gdf.crs is None:
            detected_crs = _detect_crs(geojson)
            gdf = gdf.set_crs(detected_crs)

        # Merge with the data DataFrame
        merged = _merge_dataframes(df, gdf, spatial_key, geojson)

        if merged is None:
            raise SpatialError("Spatial merge failed.")

        return merged

    except (KeyError, ValueError) as e:
        raise SpatialError(f"Error creating GeoDataFrame: {e}") from e


def create_met_geodataframe(
    df: pd.DataFrame,
    spatial_key: str | None = None,
) -> gpd.GeoDataFrame:
    """Create a GeoDataFrame by merging a DataFrame with weather station coordinates.

    Met Eireann-derived datasets (table codes MT) do not include
    spatial boundary data in the PxStat API. This function manually geocodes
    such datasets by merging with built-in weather station coordinates.

    The merge uses a left join, so all rows from the input DataFrame are
    preserved. Rows that do not match a known weather station will have
    null geometries.

    Args:
        df: The DataFrame to merge with weather station data. Must contain
            a column matching spatial_key (typically
            "Meteorological Weather Station").
        spatial_key: The column name for the spatial join key. Defaults to
            the standard Met Éireann spatial key if not specified.

    Returns:
        A GeoDataFrame with point geometry for each weather station,
        using WGS84 (EPSG:4326) coordinate reference system.

    Raises:
        SpatialError: If the spatial key column is not found in the DataFrame,
            or if the merge produces no geometry column.
    """
    if spatial_key is None:
        spatial_key = MET_EIREANN_SPATIAL_KEY

    if spatial_key not in df.columns:
        raise SpatialError(
            f"Column '{spatial_key}' not found in DataFrame. Available columns: {list(df.columns)}"
        )

    stations_gdf = _build_weather_stations_gdf()

    try:
        # Merge on spatial_key and station_id both titlecased to match typical dataset values
        df[f"{spatial_key}_titlecased"] = df[spatial_key].str.strip().str.title()

        merged = df.merge(
            stations_gdf[["station_id", "geometry"]],
            left_on=f"{spatial_key}_titlecased",
            right_on="station_id",
            how="left",
            validate="many_to_one",
        ).drop(columns=["station_id", f"{spatial_key}_titlecased"])

        if "geometry" not in merged.columns:
            raise SpatialError(
                "Merged DataFrame has no geometry column after weather station merge."
            )

        return gpd.GeoDataFrame(merged, geometry="geometry", crs=DEFAULT_CRS)

    except SpatialError:
        raise
    except Exception as e:
        raise SpatialError(f"Error during weather station spatial merge: {e}") from e


def _build_weather_stations_gdf() -> gpd.GeoDataFrame:
    """Build a GeoDataFrame of weather station point geometries.

    Parses the built-in WEATHER_STATIONS CSV data and creates a
    GeoDataFrame with point geometries from latitude/longitude values.
    Station names are stored in Title Case to match CSO dataset values.

    Returns:
        A GeoDataFrame with columns 'station_id' and 'geometry',
        using WGS84 CRS.
    """
    stations_df = pd.read_csv(io.StringIO(WEATHER_STATIONS.strip()))
    stations_df["station_id"] = stations_df["station_id"].str.strip()

    return gpd.GeoDataFrame(
        stations_df[["station_id"]],
        geometry=gpd.points_from_xy(
            stations_df["Longitude"].astype(float),
            stations_df["Latitude"].astype(float),
        ),
        crs=DEFAULT_CRS,
    )


# =============================================================================
# Private Implementation
# =============================================================================


def _detect_crs(geojson: dict[str, Any]) -> str:
    """Detect CRS from a GeoJSON structure.

    Looks for CRS information in the GeoJSON metadata. Falls back to
    WGS84 (EPSG:4326) if no CRS is specified.

    Args:
        geojson: The GeoJSON dictionary.

    Returns:
        The detected CRS string (e.g., "EPSG:4326"), or DEFAULT_CRS
        if not found.
    """
    crs_info = geojson.get("crs")

    if isinstance(crs_info, dict):
        # Try properties.name first
        if "properties" in crs_info and "name" in crs_info["properties"]:
            return str(crs_info["properties"]["name"])
        # Fall back to direct name
        if "name" in crs_info:
            return str(crs_info["name"])

    return DEFAULT_CRS


def _merge_dataframes(
    df: pd.DataFrame,
    gdf: gpd.GeoDataFrame,
    spatial_key: str,
    geojson: dict[str, Any],
) -> gpd.GeoDataFrame | None:
    """Merge a DataFrame with a GeoDataFrame on the spatial key.

    Performs a left join to preserve all rows from the input DataFrame.
    Rows without matching geometries (e.g., aggregate regions like "State")
    will have null geometries.

    Tries multiple merge strategies based on available columns:
    1. Join on ID column (e.g., "County ID") to "code" column in GeoJSON.
    2. Join on label column (e.g., "County") if matching column exists.

    Args:
        df: The data DataFrame.
        gdf: The spatial GeoDataFrame from GeoJSON.
        spatial_key: The dimension label for joining (e.g., "County").
        geojson: The original GeoJSON (for CRS fallback).

    Returns:
        The merged GeoDataFrame with all rows from df preserved.
        Rows without matching spatial data will have null geometries.

    Raises:
        SpatialError: If no suitable columns for merging are found,
            or if the merge produces no geometry column.
    """
    id_column = f"{spatial_key}{ID_COLUMN_SUFFIX}"

    try:
        merged = None

        # Strategy 1: Join on ID column to 'code' column
        if id_column in df.columns and "code" in gdf.columns:
            merged = df.merge(
                gdf[["code", "geometry"]],
                left_on=id_column,
                right_on="code",
                how="left",
                validate="many_to_one",
            ).drop(columns=["code"])

        # Strategy 2: Join on label column
        elif spatial_key in df.columns and spatial_key in gdf.columns:
            merged = df.merge(
                gdf[[spatial_key, "geometry"]],
                on=spatial_key,
                how="left",
                validate="many_to_one",
            )

        if merged is None:
            raise SpatialError("Could not find suitable columns for spatial merge.")

        if "geometry" not in merged.columns:
            raise SpatialError("Merged DataFrame has no geometry column.")

        # Determine CRS for the result
        result_crs = gdf.crs if gdf.crs is not None else _detect_crs(geojson)

        return gpd.GeoDataFrame(merged, geometry="geometry", crs=result_crs)

    except Exception as e:
        raise SpatialError(f"Error during spatial merge operation: {e}") from e
