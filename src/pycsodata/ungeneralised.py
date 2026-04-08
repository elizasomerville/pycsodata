"""Ungeneralised geometry support for CSO datasets.

This module provides functionality for downloading, caching, and merging
ungeneralised (high-resolution) geometry data from Tailte Éireann ArcGIS
Feature Services and OSNI (Ordnance Survey Northern Ireland) open data.

The ungeneralised geometries are significantly more detailed than the
default generalised geometries provided by the CSO API, but are also
larger files that require downloading from external services.

Downloads use the ArcGIS REST API directly via ``requests`` with
concurrent pagination for fast downloads, with no dependency on the
heavyweight ``arcgis`` Python package.

Public Functions:
    create_ungeneralised_geodataframe: Create a GeoDataFrame by merging
        statistical data with ungeneralised spatial boundaries.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import defusedxml.ElementTree as ET
import geopandas as gpd
import pandas as pd
import requests
import shapely
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

from pycsodata.constants import DEFAULT_RETRIES, DEFAULT_TIMEOUT, RETRY_DELAY_MULTIPLIER
from pycsodata.exceptions import SpatialError
from pycsodata.fetchers import fetch_json
from pycsodata.spatial import _detect_crs, _merge_dataframes

logger = logging.getLogger(__name__)


# =============================================================================
# Cache Configuration
# =============================================================================

# Maximum number of concurrent page download threads.
_MAX_WORKERS: int = 8


def _get_cache_dir() -> Path:
    """Get the cache directory for ungeneralised geometry files.

    Uses the ``PYCSODATA_CACHE_DIR`` environment variable if set,
    otherwise falls back to ``~/.pycsodata/cache/ungeneralised``.

    Returns:
        Path to the cache directory.
    """
    env_dir = os.environ.get("PYCSODATA_CACHE_DIR")
    if env_dir:
        return Path(env_dir) / "ungeneralised"
    return Path.home() / ".pycsodata" / "cache" / "ungeneralised"


# =============================================================================
# Mapping Data: CSO Filecodes → Ungeneralised Geometry Sources
# =============================================================================
#
# This data is derived from data/CSOtoTailte.csv and embedded here for
# efficient access without file I/O at import time.
#
# Simple mappings: filecode → (id_field_in_tailte, feature_service_url)
# For these, the merge is: 'code' in CSO GeoJSON ← id_field in Tailte data
#

_SIMPLE_MAPPINGS: dict[str, tuple[str, str]] = {
    "440c36d3b86e067e97ffb2fabf55900e": (
        "GUID",
        "https://services-eu1.arcgis.com/FH5XCsx8rYXqnjF5/ArcGIS/rest/services/"
        "Administrative_Areas_Ungeneralised/FeatureServer/0",
    ),
    "e4d3585d979c15653c7317a18d73b511": (
        "GUID",
        "https://services-eu1.arcgis.com/FH5XCsx8rYXqnjF5/ArcGIS/rest/services/"
        "Administrative_Areas_Ungeneralised/FeatureServer/0",
    ),
    "81abcc6918ea811bcd0d3fe9aea320f1": (
        "GUID",
        "https://services-eu1.arcgis.com/FH5XCsx8rYXqnjF5/ArcGIS/rest/services/"
        "Constituency_Boundaries_Ungeneralised_2017/FeatureServer/1",
    ),
    "2b6b493d675f17c75de3d2a76e69ef34": (
        "GUID",
        "https://services-eu1.arcgis.com/FH5XCsx8rYXqnjF5/ArcGIS/rest/services/"
        "Constituency_Boundaries_Ungeneralised_2017/FeatureServer/1",
    ),
    "91d1bb9ad7b0af2b8ca4361f944c3f57": (
        "GUID",
        "https://services-eu1.arcgis.com/FH5XCsx8rYXqnjF5/ArcGIS/rest/services/"
        "Constituency_Boundaries_Ungeneralised/FeatureServer/0",
    ),
    "3d2de896c415cfde086df8ae574df209": (
        "geo_code",
        "https://services-eu1.arcgis.com/BuS9rtTsYEV5C0xh/ArcGIS/rest/services/"
        "Census_1926_County/FeatureServer/2",
    ),
    "c66e74040a8c357b29538ec3021d7ad4": (
        "geocode",
        "https://services-eu1.arcgis.com/BuS9rtTsYEV5C0xh/ArcGIS/rest/services/"
        "Census1911_County/FeatureServer/0",
    ),
    "ba6f4dc7fba1888d5abc63dadc5a243b": (
        "GUID",
        "https://services-eu1.arcgis.com/FH5XCsx8rYXqnjF5/ArcGIS/rest/services/"
        "Administrative_Areas_Ungeneralised/FeatureServer/0",
    ),
    "0fde62f1ceeea0d85105951ac05bd746": (
        "geo_code",
        "https://services-eu1.arcgis.com/BuS9rtTsYEV5C0xh/ArcGIS/rest/services/"
        "Census_1926_DED/FeatureServer/1",
    ),
    "8b8ce7e9b82656cabad440da0160293d": (
        "geocode",
        "https://services-eu1.arcgis.com/BuS9rtTsYEV5C0xh/ArcGIS/rest/services/"
        "Census1911_DED/FeatureServer/0",
    ),
    "e15ef6ed9cdffef0edd313367e50ca7b": (
        "geocode",
        "https://services-eu1.arcgis.com/BuS9rtTsYEV5C0xh/ArcGIS/rest/services/"
        "Census1911_DED/FeatureServer/0",
    ),
    "6ad4f179939b219fda9c5e921a890644": (
        "ED_GUID",
        "https://services-eu1.arcgis.com/BuS9rtTsYEV5C0xh/ArcGIS/rest/services/"
        "CSO_Electoral_Divisions_National_Statistical_Boundaries_2022_"
        "Ungeneralised_view/FeatureServer/1",
    ),
    "8618bd9a9b8b23c966fdd8a37a1b3204": (
        "GUID_",
        "https://services-eu1.arcgis.com/FH5XCsx8rYXqnjF5/ArcGIS/rest/services/"
        "CSO_Electoral_Divisions_Ungeneralised/FeatureServer/0",
    ),
    "c5b950f2f3ab85cc657c4c0082b9fd05": (
        "ED_GUID",
        "https://services-eu1.arcgis.com/BuS9rtTsYEV5C0xh/ArcGIS/rest/services/"
        "CSO_Electoral_Divisions_National_Statistical_Boundaries_2022_"
        "Ungeneralised_view/FeatureServer/1",
    ),
    "ea4d7bf2683f1bbcafc8428c715235b6": (
        "GUID_",
        "https://services-eu1.arcgis.com/FH5XCsx8rYXqnjF5/ArcGIS/rest/services/"
        "CSO_Electoral_Divisions_Ungeneralised/FeatureServer/0",
    ),
    "feba4375fbb00dc945abab0e4477141f": (
        "GUID",
        "https://services-eu1.arcgis.com/FH5XCsx8rYXqnjF5/ArcGIS/rest/services/"
        "Gaeltacht_Boundaries_Ungeneralised___2015/FeatureServer/0",
    ),
    "295fa6b26cb0e26f75e316b64e4c22b4": (
        "IHA_guid",
        "https://services-eu1.arcgis.com/BuS9rtTsYEV5C0xh/ArcGIS/rest/services/"
        "HealthGeographies_2025_IHA/FeatureServer/1",
    ),
    "8c1e622fa6a74cb468237a1273de9c2a": (
        "LEA_GUID",
        "https://services-eu1.arcgis.com/BuS9rtTsYEV5C0xh/ArcGIS/rest/services/"
        "CSO_Local_Electoral_Areas_National_Statistical_Boundaries_2022_"
        "Ungeneralised_view/FeatureServer/3",
    ),
    "9796a5bc22b2e37415c2000f28eeddc8": (
        "LEA_GUID",
        "https://services-eu1.arcgis.com/BuS9rtTsYEV5C0xh/ArcGIS/rest/services/"
        "CSO_Local_Electoral_Areas_National_Statistical_Boundaries_2022_"
        "Ungeneralised_view/FeatureServer/3",
    ),
    "e34a94319c050ca52766e193036eecaa": (
        "GUID",
        "https://services-eu1.arcgis.com/FH5XCsx8rYXqnjF5/ArcGIS/rest/services/"
        "Local_Electoral_Areas_Boundaries_Ungeneralised/FeatureServer/0",
    ),
    "f381d63507530cfc61df96fa5f766e31": (
        "LEA_GUID",
        "https://services-eu1.arcgis.com/BuS9rtTsYEV5C0xh/ArcGIS/rest/services/"
        "CSO_Local_Electoral_Areas_National_Statistical_Boundaries_2022_"
        "Ungeneralised_view/FeatureServer/3",
    ),
    "e8ded10cf938057e222303947b1747dc": (
        "GUID",
        "https://services-eu1.arcgis.com/FH5XCsx8rYXqnjF5/ArcGIS/rest/services/"
        "Local_Electoral_Areas___OSi_National_Statutory_Boundaries/FeatureServer/0",
    ),
    "57bf25130c4f5e8d086f314bbb98ef72": (
        "GUID",
        "https://services-eu1.arcgis.com/FH5XCsx8rYXqnjF5/ArcGIS/rest/services/"
        "NUTS3_Boundaries_Ungeneralised/FeatureServer/0",
    ),
    "1781a48b462bacb8eb860c716e24f609": (
        "GUID",
        "https://services-eu1.arcgis.com/FH5XCsx8rYXqnjF5/ArcGIS/rest/services/"
        "Settlements_Ungeneralised/FeatureServer/0",
    ),
    "07c62efd15e26aa0ccefda89a69e8052": (
        "SA_GUID_2016",
        "https://services-eu1.arcgis.com/BuS9rtTsYEV5C0xh/ArcGIS/rest/services/"
        "Small_Area_National_Statistical_Boundaries_2022_"
        "Ungeneralised_view/FeatureServer/0",
    ),
    "988e6c798cfe938b89771ad4e4769167": (
        "GEOGID",
        "https://services-eu1.arcgis.com/FH5XCsx8rYXqnjF5/ArcGIS/rest/services/"
        "Small_Areas_Ungeneralised/FeatureServer/0",
    ),
    "a9c563c15cc611817af939f70d1d1f04": (
        "GUID",
        "https://services-eu1.arcgis.com/FH5XCsx8rYXqnjF5/ArcGIS/rest/services/"
        "Small_Areas_Ungeneralised/FeatureServer/0",
    ),
    "dc342ded3e0ec8884e99eebd766f8233": (
        "SA_GUID_2022",
        "https://services-eu1.arcgis.com/BuS9rtTsYEV5C0xh/ArcGIS/rest/services/"
        "Small_Area_National_Statistical_Boundaries_2022_"
        "Ungeneralised_view/FeatureServer/0",
    ),
    "01bf8f2912f9795a10c35581cba0dff4": (
        "URBAN_AREA_GUID",
        "https://services-eu1.arcgis.com/BuS9rtTsYEV5C0xh/ArcGIS/rest/services/"
        "Urban_Areas_National_Statistical_Boundaries_2022_"
        "Ungeneralised_View/FeatureServer/0",
    ),
}

# Complex mappings: filecode → (feature_service_url, ni_url_or_none)
# These require special merge logic implemented in dedicated functions.
_COMPLEX_MAPPINGS: dict[str, tuple[str, str | None]] = {
    "c0ad28a75e6fd0c4cc76a50ba859def4": (
        "https://services-eu1.arcgis.com/FH5XCsx8rYXqnjF5/ArcGIS/rest/services/"
        "Administrative_Areas_Ungeneralised/FeatureServer/0",
        "https://admin.opendatani.gov.uk/dataset/"
        "76fa160c-f473-4006-bd73-31849b6f1160/resource/"
        "eb287b2c-9213-4501-bcb0-c4525a34c65b/download/"
        "osni_open_data_largescale_boundaries_local_government_districts_2012.geojson",
    ),
    "526860fb25a6567dae4dbaff1e6d48d3": (
        "https://services-eu1.arcgis.com/FH5XCsx8rYXqnjF5/ArcGIS/rest/services/"
        "Counties_NationalStatutoryBoundaries_Ungeneralised_2024/FeatureServer/1",
        "https://admin.opendatani.gov.uk/dataset/"
        "d0385f2d-6beb-4aff-87dc-f1bf357d792d/resource/"
        "108d8567-3ec7-4403-8912-bcc6233bf361/download/"
        "osni_open_data_largescale_boundaries_county_boundaries.geojson",
    ),
    "893a6eb4f4a6f907410396ec8d8b738b": (
        "https://services-eu1.arcgis.com/FH5XCsx8rYXqnjF5/ArcGIS/rest/services/"
        "Counties_NationalStatutoryBoundaries_Ungeneralised_2024/FeatureServer/1",
        None,
    ),
    "af2d32358e02fff16dfe1f54ecc5225d": (
        "https://services-eu1.arcgis.com/FH5XCsx8rYXqnjF5/ArcGIS/rest/services/"
        "Gaeltacht_Boundaries_Ungeneralised___2015/FeatureServer/0",
        None,
    ),
    "9b504eb50b10e0087c2b4913ade4d10d": (
        "https://services-eu1.arcgis.com/FH5XCsx8rYXqnjF5/ArcGIS/rest/services/"
        "GaeltachtLanguagePlanningAreas_National_AdministrativeBoundaries_"
        "Ungeneralised_2024/FeatureServer/1",
        None,
    ),
    "9ae1df4db5df6639ed4724f3a1b314ee": (
        "https://services-eu1.arcgis.com/FH5XCsx8rYXqnjF5/ArcGIS/rest/services/"
        "Province_Boundaries_Ungeneralised/FeatureServer/0",
        None,
    ),
    "9f352336de5e2a0d42455237888478b7": (
        "https://services-eu1.arcgis.com/FH5XCsx8rYXqnjF5/ArcGIS/rest/services/"
        "Provinces___OSi_National_Statutory_Boundaries/FeatureServer/0",
        None,
    ),
}

# Filecodes for which no ungeneralised geometry is available.
_UNAVAILABLE_FILECODES: frozenset[str] = frozenset(
    {
        "09a3c5e1c9d0ac5fc1ac4cfaa4506e51",
        "9c27b24dcde268707d4cc7d49e7592ea",
        "fe3543e6f1896881f86f1bf126383dd6",
        "46d5c26126f7abf86a319dad3595e406",
        "b7e51a152782b344395bda90a567c9ec",
        "c41e8d6c2dc59550fe8549c59659d1ad",
        "c1701095ec5e222079b52f63beb593aa",
    }
)

# Set of ALL known filecodes for validation.
_ALL_KNOWN_FILECODES: frozenset[str] = (
    frozenset(_SIMPLE_MAPPINGS) | frozenset(_COMPLEX_MAPPINGS) | _UNAVAILABLE_FILECODES
)


# =============================================================================
# Copyright & Licence Information
# =============================================================================
#
# Copyright and licence information is parsed dynamically from the
# cached metadata files (XML, JSON, or text) that are stored alongside
# the geometry data when it is first downloaded.
#
# All Tailte Éireann data is published under the Creative Commons
# Attribution 4.0 International (CC BY 4.0) licence:
#   https://creativecommons.org/licenses/by/4.0/
#
# OSNI data is published under the Open Government Licence v3.0:
#   https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/
#

_DEFAULT_TAILTE_COPYRIGHT: str = "Tailte \u00c9ireann"
_DEFAULT_TAILTE_LICENCE: str = (
    "Not included in metadata; assumed CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)"
)

_OSNI_COPYRIGHT: str = "Ordnance Survey of Northern Ireland"
_OSNI_LICENCE: str = (
    "Open Government Licence v3.0 "
    "(https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/)"
)


# =============================================================================
# Internal: Metadata Writing & Parsing
# =============================================================================


def _write_metadata_txt(metadata_dir: Path, url: str) -> None:
    """Write a plain-text metadata fallback file.

    Called by :func:`_cache_feature_service_metadata` when neither XML
    nor usable JSON metadata is available.  Stores the service URL
    and default copyright/licence information so that
    :func:`_read_cached_copyright` always has *something* to parse.

    Args:
        metadata_dir: Directory in which to write ``metadata.txt``.
        url: The Feature Service URL.
    """
    from datetime import datetime, timezone

    txt_path = metadata_dir / "metadata.txt"
    txt_path.write_text(
        f"URL: {url}\n"
        f"Copyright: {_DEFAULT_TAILTE_COPYRIGHT}\n"
        f"Licence: {_DEFAULT_TAILTE_LICENCE}\n"
        f"Downloaded: {datetime.now(timezone.utc).isoformat()}\n",
        encoding="utf-8",
    )
    logger.debug("Metadata text fallback saved to %s", txt_path)


def _parse_copyright_from_xml(xml_path: Path) -> tuple[str | None, str | None]:
    """Parse copyright and licence from a cached metadata XML file.

    Searches for the ``<credit>`` / ``<idCredit>`` element (copyright)
    and the ``<useLimitation>`` / ``<useLimit>`` element (licence) in
    the XML tree, stripping any embedded HTML tags from the values.

    Args:
        xml_path: Path to the metadata XML file.

    Returns:
        A ``(copyright, licence)`` tuple.  Either value may be ``None``
        if the corresponding element was not found.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except (ET.ParseError, OSError):
        return None, None

    if root is None:
        return None, None

    copyright_text: str | None = None
    licence_text: str | None = None

    for elem in root.iter():
        # Strip namespace prefix
        tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
        tag_lower = tag.lower()

        if tag_lower in ("idcredit", "credit") and not copyright_text:
            raw = (elem.text or "").strip()
            if not raw:
                # Check for a nested <gco:CharacterString> child
                for child in elem:
                    child_tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
                    if child_tag == "CharacterString" and child.text:
                        raw = child.text.strip()
                        break
            if raw:
                copyright_text = re.sub(r"<[^>]+>", "", raw).strip()

        if tag_lower in ("uselimitation", "uselimit", "uselimits") and not licence_text:
            raw = (elem.text or "").strip()
            if not raw:
                for child in elem:
                    child_tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
                    if child_tag == "CharacterString" and child.text:
                        raw = child.text.strip()
                        break
            if raw:
                licence_text = re.sub(r"<[^>]+>", " ", raw).strip()
                licence_text = re.sub(r"\s+", " ", licence_text)

    return copyright_text, licence_text


def _parse_copyright_from_json(json_path: Path) -> tuple[str | None, str | None]:
    """Parse copyright from a cached service properties JSON file.

    Reads the ``copyrightText`` field from the JSON.  Licence
    information is not available in service properties JSON, so
    only the copyright is returned.

    Args:
        json_path: Path to the properties JSON file.

    Returns:
        A ``(copyright, licence)`` tuple.  ``licence`` is always
        ``None`` since the JSON format does not include it.
    """
    try:
        with json_path.open() as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None, None

    copyright_text = (data.get("copyrightText") or "").strip() or None
    return copyright_text, None


def _parse_copyright_from_txt(txt_path: Path) -> tuple[str | None, str | None]:
    """Parse copyright and licence from a cached metadata text file.

    The text file uses a simple ``Key: Value`` format, one per line.

    Args:
        txt_path: Path to the metadata text file.

    Returns:
        A ``(copyright, licence)`` tuple.  Either value may be ``None``
        if the corresponding line was not found.
    """
    try:
        text = txt_path.read_text(encoding="utf-8")
    except OSError:
        return None, None

    copyright_text: str | None = None
    licence_text: str | None = None

    for line in text.splitlines():
        if line.lower().startswith("copyright:"):
            copyright_text = line.split(":", 1)[1].strip() or None
        elif line.lower().startswith("licence:") or line.lower().startswith("license:"):
            licence_text = line.split(":", 1)[1].strip() or None

    return copyright_text, licence_text


def _read_cached_copyright(url: str, prefix: str = "tailte") -> tuple[str | None, str | None]:
    """Read copyright and licence from cached metadata for a URL.

    Checks the cache directory for metadata files in priority order:
    XML → JSON → text.  Returns the first successfully parsed result.

    Args:
        url: The Feature Service or data URL.
        prefix: Cache subdirectory prefix (``'tailte'`` or ``'osni'``).

    Returns:
        A ``(copyright, licence)`` tuple.  Either value may be ``None``
        if no metadata is cached or it could not be parsed.
    """
    cache_dir = _get_cache_dir()
    cache_key = _url_cache_key(url)
    metadata_dir = cache_dir / f"{prefix}_{cache_key}" / "metadata"

    if not metadata_dir.is_dir():
        return None, None

    # Priority 1: XML metadata
    xml_path = metadata_dir / "metadata.xml"
    if xml_path.exists():
        result = _parse_copyright_from_xml(xml_path)
        if result != (None, None):
            return result

    # Priority 2: JSON properties
    json_path = metadata_dir / "properties.json"
    if json_path.exists():
        result = _parse_copyright_from_json(json_path)
        if result != (None, None):
            return result

    # Priority 3: Text metadata
    txt_path = metadata_dir / "metadata.txt"
    if txt_path.exists():
        result = _parse_copyright_from_txt(txt_path)
        if result != (None, None):
            return result

    return None, None


def _log_copyright_info(filecode: str) -> None:
    """Log copyright and licence information for ungeneralised geometry sources.

    Parses copyright and licence dynamically from the cached metadata
    files (XML, JSON, or text) that were saved alongside the geometry
    data.  Falls back to default values if no metadata is available.

    Called each time ungeneralised geometries are accessed (whether
    freshly downloaded or loaded from cache) to ensure users are
    aware of the data licensing terms.

    Args:
        filecode: The CSO filecode identifying the geometry source.
    """
    tailte_url: str | None = None
    ni_url: str | None = None

    if filecode in _SIMPLE_MAPPINGS:
        _, tailte_url = _SIMPLE_MAPPINGS[filecode]
    elif filecode in _COMPLEX_MAPPINGS:
        tailte_url, ni_url = _COMPLEX_MAPPINGS[filecode]

    if tailte_url is None:
        return

    # Parse copyright and licence from cached metadata
    copyright_text, licence_text = _read_cached_copyright(tailte_url, prefix="tailte")
    copyright_text = copyright_text or _DEFAULT_TAILTE_COPYRIGHT
    licence_text = licence_text or _DEFAULT_TAILTE_LICENCE

    print(
        f"Ungeneralised boundary data: {copyright_text}.\nLicence: {licence_text}.",
        file=sys.stderr,
    )

    # Log OSNI copyright if Northern Ireland data is included
    if ni_url is not None:
        print(
            f"Northern Ireland boundary data: \u00a9 {_OSNI_COPYRIGHT}.\nLicence: {_OSNI_LICENCE}.",
            file=sys.stderr,
        )


# =============================================================================
# Public API
# =============================================================================


def create_ungeneralised_geodataframe(
    df: pd.DataFrame,
    spatial_url: str | None,
    spatial_key: str | None,
    *,
    cache: bool = True,
    force_reload: bool = False,
) -> gpd.GeoDataFrame:
    """Create a GeoDataFrame by merging data with ungeneralised geometries.

    Downloads high-resolution geometry data from Tailte Éireann ArcGIS
    Feature Services (and OSNI for Northern Ireland boundaries where
    applicable), replacing the generalised geometries from the CSO API.

    Downloaded geometry files are cached to disk for subsequent use.
    A coordinate count comparison is performed to verify that the
    ungeneralised geometry is indeed more detailed than the default.

    Args:
        df: The statistical DataFrame to merge with spatial boundaries.
        spatial_url: URL to the CSO GeoJSON (contains the filecode as
            the last path segment).
        spatial_key: The dimension label for the spatial join key.
        cache: Whether to use cached files if available.
        force_reload: Whether to force re-download even if cached
            files exist.

    Returns:
        A GeoDataFrame with ungeneralised geometry. Rows without
        matching spatial boundaries will have null geometries.

    Raises:
        SpatialError: If spatial data is unavailable, ungeneralised
            geometry is not available for the dataset's filecode, the
            filecode is unrecognised, or the merge fails.
    """
    if not spatial_url or not spatial_key:
        raise SpatialError("Dataset has no spatial information available.")

    filecode = _extract_filecode(spatial_url)

    # Check availability
    if filecode in _UNAVAILABLE_FILECODES:
        raise SpatialError(
            f"Ungeneralised geometry is not available for this dataset's spatial "
            f"boundaries (filecode: {filecode}). Consider using ungeneralised=False "
            f"to use the standard generalised geometries from the CSO API instead."
        )

    if filecode not in _ALL_KNOWN_FILECODES:
        raise SpatialError(
            f"Filecode '{filecode}' is not recognised in the ungeneralised geometry "
            f"mapping data. Ungeneralised geometry may not be available for this "
            f"dataset. Consider using ungeneralised=False instead."
        )

    # Log copyright and licence information for the geometry sources
    _log_copyright_info(filecode)

    try:
        # Fetch CSO GeoJSON for attribute columns and default geometry comparison
        geojson = fetch_json(spatial_url, cache=cache)
        features = geojson.get("features", [])

        if not features:
            raise SpatialError("No features found in CSO GeoJSON data.")

        cso_geo_gdf = gpd.GeoDataFrame.from_features(features)
        if cso_geo_gdf.crs is None:
            cso_geo_gdf = cso_geo_gdf.set_crs(_detect_crs(geojson))

        # Count default geometry coordinates for later comparison
        default_coord_count = _count_coordinates(cso_geo_gdf)

        # Drop geometry from CSO data (keep attributes for merging)
        cso_attrs = pd.DataFrame(cso_geo_gdf.drop(columns=["geometry"]))

        # Build ungeneralised geometry
        geo_gdf = _build_ungeneralised_gdf(filecode, cso_attrs, force_reload=force_reload)

        # Compare coordinate counts
        ungeneralised_coord_count = _count_coordinates(geo_gdf)
        if ungeneralised_coord_count <= default_coord_count:
            logger.warning(
                "Ungeneralised geometry for filecode '%s' has %d coordinates, "
                "which is not more than the default geometry (%d coordinates). "
                "The ungeneralised data may not be more detailed.",
                filecode,
                ungeneralised_coord_count,
                default_coord_count,
            )
        else:
            print(
                f"Ungeneralised geometry has {ungeneralised_coord_count} coordinates "
                f"vs {default_coord_count} for default "
                f"({ungeneralised_coord_count // max(default_coord_count, 1)}x more detailed).",
                file=sys.stderr,
            )

        # Merge with statistical data using existing merge logic
        merged = _merge_dataframes(df, geo_gdf, spatial_key, geojson)

        if merged is None:
            raise SpatialError("Spatial merge with ungeneralised geometry failed.")

        return merged

    except SpatialError:
        raise
    except (KeyError, ValueError) as e:
        raise SpatialError(f"Error creating ungeneralised GeoDataFrame: {e}") from e


# =============================================================================
# Internal: Filecode Extraction
# =============================================================================


def _extract_filecode(spatial_url: str) -> str:
    """Extract the CSO filecode from a spatial GeoJSON URL.

    The filecode is the last path segment of the URL, e.g.::

        https://ws.cso.ie/public/api.static/PxStat.Data.GeoMap_API.Read/abc123
        → 'abc123'

    Args:
        spatial_url: The CSO GeoJSON URL.

    Returns:
        The filecode string.
    """
    return spatial_url.rstrip("/").split("/")[-1]


# =============================================================================
# Internal: Coordinate Counting
# =============================================================================


def _force_2d(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Force all geometries in a GeoDataFrame to 2D.

    If any geometry has Z coordinates (e.g. POLYGON Z, MULTIPOLYGON Z),
    applies ``shapely.force_2d`` to strip the Z dimension.

    Args:
        gdf: The GeoDataFrame to process.

    Returns:
        The GeoDataFrame with only 2D geometries.
    """
    if gdf.geometry.isna().all():
        return gdf

    has_z = gdf.geometry.dropna().apply(lambda g: g.has_z).any()
    if has_z:
        logger.debug("Detected 3D geometries (Z coordinates); forcing to 2D.")
        gdf = gdf.copy()
        gdf["geometry"] = gdf.geometry.apply(lambda g: shapely.force_2d(g) if g is not None else g)
    return gdf


def _count_coordinates(gdf: gpd.GeoDataFrame) -> int:
    """Count total coordinates across all non-null geometries.

    Uses ``GeoSeries.count_coordinates()`` if available (geopandas >= 1.0),
    falling back to ``shapely.get_num_coordinates`` for older versions.

    Args:
        gdf: The GeoDataFrame to count coordinates in.

    Returns:
        Total number of coordinates across all geometries.
    """
    non_null = gdf[~gdf.geometry.isna()]
    if len(non_null) == 0:
        return 0

    try:
        return int(non_null.geometry.count_coordinates().sum())
    except AttributeError:
        pass

    try:
        import shapely

        return int(sum(shapely.get_num_coordinates(g) for g in non_null.geometry if g is not None))
    except (ImportError, AttributeError):
        return 0


# =============================================================================
# Internal: Build Ungeneralised GeoDataFrame
# =============================================================================


def _build_ungeneralised_gdf(
    filecode: str,
    cso_attrs: pd.DataFrame,
    *,
    force_reload: bool = False,
) -> gpd.GeoDataFrame:
    """Build a GeoDataFrame with ungeneralised geometry for a filecode.

    Dispatches to the appropriate merge strategy (simple or complex).

    Args:
        filecode: The CSO filecode.
        cso_attrs: CSO GeoJSON attributes without geometry.
        force_reload: Whether to force re-download.

    Returns:
        A GeoDataFrame with CSO attributes and ungeneralised geometry.
    """
    if filecode in _SIMPLE_MAPPINGS:
        return _simple_merge(filecode, cso_attrs, force_reload=force_reload)

    if filecode in _COMPLEX_MAPPINGS:
        return _complex_merge(filecode, cso_attrs, force_reload=force_reload)

    raise SpatialError(f"No merge strategy found for filecode '{filecode}'.")


def _simple_merge(
    filecode: str,
    cso_attrs: pd.DataFrame,
    *,
    force_reload: bool = False,
) -> gpd.GeoDataFrame:
    """Perform a simple merge on the 'code' column.

    Downloads the Tailte Feature Service data, renames the ID field
    to 'code', and merges with CSO attributes.

    Args:
        filecode: The CSO filecode.
        cso_attrs: CSO GeoJSON attributes without geometry.
        force_reload: Whether to force re-download.

    Returns:
        A GeoDataFrame with CSO attributes and ungeneralised geometry.
    """
    id_field, tailte_url = _SIMPLE_MAPPINGS[filecode]
    tailte_gdf = _download_feature_service(
        tailte_url, out_fields=id_field, force_reload=force_reload
    )
    tailte_gdf = tailte_gdf[[id_field, "geometry"]].rename(columns={id_field: "code"})
    merged = pd.merge(cso_attrs, tailte_gdf, on="code", how="left")
    result = gpd.GeoDataFrame(merged, geometry="geometry", crs=tailte_gdf.crs)
    return result


# =============================================================================
# Internal: Complex Merge Strategies
# =============================================================================


def _complex_merge(
    filecode: str,
    cso_attrs: pd.DataFrame,
    *,
    force_reload: bool = False,
) -> gpd.GeoDataFrame:
    """Dispatch to the appropriate complex merge function.

    Args:
        filecode: The CSO filecode identifying the complex case.
        cso_attrs: CSO GeoJSON attributes without geometry.
        force_reload: Whether to force re-download.

    Returns:
        A GeoDataFrame with CSO attributes and ungeneralised geometry.
    """
    dispatch: dict[str, Any] = {
        "c0ad28a75e6fd0c4cc76a50ba859def4": _merge_admin_areas_with_ni_lgd,
        "893a6eb4f4a6f907410396ec8d8b738b": _merge_counties_dissolve,
        "526860fb25a6567dae4dbaff1e6d48d3": _merge_counties_with_ni,
        "af2d32358e02fff16dfe1f54ecc5225d": _merge_gaeltacht_county_title,
        "9ae1df4db5df6639ed4724f3a1b314ee": _merge_provinces_ulster,
        "9f352336de5e2a0d42455237888478b7": _merge_provinces_guid_lower,
        "9b504eb50b10e0087c2b4913ade4d10d": _merge_gaeltacht_lp_dissolve,
    }

    merge_fn = dispatch.get(filecode)
    if merge_fn is None:
        raise SpatialError(f"No complex merge function for filecode '{filecode}'.")

    return merge_fn(filecode, cso_attrs, force_reload=force_reload)


def _merge_admin_areas_with_ni_lgd(
    filecode: str,
    cso_attrs: pd.DataFrame,
    *,
    force_reload: bool = False,
) -> gpd.GeoDataFrame:
    """Merge RoI Administrative Areas on GUID, NI on LGDFilecode.

    Filecode: c0ad28a75e6fd0c4cc76a50ba859def4
    """
    tailte_url, ni_url = _COMPLEX_MAPPINGS[filecode]

    # RoI: merge on GUID → code
    tailte_gdf = _download_feature_service(tailte_url, out_fields="GUID", force_reload=force_reload)
    tailte_gdf = tailte_gdf[["GUID", "geometry"]].rename(columns={"GUID": "code"})

    # NI: merge on LGDFilecode → code
    if ni_url is None:
        raise SpatialError("NI URL is required for admin areas merge.")
    ni_gdf = _download_ni_geojson(ni_url, force_reload=force_reload)
    ni_gdf = ni_gdf[["LGDFilecode", "geometry"]].rename(columns={"LGDFilecode": "code"})
    if tailte_gdf.crs:
        ni_gdf = ni_gdf.to_crs(tailte_gdf.crs)
    else:
        raise SpatialError("Tailte geometry has no CRS; cannot reproject NI data.")

    # Combine RoI + NI geometries
    geo_gdf = pd.concat([tailte_gdf, ni_gdf], ignore_index=True)[["code", "geometry"]]

    merged = pd.merge(cso_attrs, geo_gdf, on="code", how="left")
    return gpd.GeoDataFrame(merged, geometry="geometry", crs=tailte_gdf.crs)


def _merge_counties_dissolve(
    filecode: str,
    cso_attrs: pd.DataFrame,
    *,
    force_reload: bool = False,
) -> gpd.GeoDataFrame:
    """Dissolve Tailte counties by ENG_NAME_VALUE, merge on 'en'.

    Filecode: 893a6eb4f4a6f907410396ec8d8b738b
    """
    tailte_url, _ = _COMPLEX_MAPPINGS[filecode]

    tailte_gdf = _download_feature_service(
        tailte_url, out_fields="ENG_NAME_VALUE", force_reload=force_reload
    )
    tailte_gdf = tailte_gdf[["ENG_NAME_VALUE", "geometry"]].rename(columns={"ENG_NAME_VALUE": "en"})
    tailte_gdf = tailte_gdf.dissolve(by="en", as_index=False)

    merged = pd.merge(cso_attrs, tailte_gdf, on="en", how="left")
    return gpd.GeoDataFrame(merged, geometry="geometry", crs=tailte_gdf.crs)


def _merge_counties_with_ni(
    filecode: str,
    cso_attrs: pd.DataFrame,
    *,
    force_reload: bool = False,
) -> gpd.GeoDataFrame:
    """Dissolve Tailte counties + NI counties; merge DERRY/LONDONDERRY.

    Filecode: 526860fb25a6567dae4dbaff1e6d48d3
    """
    tailte_url, ni_url = _COMPLEX_MAPPINGS[filecode]

    # RoI counties: dissolve by ENG_NAME_VALUE
    tailte_gdf = _download_feature_service(
        tailte_url, out_fields="ENG_NAME_VALUE", force_reload=force_reload
    )
    tailte_gdf = tailte_gdf[["ENG_NAME_VALUE", "geometry"]].rename(columns={"ENG_NAME_VALUE": "en"})
    tailte_gdf = tailte_gdf.dissolve(by="en", as_index=False)

    # NI counties: rename CountyName → en, merge DERRY/LONDONDERRY
    if ni_url is None:
        raise SpatialError("NI URL is required for counties merge.")
    ni_gdf = _download_ni_geojson(ni_url, force_reload=force_reload)
    ni_gdf = ni_gdf[["CountyName", "geometry"]].rename(columns={"CountyName": "en"})
    if tailte_gdf.crs:
        ni_gdf = ni_gdf.to_crs(tailte_gdf.crs)
    else:
        raise SpatialError("Tailte geometry has no CRS; cannot reproject NI data.")
    ni_gdf["en"] = ni_gdf["en"].replace({"LONDONDERRY": "DERRY/LONDONDERRY"})

    # Combine
    geo_gdf = pd.concat([tailte_gdf, ni_gdf], ignore_index=True)
    merged = pd.merge(cso_attrs, geo_gdf, on="en", how="left")
    return gpd.GeoDataFrame(merged, geometry="geometry", crs=tailte_gdf.crs)


def _merge_gaeltacht_county_title(
    filecode: str,
    cso_attrs: pd.DataFrame,
    *,
    force_reload: bool = False,
) -> gpd.GeoDataFrame:
    """Match 'en' with COUNTY.title().

    Filecode: af2d32358e02fff16dfe1f54ecc5225d
    """
    tailte_url, _ = _COMPLEX_MAPPINGS[filecode]

    tailte_gdf = _download_feature_service(
        tailte_url, out_fields="COUNTY", force_reload=force_reload
    )
    tailte_gdf = tailte_gdf[["COUNTY", "geometry"]].rename(columns={"COUNTY": "en"})
    tailte_gdf["en"] = tailte_gdf["en"].str.title()

    merged = pd.merge(cso_attrs, tailte_gdf, on="en", how="left")
    return gpd.GeoDataFrame(merged, geometry="geometry", crs=tailte_gdf.crs)


def _merge_provinces_ulster(
    filecode: str,
    cso_attrs: pd.DataFrame,
    *,
    force_reload: bool = False,
) -> gpd.GeoDataFrame:
    """Map 'Ulster' → 'Ulster (part of)' for province matching.

    Filecode: 9ae1df4db5df6639ed4724f3a1b314ee
    """
    tailte_url, _ = _COMPLEX_MAPPINGS[filecode]

    tailte_gdf = _download_feature_service(
        tailte_url, out_fields="PROVINCE", force_reload=force_reload
    )
    tailte_gdf = tailte_gdf[["PROVINCE", "geometry"]].rename(columns={"PROVINCE": "en"})
    tailte_gdf["en"] = tailte_gdf["en"].replace({"Ulster": "Ulster (part of)"})

    merged = pd.merge(cso_attrs, tailte_gdf, on="en", how="left")
    return gpd.GeoDataFrame(merged, geometry="geometry", crs=tailte_gdf.crs)


def _merge_provinces_guid_lower(
    filecode: str,
    cso_attrs: pd.DataFrame,
    *,
    force_reload: bool = False,
) -> gpd.GeoDataFrame:
    """Match 'code' with GUID.lower().

    Filecode: 9f352336de5e2a0d42455237888478b7
    """
    tailte_url, _ = _COMPLEX_MAPPINGS[filecode]

    tailte_gdf = _download_feature_service(tailte_url, out_fields="GUID", force_reload=force_reload)
    tailte_gdf = tailte_gdf[["GUID", "geometry"]].rename(columns={"GUID": "code"})
    tailte_gdf["code"] = tailte_gdf["code"].str.lower()

    merged = pd.merge(cso_attrs, tailte_gdf, on="code", how="left")
    return gpd.GeoDataFrame(merged, geometry="geometry", crs=tailte_gdf.crs)


def _merge_gaeltacht_lp_dissolve(
    filecode: str,
    cso_attrs: pd.DataFrame,
    *,
    force_reload: bool = False,
) -> gpd.GeoDataFrame:
    """Dissolve Gaeltacht Language Planning Areas by ENG_NAME_VALUE.

    Filecode: 9b504eb50b10e0087c2b4913ade4d10d
    """
    tailte_url, _ = _COMPLEX_MAPPINGS[filecode]

    tailte_gdf = _download_feature_service(
        tailte_url, out_fields="ENG_NAME_VALUE", force_reload=force_reload
    )
    tailte_gdf = tailte_gdf[["ENG_NAME_VALUE", "geometry"]].rename(columns={"ENG_NAME_VALUE": "en"})
    tailte_gdf = tailte_gdf.dissolve(by="en", as_index=False)

    merged = pd.merge(cso_attrs, tailte_gdf, on="en", how="left")
    return gpd.GeoDataFrame(merged, geometry="geometry", crs=tailte_gdf.crs)


# =============================================================================
# Internal: Download and Cache Functions
# =============================================================================


def _url_cache_key(url: str) -> str:
    """Create a filesystem-safe cache key from a URL.

    Args:
        url: The URL to hash.

    Returns:
        A 16-character hexadecimal hash string.
    """
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def _create_session() -> requests.Session:
    """Create a requests session with retry logic and connection pooling.

    Returns:
        A configured :class:`requests.Session`.
    """
    session = requests.Session()
    retry_strategy = Retry(
        total=DEFAULT_RETRIES,
        backoff_factor=RETRY_DELAY_MULTIPLIER,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=_MAX_WORKERS,
        pool_maxsize=_MAX_WORKERS,
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _download_feature_service(
    url: str,
    out_fields: str = "*",
    *,
    force_reload: bool = False,
) -> gpd.GeoDataFrame:
    """Download geometry from a Tailte Éireann ArcGIS Feature Service.

    Uses the ArcGIS REST API directly with paginated queries and
    concurrent page downloads for fast retrieval.  Downloads are
    cached to disk as GeoPackage files.  On subsequent calls the
    cached file is loaded unless ``force_reload=True``.

    The Feature Service metadata (XML where available, otherwise
    service properties as JSON) is cached alongside the geometry.

    Args:
        url: The Feature Service layer URL.
        out_fields: Fields to request.  Use ``'*'`` for all fields.
            Individual fields are selected from the cached result.
        force_reload: Whether to force re-download.

    Returns:
        A GeoDataFrame with the requested fields and geometry.

    Raises:
        SpatialError: If the download or conversion fails.
    """
    cache_dir = _get_cache_dir()
    cache_key = _url_cache_key(url)
    cache_subdir = cache_dir / f"tailte_{cache_key}"
    cache_file = cache_subdir / "features.gpkg"

    # Try loading from cache
    if cache_file.exists() and not force_reload:
        print(f"Loading cached ungeneralised geometry from {cache_file}", file=sys.stderr)
        gdf = gpd.read_file(cache_file)
        # Select requested fields (if specific fields were requested)
        if out_fields != "*" and out_fields in gdf.columns:
            return gdf[[out_fields, "geometry"]]
        return gdf

    # Download from Feature Service via the ArcGIS REST API
    print(
        "Downloading ungeneralised geometry from Tailte Éireann Feature Service...",
        file=sys.stderr,
    )
    print(f"  URL: {url}", file=sys.stderr)

    session = _create_session()
    try:
        # ------------------------------------------------------------------
        # 1. Query service information for CRS and max record count
        # ------------------------------------------------------------------
        service_resp = session.get(url, params={"f": "json"}, timeout=DEFAULT_TIMEOUT)
        service_resp.raise_for_status()
        service_info = service_resp.json()

        max_record_count = service_info.get("maxRecordCount", 1000)
        sr = service_info.get("extent", {}).get("spatialReference", {})
        native_wkid = sr.get("latestWkid") or sr.get("wkid", 4326)

        # ------------------------------------------------------------------
        # 2. Get total feature count
        # ------------------------------------------------------------------
        count_resp = session.get(
            f"{url}/query",
            params={"where": "1=1", "returnCountOnly": "true", "f": "json"},
            timeout=DEFAULT_TIMEOUT,
        )
        count_resp.raise_for_status()
        total_count = count_resp.json()["count"]

        print(f"  Feature count: {total_count}", file=sys.stderr)

        # ------------------------------------------------------------------
        # 3. Download feature pages with page-level progress
        # ------------------------------------------------------------------
        page_size = max(int(max_record_count or 1000), 1)
        offsets = list(range(0, total_count, page_size))
        n_pages = len(offsets)
        query_out_fields = "*" if out_fields == "*" else out_fields

        def _fetch_page(offset: int) -> list[dict[str, Any]]:
            """Download a single page and return its GeoJSON features."""
            resp = session.get(
                f"{url}/query",
                params={
                    "where": "1=1",
                    "outFields": query_out_fields,
                    "f": "geojson",
                    "resultOffset": str(offset),
                    "resultRecordCount": str(page_size),
                },
                timeout=DEFAULT_TIMEOUT * 3,
            )
            resp.raise_for_status()
            payload = resp.json()
            features = payload.get("features", [])
            if not isinstance(features, list):
                raise SpatialError(f"Unexpected page payload from {url}")
            return features

        def _fetch_object_id_batch(object_ids: list[int]) -> list[dict[str, Any]]:
            """Download a batch of features by object IDs."""
            resp = session.get(
                f"{url}/query",
                params={
                    "objectIds": ",".join(str(object_id) for object_id in object_ids),
                    "outFields": query_out_fields,
                    "f": "geojson",
                },
                timeout=DEFAULT_TIMEOUT * 3,
            )
            resp.raise_for_status()
            payload = resp.json()
            features = payload.get("features", [])
            if not isinstance(features, list):
                raise SpatialError(f"Unexpected object-id payload from {url}")
            return features

        def _choose_object_id_batch_size() -> int:
            """Select object-id batch size by layer size.

            Smaller batches are faster for tiny layers with heavy geometries,
            while medium batches avoid excessive request overhead for larger layers.
            """
            if total_count <= 64:
                return 1
            if total_count <= 2048:
                return 25
            return 100

        def _choose_object_id_workers(n_batches: int) -> int:
            """Select a conservative worker count for object-id sharding."""
            if total_count > 2048:
                return min(_MAX_WORKERS + 2, n_batches)
            return min(_MAX_WORKERS * 2, n_batches)

        def _try_object_id_fetch() -> list[dict[str, Any]] | None:
            """Try object-id sharding across single- and multi-page layers."""
            ids_resp = session.get(
                f"{url}/query",
                params={"where": "1=1", "returnIdsOnly": "true", "f": "json"},
                timeout=DEFAULT_TIMEOUT,
            )
            ids_resp.raise_for_status()
            ids_data = ids_resp.json()

            object_ids = ids_data.get("objectIds")
            if not isinstance(object_ids, list):
                return None
            if len(object_ids) != total_count:
                return None

            batch_size = _choose_object_id_batch_size()
            batches = [
                object_ids[i : i + batch_size] for i in range(0, len(object_ids), batch_size)
            ]
            max_workers = max(1, _choose_object_id_workers(len(batches)))

            all_features_local: list[dict[str, Any]] = []
            if n_pages == 1:
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [executor.submit(_fetch_object_id_batch, batch) for batch in batches]
                    for future in concurrent.futures.as_completed(futures):
                        all_features_local.extend(future.result())
                return all_features_local

            # Keep progress in page units for multi-page layers.
            with (
                tqdm(
                    total=n_pages,
                    desc=f"Downloading features ({n_pages} pages)",
                    unit="page",
                    leave=True,
                ) as pbar,
                concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor,
            ):
                future_to_batch_idx = {
                    executor.submit(_fetch_object_id_batch, batch): idx
                    for idx, batch in enumerate(batches)
                }
                completed_batches: dict[int, list[dict[str, Any]]] = {}
                downloaded_count = 0
                pages_completed = 0

                for future in concurrent.futures.as_completed(future_to_batch_idx):
                    batch_idx = future_to_batch_idx[future]
                    batch_features = future.result()
                    completed_batches[batch_idx] = batch_features
                    downloaded_count += len(batch_features)

                    target_pages = min(n_pages, downloaded_count // page_size)
                    if target_pages > pages_completed:
                        pbar.update(target_pages - pages_completed)
                        pages_completed = target_pages

                if pages_completed < n_pages:
                    pbar.update(n_pages - pages_completed)

                for batch_idx in range(len(batches)):
                    all_features_local.extend(completed_batches.get(batch_idx, []))

            return all_features_local

        object_id_features = _try_object_id_fetch()
        if object_id_features is not None:
            all_features = object_id_features
        elif n_pages == 1:
            all_features = _fetch_page(offsets[0])
        else:
            page_features: dict[int, list[dict[str, Any]]] = {}
            with (
                tqdm(
                    total=n_pages,
                    desc=f"Downloading features ({n_pages} pages)",
                    unit="page",
                    leave=True,
                ) as pbar,
                concurrent.futures.ThreadPoolExecutor(
                    max_workers=min(_MAX_WORKERS, n_pages)
                ) as executor,
            ):
                futures = {executor.submit(_fetch_page, offset): offset for offset in offsets}
                for future in concurrent.futures.as_completed(futures):
                    offset = futures[future]
                    page_features[offset] = future.result()
                    pbar.update(1)

            all_features = []
            for offset in offsets:
                all_features.extend(page_features.get(offset, []))

        if not all_features:
            raise SpatialError(f"No features returned from {url}")

        # ------------------------------------------------------------------
        # 4. Build GeoDataFrame (GeoJSON is WGS 84 by default)
        # ------------------------------------------------------------------
        gdf = gpd.GeoDataFrame.from_features(all_features, crs="EPSG:4326")

        # Strip Z dimension if present (POLYGON Z / MULTIPOLYGON Z → 2D)
        gdf = _force_2d(gdf)

        # Reproject to the service's native CRS if it is not WGS 84
        if native_wkid != 4326:
            gdf = gdf.to_crs(epsg=native_wkid)

        print(f"  Download complete ({len(gdf)} features).", file=sys.stderr)

    except SpatialError:
        raise
    except Exception as e:
        raise SpatialError(f"Failed to download Feature Service from {url}: {e}") from e
    finally:
        session.close()

    # Cache the result
    try:
        cache_subdir.mkdir(parents=True, exist_ok=True)
        gdf.to_file(cache_file, driver="GPKG")
        print(f"  Cached to {cache_file}", file=sys.stderr)
    except Exception as e:
        logger.warning("Failed to cache geometry to %s: %s", cache_file, e)

    # Cache metadata
    _cache_feature_service_metadata(url, cache_subdir)

    # Update README
    _update_readme(cache_dir, f"tailte_{cache_key}/features.gpkg", url)

    # Select requested fields
    if out_fields != "*" and out_fields in gdf.columns:
        return gdf[[out_fields, "geometry"]]
    return gdf


def _download_ni_geojson(
    url: str,
    *,
    force_reload: bool = False,
) -> gpd.GeoDataFrame:
    """Download geometry from an OSNI GeoJSON URL.

    Downloads are cached to disk as GeoPackage files.

    Args:
        url: The OSNI GeoJSON URL.
        force_reload: Whether to force re-download.

    Returns:
        A GeoDataFrame with all columns from the GeoJSON.

    Raises:
        SpatialError: If the download fails.
    """
    cache_dir = _get_cache_dir()
    cache_key = _url_cache_key(url)
    cache_subdir = cache_dir / f"osni_{cache_key}"
    cache_file = cache_subdir / "features.gpkg"

    # Try loading from cache
    if cache_file.exists() and not force_reload:
        print(f"Loading cached NI geometry from {cache_file}", file=sys.stderr)
        return gpd.read_file(cache_file)

    # Download
    print("Downloading geometry from OSNI Open Data...", file=sys.stderr)
    print(f"  URL: {url}", file=sys.stderr)

    try:
        gdf = gpd.read_file(url)
        # Strip Z dimension if present
        gdf = _force_2d(gdf)
        print(f"  Download complete ({len(gdf)} features).", file=sys.stderr)
    except Exception as e:
        raise SpatialError(f"Failed to download NI geometry from {url}: {e}") from e

    # Cache the result
    try:
        cache_subdir.mkdir(parents=True, exist_ok=True)
        gdf.to_file(cache_file, driver="GPKG")
        print(f"  Cached to {cache_file}", file=sys.stderr)
    except Exception as e:
        logger.warning("Failed to cache NI geometry to %s: %s", cache_file, e)

    # Update README
    _update_readme(cache_dir, f"osni_{cache_key}/features.gpkg", url)

    return gdf


def _cache_feature_service_metadata(url: str, cache_subdir: Path) -> None:
    """Cache Feature Service metadata to disk.

    Attempts to download metadata in the following priority order:

    1. **XML metadata** from the service's ``/metadata`` endpoint
       (FGDC/ISO format with ``<credit>`` and ``<useLimitation>``).
    2. **JSON service properties** from the service endpoint with
       ``?f=pjson`` (contains ``copyrightText``).
    3. **Plain text fallback** containing the service URL and default
       copyright/licence information.

    At least one of these will always be written so that
    :func:`_read_cached_copyright` can find something on disk.

    Args:
        url: The Feature Service layer URL.
        cache_subdir: Directory to save metadata in.
    """
    metadata_dir = cache_subdir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    # ---- Priority 1: XML metadata ----
    try:
        resp = requests.get(f"{url}/metadata", timeout=DEFAULT_TIMEOUT)
        content_type = resp.headers.get("Content-Type", "").lower()
        if resp.status_code == 200 and (
            "xml" in content_type or resp.text.strip().startswith("<?xml")
        ):
            metadata_path = metadata_dir / "metadata.xml"
            with metadata_path.open("wb") as f:
                f.write(resp.content)
            logger.debug("Feature Service metadata (XML) saved to %s", metadata_path)
            return
    except Exception as e:
        logger.debug("Failed to download metadata XML from %s: %s", url, e)

    # ---- Priority 2: JSON service properties ----
    try:
        props_resp = requests.get(url, params={"f": "pjson"}, timeout=DEFAULT_TIMEOUT)
        props_resp.raise_for_status()
        props_data = props_resp.json()
        props_path = metadata_dir / "properties.json"
        with props_path.open("w") as f:
            json.dump(props_data, f, indent=2, default=str)
        logger.debug("Feature Service properties (JSON) saved to %s", props_path)

        # Only treat JSON as sufficient if it actually contains copyright info
        if (props_data.get("copyrightText") or "").strip():
            return
    except Exception as e:
        logger.debug("Failed to download service properties JSON from %s: %s", url, e)

    # ---- Priority 3: Plain text fallback ----
    _write_metadata_txt(metadata_dir, url)


def _update_readme(cache_dir: Path, filename: str, url: str) -> None:
    """Append a download record to the cache README.txt.

    Args:
        cache_dir: The base cache directory.
        filename: The name/path of the cached file.
        url: The URL from which the file was downloaded.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    readme_path = cache_dir / "README.txt"

    timestamp = datetime.now(tz=timezone.utc).isoformat()

    # Create header if file doesn't exist
    if not readme_path.exists():
        header = (
            "pycsodata Ungeneralised Geometry Cache\n"
            "======================================\n"
            "\n"
            "This directory contains cached ungeneralised geometry files\n"
            "downloaded by pycsodata from Tailte Éireann ArcGIS Feature\n"
            "Services and OSNI Open Data.\n"
            "\n"
            "These files are used when ungeneralised=True is specified in\n"
            "CSODataset.gdf(). They can be safely deleted; they will be\n"
            "re-downloaded when needed.\n"
            "\n"
            "Download Log:\n"
            "Timestamp | Filename | Source URL\n"
            "----------|----------|-----------\n"
        )
        with readme_path.open("w") as f:
            f.write(header)

    # Append download record
    with readme_path.open("a") as f:
        f.write(f"{timestamp} | {filename} | {url}\n")
