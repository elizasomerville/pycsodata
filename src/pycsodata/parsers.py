"""Parsing utilities for CSO API responses.

This module handles parsing and transformation of data from the CSO API,
including text repair for encoding issues, metadata extraction, and
temporal column processing.

Public Functions:
    repair_text: Fix incorrectly encoded Irish characters.
    repair_json: Recursively repair encoding in JSON structures.
    extract_spatial_info: Extract spatial configuration from metadata.
    extract_id_mapping: Create label-to-ID mapping for a dimension.
    parse_metadata: Parse raw API metadata into structured format.
    parse_temporal_column: Convert temporal columns to appropriate types.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from pycsodata._types import DatasetMetadata, SpatialInfo
from pycsodata.constants import MISENCODED_CHARACTER_MAP, STATISTIC_LABELS

# =============================================================================
# Text Repair Functions
# =============================================================================


def repair_text(text: str) -> str:
    """Fix incorrectly encoded Irish characters in text.

    The CSO API sometimes returns Irish characters (fadas) with incorrect
    encoding. This function repairs those characters.

    Args:
        text: The text to repair.

    Returns:
        The text with corrected character encoding.

    Examples:
        >>> repair_text("┴ras an Uachtarßin")
        'Áras an Uachtaráin'
    """
    for bad_char, good_char in MISENCODED_CHARACTER_MAP.items():
        text = text.replace(bad_char, good_char)
    return text


def repair_json(obj: Any) -> Any:
    """Recursively repair encoding issues in a JSON-like structure.

    Traverses dictionaries, lists, and strings, applying character
    encoding fixes throughout.

    Args:
        obj: A JSON-like object (dict, list, str, or primitive).

    Returns:
        The same structure with all strings repaired.

    Examples:
        >>> repair_json({"name": "╔ire", "values": ["Θire"]})
        {'name': 'Éire', 'values': ['éire']}
    """
    if isinstance(obj, dict):
        return {key: repair_json(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [repair_json(item) for item in obj]
    if isinstance(obj, str):
        return repair_text(obj)
    return obj


# =============================================================================
# Metadata Extraction Functions
# =============================================================================


def extract_spatial_info(metadata: dict[str, Any]) -> SpatialInfo:
    """Extract spatial data configuration from metadata.

    Searches the dimension metadata for a link to spatial boundary data.

    Args:
        metadata: The full metadata dictionary from the CSO API.

    Returns:
        A SpatialInfo object with URL and key, or empty if not found.

    Examples:
        >>> info = extract_spatial_info(metadata)
        >>> if info.is_available:
        ...     print(f"Spatial data at: {info.url}")
    """
    for dimension in metadata.get("dimension", {}).values():
        enclosure = dimension.get("link", {}).get("enclosure", [])
        if enclosure and "href" in enclosure[0]:
            return SpatialInfo(
                url=enclosure[0]["href"],
                key=dimension.get("label"),
            )
    return SpatialInfo()


def extract_id_mapping(dimension_info: dict[str, Any]) -> dict[str, str]:
    """Create a mapping from labels to IDs for a dimension.

    Args:
        dimension_info: The dimension metadata dictionary.

    Returns:
        A dict mapping label strings to their corresponding ID codes.

    Examples:
        >>> mapping = extract_id_mapping(dim_info)
        >>> mapping["Dublin"]
        'IE061'
    """
    category = dimension_info.get("category", {})
    labels = category.get("label", {})
    # Invert the label mapping: {id: label} -> {label: id}
    return {label: code for code, label in labels.items()}


def parse_metadata(raw_metadata: dict[str, Any]) -> DatasetMetadata:
    """Parse raw API metadata into a structured format.

    Extracts all relevant fields from the CSO metadata response and
    returns them in a typed dictionary structure.

    Args:
        raw_metadata: The raw metadata dictionary from the CSO API.

    Returns:
        A DatasetMetadata TypedDict with all extracted fields.
    """
    extension = raw_metadata.get("extension", {})
    dimensions = raw_metadata.get("dimension", {})

    # Find the statistic dimension
    statistic_dim = _get_statistic_dimension(dimensions)

    # Extract units from statistic dimension
    units = _extract_units(statistic_dim)

    # Extract statistic labels
    statistics = list(statistic_dim.get("category", {}).get("label", {}).values())

    # Extract time variable
    time_variable = _extract_time_variable(raw_metadata, dimensions)

    # Extract and clean notes
    notes = _process_notes(raw_metadata.get("note", []))

    # Extract spatial info
    spatial_info = extract_spatial_info(raw_metadata)

    # Build tags list
    tags = _build_tags(extension, spatial_info.is_available)

    # Extract contact info
    contact = extension.get("contact", {})

    # Parse update timestamp
    updated_str = raw_metadata.get("updated")
    last_updated = pd.to_datetime(updated_str, format="mixed") if updated_str else None

    # Extract copyright
    copyright_block = extension.get("copyright", {})

    return DatasetMetadata(
        table_code=extension.get("matrix"),
        title=raw_metadata.get("label"),
        units=units,
        time_variable=time_variable,
        reasons=extension.get("reasons", []),
        official=extension.get("official", False),
        experimental=extension.get("experimental", False),
        reservation=extension.get("reservation", False),
        archive=extension.get("archive", False),
        analytical=extension.get("analytical", False),
        geographic=spatial_info.is_available,
        tags=tags,
        variables=[dim.get("label", name) for name, dim in dimensions.items()],
        statistics=statistics,
        last_updated=last_updated,
        notes=notes,
        copyright_name=copyright_block.get("name"),
        copyright_href=copyright_block.get("href"),
        contact_name=contact.get("name"),
        contact_email=contact.get("email"),
        contact_phone=contact.get("phone"),
        spatial_url=spatial_info.url,
        spatial_key=spatial_info.key,
    )


def parse_temporal_column(
    df: pd.DataFrame,
    time_variable: str | None,
) -> pd.DataFrame:
    """Parse temporal columns into appropriate pandas types.

    Detects the format of the time variable and converts it to the
    appropriate pandas datetime type (datetime, Period, or date).

    Supported formats:
        - Year only (e.g., "2022"): Converted to integer year.
        - Monthly (e.g., "2022M01"): Converted to pandas Period with 'M' frequency.
        - Quarterly: Converted to pandas Period with 'Q' frequency.
        - Weekly: Converted to date.
        - Full date: Converted to date or datetime as appropriate.

    Args:
        df: The DataFrame to process.
        time_variable: The name of the time column, or None.

    Returns:
        The DataFrame with the time column converted (if applicable).
        Returns the original DataFrame unchanged if no time variable
        is specified or if parsing is not possible.
    """
    if df is None or df.empty:
        return df

    if not time_variable or time_variable not in df.columns:
        return df

    sample_value = str(df[time_variable].iloc[0])
    time_label_lower = time_variable.lower()

    # Year only (e.g., "2022")
    if re.match(r"^\d{4}$", sample_value):
        df[time_variable] = pd.to_datetime(df[time_variable], format="%Y", errors="coerce").dt.year
        return df

    # Monthly data (e.g., "2022M01")
    if "month" in time_label_lower:
        if re.match(r"^\d{4}M\d{2}$", sample_value):
            df[time_variable] = pd.to_datetime(
                df[time_variable], format="%YM%m", errors="coerce"
            ).dt.to_period("M")
        else:
            df[time_variable] = pd.to_datetime(
                df[time_variable], format="mixed", errors="coerce"
            ).dt.to_period("M")
        return df

    # Quarterly data
    if "quarter" in time_label_lower:
        df[time_variable] = pd.to_datetime(
            df[time_variable], format="mixed", errors="coerce"
        ).dt.to_period("Q")
        return df

    # Weekly data
    if "week" in time_label_lower:
        df[time_variable] = pd.to_datetime(
            df[time_variable], format="mixed", errors="coerce"
        ).dt.date
        return df

    # Skip non-standard time formats
    skip_patterns = ("influenza season", "academic year", "halfyear")
    if any(pattern in time_label_lower for pattern in skip_patterns):
        return df

    # Default: try to parse as datetime
    try:
        parsed = pd.to_datetime(df[time_variable], format="mixed", errors="coerce")
        # If all times are midnight, convert to date only
        if pd.api.types.is_datetime64_any_dtype(parsed):
            if (parsed.dt.time == pd.Timestamp("00:00:00").time()).all():
                df[time_variable] = parsed.dt.date
            else:
                df[time_variable] = parsed
    except (ValueError, TypeError, AttributeError):
        # Parsing failed - leave the column as-is
        pass

    return df


# =============================================================================
# Private Helper Functions
# =============================================================================


def _get_statistic_dimension(dimensions: dict[str, Any]) -> dict[str, Any]:
    """Get the statistic dimension from the dimensions dict.

    The CSO API uses both "Statistic" and "STATISTIC" as dimension names.
    This function finds and returns the statistic dimension regardless
    of the capitalisation used.

    Args:
        dimensions: The dimensions dictionary from the API response.

    Returns:
        The statistic dimension dictionary, or an empty dict if not found.
    """
    for label in STATISTIC_LABELS:
        if label in dimensions:
            result = dimensions[label]
            return result if isinstance(result, dict) else {}
    return {}


def _extract_units(statistic_dim: dict[str, Any]) -> list[str]:
    """Extract unit labels from the statistic dimension.

    Args:
        statistic_dim: The statistic dimension dictionary.

    Returns:
        A list of unit labels (e.g., ["Persons", "Percentage"]).
    """
    unit_info = statistic_dim.get("category", {}).get("unit", {})
    return [info.get("label") for info in unit_info.values() if info.get("label")]


def _extract_time_variable(
    metadata: dict[str, Any],
    dimensions: dict[str, Any],
) -> str | None:
    """Extract the time variable label from metadata.

    Uses the 'role' field to identify which dimension represents time.

    Args:
        metadata: The full metadata dictionary.
        dimensions: The dimensions dictionary.

    Returns:
        The time variable label (e.g., "Census Year"), or None if not found.
    """
    time_dims = metadata.get("role", {}).get("time", [])
    if time_dims:
        time_key = time_dims[0]
        label = dimensions.get(time_key, {}).get("label")
        return str(label) if label is not None else None
    return None


def _process_notes(notes: list[str]) -> list[str]:
    """Clean and format notes from metadata.

    Removes formatting tags, normalises whitespace, and converts
    [url=...] tags to readable format.

    Args:
        notes: The raw notes list from the API.

    Returns:
        A list of cleaned note strings.
    """
    processed = []
    for note in notes:
        if not note:
            continue

        # Remove formatting tags and normalise whitespace
        cleaned = note.strip()
        cleaned = cleaned.replace("[i]", "").replace("[/i]", "")
        cleaned = cleaned.replace("\n", " ")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        cleaned = cleaned.replace("[b]", "").replace("[/b]", "")

        # Convert [url=...] tags to readable format
        def replace_url(match: re.Match[str]) -> str:
            url = match.group(1).strip()
            text = match.group(2).strip()
            return f" {text} ({url}) "

        cleaned = re.sub(r"\s*\[url=(.*?)\](.*?)\[/url\]\s*", replace_url, cleaned)
        processed.append(cleaned)

    return processed


def _build_tags(extension: dict[str, Any], has_spatial: bool) -> list[str]:
    """Build a list of descriptive tags from extension data.

    Creates human-readable tags based on dataset classification flags
    (experimental, reservation, archive, etc.).

    Args:
        extension: The extension dictionary from the API response.
        has_spatial: Whether the dataset has spatial data available.

    Returns:
        A list of tag strings (e.g., ["Official Statistics", "Geographic Data"]).
    """
    tags = []

    tag_mapping = [
        ("experimental", "Experimental Statistics"),
        ("reservation", "Reservation Statistics"),
        ("archive", "Archive Statistics"),
        ("analytical", "Analytical Statistics"),
        ("official", "Official Statistics"),
    ]

    for key, label in tag_mapping:
        if extension.get(key, False):
            tags.append(label)

    if has_spatial:
        tags.append("Geographic Data")

    return tags
