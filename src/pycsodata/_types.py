"""Type definitions and enums for pycsodata.

This module contains all shared type definitions, enums, and TypedDict
structures used throughout the package.

Classes:
    IncludeIDs: Enum for controlling which ID columns to include in output.
    PivotFormat: Enum for specifying DataFrame output format.
    DatasetMetadata: TypedDict for structured dataset metadata.
    SpatialInfo: Dataclass for spatial data configuration.

Type Aliases:
    IncludeIDsSpec: Union type for include_ids parameter.
    FilterSpec: Type for filter dictionaries.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import TypedDict


class IncludeIDs(str, Enum):
    """Options for including ID columns in dataset outputs.

    ID columns contain the CSO's internal codes for each category value.
    For example, "Dublin" might have ID "IE061".

    Attributes:
        ALL: Include all ID columns for every dimension.
        SPATIAL_ONLY: Include only the ID column for the spatial dimension.
        NONE: Exclude all ID columns.

    Examples:
        >>> from pycsodata import CSODataset, IncludeIDs
        >>> dataset = CSODataset("FY003A", include_ids=IncludeIDs.SPATIAL_ONLY)
    """

    ALL = "all"
    SPATIAL_ONLY = "spatial_only"
    NONE = "none"


# Type alias for include_ids parameter - can be string, enum, or list of column names
IncludeIDsSpec = str | IncludeIDs | list[str] | None


class PivotFormat(str, Enum):
    """Output format options for DataFrame/GeoDataFrame pivoting.

    Attributes:
        LONG: Default long format with one row per observation.
            Each row represents a single data point.
        WIDE: Time periods as columns. Useful for time series analysis
            where each column represents a different time period.
        TIDY: Statistics as columns. Useful when you want each statistic
            as a separate column for easier comparison.

    Examples:
        >>> from pycsodata import CSODataset, PivotFormat
        >>> dataset = CSODataset("FY003A")
        >>> df_long = dataset.df(PivotFormat.LONG)
        >>> df_wide = dataset.df("wide")  # String values also accepted
    """

    LONG = "long"
    WIDE = "wide"
    TIDY = "tidy"


class DatasetMetadata(TypedDict, total=False):
    """Structured metadata for a CSO dataset.

    All fields are optional since not all datasets have complete metadata.
    """

    table_code: str
    title: str | None
    units: list[str]
    time_variable: str | None
    reasons: list[str]
    official: bool
    experimental: bool
    reservation: bool
    archive: bool
    analytical: bool
    geographic: bool
    tags: list[str]
    variables: list[str]
    statistics: list[str]
    last_updated: datetime | None
    notes: list[str]
    copyright_name: str | None
    copyright_href: str | None
    contact_name: str | None
    contact_email: str | None
    contact_phone: str | None
    spatial_url: str | None
    spatial_key: str | None


@dataclass(frozen=True, slots=True)
class SpatialInfo:
    """Container for spatial data configuration.

    This dataclass holds the URL and key needed to join CSO statistical
    data with geographic boundary data.

    Attributes:
        url: URL to the GeoJSON source for spatial boundaries.
        key: The dimension label that corresponds to spatial data
            (e.g., "County", "Electoral Division").

    Examples:
        >>> dataset = CSODataset("FY003A")
        >>> if dataset.spatial_info.is_available:
        ...     print(f"Spatial key: {dataset.spatial_info.key}")
    """

    url: str | None = None
    key: str | None = None

    @property
    def is_available(self) -> bool:
        """Check if spatial data is available for this dataset."""
        return self.url is not None and self.key is not None


# Type aliases for common patterns
FilterValue = str | int | float
"""Type for individual filter values in filter specifications."""

FilterSpec = dict[str, list[FilterValue] | FilterValue | None]
"""Type for filter dictionaries passed to CSODataset.

Keys are dimension names, values are lists of values to include.
Single values are also accepted and will be wrapped in a list.

Examples::

    filters: FilterSpec = {
        "County": ["Dublin", "Cork"],
        "Census Year": "2022",  # Single value
    }
"""
