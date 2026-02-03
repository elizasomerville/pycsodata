"""pycsodata - Load CSO datasets with optional spatial joins.

This package provides a convenient interface for accessing datasets from
Ireland's Central Statistics Office (CSO), with optional spatial data
integration for geographic analysis.

Main Classes:
    CSODataset: Load and work with individual CSO datasets.
    CSOCatalogue: Browse and search available CSO datasets.
    CSOCache: Manage the HTTP response cache.

Examples:
    >>> from pycsodata import CSODataset, CSOCatalogue, CSOCache
    >>> # Search for datasets
    >>> catalogue = CSOCatalogue()
    >>> results = catalogue.search(title="population")
    >>> # Load a specific dataset
    >>> dataset = CSODataset(
    ...     "FY003A",
    ...     filters={"Statistic": ["Population"], "Census Year": ["2022"]},
    ...     include_ids="spatial_only",
    ...     sanitise=True
    ... )
    >>> dataset.describe()
    >>> df = dataset.df()
    >>> gdf = dataset.gdf("wide")
    >>> # Manage the cache
    >>> cache = CSOCache()
    >>> cache.info()  # Check cache statistics
    >>> cache.flush()  # Clear the cache
"""

from __future__ import annotations

import importlib.metadata

__version__ = importlib.metadata.version("pycsodata")

# Enums for type-safe parameters
from pycsodata._types import IncludeIDs, IncludeIDsSpec, PivotFormat
from pycsodata.cache import CacheInfo, CSOCache
from pycsodata.catalogue import CSOCatalogue
from pycsodata.dataset import CSODataset

# Exceptions
from pycsodata.exceptions import APIError, DataError, SpatialError, ValidationError

__all__ = [
    "APIError",
    "CSOCache",
    "CSOCatalogue",
    "CSODataset",
    "CacheInfo",
    "DataError",
    "IncludeIDs",
    "IncludeIDsSpec",
    "PivotFormat",
    "SpatialError",
    "ValidationError",
]
