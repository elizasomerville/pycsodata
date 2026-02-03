"""Dataset classes for CSO data with optional spatial integration.

This module provides the main CSODataset class for loading and working
with datasets from Ireland's Central Statistics Office.

Examples:
    >>> from pycsodata import CSODataset
    >>> dataset = CSODataset("FY003A", sanitise=True, filters={"Census Year": ["2022"]})
    >>> df = dataset.df()
    >>> gdf = dataset.gdf()  # GeoDataFrame with spatial boundaries
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
from pyjstat import pyjstat

from pycsodata._types import (
    DatasetMetadata,
    FilterSpec,
    IncludeIDs,
    IncludeIDsSpec,
    PivotFormat,
    SpatialInfo,
)
from pycsodata.constants import (
    ID_COLUMN_SUFFIX,
    NATIONAL_AREA_CODE,
    NATIONAL_AREA_LABELS,
)
from pycsodata.exceptions import SpatialError, ValidationError
from pycsodata.fetchers import load_dataset, load_metadata
from pycsodata.parsers import (
    extract_id_mapping,
    extract_spatial_info,
    parse_metadata,
    parse_temporal_column,
)
from pycsodata.printer import MetadataPrinter
from pycsodata.sanitise import (
    create_reverse_mapping,
    sanitise_list,
    sanitise_string,
)
from pycsodata.spatial import create_geodataframe


class CSODataset:
    """A dataset from Ireland's Central Statistics Office.

    This class provides a convenient interface for loading CSO datasets,
    with optional spatial data integration. Data is lazily loaded on first
    access, and results are cached for subsequent access.

    Args:
        table_code: The CSO table code (e.g., 'FY003A').
        filters (dict): Filters to apply to the dataset dimensions.
        include_ids (str): Which ID columns to include in output. Can be:
            - "all": Include all ID columns for every dimension.
            - "spatial_only": Include only the ID column for the spatial dimension.
            - "none" (default): Exclude all ID columns.
            - A list of column names: Include only ID columns for the specified
              columns (e.g., ["County", "Sex"] to include "County ID" and "Sex ID").
              If the list contains column names that do not correspond to
              dimensions in the dataset, a ValidationError is raised.
        drop_filtered_cols: Whether to drop columns for filtered dimensions.
        drop_national_data: Whether to exclude national-level (Ireland) rows.
        convert_dates: Whether to parse temporal columns as datetime.
        sanitise: Whether to sanitise column names for consistency. When True,
            applies standardised transformations: replacing '&' with 'and',
            normalising slashes and spaces, and applying standard name mappings.
        cache: Whether to cache API responses. Defaults to True.

    Methods:
        df: Get the dataset as a pandas DataFrame.
        gdf: Get the dataset as a GeoDataFrame with spatial data.
        describe: Print a summary of the dataset metadata.

    Raises:
        APIError: If the dataset cannot be loaded from the CSO API.
        ValidationError: If invalid parameters are provided.

    Examples:
        >>> dataset = CSODataset(
        ...     "FY003A",
        ...     filters={"CensusYear": ["2022"], "Sex": ["Both sexes"]},
        ...     include_ids="spatial_only",
        ... )
        >>> df = dataset.df()
        >>> gdf = dataset.gdf()
        >>> # Include specific ID columns
        >>> dataset = CSODataset("FY003A", include_ids=["County", "Sex"])
    """

    def __init__(
        self,
        table_code: str,
        *,
        filters: FilterSpec | None = None,
        include_ids: IncludeIDsSpec = None,
        drop_filtered_cols: bool = False,
        drop_national_data: bool = False,
        convert_dates: bool = False,
        sanitise: bool = False,
        cache: bool = True,
    ) -> None:
        # Normalise include_ids (will be IncludeIDs enum or list of column names)
        self._include_ids = self._normalise_include_ids(include_ids)

        self.table_code = table_code.upper()
        self._filters = filters
        self._drop_filtered_cols = drop_filtered_cols
        self._drop_national_data = drop_national_data
        self._convert_dates = convert_dates
        self._sanitise = sanitise
        self._cache_enabled = cache

        # Load metadata (eagerly, to fail fast on invalid table codes)
        self._raw_metadata = load_metadata(table_code, cache=cache)

        # Extract spatial info (sanitise key if enabled)
        self._spatial_info = extract_spatial_info(self._raw_metadata)
        if self._sanitise and self._spatial_info.key:
            sanitised_key = sanitise_string(self._spatial_info.key)
            self._spatial_info = SpatialInfo(
                url=self._spatial_info.url,
                key=sanitised_key,
            )

        # Build column name mappings for filter translation
        # Maps sanitised names back to original names for API compatibility
        self._original_column_names: list[str] = []
        self._sanitise_to_original_map: dict[str, str] = {}

        # Lazy-loaded data
        self._cached_base_df: pd.DataFrame | None = None
        self._cached_df: pd.DataFrame | None = None
        self._cached_gdf: gpd.GeoDataFrame | None = None

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def metadata(self) -> DatasetMetadata:
        """Get structured metadata for this dataset.

        Returns a TypedDict containing information about the dataset including
        its title, variables, units, time variable, and other attributes.

        If sanitise=True was specified during initialisation, variable names
        and other fields are sanitised for consistency.

        Returns:
            A DatasetMetadata TypedDict with all available metadata fields.
        """
        meta = parse_metadata(self._raw_metadata)
        if self._sanitise:
            meta = self._sanitise_metadata(meta)
        return meta

    def _sanitise_metadata(self, meta: DatasetMetadata) -> DatasetMetadata:
        """Apply sanitisation to metadata fields."""
        sanitised = meta
        if "variables" in sanitised:
            sanitised["variables"] = sanitise_list(sanitised["variables"])
        if "statistics" in sanitised:
            sanitised["statistics"] = sanitise_list(sanitised["statistics"])
        time_variable = sanitised.get("time_variable")
        if time_variable:
            sanitised["time_variable"] = sanitise_string(time_variable)
        spatial_key = sanitised.get("spatial_key")
        if spatial_key:
            sanitised["spatial_key"] = sanitise_string(spatial_key)
        return DatasetMetadata(**sanitised)

    @property
    def spatial_info(self) -> SpatialInfo:
        """Get spatial data configuration.

        Returns:
            A SpatialInfo object containing the URL and key for
            spatial boundary data, or empty values if not available.
        """
        return self._spatial_info

    @property
    def has_spatial_data(self) -> bool:
        """Check if this dataset has spatial data available.

        Returns:
            True if the dataset has linked geographic boundary data,
            False otherwise.
        """
        return self._spatial_info.is_available

    # =========================================================================
    # Public Methods
    # =========================================================================

    def df(self, pivot_format: str | PivotFormat = "long", *, copy: bool = True) -> pd.DataFrame:
        """Get the dataset as a DataFrame.

        Args:
            pivot_format (str): The output format for the data.
                Options: "long" (default), "wide", "tidy".
            copy: Whether to return a copy of the cached DataFrame.
                Defaults to True to prevent accidental mutation of cached data.
                Set to False for better performance if you won't modify the result.

        Returns:
            The dataset as a pandas DataFrame.

        Raises:
            ValidationError: If an invalid pivot format is provided.

        Examples:
            >>> df = dataset.df("wide")
        """
        fmt = self._normalise_pivot_format(pivot_format)

        if self._cached_df is None:
            self._cached_df = self._build_df()

        if fmt == PivotFormat.LONG:
            return self._cached_df.copy() if copy else self._cached_df

        if fmt == PivotFormat.WIDE:
            result = self._pivot_wide(self._cached_df)
            return result.copy() if copy else result

        if fmt == PivotFormat.TIDY:
            result = self._pivot_tidy(self._cached_df)
            return result.copy() if copy else result

        return self._cached_df.copy() if copy else self._cached_df

    def gdf(
        self, pivot_format: str | PivotFormat = "long", *, copy: bool = True
    ) -> gpd.GeoDataFrame:
        """Get the dataset as a GeoDataFrame with spatial data.

        The returned GeoDataFrame contains all rows from the dataset,
        including aggregate regions (e.g., "State", "Leinster") that may
        not have corresponding geometries in the spatial data. These rows
        will have null (None) geometries.

        Args:
            pivot_format (str): The output format for the data.
                Options: "long" (default), "wide", "tidy".
            copy: Whether to return a copy of the cached GeoDataFrame.
                Defaults to True to prevent accidental mutation of cached data.
                Set to False for better performance if you won't modify the result.

        Returns:
            The dataset as a GeoDataFrame with geometry column. Rows for
                aggregate regions without spatial boundaries will have null
                geometries.

        Raises:
            SpatialError: If spatial data is not available or merge fails.
            ValidationError: If an invalid pivot format is provided.

        Examples:
            >>> gdf = dataset.gdf()
            >>> gdf.plot(column="value")
            >>> # Check for null geometries
            >>> gdf[gdf.geometry.isna()]
        """
        fmt = self._normalise_pivot_format(pivot_format)

        if not self._spatial_info.is_available:
            raise SpatialError(
                f"Spatial data is not available for dataset '{self.table_code}'. "
                "This dataset does not have linked geographic boundaries.",
                table_code=self.table_code,
            )

        if self._cached_gdf is None:
            self._cached_gdf = self._build_gdf()

        if fmt == PivotFormat.LONG:
            return self._cached_gdf.copy() if copy else self._cached_gdf

        if fmt == PivotFormat.WIDE:
            result = self._gdf_pivot_wide(self._cached_gdf)
            return result.copy() if copy else result

        if fmt == PivotFormat.TIDY:
            result = self._gdf_pivot_tidy(self._cached_gdf)
            return result.copy() if copy else result

        return self._cached_gdf.copy() if copy else self._cached_gdf

    def describe(self) -> None:
        """Print a summary of the dataset metadata.

        Displays information about the dataset including its code, title,
        variables, units, tags, and other relevant metadata.

        Examples:
            >>> dataset = CSODataset("FY003A")
            >>> dataset.describe()
        """
        meta = self.metadata
        printer = MetadataPrinter(meta, self._filters, self._drop_filtered_cols)
        printer.print_all()

    def __repr__(self) -> str:
        """Return a string representation of the dataset."""
        spatial = "yes" if self._spatial_info.is_available else "no"

        return f"<CSODataset(table_code='{self.table_code}', spatial={spatial})>"

    # =========================================================================
    # Private: Data Loading
    # =========================================================================

    # -------------------------------------------------------------------------
    # Parameter normalisation
    # -------------------------------------------------------------------------

    @staticmethod
    def _normalise_include_ids(value: IncludeIDsSpec) -> IncludeIDs | list[str]:
        """Normalise include_ids to IncludeIDs enum or list of column names.

        Args:
            value: String value like "all", "spatial_only", "none",
                   or a list of column names.

        Returns:
            The corresponding IncludeIDs enum member, or a list of column names.

        Raises:
            ValidationError: If the value is not valid.
        """
        # Already an enum
        if isinstance(value, IncludeIDs):
            return value

        # List of column names
        if isinstance(value, list):
            if not all(isinstance(item, str) for item in value):
                raise ValidationError(
                    "include_ids list must contain only strings (column names).",
                    parameter="include_ids",
                    value=value,
                )
            return value

        # String - try to parse as enum
        normalised = str(value).lower().strip()
        try:
            return IncludeIDs(normalised)
        except ValueError:
            valid = ", ".join(f'"{m.value}"' for m in IncludeIDs)
            raise ValidationError(
                f"Invalid include_ids value: {value!r}. "
                f"Valid options are: {valid}, or a list of column names.",
                parameter="include_ids",
                value=value,
            ) from None

    @staticmethod
    def _normalise_pivot_format(value: str) -> PivotFormat:
        """Normalise pivot_format string to PivotFormat enum.

        Args:
            value: String value like "long", "wide", "tidy".

        Returns:
            The corresponding PivotFormat enum member.

        Raises:
            ValidationError: If the value is not valid.
        """
        if isinstance(value, PivotFormat):
            return value

        normalised = str(value).lower().strip()
        try:
            return PivotFormat(normalised)
        except ValueError:
            valid = ", ".join(f'"{m.value}"' for m in PivotFormat)
            raise ValidationError(
                f"Invalid pivot_format value: {value!r}. Valid options are: {valid}.",
                parameter="pivot_format",
                value=value,
            ) from None

    # -------------------------------------------------------------------------
    # Data Loading
    # -------------------------------------------------------------------------

    def _get_base_df(self) -> pd.DataFrame:
        """Load and cache the base DataFrame (before ID column filtering).

        This method handles the initial data loading, including optional
        removal of national-level data and date parsing.

        Returns:
            The base DataFrame with all columns before ID filtering.
        """
        if self._cached_base_df is None:
            df = self._load_raw_data()

            if self._drop_national_data:
                df = self._remove_national_rows(df)

            if self._convert_dates:
                time_var = self.metadata.get("time_variable")
                df = parse_temporal_column(df, time_var)

            self._cached_base_df = df

        return self._cached_base_df

    def _load_raw_data(self) -> pd.DataFrame:
        """Load the raw dataset from the API.

        Fetches the dataset from the CSO API, normalises column names,
        adds ID columns, and applies any configured filters.

        Returns:
            The processed DataFrame ready for further transformation.

        Raises:
            TypeError: If the API response cannot be parsed into a DataFrame.
        """
        dataset_json = load_dataset(self.table_code, cache=self._cache_enabled)

        # Parse JSON-stat to DataFrame
        df = pyjstat.from_json_stat(dataset_json)[0]

        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected DataFrame, got {type(df).__name__}")

        # normalise column names FIRST (STATISTIC -> Statistic, etc.)
        # This must happen before adding ID columns and applying filters
        df = self._normalise_dataframe(df)

        # Store original column names for mapping before any sanitisation
        self._original_column_names = list(df.columns)

        # Sanitise column names and values if enabled
        if self._sanitise:
            df = self._sanitise_dataframe(df)
            # Build reverse mapping for filter translation
            self._sanitise_to_original_map = create_reverse_mapping(self._original_column_names)

        # Add ID columns
        df = self._add_id_columns(df)

        # Apply filters (after normalisation so "Statistic" works consistently)
        if self._filters:
            df = self._apply_filters(df)

        return df

    def _build_df(self) -> pd.DataFrame:
        """Build the final DataFrame with ID column handling.

        Applies filtering of ID columns based on the include_ids setting
        and optionally drops columns for filtered dimensions.

        Returns:
            The final DataFrame ready for output.
        """
        base_df = self._get_base_df().copy()

        # Drop filtered columns if requested
        if self._drop_filtered_cols and self._filters:
            base_df = self._drop_filter_columns(base_df)

        return self._filter_id_columns(base_df)

    def _build_gdf(self) -> gpd.GeoDataFrame:
        """Build the GeoDataFrame with spatial data.

        Merges the statistical data with geographic boundary data
        to create a GeoDataFrame suitable for spatial analysis.

        Returns:
            A GeoDataFrame with geometry column.

        Raises:
            SpatialError: If spatial merge fails or produces invalid geometry.
        """
        base_df = self._get_base_df().copy()

        # Create GeoDataFrame BEFORE dropping columns, so spatial merge works
        gdf = create_geodataframe(
            base_df,
            self._spatial_info.url,
            self._spatial_info.key,
            cache=self._cache_enabled,
        )

        if gdf is None or not isinstance(gdf, gpd.GeoDataFrame):
            raise SpatialError(
                f"Failed to create GeoDataFrame for dataset '{self.table_code}'.",
                table_code=self.table_code,
            )

        if "geometry" not in gdf.columns:
            raise SpatialError(
                f"Spatial merge produced no geometry column for dataset '{self.table_code}'.",
                table_code=self.table_code,
            )

        # Check for empty geometries
        if len(gdf) > 0:
            is_missing = gdf.geometry.isna() | gdf.geometry.is_empty
            if is_missing.all():
                raise SpatialError(
                    f"All geometries are missing or empty for dataset '{self.table_code}'.",
                    table_code=self.table_code,
                )

        # Drop filtered columns if requested
        if self._drop_filtered_cols and self._filters:
            gdf = self._drop_filter_columns(gdf)

        gdf = self._filter_id_columns(gdf)

        if type(gdf) is not gpd.GeoDataFrame:
            gdf = gpd.GeoDataFrame(gdf, geometry="geometry")

        return gdf.reset_index(drop=True)

    # =========================================================================
    # Private: Data Transformation
    # =========================================================================

    def _add_id_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ID columns for each dimension based on metadata.

        For each dimension in the dataset, adds a corresponding ID column
        containing the CSO's internal codes. For example, if a "County"
        column contains "Dublin", the "County ID" column will contain "IE061".

        Args:
            df: The DataFrame to add ID columns to.

        Returns:
            The DataFrame with ID columns added after each dimension column.
        """
        dimensions = self._raw_metadata.get("dimension", {})

        # Build mapping from label to ID mapper
        # The label needs to match the DataFrame column names after normalisation
        label_to_id_map: dict[str, dict[str, str]] = {}
        for dim_info in dimensions.values():
            label = dim_info.get("label")
            if label:
                # Normalise STATISTIC -> Statistic to match DataFrame column names
                normalised_label = "Statistic" if label == "STATISTIC" else label

                if self._sanitise:
                    # If sanitise is enabled, further sanitise the label
                    sanitised_label = sanitise_string(normalised_label)
                    # Also sanitise the ID mapping keys (category labels)
                    original_mapping = extract_id_mapping(dim_info)
                    sanitised_mapping = {sanitise_string(k): v for k, v in original_mapping.items()}
                    label_to_id_map[sanitised_label] = sanitised_mapping
                else:
                    label_to_id_map[normalised_label] = extract_id_mapping(dim_info)

        # Add ID columns in order
        new_columns = []
        for col in df.columns:
            new_columns.append(col)
            if col in label_to_id_map:
                id_col = f"{col}{ID_COLUMN_SUFFIX}"
                if id_col not in df.columns:
                    df[id_col] = df[col].map(label_to_id_map[col])
                    new_columns.append(id_col)

        return df[new_columns]

    def _filter_id_columns(
        self, df: pd.DataFrame | gpd.GeoDataFrame
    ) -> pd.DataFrame | gpd.GeoDataFrame:
        """Filter ID columns based on include_ids setting.

        Handles three cases:
        1. IncludeIDs.ALL: Keep all ID columns.
        2. IncludeIDs.SPATIAL_ONLY: Keep only the spatial dimension ID column.
        3. IncludeIDs.NONE: Remove all ID columns.
        4. List of column names: Keep only ID columns for the specified columns.

        Raises:
            ValidationError: If include_ids is a list containing column names
                that do not correspond to dimensions in the dataset.
        """
        # Case 1: Keep all ID columns
        if self._include_ids == IncludeIDs.ALL:
            return df

        id_columns = [col for col in df.columns if col.endswith(ID_COLUMN_SUFFIX)]

        # Case 2: Keep only spatial ID column
        if self._include_ids == IncludeIDs.SPATIAL_ONLY and self._spatial_info.key:
            spatial_id_col = f"{self._spatial_info.key}{ID_COLUMN_SUFFIX}"
            cols_to_drop = [col for col in id_columns if col != spatial_id_col]

        # Case 3: Drop all ID columns
        elif self._include_ids == IncludeIDs.NONE:
            cols_to_drop = id_columns

        # Case 4: Keep ID columns for specific column names
        elif isinstance(self._include_ids, list):
            # Get base column names (without ID suffix) that have ID columns
            valid_dimensions = {col[: -len(ID_COLUMN_SUFFIX)] for col in id_columns}
            # Validate that all requested columns exist as dimensions
            invalid_cols = [col for col in self._include_ids if col not in valid_dimensions]
            if invalid_cols:
                raise ValidationError(
                    f"include_ids contains column names that are not dimensions "
                    f"in dataset '{self.table_code}': {invalid_cols}. "
                    f"Valid dimensions are: {sorted(valid_dimensions)}.",
                    parameter="include_ids",
                    value=self._include_ids,
                )
            # Build set of ID columns to keep
            cols_to_keep = {f"{col}{ID_COLUMN_SUFFIX}" for col in self._include_ids}
            cols_to_drop = [col for col in id_columns if col not in cols_to_keep]

        else:
            # Fallback - drop all ID columns
            cols_to_drop = id_columns

        return df.drop(columns=cols_to_drop, errors="ignore").reset_index(drop=True)

    def _normalise_filter_keys(self, filters: FilterSpec) -> FilterSpec:
        """Normalise filter keys, particularly STATISTIC -> Statistic.

        This ensures users can use either "STATISTIC" or "Statistic" in filters,
        and it will be normalised to "Statistic" to match the normalised column names.

        If sanitise=True, filter keys and values are also sanitised to match
        the sanitised column names and values in the DataFrame.

        Args:
            filters: The original filter specification.

        Returns:
            A new filter specification with normalised keys.
        """
        if not filters:
            return filters

        normalised: FilterSpec = {}
        for key, value in filters.items():
            # normalise STATISTIC to Statistic
            if key.upper() == "STATISTIC":
                normalised_key = "Statistic"
            elif key.upper() == "STATISTIC ID":
                normalised_key = "Statistic ID"
            else:
                normalised_key = key

            # If sanitise is enabled, also sanitise the key
            if self._sanitise:
                normalised_key = sanitise_string(normalised_key)

            # Sanitise values if enabled
            if self._sanitise and value is not None:
                if isinstance(value, list | tuple | set):
                    value = [sanitise_string(str(v)) if isinstance(v, str) else v for v in value]
                elif isinstance(value, str):
                    value = sanitise_string(value)

            normalised[normalised_key] = value

        return normalised

    def _apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply client-side filtering to the DataFrame.

        Filters are applied directly to the column specified in the filter key.
        To filter on an ID column, use the full column name including " ID" suffix.
        Both "STATISTIC" and "Statistic" are accepted and normalised to "Statistic".

        Args:
            df: The DataFrame to filter.

        Returns:
            The filtered DataFrame.

        Raises:
            ValidationError: If a filter dimension is not found in the dataset,
                or if no values match the filter.

        Examples:
            filters={"County": ["Dublin"]}  # Filter by label
            filters={"County ID": ["IE0123"]}  # Filter by ID
            filters={"Statistic": ["Population"]}  # Filter by statistic
        """
        if not self._filters:
            return df

        # normalise filter keys (STATISTIC -> Statistic)
        normalised_filters = self._normalise_filter_keys(self._filters)

        for dim, values in normalised_filters.items():
            if values is None:
                continue

            # normalise values to list
            value_list = list(values) if isinstance(values, (list | tuple | set)) else [values]
            value_strs = {str(v).strip() for v in value_list}

            # Filter on the exact column specified
            if dim in df.columns:
                mask = df[dim].isin(value_list) | df[dim].astype(str).isin(value_strs)
            else:
                raise ValidationError(
                    f"Filter dimension {dim!r} not found in dataset '{self.table_code}'.",
                    parameter="filters",
                    value=dim,
                )

            if not mask.any():
                raise ValidationError(
                    f"No matching values for filter {dim}={value_list} "
                    f"in dataset '{self.table_code}'."
                )

            df = df[mask]

        return df.reset_index(drop=True)

    def _drop_filter_columns(
        self,
        df: pd.DataFrame,
        preserve_spatial: bool = False,
    ) -> pd.DataFrame:
        """Remove columns corresponding to filtered dimensions.

        When drop_filtered_cols=True is specified, this method removes
        columns that have been filtered to a single value, reducing
        redundancy in the output.

        Args:
            df: The DataFrame to modify.
            preserve_spatial: If True, do not drop the spatial key column
                or its ID. This is used for GeoDataFrame creation where
                the spatial column is needed for the merge.

        Returns:
            The DataFrame with filtered dimension columns removed.
        """
        if not self._filters:
            return df

        # Use normalised filters for consistency
        normalised_filters = self._normalise_filter_keys(self._filters)

        spatial_key = self._spatial_info.key if preserve_spatial else None
        spatial_id = f"{spatial_key}{ID_COLUMN_SUFFIX}" if spatial_key else None

        cols_to_drop = []
        for dim in normalised_filters:
            # Skip spatial columns if preserving
            if preserve_spatial and dim in (spatial_key, spatial_id):
                continue

            # Drop the exact column specified in the filter
            if dim in df.columns:
                cols_to_drop.append(dim)

            # Also drop the corresponding ID/label column
            if dim.endswith(ID_COLUMN_SUFFIX):
                label_col = dim[: -len(ID_COLUMN_SUFFIX)]
                if label_col in df.columns and label_col not in (spatial_key, spatial_id):
                    cols_to_drop.append(label_col)
            else:
                id_col = f"{dim}{ID_COLUMN_SUFFIX}"
                if id_col in df.columns and id_col not in (spatial_key, spatial_id):
                    cols_to_drop.append(id_col)

        return df.drop(columns=cols_to_drop, errors="ignore")

    def _remove_national_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows corresponding to national-level data.

        Filters out rows that represent national-level aggregates (e.g.,
        "Ireland" or "State") based on either the ID column (checking
        for "IE0") or the label column (checking for known national labels).

        This is applied when drop_national_data=True is specified.

        Args:
            df: The DataFrame to filter.

        Returns:
            The DataFrame with national-level rows removed.
        """
        spatial_key = self._spatial_info.key
        if not spatial_key:
            return df

        id_col = f"{spatial_key}{ID_COLUMN_SUFFIX}"

        # Build a mask for rows to keep (non-national rows)
        # We need to check BOTH ID and label columns because some datasets
        # use "IE0" as the national ID, while others use different IDs (e.g., "-")
        # but still have national labels like "State" or "Ireland"
        mask = pd.Series(True, index=df.index)

        # Filter by ID column if present
        if id_col in df.columns:
            mask &= df[id_col] != NATIONAL_AREA_CODE

        # Also filter by label column if present
        if spatial_key in df.columns:
            mask &= ~df[spatial_key].isin(NATIONAL_AREA_LABELS)

        return df[mask].reset_index(drop=True)

    def _normalise_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalise column names and types.

        Converts the 'value' column to numeric type and standardises
        column names (e.g., 'STATISTIC' -> 'Statistic').

        Args:
            df: The DataFrame to normalise.

        Returns:
            The normalised DataFrame.
        """
        # normalise 'value' column to numeric
        if "value" in df.columns:
            df["value"] = pd.to_numeric(df["value"], errors="coerce")

        # normalise 'STATISTIC' to 'Statistic'
        rename_map = {}
        if "STATISTIC" in df.columns:
            rename_map["STATISTIC"] = "Statistic"
        if "STATISTIC ID" in df.columns:
            rename_map["STATISTIC ID"] = "Statistic ID"

        if rename_map:
            df = df.rename(columns=rename_map)

        return df

    def _sanitise_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sanitise column names and categorical string values in the DataFrame.

        This method:
        1. Renames columns using sanitise_string
        2. Sanitises string values in categorical/dimension columns (not 'value')

        Args:
            df: The DataFrame to sanitise.

        Returns:
            The sanitised DataFrame.
        """
        # Step 1: Sanitise column names
        new_columns = {col: sanitise_string(col) for col in df.columns}
        df = df.rename(columns=new_columns)

        # Step 2: Sanitise string values in dimension columns (excluding 'value')
        for col in df.columns:
            if col == "value":
                continue
            if df[col].dtype == "object":
                df[col] = df[col].apply(lambda x: sanitise_string(x) if isinstance(x, str) else x)

        return df

    # -------------------------------------------------------------------------
    # Pivoting Methods
    # -------------------------------------------------------------------------

    def _pivot_wide(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pivot to wide format with time periods as columns.

        Transforms the DataFrame so that each unique time period becomes
        a separate column, making it easier to compare values across time.

        Args:
            df: The DataFrame in long format.

        Returns:
            The pivoted DataFrame in wide format.

        Raises:
            ValidationError: If the time variable is not present or if
                duplicate entries prevent pivoting.
        """
        time_var = self.metadata.get("time_variable")

        if not time_var or time_var not in df.columns:
            raise ValidationError(
                "Cannot pivot to wide format: time variable is not defined "
                "or not present in the data.",
                parameter="pivot_format",
                value="wide",
            )

        # Exclude time variable, its ID column, and value from index columns
        time_var_id = f"{time_var}{ID_COLUMN_SUFFIX}"
        index_cols = [col for col in df.columns if col not in (time_var, time_var_id, "value")]

        # Check for duplicates
        if df.duplicated(subset=[*index_cols, time_var]).any():
            raise ValidationError(
                "Cannot pivot to wide format: duplicate entries exist for "
                "some combinations of index columns and time variable.",
                parameter="pivot_format",
                value="wide",
            )

        # Drop time variable ID before pivoting if present
        df_to_pivot = df.drop(columns=[time_var_id], errors="ignore")

        # Preserve original row order by creating a sort key
        df_to_pivot = df_to_pivot.copy()
        df_to_pivot["_original_order"] = range(len(df_to_pivot))

        # Get the first occurrence order for each combination of index columns
        order_df = (
            df_to_pivot.groupby(index_cols, sort=False)["_original_order"].first().reset_index()
        )
        order_df = order_df.rename(columns={"_original_order": "_sort_key"})

        df_to_pivot = df_to_pivot.drop(columns=["_original_order"])

        pivoted = df_to_pivot.pivot_table(
            index=index_cols,
            columns=time_var,
            values="value",
            aggfunc="first",  # type: ignore
            sort=False,
        ).reset_index()

        # Merge with order and sort to restore original order
        pivoted = pivoted.merge(order_df, on=index_cols, how="left")
        pivoted = (
            pivoted.sort_values("_sort_key").drop(columns=["_sort_key"]).reset_index(drop=True)
        )

        pivoted.columns.name = None
        return pivoted

    def _pivot_tidy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pivot to tidy format with statistics as columns.

        Transforms the DataFrame so that each unique statistic becomes
        a separate column, making it easier to compare different measures.

        Args:
            df: The DataFrame in long format.

        Returns:
            The pivoted DataFrame in tidy format.

        Raises:
            ValidationError: If the Statistic column is not present or if
                duplicate entries prevent pivoting.
        """
        if "Statistic" not in df.columns:
            raise ValidationError(
                "Cannot pivot to tidy format: 'Statistic' column is not present in the data.",
                parameter="pivot_format",
                value="tidy",
            )

        # Exclude Statistic, Statistic ID, and value from index columns
        index_cols = [
            col for col in df.columns if col not in ("Statistic", "Statistic ID", "value")
        ]

        # For duplicate checking, use the cleaned index columns
        if df.duplicated(subset=[*index_cols, "Statistic"]).any():
            raise ValidationError(
                "Cannot pivot to tidy format: duplicate entries exist for "
                "some combinations of index columns and Statistic.",
                parameter="pivot_format",
                value="tidy",
            )

        # Preserve original statistic order
        stat_order = list(dict.fromkeys(df["Statistic"].tolist()))

        # Drop Statistic ID before pivoting if present
        df_to_pivot = df.drop(columns=["Statistic ID"], errors="ignore")

        # Preserve original row order by creating a sort key
        df_to_pivot = df_to_pivot.copy()
        df_to_pivot["_original_order"] = range(len(df_to_pivot))

        # Get the first occurrence order for each combination of index columns
        order_df = (
            df_to_pivot.groupby(index_cols, sort=False)["_original_order"].first().reset_index()
        )
        order_df = order_df.rename(columns={"_original_order": "_sort_key"})

        df_to_pivot = df_to_pivot.drop(columns=["_original_order"])

        pivoted = df_to_pivot.pivot_table(
            index=index_cols,
            columns="Statistic",
            values="value",
            aggfunc="first",  # type: ignore
            sort=False,
        ).reset_index()

        # Merge with order and sort to restore original order
        pivoted = pivoted.merge(order_df, on=index_cols, how="left")
        pivoted = (
            pivoted.sort_values("_sort_key").drop(columns=["_sort_key"]).reset_index(drop=True)
        )

        # Reorder columns to preserve statistic order
        other_cols = [col for col in pivoted.columns if col not in stat_order]
        ordered_stats = [col for col in stat_order if col in pivoted.columns]
        pivoted = pivoted[other_cols + ordered_stats]

        pivoted.columns.name = None
        return pivoted

    def _gdf_pivot_wide(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Pivot GeoDataFrame to wide format, preserving geometry.

        Handles null geometries correctly by extracting geometry before
        pivoting and merging it back afterwards.

        Args:
            gdf: The GeoDataFrame in long format.

        Returns:
            The pivoted GeoDataFrame with geometry preserved.
        """
        geometry_col = gdf.geometry.name
        crs = gdf.crs
        spatial_key = self._spatial_info.key
        spatial_id_col = f"{spatial_key}{ID_COLUMN_SUFFIX}" if spatial_key else None

        # Extract geometry mapping before pivoting to avoid losing rows with null geometry
        # The spatial ID column is more reliable for joining if available
        if spatial_id_col and spatial_id_col in gdf.columns:
            geometry_map = gdf[[spatial_id_col, geometry_col]].drop_duplicates(
                subset=[spatial_id_col]
            )
            join_col = spatial_id_col
        elif spatial_key and spatial_key in gdf.columns:
            geometry_map = gdf[[spatial_key, geometry_col]].drop_duplicates(subset=[spatial_key])
            join_col = spatial_key
        else:
            # Fall back to original behavior if no spatial key
            pivoted = self._pivot_wide(gdf)
            if geometry_col in pivoted.columns:
                cols = [col for col in pivoted.columns if col != geometry_col] + [geometry_col]
                pivoted = pivoted[cols]
                return gpd.GeoDataFrame(pivoted, geometry=geometry_col, crs=crs)
            return pivoted  # type: ignore

        # Drop geometry before pivoting to avoid issues with null geometries
        df_no_geom = gdf.drop(columns=[geometry_col])
        pivoted = self._pivot_wide(df_no_geom)

        # Merge geometry back in
        pivoted = pivoted.merge(geometry_map, on=join_col, how="left")

        # Make geometry column come last
        cols = [col for col in pivoted.columns if col != geometry_col] + [geometry_col]
        pivoted = pivoted[cols]

        return gpd.GeoDataFrame(pivoted, geometry=geometry_col, crs=crs)

    def _gdf_pivot_tidy(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Pivot GeoDataFrame to tidy format, preserving geometry.

        Handles null geometries correctly by extracting geometry before
        pivoting and merging it back afterwards.

        Args:
            gdf: The GeoDataFrame in long format.

        Returns:
            The pivoted GeoDataFrame with geometry preserved.
        """
        geometry_col = gdf.geometry.name
        crs = gdf.crs
        spatial_key = self._spatial_info.key
        spatial_id_col = f"{spatial_key}{ID_COLUMN_SUFFIX}" if spatial_key else None

        # Extract geometry mapping before pivoting to avoid losing rows with null geometry
        # The spatial ID column is more reliable for joining if available
        if spatial_id_col and spatial_id_col in gdf.columns:
            geometry_map = gdf[[spatial_id_col, geometry_col]].drop_duplicates(
                subset=[spatial_id_col]
            )
            join_col = spatial_id_col
        elif spatial_key and spatial_key in gdf.columns:
            geometry_map = gdf[[spatial_key, geometry_col]].drop_duplicates(subset=[spatial_key])
            join_col = spatial_key
        else:
            # Fall back to original behavior if no spatial key
            pivoted = self._pivot_tidy(gdf)
            if geometry_col in pivoted.columns:
                cols = [col for col in pivoted.columns if col != geometry_col] + [geometry_col]
                pivoted = pivoted[cols]
                return gpd.GeoDataFrame(pivoted, geometry=geometry_col, crs=crs)
            return pivoted  # type: ignore

        # Drop geometry before pivoting to avoid issues with null geometries
        df_no_geom = gdf.drop(columns=[geometry_col])
        pivoted = self._pivot_tidy(df_no_geom)

        # Merge geometry back in
        pivoted = pivoted.merge(geometry_map, on=join_col, how="left")

        # Make geometry column come last
        cols = [col for col in pivoted.columns if col != geometry_col] + [geometry_col]
        pivoted = pivoted[cols]

        return gpd.GeoDataFrame(pivoted, geometry=geometry_col, crs=crs)
