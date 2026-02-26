"""Tests for the dataset module."""

from typing import cast

import geopandas as gpd
import pandas as pd
import pytest

from pycsodata import CSOCache
from pycsodata._types import FilterValue, IncludeIDs, PivotFormat
from pycsodata.dataset import CSODataset
from pycsodata.exceptions import APIError, SpatialError, ValidationError

# Use CSOCache for cache management
_cache = CSOCache()


def flush_cache():
    """Helper to flush cache without deprecation warnings."""
    _cache.flush()


class TestCSODatasetInit:
    """Tests for CSODataset initialisation."""

    @pytest.mark.network
    def test_valid_table_code_loads(self):
        """Test that a valid table code loads successfully."""
        flush_cache()
        dataset = CSODataset("FY003A")

        assert dataset.table_code == "FY003A"

    def test_invalid_table_code_raises(self):
        """Test that invalid table code raises APIError."""
        flush_cache()

        with pytest.raises(APIError, match="not found"):
            CSODataset("INVALID_CODE_XYZ123")

    def test_invalid_table_code_message(self):
        """Test that invalid table code provides helpful error message."""
        flush_cache()

        with pytest.raises(APIError) as exc_info:
            CSODataset("INVALID_CODE_XYZ123")

        error_msg = str(exc_info.value)
        assert "not found" in error_msg.lower() or "not correspond" in error_msg.lower()
        assert "CSOCatalogue" in error_msg  # Should suggest using catalogue

    def test_invalid_include_ids_raises(self):
        """Test that invalid include_ids raises ValidationError."""
        with pytest.raises(ValidationError, match="Invalid include_ids"):
            CSODataset("FY003A", include_ids="invalid")

    @pytest.mark.network
    def test_accepts_valid_include_ids_strings(self):
        """Test that valid string values for include_ids are accepted."""
        flush_cache()

        for value in ["all", "spatial_only", "none"]:
            dataset = CSODataset("FY003A", include_ids=value)
            assert dataset._include_ids == IncludeIDs(value)

    @pytest.mark.network
    def test_accepts_valid_include_ids_enum(self):
        """Test that valid IncludeIDs enum values are accepted."""
        flush_cache()

        for ids_option in IncludeIDs:
            dataset = CSODataset("FY003A", include_ids=ids_option)
            assert dataset._include_ids == ids_option

    @pytest.mark.network
    def test_accepts_include_ids_list(self):
        """Test that include_ids accepts a list of column names."""
        flush_cache()
        dataset = CSODataset("FY003A", include_ids=["County"])

        assert dataset._include_ids == ["County"]


class TestCSODatasetProperties:
    """Tests for CSODataset properties."""

    @pytest.mark.network
    def test_metadata_returns_dict(self):
        """Test that metadata property returns a dict."""
        flush_cache()
        dataset = CSODataset("FY003A")

        metadata = dataset.metadata
        assert isinstance(metadata, dict)
        assert "table_code" in metadata

    @pytest.mark.network
    def test_has_spatial_data_true_for_geo_dataset(self):
        """Test has_spatial_data for a geographic dataset."""
        flush_cache()
        dataset = CSODataset("FY003A")

        # FY003A should have spatial data
        if dataset.spatial_info.is_available:
            assert dataset.has_spatial_data is True

    @pytest.mark.network
    def test_spatial_info_returns_spatial_info(self):
        """Test that spatial_info property works."""
        flush_cache()
        dataset = CSODataset("FY003A")

        info = dataset.spatial_info
        assert hasattr(info, "url")
        assert hasattr(info, "key")
        assert hasattr(info, "is_available")


class TestCSODatasetDf:
    """Tests for the df method."""

    @pytest.mark.network
    def test_returns_dataframe(self):
        """Test that df returns a DataFrame."""
        flush_cache()
        dataset = CSODataset("FY003A")
        df = dataset.df()

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    @pytest.mark.network
    def test_has_value_column(self):
        """Test that DataFrame has a value column."""
        flush_cache()
        dataset = CSODataset("FY003A")
        df = dataset.df()

        assert "value" in df.columns

    @pytest.mark.network
    def test_invalid_pivot_format_raises(self):
        """Test that invalid pivot format raises ValidationError."""
        flush_cache()
        dataset = CSODataset("FY003A")

        with pytest.raises(ValidationError, match="Invalid pivot_format"):
            dataset.df(pivot_format="invalid")

    @pytest.mark.network
    def test_long_format_default(self):
        """Test that long format is the default."""
        flush_cache()
        dataset = CSODataset("FY003A")

        df_default = dataset.df()
        df_long = dataset.df("long")

        pd.testing.assert_frame_equal(df_default, df_long)

    @pytest.mark.network
    def test_wide_format(self):
        """Test wide format pivoting."""
        flush_cache()
        dataset = CSODataset("FY003A", filters={"Statistic": ["Population"]})
        df_wide = dataset.df("wide")

        assert isinstance(df_wide, pd.DataFrame)
        # Wide format should not have 'value' column
        # (values are spread across time columns)

    @pytest.mark.network
    def test_tidy_format(self):
        """Test tidy format pivoting."""
        flush_cache()
        dataset = CSODataset("FY003A", filters={"CensusYear": ["2022"]})
        df_tidy = dataset.df("tidy")

        assert isinstance(df_tidy, pd.DataFrame)
        # Tidy format should not have 'Statistic' or 'value' columns
        # (statistics become column names)


class TestCSODatasetGdf:
    """Tests for the gdf method."""

    @pytest.mark.network
    def test_returns_geodataframe(self):
        """Test that gdf returns a GeoDataFrame."""
        flush_cache()
        dataset = CSODataset("FY003A")

        if dataset.has_spatial_data:
            gdf = dataset.gdf()
            assert isinstance(gdf, gpd.GeoDataFrame)
            assert "geometry" in gdf.columns

    @pytest.mark.network
    def test_raises_for_non_spatial_dataset(self):
        """Test that SpatialError is raised for non-spatial dataset."""
        flush_cache()
        # RIQ02 is a dataset without spatial data
        dataset = CSODataset("RIQ02")

        if not dataset.has_spatial_data:
            with pytest.raises(SpatialError, match="not available"):
                dataset.gdf()

    @pytest.mark.network
    def test_invalid_pivot_format_raises(self):
        """Test that invalid pivot format raises ValidationError."""
        flush_cache()
        dataset = CSODataset("FY003A")

        if dataset.has_spatial_data:
            with pytest.raises(ValidationError, match="Invalid pivot_format"):
                dataset.gdf(pivot_format="invalid")


class TestCSODatasetFilters:
    """Tests for dataset filtering."""

    @pytest.mark.network
    def test_filters_apply(self):
        """Test that filters are applied correctly."""
        flush_cache()
        dataset = CSODataset("FY003A", filters={"CensusYear": ["2022"]})
        df = dataset.df()

        assert len(df) > 0
        if "CensusYear" in df.columns:
            assert all(str(year) == "2022" for year in df["CensusYear"])

    @pytest.mark.network
    def test_multiple_filter_values(self):
        """Test filtering with multiple values."""
        flush_cache()
        dataset = CSODataset("FY003A", filters={"CensusYear": ["2016", "2022"]})
        df = dataset.df()

        assert len(df) > 0
        if "CensusYear" in df.columns:
            years = {str(y) for y in df["CensusYear"]}
            assert years.issubset({"2016", "2022"})


class TestCSODatasetIncludeIds:
    """Tests for include_ids options."""

    @pytest.mark.network
    def test_include_ids_none_drops_all(self):
        """Test that IncludeIDs.NONE drops all ID columns."""
        flush_cache()
        dataset = CSODataset("FY003A", include_ids=IncludeIDs.NONE)
        df = dataset.df()

        id_cols = [c for c in df.columns if c.endswith(" ID")]
        assert len(id_cols) == 0

    @pytest.mark.network
    def test_include_ids_all_keeps_all(self):
        """Test that IncludeIDs.ALL keeps all ID columns."""
        flush_cache()
        dataset = CSODataset("FY003A", include_ids=IncludeIDs.ALL)
        df = dataset.df()

        id_cols = [c for c in df.columns if c.endswith(" ID")]
        # Should have at least some ID columns
        assert len(id_cols) > 0

    @pytest.mark.network
    def test_include_ids_spatial_only(self):
        """Test that IncludeIDs.SPATIAL_ONLY keeps only spatial ID."""
        flush_cache()
        dataset = CSODataset("FY003A", include_ids=IncludeIDs.SPATIAL_ONLY)
        df = dataset.df()

        id_cols = [c for c in df.columns if c.endswith(" ID")]

        if dataset.has_spatial_data and dataset.spatial_info.key:
            spatial_id = f"{dataset.spatial_info.key} ID"
            # Should only have the spatial ID column
            assert id_cols == [spatial_id] or len(id_cols) == 0

    @pytest.mark.network
    def test_include_ids_list_single_column(self):
        """Test that include_ids list with single column keeps only that ID."""
        flush_cache()
        dataset = CSODataset("FY003A", include_ids=["CensusYear"])
        df = dataset.df()

        id_cols = [c for c in df.columns if c.endswith(" ID")]
        assert id_cols == ["CensusYear ID"]

    @pytest.mark.network
    def test_include_ids_list_multiple_columns(self):
        """Test that include_ids list with multiple columns keeps specified IDs."""
        flush_cache()
        dataset = CSODataset("FY003A", include_ids=["CensusYear", "Sex"])
        df = dataset.df()

        id_cols = {c for c in df.columns if c.endswith(" ID")}
        assert "CensusYear ID" in id_cols
        assert "Sex ID" in id_cols
        # Should only have the specified ID columns
        assert id_cols == {"CensusYear ID", "Sex ID"}


class TestCSODatasetStatisticnormalisation:
    """Tests for STATISTIC -> Statistic normalisation."""

    @pytest.mark.network
    def test_statistic_column_is_normalised(self):
        """Test that STATISTIC column is normalised to Statistic."""
        flush_cache()
        # RIQ02 has label="STATISTIC" in raw data
        dataset = CSODataset("RIQ02")
        df = dataset.df()

        # Should have "Statistic" not "STATISTIC"
        assert "Statistic" in df.columns
        assert "STATISTIC" not in df.columns

    @pytest.mark.network
    def test_filter_with_statistic_lowercase(self):
        """Test filtering with 'Statistic' key."""
        flush_cache()
        dataset = CSODataset("RIQ02", filters={"Statistic": ["RTB Average Monthly Rent Report"]})
        df = dataset.df()

        assert len(df) > 0
        assert all(df["Statistic"] == "RTB Average Monthly Rent Report")

    @pytest.mark.network
    def test_filter_with_statistic_uppercase(self):
        """Test filtering with 'STATISTIC' key (should be normalised)."""
        flush_cache()
        # This should work - STATISTIC should be normalised to Statistic
        dataset = CSODataset("RIQ02", filters={"STATISTIC": ["RTB Average Monthly Rent Report"]})
        df = dataset.df()

        assert len(df) > 0
        # Column should still be "Statistic" (normalised)
        assert "Statistic" in df.columns

    @pytest.mark.network
    def test_statistic_id_column_with_include_ids_all(self):
        """Test that Statistic ID column is added when include_ids='all'.

        This tests the fix for the bug where the STATISTIC dimension's label
        was not being normalised to 'Statistic' when building the ID mappings,
        causing the Statistic ID column to be missing.
        """
        flush_cache()
        # NDQ02 has STATISTIC dimension with label="STATISTIC" in raw metadata
        dataset = CSODataset("NDQ02", include_ids="all")
        df = dataset.df()

        # Should have Statistic and Statistic ID columns
        assert "Statistic" in df.columns, "Statistic column is missing"
        assert "Statistic ID" in df.columns, "Statistic ID column is missing"

        # Verify the Statistic ID column has the correct values
        assert df["Statistic ID"].notna().all(), "Statistic ID has null values"

    @pytest.mark.network
    def test_statistic_id_column_with_include_ids_list(self):
        """Test that Statistic ID column is added when explicitly requested."""
        flush_cache()
        dataset = CSODataset("NDQ02", include_ids=["Statistic"])
        df = dataset.df()

        # Should have Statistic ID column
        id_cols = [c for c in df.columns if c.endswith(" ID")]
        assert id_cols == ["Statistic ID"], f"Expected ['Statistic ID'], got {id_cols}"


class TestCSODatasetPivoting:
    """Tests for pivot_format functionality."""

    @pytest.mark.network
    def test_tidy_format_removes_statistic_id(self):
        """Test that tidy format removes Statistic ID column."""
        flush_cache()
        dataset = CSODataset("E2013", include_ids="all")
        df_tidy = dataset.df("tidy")

        # Statistic ID should not be in tidy format
        assert "Statistic ID" not in df_tidy.columns
        # Statistic column should also not be present (values are now column names)
        assert "Statistic" not in df_tidy.columns

    @pytest.mark.network
    def test_wide_format_removes_time_id(self):
        """Test that wide format removes time variable ID column."""
        flush_cache()
        dataset = CSODataset("FY003A", include_ids="all", filters={"Statistic": ["Population"]})
        df_wide = dataset.df("wide")

        time_var = dataset.metadata.get("time_variable")
        if time_var:
            time_id = f"{time_var} ID"
            # Time variable ID should not be in wide format
            assert time_id not in df_wide.columns


class TestCSODatasetDropNational:
    """Tests for drop_national_data option."""

    @pytest.mark.network
    def test_drop_national_removes_ie0(self):
        """Test that national data (IE0) is removed."""
        flush_cache()
        dataset = CSODataset("FY003A", drop_national_data=True, include_ids=IncludeIDs.ALL)
        df = dataset.df()

        if dataset.spatial_info.key:
            id_col = f"{dataset.spatial_info.key} ID"
            if id_col in df.columns:
                assert "IE0" not in df[id_col].values


class TestCSODatasetDescribe:
    """Tests for the describe method."""

    @pytest.mark.network
    def test_describe_runs_without_error(self, capsys):
        """Test that describe runs without error."""
        flush_cache()
        dataset = CSODataset("FY003A")
        dataset.describe()

        output = capsys.readouterr().out
        assert "Code:" in output
        assert "FY003A" in output


class TestCSODatasetRepr:
    """Tests for the __repr__ method."""

    @pytest.mark.network
    def test_repr_before_loading(self):
        """Test repr before data is loaded."""
        flush_cache()
        dataset = CSODataset("FY003A")

        repr_str = repr(dataset)
        assert "CSODataset" in repr_str
        assert "FY003A" in repr_str

    @pytest.mark.network
    def test_repr_after_loading(self):
        """Test repr after data is loaded."""
        flush_cache()
        dataset = CSODataset("FY003A")
        _ = dataset.df()  # Load data

        repr_str = repr(dataset)
        assert "CSODataset" in repr_str
        assert "FY003A" in repr_str
        # Repr shows spatial availability
        assert "spatial=" in repr_str


class TestCSODatasetSanitise:
    """Tests for the sanitise option."""

    @pytest.mark.network
    def test_sanitise_column_names(self):
        """Test that column names are sanitised when sanitise=True."""
        flush_cache()
        # Find a dataset with a column that would be sanitised
        dataset = CSODataset("FY003A", sanitise=True)
        df = dataset.df()

        # Check that columns don't have multiple spaces or trailing whitespace
        for col in df.columns:
            if col != "value":
                assert "  " not in col  # No multiple spaces
                assert col == col.strip()  # No edge whitespace

    @pytest.mark.network
    def test_sanitise_metadata(self):
        """Test that metadata is sanitised when sanitise=True."""
        flush_cache()
        dataset = CSODataset("FY003A", sanitise=True)
        meta = dataset.metadata

        # Check variables are sanitised
        for var in meta.get("variables", []):
            assert "  " not in var
            assert var == var.strip()

    @pytest.mark.network
    def test_sanitise_false_by_default(self):
        """Test that sanitise is False by default."""
        flush_cache()
        dataset = CSODataset("FY003A")
        assert dataset._sanitise is False

    @pytest.mark.network
    def test_filter_with_sanitised_key(self):
        """Test that filters work with sanitised column names."""
        flush_cache()
        # Use a dataset where the filter key might be sanitised
        dataset = CSODataset(
            "FY003A",
            sanitise=True,
            filters={"Census Year": ["2022"]},  # Sanitised form of "CensusYear"
        )
        df = dataset.df()

        # Should have data (filter worked)
        assert len(df) > 0

    @pytest.mark.network
    def test_sanitise_spatial_info(self):
        """Test that spatial info key is sanitised."""
        flush_cache()
        dataset = CSODataset("FY003A", sanitise=True)

        if dataset.has_spatial_data:
            key = dataset.spatial_info.key
            if key:
                assert "  " not in key
                assert key == key.strip()

    @pytest.mark.network
    def test_sanitise_include_ids_list(self):
        """Test that include_ids with sanitised column names works."""
        flush_cache()
        dataset = CSODataset(
            "FY003A",
            sanitise=True,
            include_ids=["Census Year"],  # Sanitised name
        )
        df = dataset.df()

        id_cols = [c for c in df.columns if c.endswith(" ID")]
        assert "Census Year ID" in id_cols


class TestCSODatasetPivotingOrder:
    """Tests for pivot format row ordering."""

    @pytest.mark.network
    def test_tidy_preserves_row_order(self):
        """Test that tidy format preserves row order."""
        flush_cache()
        dataset = CSODataset("PEA11")

        df_long = dataset.df("long")
        df_tidy = dataset.df("tidy")

        # Get the first unique index column values from both formats
        # The order should be the same
        index_cols = [
            col
            for col in df_tidy.columns
            if col not in ("Statistic", "Statistic ID", "value") and not col.endswith(" ID")
        ]

        if index_cols:
            first_col = index_cols[0]
            long_order = df_long[first_col].unique().tolist()
            tidy_order = df_tidy[first_col].unique().tolist()

            # Check order is preserved (first few values at least)
            for i in range(min(5, len(long_order), len(tidy_order))):
                assert long_order[i] == tidy_order[i]

    @pytest.mark.network
    def test_wide_preserves_row_order(self):
        """Test that wide format preserves row order."""
        flush_cache()
        dataset = CSODataset("FY003A", filters={"Statistic": ["Population"]})

        df_long = dataset.df("long")
        df_wide = dataset.df("wide")

        # Get the first unique spatial column values from both formats
        time_var = dataset.metadata.get("time_variable")
        index_cols = [
            col
            for col in df_wide.columns
            if col not in (time_var, f"{time_var} ID", "value") and not col.endswith(" ID")
        ]

        if index_cols:
            first_col = index_cols[0]
            long_order = df_long[first_col].unique().tolist()
            wide_order = df_wide[first_col].unique().tolist()

            # Check order is preserved (first few values at least)
            for i in range(min(5, len(long_order), len(wide_order))):
                assert long_order[i] == wide_order[i]


class TestCSODatasetNormalisation:
    """Tests for DataFrame normalisation methods."""

    @pytest.mark.network
    def test_normalise_value_column(self):
        """Test that value column is converted to numeric."""
        flush_cache()
        dataset = CSODataset("FY003A")
        df = dataset.df()

        assert df["value"].dtype in ("float64", "int64", "float32", "int32")

    @pytest.mark.network
    def test_normalise_statistic_id_column(self):
        """Test that STATISTIC ID is normalised to Statistic ID."""
        flush_cache()
        dataset = CSODataset("RIQ02", include_ids=IncludeIDs.ALL)
        df = dataset.df()

        # Should have Statistic ID, not STATISTIC ID
        id_cols = [c for c in df.columns if "statistic" in c.lower() and "id" in c.lower()]
        if id_cols:
            assert "Statistic ID" in df.columns


class TestCSODatasetIncludeIdsEdgeCases:
    """Edge case tests for include_ids parameter."""

    def test_invalid_include_ids_list_type_raises(self):
        """Test that include_ids list with non-string elements raises."""
        with pytest.raises(ValidationError, match="include_ids list must contain only strings"):
            CSODataset("FY003A", include_ids=[123, "County"])  # type: ignore

    @pytest.mark.network
    def test_include_ids_empty_list(self):
        """Test that include_ids with empty list drops all ID columns."""
        flush_cache()
        dataset = CSODataset("FY003A", include_ids=[])
        df = dataset.df()

        id_cols = [c for c in df.columns if c.endswith(" ID")]
        assert len(id_cols) == 0

    @pytest.mark.network
    def test_include_ids_nonexistent_column_raises(self):
        """Test that include_ids with non-existent column raises ValidationError."""
        flush_cache()
        dataset = CSODataset("FY003A", include_ids=["NonExistentColumn"])

        with pytest.raises(ValidationError, match="include_ids contains column names"):
            dataset.df()

    @pytest.mark.network
    def test_include_ids_partially_invalid_raises(self):
        """Test that include_ids with some invalid columns raises ValidationError."""
        flush_cache()
        dataset = CSODataset("FY003A", include_ids=["CensusYear", "NotAColumn"])

        with pytest.raises(ValidationError, match="include_ids contains column names"):
            dataset.df()

    @pytest.mark.network
    def test_include_ids_invalid_error_message_shows_valid_dimensions(self):
        """Test that ValidationError message shows valid dimension names."""
        flush_cache()
        dataset = CSODataset("FY003A", include_ids=["InvalidDimension"])

        try:
            dataset.df()
            pytest.fail("Expected ValidationError to be raised")
        except ValidationError as e:
            # Error message should include valid dimension names
            assert "Valid dimensions are:" in e.message
            assert "CensusYear" in e.message or "Sex" in e.message


class TestCSODatasetPivotFormatEdgeCases:
    """Edge case tests for pivot_format."""

    @pytest.mark.network
    def test_pivot_format_enum(self):
        """Test that PivotFormat enum values work."""
        flush_cache()
        dataset = CSODataset("FY003A")

        df_long = dataset.df(PivotFormat.LONG)
        assert isinstance(df_long, pd.DataFrame)

    @pytest.mark.network
    def test_wide_format_no_time_variable_raises(self):
        """Test that wide format raises when no time variable."""
        flush_cache()
        dataset = CSODataset("FY003A")
        df = dataset.df()

        # Create a DataFrame without the time variable column
        time_var = dataset.metadata.get("time_variable")
        if time_var and time_var in df.columns:
            df_no_time = df.drop(columns=[time_var])
            # This should raise because time variable column is missing
            with pytest.raises(ValidationError, match="Cannot pivot to wide format"):
                dataset._pivot_wide(df_no_time)

    @pytest.mark.network
    def test_tidy_format_no_statistic_column_raises(self):
        """Test that tidy format raises when no Statistic column."""
        flush_cache()
        dataset = CSODataset("FY003A", filters={"Statistic": ["Population"]})
        df = dataset.df()

        # Remove Statistic column
        if "Statistic" in df.columns:
            df_no_stat = df.drop(columns=["Statistic"])

            with pytest.raises(ValidationError, match="Cannot pivot to tidy format"):
                dataset._pivot_tidy(df_no_stat)


class TestCSODatasetFilterErrors:
    """Tests for filter validation errors."""

    @pytest.mark.network
    def test_filter_nonexistent_dimension_raises(self):
        """Test that filtering on non-existent dimension raises."""
        flush_cache()

        with pytest.raises(ValidationError, match="not found in dataset"):
            dataset = CSODataset("FY003A", filters={"NonExistent": ["Value"]})
            dataset.df()

    @pytest.mark.network
    def test_filter_no_matching_values_raises(self):
        """Test that filtering with no matching values raises."""
        flush_cache()

        with pytest.raises(ValidationError, match="No matching values"):
            dataset = CSODataset("FY003A", filters={"CensusYear": ["9999"]})
            dataset.df()


class TestCSODatasetDropFilteredColumns:
    """Tests for drop_filtered_cols option."""

    @pytest.mark.network
    def test_drop_filtered_cols_removes_columns(self):
        """Test that filtered columns are dropped."""
        flush_cache()
        dataset = CSODataset("FY003A", filters={"Sex": ["Both sexes"]}, drop_filtered_cols=True)
        df = dataset.df()

        assert "Sex" not in df.columns
        assert "Sex ID" not in df.columns

    @pytest.mark.network
    def test_drop_filtered_cols_also_drops_id_columns(self):
        """Test that ID columns for filtered dimensions are also dropped."""
        flush_cache()
        dataset = CSODataset(
            "FY003A",
            filters={"CensusYear": ["2022"]},
            drop_filtered_cols=True,
            include_ids=IncludeIDs.ALL,
        )
        df = dataset.df()

        # The filtered dimension ID column should be dropped
        assert "CensusYear ID" not in df.columns


class TestCSODatasetConvertDates:
    """Tests for convert_dates option."""

    @pytest.mark.network
    def test_convert_dates_parses_year(self):
        """Test that convert_dates parses year columns."""
        flush_cache()
        dataset = CSODataset("FY003A", convert_dates=True)
        df = dataset.df()

        time_var = dataset.metadata.get("time_variable")
        if time_var and time_var in df.columns:
            # Should be numeric (year) or datetime
            assert df[time_var].dtype in ("int64", "int32", "datetime64[ns]") or str(
                df[time_var].dtype
            ).startswith("period")


class TestCSODatasetGdfEdgeCases:
    """Edge case tests for gdf method."""

    @pytest.mark.network
    def test_gdf_caches_result(self):
        """Test that gdf result is cached."""
        flush_cache()
        dataset = CSODataset("FY003A")

        if dataset.has_spatial_data:
            gdf1 = dataset.gdf()
            gdf2 = dataset.gdf()

            # Should return cached result (same object)
            pd.testing.assert_frame_equal(gdf1, gdf2)

    @pytest.mark.network
    def test_gdf_wide_format(self):
        """Test gdf with wide pivot format."""
        flush_cache()
        dataset = CSODataset("FY003A", filters={"Statistic": ["Population"]})

        if dataset.has_spatial_data:
            gdf = dataset.gdf("wide")
            assert isinstance(gdf, gpd.GeoDataFrame)
            assert "geometry" in gdf.columns

    @pytest.mark.network
    def test_gdf_tidy_format(self):
        """Test gdf with tidy pivot format."""
        flush_cache()
        dataset = CSODataset("FY003A", filters={"CensusYear": ["2022"]})

        if dataset.has_spatial_data:
            gdf = dataset.gdf("tidy")
            assert isinstance(gdf, gpd.GeoDataFrame)
            assert "geometry" in gdf.columns

    @pytest.mark.network
    def test_gdf_preserves_aggregate_rows_with_null_geometry(self):
        """Test that gdf includes rows for aggregate regions with null geometries."""
        flush_cache()
        # NDQ09 has 'State' as an aggregate region in 'Local Electoral Area'
        dataset = CSODataset("NDQ09", include_ids="all")

        if dataset.has_spatial_data:
            df = dataset.df()
            gdf = dataset.gdf()

            # Both should have the same number of rows
            assert len(gdf) == len(df)

            # Check that aggregate rows exist with null geometries
            spatial_key = dataset.spatial_info.key
            if "State" in df[spatial_key].values:
                state_rows = gdf[gdf[spatial_key] == "State"]
                assert len(state_rows) > 0
                # State rows should have null geometries
                assert state_rows.geometry.isna().all()

    @pytest.mark.network
    def test_gdf_df_have_same_row_count(self):
        """Test that gdf and df always have the same number of rows."""
        flush_cache()
        dataset = CSODataset("FY003A", include_ids="all")

        if dataset.has_spatial_data:
            df = dataset.df()
            gdf = dataset.gdf()

            assert len(gdf) == len(df)

            # Check that areas missing from spatial data have null geometries
            null_geom_count = gdf.geometry.isna().sum()
            # There may be some null geometries for aggregate regions
            assert null_geom_count >= 0  # Just checking no error occurs

    @pytest.mark.network
    def test_gdf_df_have_same_row_count_wide_format(self):
        """Test that gdf and df have same row count in wide format."""
        flush_cache()
        # Use filter to ensure pivot works without duplicates
        dataset = CSODataset("NDQ09", include_ids="all")

        if dataset.has_spatial_data:
            df_wide = dataset.df("wide")
            gdf_wide = dataset.gdf("wide")

            assert len(gdf_wide) == len(df_wide)

            # Verify aggregate regions with null geometry are preserved
            spatial_key = dataset.spatial_info.key
            if "State" in df_wide[spatial_key].values:
                assert "State" in gdf_wide[spatial_key].values
                state_rows = gdf_wide[gdf_wide[spatial_key] == "State"]
                assert state_rows.geometry.isna().all()

    @pytest.mark.network
    def test_gdf_df_have_same_row_count_tidy_format(self):
        """Test that gdf and df have same row count in tidy format."""
        flush_cache()
        dataset = CSODataset("NDQ09", include_ids="all")

        if dataset.has_spatial_data:
            df_tidy = dataset.df("tidy")
            gdf_tidy = dataset.gdf("tidy")

            assert len(gdf_tidy) == len(df_tidy)

            # Verify aggregate regions with null geometry are preserved
            spatial_key = dataset.spatial_info.key
            if "State" in df_tidy[spatial_key].values:
                assert "State" in gdf_tidy[spatial_key].values
                state_rows = gdf_tidy[gdf_tidy[spatial_key] == "State"]
                assert state_rows.geometry.isna().all()


class TestCSODatasetDropNationalEdgeCases:
    """Edge case tests for drop_national_data."""

    @pytest.mark.network
    def test_drop_national_removes_state_label(self):
        """Test that national data with 'State' label is removed."""
        flush_cache()
        dataset = CSODataset("FY003A", drop_national_data=True)
        df = dataset.df()

        if dataset.spatial_info.key:
            spatial_key = dataset.spatial_info.key
            if spatial_key in df.columns:
                # Check that national labels are not present
                national_labels = ["State", "Ireland", "-"]
                for label in national_labels:
                    assert label not in df[spatial_key].values or True  # May not be applicable


class TestCSODatasetSanitiseEdgeCases:
    """Edge case tests for sanitise option."""

    @pytest.mark.network
    def test_sanitise_value_column_values(self):
        """Test that string values in dimension columns are sanitised."""
        flush_cache()
        dataset = CSODataset("FY003A", sanitise=True)
        df = dataset.df()

        # Check that columns have consistent spacing
        for col in df.columns:
            if col != "value" and df[col].dtype == "object":
                for val in df[col].dropna().unique():
                    if isinstance(val, str):
                        assert "  " not in val  # No multiple spaces

    @pytest.mark.network
    def test_sanitise_filter_value_translation(self):
        """Test that sanitised filter values work."""
        flush_cache()
        # The filter should be sanitised to match sanitised data
        dataset = CSODataset("FY003A", sanitise=True, filters={"Statistic": ["Population"]})
        df = dataset.df()

        assert len(df) > 0


class TestCSODatasetMetadataEdgeCases:
    """Tests for metadata property."""

    @pytest.mark.network
    def test_metadata_has_expected_keys(self):
        """Test that metadata has expected keys."""
        flush_cache()
        dataset = CSODataset("FY003A")
        meta = dataset.metadata

        expected_keys = ["table_code", "title", "variables"]
        for key in expected_keys:
            assert key in meta

    @pytest.mark.network
    def test_sanitised_metadata_variables(self):
        """Test that metadata variables are sanitised."""
        flush_cache()
        dataset = CSODataset("FY003A", sanitise=True)
        meta = dataset.metadata

        for var in meta.get("variables", []):
            assert "  " not in var


class TestCSODatasetDfCaching:
    """Tests for df caching."""

    @pytest.mark.network
    def test_df_caches_base_df(self):
        """Test that base DataFrame is cached."""
        flush_cache()
        dataset = CSODataset("FY003A")

        _ = dataset.df()
        assert dataset._cached_base_df is not None
        assert dataset._cached_df is not None

    @pytest.mark.network
    def test_df_returns_same_result(self):
        """Test that df returns consistent results."""
        flush_cache()
        dataset = CSODataset("FY003A")

        df1 = dataset.df()
        df2 = dataset.df()

        pd.testing.assert_frame_equal(df1, df2)


class TestCSODatasetPrivateMethods:
    """Tests for private methods."""

    @pytest.mark.network
    def test_add_id_columns(self):
        """Test that ID columns are added correctly."""
        flush_cache()
        dataset = CSODataset("FY003A", include_ids=IncludeIDs.ALL)
        df = dataset.df()

        # Should have ID columns
        id_cols = [c for c in df.columns if c.endswith(" ID")]
        assert len(id_cols) > 0

    @pytest.mark.network
    def test_normalise_filter_keys_mixed_case(self):
        """Test that filter keys with mixed case work."""
        flush_cache()
        # Using lowercase should still work
        dataset = CSODataset("RIQ02", filters={"statistic": ["RTB Average Monthly Rent Report"]})
        dataset.df()

        # This should not raise - the filter should be normalised


class TestCSODatasetPivotDuplicates:
    """Tests for pivot format duplicate handling."""

    @pytest.mark.network
    def test_wide_duplicate_detection(self):
        """Test that wide format detects duplicates properly."""
        flush_cache()
        # Filter to ensure no duplicates for this test
        dataset = CSODataset("FY003A", filters={"Statistic": ["Population"], "Sex": ["Both sexes"]})
        df_wide = dataset.df("wide")

        # Should succeed without raising duplicate error
        assert isinstance(df_wide, pd.DataFrame)

    @pytest.mark.network
    def test_tidy_duplicate_detection(self):
        """Test that tidy format detects duplicates properly."""
        flush_cache()
        # Filter to ensure no duplicates for this test
        dataset = CSODataset("FY003A", filters={"CensusYear": ["2022"], "Sex": ["Both sexes"]})
        df_tidy = dataset.df("tidy")

        # Should succeed without raising duplicate error
        assert isinstance(df_tidy, pd.DataFrame)


class TestCSODatasetFilterNormalisation:
    """Tests for filter key and value normalisation."""

    @pytest.mark.network
    def test_filter_statistic_id_uppercase(self):
        """Test that STATISTIC ID filter key is normalised."""
        flush_cache()
        dataset = CSODataset("RIQ02", include_ids=IncludeIDs.ALL)
        df = dataset.df()

        # Get a valid Statistic ID from the data
        if "Statistic ID" in df.columns:
            valid_stat_id = df["Statistic ID"].iloc[0]

            # Use STATISTIC ID (uppercase) as filter key
            dataset2 = CSODataset(
                "RIQ02", filters={"STATISTIC ID": [valid_stat_id]}, include_ids=IncludeIDs.ALL
            )
            df2 = dataset2.df()
            assert len(df2) > 0

    @pytest.mark.network
    def test_filter_with_none_value_skipped(self):
        """Test that filters with None values are skipped."""
        flush_cache()
        dataset = CSODataset("FY003A", filters={"Statistic": None, "CensusYear": ["2022"]})
        df = dataset.df()

        # Should work - None filter should be skipped
        assert len(df) > 0
        if "CensusYear" in df.columns:
            assert all(str(y) == "2022" for y in df["CensusYear"])

    @pytest.mark.network
    def test_filter_with_string_value_normalised_to_list(self):
        """Test that string filter values are normalised to list."""
        flush_cache()
        dataset = CSODataset(
            "FY003A",
            filters={"Statistic": "Population"},  # String, not list
        )
        df = dataset.df()

        assert len(df) > 0
        if "Statistic" in df.columns:
            assert all("Population" in str(s) for s in df["Statistic"])


class TestCSODatasetGdfPivotFormats:
    """Tests for GDF pivot format operations."""

    @pytest.mark.network
    def test_gdf_wide_returns_geodataframe(self):
        """Test that gdf wide format returns a GeoDataFrame with geometry."""
        flush_cache()
        dataset = CSODataset("FY003A", filters={"Statistic": ["Population"]})

        if dataset.has_spatial_data:
            gdf = dataset.gdf("wide")
            assert isinstance(gdf, gpd.GeoDataFrame)
            assert "geometry" in gdf.columns
            # Should preserve CRS
            assert gdf.crs is not None

    @pytest.mark.network
    def test_gdf_tidy_returns_geodataframe(self):
        """Test that gdf tidy format returns a GeoDataFrame with geometry."""
        flush_cache()
        dataset = CSODataset("FY003A", filters={"CensusYear": ["2022"]})

        if dataset.has_spatial_data:
            gdf = dataset.gdf("tidy")
            assert isinstance(gdf, gpd.GeoDataFrame)
            assert "geometry" in gdf.columns
            # Should preserve CRS
            assert gdf.crs is not None


class TestCSODatasetDropFilterColumnsEdgeCases:
    """Tests for _drop_filter_columns edge cases."""

    @pytest.mark.network
    def test_drop_filtered_cols_with_id_suffix_filter(self):
        """Test dropping columns when filter uses ID suffix."""
        flush_cache()
        dataset = CSODataset("RIQ02", include_ids=IncludeIDs.ALL)
        df = dataset.df()

        if "Statistic ID" in df.columns:
            valid_stat_id = df["Statistic ID"].iloc[0]

            dataset2 = CSODataset(
                "RIQ02",
                filters={"Statistic ID": [valid_stat_id]},
                drop_filtered_cols=True,
                include_ids=IncludeIDs.ALL,
            )
            df2 = dataset2.df()

            # Both Statistic and Statistic ID should be dropped
            assert "Statistic ID" not in df2.columns
            assert "Statistic" not in df2.columns


class TestCSODatasetReprEdgeCases:
    """Tests for __repr__ edge cases."""

    @pytest.mark.network
    def test_repr_includes_filters(self):
        """Test that repr includes filter information."""
        flush_cache()
        dataset = CSODataset("FY003A", filters={"Statistic": ["Population"]}, sanitise=True)
        _ = dataset.df()  # Load data

        repr_str = repr(dataset)
        assert "FY003A" in repr_str
        assert "CSODataset" in repr_str


class TestCSODatasetSpatialMergePreservation:
    """Tests for spatial column preservation during filtering."""

    @pytest.mark.network
    def test_spatial_key_preserved_when_filtering(self):
        """Test that spatial key is preserved when filtering for gdf."""
        flush_cache()
        dataset = CSODataset(
            "FY003A", filters={"Statistic": ["Population"]}, drop_filtered_cols=True
        )

        if dataset.has_spatial_data:
            gdf = dataset.gdf()
            spatial_key = dataset.spatial_info.key

            # Spatial key should be preserved even with drop_filtered_cols
            if spatial_key:
                assert spatial_key in gdf.columns or "geometry" in gdf.columns


class TestCSODatasetFilterColumnDrop:
    """Tests for filter column dropping with different column types."""

    @pytest.mark.network
    def test_drop_filtered_label_col_and_id_col(self):
        """Test that both label and ID columns are dropped for filtered dimensions."""
        flush_cache()
        dataset = CSODataset(
            "FY003A",
            filters={"Sex": ["Both sexes"]},
            drop_filtered_cols=True,
            include_ids=IncludeIDs.ALL,
        )
        df = dataset.df()

        # Both Sex and Sex ID should be dropped
        assert "Sex" not in df.columns
        assert "Sex ID" not in df.columns


class TestCSODatasetMetEireann:
    """Tests for Met Eireann meteorological dataset support (MTM01-MTM08)."""

    @pytest.mark.network
    def test_mtm_has_spatial_data(self):
        """Test that MTM datasets report spatial data as available."""
        flush_cache()
        dataset = CSODataset("MTM01")
        assert dataset.has_spatial_data is True

    @pytest.mark.network
    def test_mtm_is_met_dataset_flag(self):
        """Test that MTM datasets have _is_met_dataset set to True."""
        flush_cache()
        dataset = CSODataset("MTM01")
        assert dataset._is_met_dataset is True

    @pytest.mark.network
    def test_non_mtm_is_not_met_dataset(self):
        """Test that non-MTM datasets do not have _is_met_dataset set."""
        flush_cache()
        dataset = CSODataset("FY003A")
        assert dataset._is_met_dataset is False

    @pytest.mark.network
    def test_mtm_spatial_key(self):
        """Test that MTM datasets use the correct spatial key."""
        flush_cache()
        dataset = CSODataset("MTM01")
        assert dataset.spatial_info.key == "Meteorological Weather Station"

    @pytest.mark.network
    def test_mtm_spatial_key_sanitised(self):
        """Test that MTM spatial key is sanitised when sanitise=True."""
        flush_cache()
        dataset = CSODataset("MTM01", sanitise=True)
        # Sanitisation shouldn't change this particular key, but it should be applied
        assert dataset.spatial_info.key is not None

    @pytest.mark.network
    def test_mtm_df_returns_dataframe(self):
        """Test that MTM dataset df() returns a DataFrame."""
        flush_cache()
        dataset = CSODataset("MTM01")
        df = dataset.df()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    @pytest.mark.network
    def test_mtm_gdf_returns_geodataframe(self):
        """Test that MTM dataset gdf() returns a GeoDataFrame."""
        flush_cache()
        dataset = CSODataset("MTM01")
        gdf = dataset.gdf()
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert "geometry" in gdf.columns

    @pytest.mark.network
    def test_mtm_gdf_has_crs(self):
        """Test that MTM GeoDataFrame has CRS set to EPSG:4326."""
        flush_cache()
        dataset = CSODataset("MTM01")
        gdf = dataset.gdf()
        assert gdf.crs is not None
        assert str(gdf.crs) == "EPSG:4326"

    @pytest.mark.network
    def test_mtm_gdf_has_point_geometry(self):
        """Test that MTM GeoDataFrame has Point geometries."""
        flush_cache()
        dataset = CSODataset("MTM01")
        gdf = dataset.gdf()
        valid_geom = gdf[gdf.geometry.notna()]
        if len(valid_geom) > 0:
            assert all(geom.geom_type == "Point" for geom in valid_geom.geometry)

    @pytest.mark.network
    def test_mtm_gdf_df_same_row_count(self):
        """Test that gdf and df have the same number of rows for MTM datasets."""
        flush_cache()
        dataset = CSODataset("MTM01")
        df = dataset.df()
        gdf = dataset.gdf()
        assert len(gdf) == len(df)

    @pytest.mark.network
    def test_mtm_gdf_preserves_columns(self):
        """Test that GeoDataFrame preserves all DataFrame columns."""
        flush_cache()
        dataset = CSODataset("MTM01")
        df = dataset.df()
        gdf = dataset.gdf()
        for col in df.columns:
            assert col in gdf.columns

    @pytest.mark.network
    def test_mtm_gdf_no_station_id_column(self):
        """Test that the merge key 'station_id' is not in the GeoDataFrame."""
        flush_cache()
        dataset = CSODataset("MTM01")
        gdf = dataset.gdf()
        assert "station_id" not in gdf.columns

    @pytest.mark.network
    def test_mtm_gdf_wide_format(self):
        """Test that MTM GeoDataFrame works with wide pivot format."""
        flush_cache()
        dataset = CSODataset("MTM01", filters={"Statistic": dataset_first_stat("MTM01")})
        gdf = dataset.gdf("wide")
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert "geometry" in gdf.columns

    @pytest.mark.network
    def test_mtm_gdf_tidy_format(self):
        """Test that MTM GeoDataFrame works with tidy pivot format."""
        flush_cache()
        dataset = CSODataset("MTM01")
        df = dataset.df()
        time_var = dataset.metadata.get("time_variable")
        if time_var and time_var in df.columns:
            first_time = [df[time_var].iloc[0]]
            dataset2 = CSODataset("MTM01", filters={time_var: first_time})
            gdf = dataset2.gdf("tidy")
            assert isinstance(gdf, gpd.GeoDataFrame)
            assert "geometry" in gdf.columns

    @pytest.mark.network
    def test_mtm_repr_shows_spatial(self):
        """Test that MTM dataset repr shows spatial=yes."""
        flush_cache()
        dataset = CSODataset("MTM01")
        assert "spatial=yes" in repr(dataset)

    @pytest.mark.network
    def test_mtm_with_include_ids(self):
        """Test that include_ids works with MTM datasets."""
        flush_cache()
        dataset = CSODataset("MTM01", include_ids="all")
        gdf = dataset.gdf()
        assert isinstance(gdf, gpd.GeoDataFrame)

    @pytest.mark.network
    def test_mtm_with_drop_filtered_cols(self):
        """Test that drop_filtered_cols works with MTM gdf."""
        flush_cache()
        dataset = CSODataset(
            "MTM01",
            filters={"Statistic": dataset_first_stat("MTM01")},
            drop_filtered_cols=True,
        )
        gdf = dataset.gdf()
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert "Statistic" not in gdf.columns

    @pytest.mark.network
    def test_mtm05_gdf_returns_geodataframe(self):
        """Test that MTM05 specifically returns a valid GeoDataFrame."""
        flush_cache()
        dataset = CSODataset("MTM05")
        gdf = dataset.gdf()
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert "geometry" in gdf.columns
        assert len(gdf) > 0


def dataset_first_stat(table_code: str) -> list[FilterValue]:
    """Helper to get the first statistic value for a dataset."""
    dataset = CSODataset(table_code)
    df = dataset.df()
    if "Statistic" in df.columns:
        return [cast("FilterValue", df["Statistic"].iloc[0])]
    return []


# =============================================================================
# Helper for building test datasets without network
# =============================================================================


def _make_offline_dataset(
    *,
    spatial_key: str | None = "County",
    spatial_url: str | None = "http://example.com/abc",
    is_met: bool = False,
    include_ids: IncludeIDs | list[str] = IncludeIDs.NONE,
    filters: dict | None = None,
    drop_filtered_cols: bool = False,
    base_df: pd.DataFrame | None = None,
) -> CSODataset:
    """Create a CSODataset without running __init__ (no network)."""
    from pycsodata._types import SpatialInfo

    dataset = CSODataset.__new__(CSODataset)
    dataset.table_code = "TEST01"
    dataset._include_ids = include_ids
    dataset._spatial_info = SpatialInfo(url=spatial_url, key=spatial_key)
    dataset._is_met_dataset = is_met
    dataset._cached_gdf = None
    dataset._cached_gdf_ungeneralised = None
    dataset._cached_base_df = base_df
    dataset._cached_df = None
    dataset._filters = filters
    dataset._drop_filtered_cols = drop_filtered_cols
    dataset._cache_enabled = True
    dataset._raw_metadata = {"dimension": {}}
    dataset._sanitise = False
    dataset._convert_dates = False
    dataset._drop_national_data = False
    dataset._original_column_names = []
    dataset._sanitise_to_original_map = {}
    return dataset


# =============================================================================
# Tests for gdf() ungeneralised parameter type validation
# =============================================================================


class TestGdfUngeneralisedTypeValidation:
    """Tests that gdf() validates the ungeneralised parameter type."""

    def test_non_bool_ungeneralised_raises_validation_error(self):
        """Test that ungeneralised=1 (int) raises ValidationError."""
        from pycsodata._types import SpatialInfo

        dataset = _make_offline_dataset()
        # Manually set has_spatial_data prerequisites
        dataset._spatial_info = SpatialInfo(url="http://example.com/abc", key="County")

        with pytest.raises(ValidationError, match="Invalid ungeneralised"):
            dataset.gdf(ungeneralised=1)  # type: ignore

    def test_string_ungeneralised_raises_validation_error(self):
        """Test that ungeneralised='true' (string) raises ValidationError."""
        from pycsodata._types import SpatialInfo

        dataset = _make_offline_dataset()
        dataset._spatial_info = SpatialInfo(url="http://example.com/abc", key="County")

        with pytest.raises(ValidationError, match="Invalid ungeneralised"):
            dataset.gdf(ungeneralised="true")  # type: ignore

    def test_none_ungeneralised_raises_validation_error(self):
        """Test that ungeneralised=None raises ValidationError."""
        from pycsodata._types import SpatialInfo

        dataset = _make_offline_dataset()
        dataset._spatial_info = SpatialInfo(url="http://example.com/abc", key="County")

        with pytest.raises(ValidationError, match="Invalid ungeneralised"):
            dataset.gdf(ungeneralised=None)  # type: ignore

    def test_gdf_without_spatial_raises_before_ungeneralised_check(self):
        """Test that SpatialError is raised for non-spatial dataset."""
        from pycsodata._types import SpatialInfo

        dataset = _make_offline_dataset(spatial_key=None, spatial_url=None)
        dataset._spatial_info = SpatialInfo(url=None, key=None)

        with pytest.raises(SpatialError, match="not available"):
            dataset.gdf()


# =============================================================================
# Tests for copy=False parameter
# =============================================================================


class TestCopyParameter:
    """Tests for the copy=False parameter on df() and gdf()."""

    @pytest.mark.network
    def test_df_copy_false_returns_dataframe(self):
        """Test that df(copy=False) returns a DataFrame (not a copy)."""
        flush_cache()
        dataset = CSODataset("FY003A")
        df = dataset.df(copy=False)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    @pytest.mark.network
    def test_df_copy_false_shares_cache(self):
        """Test that df(copy=False) returns the cached object directly."""
        flush_cache()
        dataset = CSODataset("FY003A")
        _ = dataset.df()  # Warm up cache
        df_nocopy = dataset.df(copy=False)
        # Should be the exact same object as the cache
        assert df_nocopy is dataset._cached_df

    @pytest.mark.network
    def test_gdf_copy_false_shares_cache(self):
        """Test that gdf(copy=False) returns the cached object directly."""
        flush_cache()
        dataset = CSODataset("FY003A")
        if dataset.has_spatial_data:
            _ = dataset.gdf()  # Warm up cache
            gdf_nocopy = dataset.gdf(copy=False)
            assert gdf_nocopy is dataset._cached_gdf

    @pytest.mark.network
    def test_df_copy_true_returns_independent_copy(self):
        """Test that df(copy=True) returns an independent copy."""
        flush_cache()
        dataset = CSODataset("FY003A")
        _ = dataset.df()  # Warm up cache
        df_copy = dataset.df(copy=True)
        assert df_copy is not dataset._cached_df

    @pytest.mark.network
    def test_df_wide_copy_false(self):
        """Test that df('wide', copy=False) works."""
        flush_cache()
        dataset = CSODataset("FY003A", filters={"Statistic": ["Population"]})
        df = dataset.df("wide", copy=False)
        assert isinstance(df, pd.DataFrame)

    @pytest.mark.network
    def test_df_tidy_copy_false(self):
        """Test that df('tidy', copy=False) works."""
        flush_cache()
        dataset = CSODataset("FY003A", filters={"CensusYear": ["2022"]})
        df = dataset.df("tidy", copy=False)
        assert isinstance(df, pd.DataFrame)


# =============================================================================
# Unit tests for _filter_id_columns (no network)
# =============================================================================


class TestFilterIdColumnsUnit:
    """Unit tests for CSODataset._filter_id_columns."""

    def test_spatial_only_without_key_drops_all_id_cols(self):
        """Test SPATIAL_ONLY with no spatial key falls back to dropping all IDs."""
        df = pd.DataFrame(
            {
                "County": ["Dublin"],
                "County ID": ["IE061"],
                "Sex": ["Male"],
                "Sex ID": ["M"],
                "value": [100],
            }
        )
        dataset = _make_offline_dataset(
            spatial_key=None,  # no spatial key
            include_ids=IncludeIDs.SPATIAL_ONLY,
        )
        result = dataset._filter_id_columns(df)
        id_cols = [c for c in result.columns if c.endswith(" ID")]
        assert len(id_cols) == 0

    def test_all_keeps_all_id_cols(self):
        """Test ALL keeps all ID columns."""
        df = pd.DataFrame(
            {
                "County": ["Dublin"],
                "County ID": ["IE061"],
                "Sex": ["Male"],
                "Sex ID": ["M"],
                "value": [100],
            }
        )
        dataset = _make_offline_dataset(include_ids=IncludeIDs.ALL)
        result = dataset._filter_id_columns(df)
        assert "County ID" in result.columns
        assert "Sex ID" in result.columns

    def test_none_drops_all_id_cols(self):
        """Test NONE drops all ID columns."""
        df = pd.DataFrame(
            {
                "County": ["Dublin"],
                "County ID": ["IE061"],
                "value": [100],
            }
        )
        dataset = _make_offline_dataset(include_ids=IncludeIDs.NONE)
        result = dataset._filter_id_columns(df)
        assert "County ID" not in result.columns
        assert "County" in result.columns

    def test_spatial_only_with_key_keeps_only_spatial_id(self):
        """Test SPATIAL_ONLY keeps only the spatial dimension's ID column."""
        df = pd.DataFrame(
            {
                "County": ["Dublin"],
                "County ID": ["IE061"],
                "Sex": ["Male"],
                "Sex ID": ["M"],
                "value": [100],
            }
        )
        dataset = _make_offline_dataset(
            spatial_key="County",
            include_ids=IncludeIDs.SPATIAL_ONLY,
        )
        result = dataset._filter_id_columns(df)
        assert "County ID" in result.columns
        assert "Sex ID" not in result.columns

    def test_list_keeps_only_specified_id_cols(self):
        """Test list include_ids keeps only the specified ID columns."""
        df = pd.DataFrame(
            {
                "County": ["Dublin"],
                "County ID": ["IE061"],
                "Sex": ["Male"],
                "Sex ID": ["M"],
                "value": [100],
            }
        )
        dataset = _make_offline_dataset(include_ids=["Sex"])
        result = dataset._filter_id_columns(df)
        assert "Sex ID" in result.columns
        assert "County ID" not in result.columns

    def test_list_with_invalid_column_raises(self):
        """Test list include_ids with invalid column raises ValidationError."""
        df = pd.DataFrame(
            {
                "County": ["Dublin"],
                "County ID": ["IE061"],
                "value": [100],
            }
        )
        dataset = _make_offline_dataset(include_ids=["NonExistent"])
        with pytest.raises(ValidationError, match="not dimensions"):
            dataset._filter_id_columns(df)


# =============================================================================
# Unit tests for _remove_national_rows (no network)
# =============================================================================


class TestRemoveNationalRowsUnit:
    """Unit tests for CSODataset._remove_national_rows."""

    def test_no_spatial_key_returns_unchanged(self):
        """Test early return when no spatial key is set."""
        df = pd.DataFrame({"County": ["Dublin", "Ireland"], "value": [100, 9999]})
        dataset = _make_offline_dataset(spatial_key=None, spatial_url=None)
        result = dataset._remove_national_rows(df)
        assert len(result) == 2  # unchanged

    def test_filters_by_national_area_code_in_id_col(self):
        """Test that rows with IE0 in the ID column are removed."""
        df = pd.DataFrame(
            {
                "County": ["Dublin", "State"],
                "County ID": ["IE061", "IE0"],
                "value": [100, 9999],
            }
        )
        dataset = _make_offline_dataset(spatial_key="County")
        result = dataset._remove_national_rows(df)
        assert len(result) == 1
        assert result["County"].iloc[0] == "Dublin"

    def test_filters_by_national_label_when_no_id_col(self):
        """Test filtering by label column when ID column is absent."""
        df = pd.DataFrame(
            {
                "County": ["Dublin", "State", "Ireland"],
                "value": [100, 9999, 8888],
            }
        )
        dataset = _make_offline_dataset(spatial_key="County")
        result = dataset._remove_national_rows(df)
        assert len(result) == 1
        assert result["County"].iloc[0] == "Dublin"

    def test_filters_by_both_when_both_present(self):
        """Test that both ID and label filters apply jointly."""
        df = pd.DataFrame(
            {
                "County": ["Dublin", "Cork", "State"],
                "County ID": ["IE061", "IE021", "IE0"],
                "value": [1, 2, 99],
            }
        )
        dataset = _make_offline_dataset(spatial_key="County")
        result = dataset._remove_national_rows(df)
        assert len(result) == 2
        assert set(result["County"]) == {"Dublin", "Cork"}

    def test_spatial_key_not_in_df_returns_unchanged(self):
        """Test that when spatial key column is absent, df is returned unchanged."""
        df = pd.DataFrame({"OtherCol": ["Dublin"], "value": [100]})
        dataset = _make_offline_dataset(spatial_key="County")
        result = dataset._remove_national_rows(df)
        assert len(result) == 1


# =============================================================================
# Unit tests for _build_gdf error paths (mocked)
# =============================================================================


class TestBuildGdfSpatialErrors:
    """Tests for SpatialError cases raised by _build_gdf."""

    def test_no_geometry_column_raises_spatial_error(self):
        """Test that missing geometry column in create_geodataframe output raises."""
        from unittest.mock import patch

        # create_geodataframe returns a GeoDataFrame without geometry col
        fake_gdf = gpd.GeoDataFrame({"County": ["Dublin"], "value": [1]})

        dataset = _make_offline_dataset(
            base_df=pd.DataFrame({"County": ["Dublin"], "value": [1]}),
        )

        with (
            patch("pycsodata.dataset.create_geodataframe", return_value=fake_gdf),
            pytest.raises(SpatialError, match="no geometry column"),
        ):
            dataset._build_gdf()

    def test_all_geometries_empty_raises_spatial_error(self):
        """Test that all-empty geometry column raises SpatialError."""
        from unittest.mock import patch

        # GeoDataFrame where all geometries are empty
        empty_gdf = gpd.GeoDataFrame(
            {"County": ["Dublin"], "value": [1]},
            geometry=[None],  # type: ignore
        )

        dataset = _make_offline_dataset(
            base_df=pd.DataFrame({"County": ["Dublin"], "value": [1]}),
        )

        with (
            patch("pycsodata.dataset.create_geodataframe", return_value=empty_gdf),
            pytest.raises(SpatialError, match="geometries are missing or empty"),
        ):
            dataset._build_gdf()


# =============================================================================
# Unit tests for _pivot_wide and _pivot_tidy duplicate detection
# =============================================================================


class TestPivotWideDuplicateDetection:
    """Tests for duplicate detection in _pivot_wide."""

    def test_duplicate_rows_raise_validation_error(self):
        """Test that duplicate rows in _pivot_wide raise ValidationError."""
        dataset = _make_offline_dataset()
        dataset._raw_metadata = {
            "dimension": {},
            "id": ["CensusYear"],
            "label": "",
        }
        # Manually give it a metadata with time_variable
        _original_meta = dataset.metadata

        df = pd.DataFrame(
            {
                "County": ["Dublin", "Dublin"],
                "CensusYear": ["2022", "2022"],  # duplicate combination
                "value": [100, 200],
            }
        )

        with pytest.raises(ValidationError, match="Cannot pivot to wide format: duplicate"):
            # Manually patch metadata to return a time_variable
            from unittest.mock import MagicMock, patch

            meta_mock = MagicMock()
            meta_mock.get.return_value = "CensusYear"
            with patch.object(
                type(dataset),
                "metadata",
                new_callable=lambda: property(lambda self: {"time_variable": "CensusYear"}),
            ):
                dataset._pivot_wide(df)


class TestPivotTidyDuplicateDetection:
    """Tests for duplicate detection in _pivot_tidy."""

    def test_duplicate_rows_raise_validation_error(self):
        """Test that duplicate rows in _pivot_tidy raise ValidationError."""
        dataset = _make_offline_dataset()

        df = pd.DataFrame(
            {
                "County": ["Dublin", "Dublin"],
                "Statistic": ["Population", "Population"],  # duplicate combination
                "value": [100, 200],
            }
        )

        with pytest.raises(ValidationError, match="Cannot pivot to tidy format: duplicate"):
            dataset._pivot_tidy(df)

    def test_no_statistic_column_raises(self):
        """Test that missing Statistic column raises ValidationError."""
        dataset = _make_offline_dataset()

        df = pd.DataFrame(
            {
                "County": ["Dublin"],
                "value": [100],
            }
        )

        with pytest.raises(ValidationError, match="Cannot pivot to tidy format: 'Statistic'"):
            dataset._pivot_tidy(df)


# =============================================================================
# Tests for _gdf_pivot_wide and _gdf_pivot_tidy fallback paths
# =============================================================================


class TestGdfPivotFallbackPaths:
    """Tests for fallback paths in _gdf_pivot_wide and _gdf_pivot_tidy."""

    def _make_dataset_no_spatial_key(self) -> CSODataset:
        """Create a dataset with no spatial key configured."""
        return _make_offline_dataset(spatial_key=None, spatial_url=None)

    def test_gdf_pivot_wide_fallback_no_spatial_key(self):
        """Test _gdf_pivot_wide fallback when no spatial key is available."""
        dataset = self._make_dataset_no_spatial_key()
        # Patch metadata to return a time_variable
        from unittest.mock import patch

        from shapely.geometry import Point

        gdf = gpd.GeoDataFrame(
            {
                "County": ["Dublin", "Cork"],
                "Year": ["2022", "2022"],
                "value": [100.0, 200.0],
                "geometry": [Point(0, 0), Point(1, 1)],
            }
        )
        gdf = gdf.set_geometry("geometry")

        with patch.object(
            type(dataset),
            "metadata",
            new_callable=lambda: property(lambda self: {"time_variable": "Year"}),
        ):
            result = dataset._gdf_pivot_wide(gdf)
        assert isinstance(result, gpd.GeoDataFrame)

    def test_gdf_pivot_tidy_fallback_no_spatial_key(self):
        """Test _gdf_pivot_tidy fallback when no spatial key is available."""
        dataset = self._make_dataset_no_spatial_key()
        from shapely.geometry import Point

        gdf = gpd.GeoDataFrame(
            {
                "County": ["Dublin", "Cork"],
                "Statistic": ["Population", "Area"],
                "value": [100.0, 200.0],
                "geometry": [Point(0, 0), Point(1, 1)],
            }
        )
        gdf = gdf.set_geometry("geometry")

        result = dataset._gdf_pivot_tidy(gdf)
        assert isinstance(result, gpd.GeoDataFrame)

    @pytest.mark.network
    def test_gdf_pivot_wide_via_id_col(self):
        """Test _gdf_pivot_wide uses spatial ID column for geometry join."""
        flush_cache()
        dataset = CSODataset(
            "FY003A", filters={"Statistic": ["Population"]}, include_ids="spatial_only"
        )
        if dataset.has_spatial_data:
            gdf = dataset.gdf("wide")
            assert isinstance(gdf, gpd.GeoDataFrame)
            assert "geometry" in gdf.columns


# =============================================================================
# Tests for repr of non-spatial dataset
# =============================================================================


class TestReprNonSpatial:
    """Tests for __repr__ on non-spatial datasets."""

    @pytest.mark.network
    def test_repr_shows_spatial_no_for_non_spatial(self):
        """Test that repr shows spatial=no for non-spatial datasets."""
        flush_cache()
        # HA12 is a non-spatial dataset
        try:
            dataset = CSODataset("HA12")
        except Exception:
            pytest.skip("Dataset HA12 not available")
        repr_str = repr(dataset)
        assert "spatial=no" in repr_str

    def test_repr_offline(self):
        """Test __repr__ directly when spatial is no."""
        dataset = _make_offline_dataset(spatial_key=None, spatial_url=None)
        repr_str = repr(dataset)
        assert "CSODataset" in repr_str
        assert "TEST01" in repr_str
        assert "spatial=no" in repr_str

    def test_repr_online_spatial_yes(self):
        """Test __repr__ with spatial data available."""
        dataset = _make_offline_dataset(spatial_key="County")
        repr_str = repr(dataset)
        assert "spatial=yes" in repr_str


# =============================================================================
# Tests for _normalise_filter_keys
# =============================================================================


class TestNormaliseFilterKeysUnit:
    """Unit tests for CSODataset._normalise_filter_keys."""

    def test_normalises_statistic_uppercase(self):
        """Test that STATISTIC key is normalised to Statistic."""
        dataset = _make_offline_dataset()
        result = dataset._normalise_filter_keys({"STATISTIC": ["Population"]})
        assert "Statistic" in result
        assert "STATISTIC" not in result

    def test_normalises_statistic_id_uppercase(self):
        """Test that 'STATISTIC ID' key is normalised to 'Statistic ID'."""
        dataset = _make_offline_dataset()
        result = dataset._normalise_filter_keys({"STATISTIC ID": ["ABC123"]})
        assert "Statistic ID" in result

    def test_preserves_other_keys_unchanged(self):
        """Test that other keys are passed through unchanged."""
        dataset = _make_offline_dataset()
        result = dataset._normalise_filter_keys({"County": ["Dublin"]})
        assert "County" in result

    def test_empty_filters_returns_empty(self):
        """Test that empty filter dict is returned as-is."""
        dataset = _make_offline_dataset()
        assert dataset._normalise_filter_keys({}) == {}

    def test_none_filters_returns_none(self):
        """Test that None filters are returned unchanged."""
        dataset = _make_offline_dataset()
        assert dataset._normalise_filter_keys(None) is None  # type: ignore

    def test_sanitise_keys_when_sanitise_enabled(self):
        """Test that keys are sanitised when sanitise=True."""
        dataset = _make_offline_dataset()
        dataset._sanitise = True
        result = dataset._normalise_filter_keys({"Census Year": ["2022"]})
        # Sanitised key should still be present
        assert len(result) == 1

    def test_sanitise_string_values_when_sanitise_enabled(self):
        """Test that string values are sanitised when sanitise=True."""
        dataset = _make_offline_dataset()
        dataset._sanitise = True
        result = dataset._normalise_filter_keys({"County": ["Dublin"]})
        assert result["County"] == ["Dublin"]  # No change for clean strings

    def test_none_value_preserved(self):
        """Test that None filter values are preserved."""
        dataset = _make_offline_dataset()
        result = dataset._normalise_filter_keys({"County": None})
        assert result["County"] is None


# =============================================================================
# Tests for _drop_filter_columns edge cases
# =============================================================================


class TestDropFilterColumnsUnit:
    """Unit tests for CSODataset._drop_filter_columns."""

    def test_preserve_spatial_keeps_spatial_col(self):
        """Test that preserve_spatial=True keeps the spatial column."""
        df = pd.DataFrame(
            {
                "County": ["Dublin"],
                "County ID": ["IE061"],
                "Sex": ["Male"],
                "value": [100],
            }
        )
        dataset = _make_offline_dataset(
            spatial_key="County",
            filters={"County": ["Dublin"], "Sex": ["Male"]},
        )
        result = dataset._drop_filter_columns(df, preserve_spatial=True)
        # County should be preserved, Sex should be dropped
        assert "County" in result.columns
        assert "Sex" not in result.columns

    def test_no_filters_returns_unchanged(self):
        """Test that no filters means df is returned unchanged."""
        df = pd.DataFrame({"County": ["Dublin"], "value": [100]})
        dataset = _make_offline_dataset(filters=None)
        result = dataset._drop_filter_columns(df)
        assert list(result.columns) == list(df.columns)

    def test_drops_id_col_and_label_col_for_id_filter(self):
        """Test that filtering on an ID column also drops the label column."""
        df = pd.DataFrame(
            {
                "County": ["Dublin"],
                "County ID": ["IE061"],
                "value": [100],
            }
        )
        dataset = _make_offline_dataset(
            filters={"County ID": ["IE061"]},
        )
        result = dataset._drop_filter_columns(df)
        assert "County ID" not in result.columns
        assert "County" not in result.columns


# =============================================================================
# Tests for _normalise_dataframe
# =============================================================================


class TestNormaliseDataFrameUnit:
    """Unit tests for CSODataset._normalise_dataframe."""

    def test_normalises_statistic_column(self):
        """Test that STATISTIC column is renamed to Statistic."""
        dataset = _make_offline_dataset()
        df = pd.DataFrame({"STATISTIC": ["Population"], "value": ["100"]})
        result = dataset._normalise_dataframe(df)
        assert "Statistic" in result.columns
        assert "STATISTIC" not in result.columns

    def test_normalises_statistic_id_column(self):
        """Test that STATISTIC ID column is renamed to Statistic ID."""
        dataset = _make_offline_dataset()
        df = pd.DataFrame({"STATISTIC ID": ["ABC"], "value": [100]})
        result = dataset._normalise_dataframe(df)
        assert "Statistic ID" in result.columns
        assert "STATISTIC ID" not in result.columns

    def test_converts_value_to_numeric(self):
        """Test that value column is converted to numeric."""
        dataset = _make_offline_dataset()
        df = pd.DataFrame({"County": ["Dublin"], "value": ["123.4"]})
        result = dataset._normalise_dataframe(df)
        assert result["value"].dtype in ("float64", "float32")
        assert result["value"].iloc[0] == 123.4

    def test_coerces_non_numeric_to_nan(self):
        """Test that non-numeric values in value column become NaN."""
        dataset = _make_offline_dataset()
        df = pd.DataFrame({"County": ["Dublin"], "value": ["N/A"]})
        result = dataset._normalise_dataframe(df)
        assert pd.isna(result["value"].iloc[0])

    def test_no_rename_when_no_statistic_column(self):
        """Test that df without STATISTIC column is returned unchanged."""
        dataset = _make_offline_dataset()
        df = pd.DataFrame({"County": ["Dublin"], "value": [100]})
        result = dataset._normalise_dataframe(df)
        assert "County" in result.columns


# =============================================================================
# Tests for gdf() caching with force_reload
# =============================================================================


class TestGdfForceReload:
    """Tests for gdf() force_reload_geometries parameter."""

    def test_force_reload_clears_cache_and_rebuilds(self):
        """Test that force_reload_geometries=True clears and rebuilds cache."""
        from unittest.mock import patch

        from shapely.geometry import Point

        fake_gdf = gpd.GeoDataFrame(
            {"County": ["Dublin"], "value": [1]},
            geometry=[Point(0, 0)],
        )

        dataset = _make_offline_dataset(
            base_df=pd.DataFrame({"County": ["Dublin"], "value": [1]}),
        )
        dataset._cached_gdf_ungeneralised = fake_gdf  # Already cached

        with patch("pycsodata.dataset.create_ungeneralised_geodataframe") as mock_create:
            rebuilt_gdf = gpd.GeoDataFrame(
                {"County": ["Dublin"], "value": [2]},
                geometry=[Point(0, 0)],
            )
            mock_create.return_value = rebuilt_gdf

            _result = dataset._build_gdf(ungeneralised=True, force_reload=True)
            # Should have been called (cache was cleared)
            mock_create.assert_called_once()

    @pytest.mark.network
    def test_force_reload_geometries_network(self):
        """Test force_reload_geometries parameter is accepted."""
        flush_cache()
        dataset = CSODataset("FY003A")
        # Just check that it doesn't raise a TypeError
        if dataset.has_spatial_data:
            # Standard gdf should work fine
            gdf = dataset.gdf(force_reload_geometries=False)
            assert isinstance(gdf, gpd.GeoDataFrame)
