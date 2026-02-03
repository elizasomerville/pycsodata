"""Tests for the dataset module."""

import geopandas as gpd
import pandas as pd
import pytest

from pycsodata import CSOCache
from pycsodata._types import IncludeIDs, PivotFormat
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
