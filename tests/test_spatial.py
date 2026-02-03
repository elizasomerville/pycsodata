"""Tests for the spatial module."""

from unittest.mock import patch

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

from pycsodata.constants import DEFAULT_CRS
from pycsodata.exceptions import SpatialError
from pycsodata.spatial import _detect_crs, _merge_dataframes, create_geodataframe


class TestDetectCRS:
    """Tests for the _detect_crs function."""

    def test_detect_crs_from_properties_name(self):
        """Test detecting CRS from properties.name."""
        geojson = {"crs": {"properties": {"name": "EPSG:4326"}}}
        assert _detect_crs(geojson) == "EPSG:4326"

    def test_detect_crs_from_direct_name(self):
        """Test detecting CRS from direct name key."""
        geojson = {"crs": {"name": "EPSG:32629"}}
        assert _detect_crs(geojson) == "EPSG:32629"

    def test_detect_crs_returns_default_when_no_crs(self):
        """Test that DEFAULT_CRS is returned when no crs info."""
        geojson = {}
        assert _detect_crs(geojson) == DEFAULT_CRS

    def test_detect_crs_returns_default_when_crs_not_dict(self):
        """Test that DEFAULT_CRS is returned when crs is not a dict."""
        geojson = {"crs": "some_string"}
        assert _detect_crs(geojson) == DEFAULT_CRS

    def test_detect_crs_returns_default_when_crs_empty(self):
        """Test that DEFAULT_CRS is returned when crs dict is empty."""
        geojson = {"crs": {}}
        assert _detect_crs(geojson) == DEFAULT_CRS

    def test_detect_crs_properties_name_priority(self):
        """Test that properties.name takes priority over direct name."""
        geojson = {"crs": {"name": "EPSG:32629", "properties": {"name": "EPSG:4326"}}}
        assert _detect_crs(geojson) == "EPSG:4326"


class TestMergeDataframes:
    """Tests for the _merge_dataframes function."""

    def test_merge_on_id_column(self):
        """Test merging on ID column with code."""
        df = pd.DataFrame(
            {"County": ["Dublin", "Cork"], "County ID": ["IE061", "IE062"], "value": [100, 200]}
        )

        gdf = gpd.GeoDataFrame({"code": ["IE061", "IE062"], "geometry": [Point(0, 0), Point(1, 1)]})

        geojson = {"features": []}

        result = _merge_dataframes(df, gdf, "County", geojson)

        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == 2
        assert "geometry" in result.columns

    def test_merge_on_label_column(self):
        """Test merging on label column."""
        df = pd.DataFrame({"County": ["Dublin", "Cork"], "value": [100, 200]})

        gdf = gpd.GeoDataFrame(
            {"County": ["Dublin", "Cork"], "geometry": [Point(0, 0), Point(1, 1)]}
        )

        geojson = {"features": []}

        result = _merge_dataframes(df, gdf, "County", geojson)

        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == 2

    def test_merge_preserves_unmatched_rows_with_null_geometry(self):
        """Test that left join preserves rows without matching geometry."""
        # Simulate aggregate region like 'State' that has no matching geometry
        df = pd.DataFrame(
            {
                "County": ["Dublin", "Cork", "State"],
                "County ID": ["IE061", "IE062", "IE0"],
                "value": [100, 200, 300],
            }
        )

        # GeoJSON only has Dublin and Cork, not State
        gdf = gpd.GeoDataFrame({"code": ["IE061", "IE062"], "geometry": [Point(0, 0), Point(1, 1)]})

        geojson = {"features": []}

        result = _merge_dataframes(df, gdf, "County", geojson)

        # Result should not be None
        assert result is not None

        # All 3 rows should be preserved
        assert len(result) == 3
        assert "State" in result["County"].values

        # State row should have null geometry
        state_row = result[result["County"] == "State"]
        assert state_row.geometry.isna().all()

        # Dublin and Cork should have valid geometries
        dublin_row = result[result["County"] == "Dublin"]
        assert not dublin_row.geometry.isna().any()

    def test_merge_fails_no_suitable_columns(self):
        """Test that merge fails when no suitable columns found."""
        df = pd.DataFrame({"Region": ["A", "B"], "value": [100, 200]})

        gdf = gpd.GeoDataFrame({"other_column": ["X", "Y"], "geometry": [Point(0, 0), Point(1, 1)]})

        geojson = {"features": []}

        with pytest.raises(SpatialError, match="Could not find suitable columns"):
            _merge_dataframes(df, gdf, "County", geojson)

    def test_merge_uses_gdf_crs_if_available(self):
        """Test that GDF CRS is used if available."""
        df = pd.DataFrame({"County ID": ["IE061"], "value": [100]})

        gdf = gpd.GeoDataFrame({"code": ["IE061"], "geometry": [Point(0, 0)]}, crs="EPSG:32629")

        geojson = {"features": []}

        result = _merge_dataframes(df, gdf, "County", geojson)

        assert result is not None
        assert result.crs is not None
        assert str(result.crs) == "EPSG:32629"

    def test_merge_uses_geojson_crs_when_gdf_has_none(self):
        """Test that GeoJSON CRS is used when GDF has none."""
        df = pd.DataFrame({"County ID": ["IE061"], "value": [100]})

        gdf = gpd.GeoDataFrame({"code": ["IE061"], "geometry": [Point(0, 0)]}, crs=None)

        geojson = {"features": [], "crs": {"properties": {"name": "EPSG:4326"}}}

        result = _merge_dataframes(df, gdf, "County", geojson)

        assert result is not None
        assert result.crs is not None


class TestCreateGeoDataFrame:
    """Tests for the create_geodataframe function."""

    def test_raises_error_when_no_url(self):
        """Test that SpatialError is raised when no URL."""
        df = pd.DataFrame({"County": ["Dublin"], "value": [100]})
        with pytest.raises(SpatialError):
            create_geodataframe(df, None, None)

    def test_raises_error_when_no_key(self):
        """Test that SpatialError is raised when no key."""
        df = pd.DataFrame({"County": ["Dublin"], "value": [100]})
        with pytest.raises(SpatialError):
            create_geodataframe(df, "http://example.com", None)


class TestCreateGeoDataFrameEdgeCases:
    """Additional edge case tests for create_geodataframe."""

    def test_raises_error_when_no_features(self):
        """Test that SpatialError is raised when no features in GeoJSON."""
        df = pd.DataFrame({"County": ["Dublin"], "value": [100]})

        with patch("pycsodata.spatial.fetch_json") as mock_fetch:
            mock_fetch.return_value = {"features": []}

            with pytest.raises(SpatialError, match="No features found"):
                create_geodataframe(df, "http://example.com/geo.json", "County")

    def test_raises_error_on_key_error(self):
        """Test that SpatialError is raised on KeyError."""
        df = pd.DataFrame({"County": ["Dublin"], "value": [100]})

        with patch("pycsodata.spatial.fetch_json") as mock_fetch:
            mock_fetch.return_value = {
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": [0, 0]},
                        "properties": {},
                    }
                ]
            }

            with patch("pycsodata.spatial._merge_dataframes") as mock_merge:
                mock_merge.side_effect = KeyError("missing key")

                with pytest.raises(SpatialError, match="Error creating GeoDataFrame"):
                    create_geodataframe(df, "http://example.com/geo.json", "County")

    def test_raises_error_on_value_error(self):
        """Test that SpatialError is raised on ValueError."""
        df = pd.DataFrame({"County": ["Dublin"], "value": [100]})

        with patch("pycsodata.spatial.fetch_json") as mock_fetch:
            mock_fetch.return_value = {
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": [0, 0]},
                        "properties": {},
                    }
                ]
            }

            with patch("pycsodata.spatial._merge_dataframes") as mock_merge:
                mock_merge.side_effect = ValueError("invalid value")

                with pytest.raises(SpatialError, match="Error creating GeoDataFrame"):
                    create_geodataframe(df, "http://example.com/geo.json", "County")

    def test_sets_crs_when_gdf_has_none(self):
        """Test that CRS is set when GDF has none."""
        df = pd.DataFrame({"County ID": ["IE061"], "value": [100]})

        mock_geojson = {
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [-6.26, 53.35]},
                    "properties": {"code": "IE061"},
                }
            ],
            "crs": {"properties": {"name": "EPSG:4326"}},
        }

        with patch("pycsodata.spatial.fetch_json") as mock_fetch:
            mock_fetch.return_value = mock_geojson

            result = create_geodataframe(df, "http://example.com/geo.json", "County")

            # Result should have CRS set
            assert result is not None
            assert result.crs is not None

    def test_raises_error_when_merge_returns_none(self):
        """Test that SpatialError is raised when merge returns None."""
        df = pd.DataFrame({"County": ["Dublin"], "value": [100]})

        with patch("pycsodata.spatial.fetch_json") as mock_fetch:
            mock_fetch.return_value = {
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": [0, 0]},
                        "properties": {},
                    }
                ]
            }

            with patch("pycsodata.spatial._merge_dataframes") as mock_merge:
                mock_merge.return_value = None

                with pytest.raises(SpatialError, match="Spatial merge failed"):
                    create_geodataframe(df, "http://example.com/geo.json", "County")
