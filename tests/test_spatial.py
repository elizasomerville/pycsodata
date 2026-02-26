"""Tests for the spatial module."""

from unittest.mock import patch

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry

from pycsodata.constants import DEFAULT_CRS, MET_EIREANN_SPATIAL_KEY
from pycsodata.exceptions import SpatialError
from pycsodata.spatial import (
    _build_weather_stations_gdf,
    _detect_crs,
    _merge_dataframes,
    create_geodataframe,
    create_met_geodataframe,
)


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


class TestBuildWeatherStationsGdf:
    """Tests for _build_weather_stations_gdf function."""

    def test_returns_geodataframe(self):
        """Test that function returns a GeoDataFrame."""
        result = _build_weather_stations_gdf()
        assert isinstance(result, gpd.GeoDataFrame)

    def test_has_station_id_column(self):
        """Test that result has station_id column."""
        result = _build_weather_stations_gdf()
        assert "station_id" in result.columns

    def test_has_geometry_column(self):
        """Test that result has geometry column."""
        result = _build_weather_stations_gdf()
        assert "geometry" in result.columns

    def test_has_point_geometries(self):
        """Test that all geometries are Points."""
        result = _build_weather_stations_gdf()
        assert all(geom.geom_type == "Point" for geom in result.geometry)

    def test_crs_is_wgs84(self):
        """Test that CRS is WGS84 (EPSG:4326)."""
        result = _build_weather_stations_gdf()
        assert result.crs is not None
        assert str(result.crs) == DEFAULT_CRS

    def test_has_expected_stations(self):
        """Test that result contains expected weather stations."""
        result = _build_weather_stations_gdf()
        station_names = set(result["station_id"])
        assert "Dublin Airport" in station_names
        assert "Cork Airport" in station_names
        assert "Shannon Airport" in station_names
        assert "Valentia Observatory" in station_names

    def test_no_extra_columns(self):
        """Test that result only has station_id and geometry columns."""
        result = _build_weather_stations_gdf()
        assert set(result.columns) == {"station_id", "geometry"}

    def test_no_null_geometries(self):
        """Test that no station has a null geometry."""
        result = _build_weather_stations_gdf()
        assert not result.geometry.isna().any()

    def test_coordinates_in_ireland_bounds(self):
        """Test that all station coordinates fall within Ireland."""
        result = _build_weather_stations_gdf()
        for _, row in result.iterrows():
            lon = row.geometry.x
            lat = row.geometry.y
            assert 51.0 <= lat <= 56.0, f"Latitude {lat} out of range for {row['station_id']}"
            assert -11.0 <= lon <= -5.0, f"Longitude {lon} out of range for {row['station_id']}"


class TestCreateMetGeoDataFrame:
    """Tests for the create_met_geodataframe function."""

    def test_returns_geodataframe(self):
        """Test that function returns a GeoDataFrame."""
        df = pd.DataFrame(
            {
                MET_EIREANN_SPATIAL_KEY: ["Dublin Airport", "Cork Airport"],
                "value": [10.5, 12.3],
            }
        )
        result = create_met_geodataframe(df)
        assert isinstance(result, gpd.GeoDataFrame)

    def test_has_geometry_column(self):
        """Test that result has geometry column."""
        df = pd.DataFrame(
            {
                MET_EIREANN_SPATIAL_KEY: ["Dublin Airport", "Cork Airport"],
                "value": [10.5, 12.3],
            }
        )
        result = create_met_geodataframe(df)
        assert "geometry" in result.columns

    def test_crs_is_wgs84(self):
        """Test that CRS is WGS84 (EPSG:4326)."""
        df = pd.DataFrame(
            {
                MET_EIREANN_SPATIAL_KEY: ["Dublin Airport"],
                "value": [10.5],
            }
        )
        result = create_met_geodataframe(df)
        assert result.crs is not None
        assert str(result.crs) == DEFAULT_CRS

    def test_preserves_all_rows(self):
        """Test that left join preserves all input rows."""
        df = pd.DataFrame(
            {
                MET_EIREANN_SPATIAL_KEY: ["Dublin Airport", "Cork Airport", "Unknown Station"],
                "value": [10.5, 12.3, 8.0],
            }
        )
        result = create_met_geodataframe(df)
        assert len(result) == 3

    def test_unmatched_stations_have_null_geometry(self):
        """Test that unmatched stations get null geometry."""
        df = pd.DataFrame(
            {
                MET_EIREANN_SPATIAL_KEY: ["Dublin Airport", "NonExistent Station"],
                "value": [10.5, 8.0],
            }
        )
        result = create_met_geodataframe(df)
        non_existent = result[result[MET_EIREANN_SPATIAL_KEY] == "NonExistent Station"]
        assert non_existent.geometry.isna().all()

    def test_matched_stations_have_valid_geometry(self):
        """Test that matched stations have valid point geometry."""
        df = pd.DataFrame(
            {
                MET_EIREANN_SPATIAL_KEY: ["Dublin Airport"],
                "value": [10.5],
            }
        )
        result = create_met_geodataframe(df)
        assert not result.geometry.isna().any()
        assert (result.geometry.geom_type == "Point").all()

    def test_preserves_original_columns(self):
        """Test that all original DataFrame columns are preserved."""
        df = pd.DataFrame(
            {
                MET_EIREANN_SPATIAL_KEY: ["Dublin Airport"],
                "Statistic": ["Mean Temperature"],
                "Month": ["January"],
                "value": [5.0],
            }
        )
        result = create_met_geodataframe(df)
        for col in df.columns:
            if col != "Meteorological Weather Station_titlecased":
                assert col in result.columns

    def test_no_station_id_column_in_result(self):
        """Test that the merge key 'station_id' is dropped from result."""
        df = pd.DataFrame(
            {
                MET_EIREANN_SPATIAL_KEY: ["Dublin Airport"],
                "value": [10.5],
            }
        )
        result = create_met_geodataframe(df)
        assert "station_id" not in result.columns

    def test_many_to_one_merge(self):
        """Test that multiple rows per station are handled correctly."""
        df = pd.DataFrame(
            {
                MET_EIREANN_SPATIAL_KEY: ["Dublin Airport", "Dublin Airport", "Cork Airport"],
                "Month": ["Jan", "Feb", "Jan"],
                "value": [5.0, 6.0, 7.0],
            }
        )
        result = create_met_geodataframe(df)
        assert len(result) == 3
        dublin_rows = result[result[MET_EIREANN_SPATIAL_KEY] == "Dublin Airport"]
        assert len(dublin_rows) == 2
        # Both Dublin rows should have the same geometry
        geom0 = dublin_rows.geometry.iloc[0]
        geom1 = dublin_rows.geometry.iloc[1]
        assert isinstance(geom0, BaseGeometry)
        assert isinstance(geom1, BaseGeometry)
        assert geom0.equals(geom1)

    def test_raises_when_spatial_key_missing(self):
        """Test that SpatialError is raised when spatial key column is missing."""
        df = pd.DataFrame({"Other Column": ["A"], "value": [1]})
        with pytest.raises(SpatialError, match="not found in DataFrame"):
            create_met_geodataframe(df)

    def test_custom_spatial_key(self):
        """Test that a custom spatial key can be used."""
        df = pd.DataFrame(
            {
                "Station": ["Dublin Airport"],
                "value": [10.5],
            }
        )
        result = create_met_geodataframe(df, spatial_key="Station")
        assert isinstance(result, gpd.GeoDataFrame)
        assert not result.geometry.isna().any()

    def test_default_spatial_key(self):
        """Test that default spatial key is MET_EIREANN_SPATIAL_KEY."""
        df = pd.DataFrame(
            {
                MET_EIREANN_SPATIAL_KEY: ["Dublin Airport"],
                "value": [10.5],
            }
        )
        # Should work without explicitly passing spatial_key
        result = create_met_geodataframe(df)
        assert isinstance(result, gpd.GeoDataFrame)
