"""Tests for the ungeneralised geometry module."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, Polygon, box
from shapely.geometry.base import BaseGeometry

from pycsodata.exceptions import SpatialError
from pycsodata.ungeneralised import (
    _ALL_KNOWN_FILECODES,
    _COMPLEX_MAPPINGS,
    _DEFAULT_TAILTE_COPYRIGHT,
    _DEFAULT_TAILTE_LICENCE,
    _OSNI_COPYRIGHT,
    _OSNI_LICENCE,
    _SIMPLE_MAPPINGS,
    _UNAVAILABLE_FILECODES,
    _build_ungeneralised_gdf,
    _cache_feature_service_metadata,
    _complex_merge,
    _count_coordinates,
    _extract_filecode,
    _force_2d,
    _get_cache_dir,
    _log_copyright_info,
    _merge_gaeltacht_lp_dissolve,
    _parse_copyright_from_json,
    _parse_copyright_from_txt,
    _parse_copyright_from_xml,
    _read_cached_copyright,
    _update_readme,
    _url_cache_key,
    _write_metadata_txt,
    create_ungeneralised_geodataframe,
)

# =============================================================================
# Test Mapping Data Completeness
# =============================================================================


class TestMappingData:
    """Tests for the embedded mapping data from CSOtoTailte.csv."""

    def test_simple_mappings_count(self):
        """Test that we have 28 simple mappings."""
        assert len(_SIMPLE_MAPPINGS) == 28

    def test_complex_mappings_count(self):
        """Test that we have 7 complex mappings."""
        assert len(_COMPLEX_MAPPINGS) == 7

    def test_unavailable_filecodes_count(self):
        """Test that we have 7 unavailable filecodes."""
        assert len(_UNAVAILABLE_FILECODES) == 7

    def test_total_known_filecodes(self):
        """Test that all filecodes sum to 42."""
        assert len(_ALL_KNOWN_FILECODES) == 42

    def test_no_overlap_between_categories(self):
        """Test that no filecode appears in multiple categories."""
        simple = set(_SIMPLE_MAPPINGS)
        complex_ = set(_COMPLEX_MAPPINGS)
        unavailable = set(_UNAVAILABLE_FILECODES)

        assert simple.isdisjoint(complex_), "Simple and complex filecodes overlap"
        assert simple.isdisjoint(unavailable), "Simple and unavailable filecodes overlap"
        assert complex_.isdisjoint(unavailable), "Complex and unavailable filecodes overlap"

    def test_all_simple_mappings_have_id_and_url(self):
        """Test that all simple mappings have both ID field and URL."""
        for filecode, (id_field, url) in _SIMPLE_MAPPINGS.items():
            assert id_field, f"Empty ID field for {filecode}"
            assert url, f"Empty URL for {filecode}"
            assert url.startswith("https://"), f"URL doesn't start with https:// for {filecode}"

    def test_all_complex_mappings_have_url(self):
        """Test that all complex mappings have a Tailte URL."""
        for filecode, (url, _) in _COMPLEX_MAPPINGS.items():
            assert url, f"Empty Tailte URL for {filecode}"
            assert url.startswith("https://"), f"URL doesn't start with https:// for {filecode}"

    def test_complex_mappings_with_ni(self):
        """Test that correct complex mappings have NI URLs."""
        # Only c0ad28a... (admin areas) and 526860f... (counties) have NI
        ni_filecodes = {fc for fc, (_, ni) in _COMPLEX_MAPPINGS.items() if ni is not None}
        assert len(ni_filecodes) == 2
        assert "c0ad28a75e6fd0c4cc76a50ba859def4" in ni_filecodes
        assert "526860fb25a6567dae4dbaff1e6d48d3" in ni_filecodes

    def test_known_filecodes_in_csv(self):
        """Test that specific known filecodes are present."""
        # Check a few known filecodes from each category
        assert "8618bd9a9b8b23c966fdd8a37a1b3204" in _SIMPLE_MAPPINGS  # Electoral Divisions
        assert "988e6c798cfe938b89771ad4e4769167" in _SIMPLE_MAPPINGS  # Small Areas
        assert "c0ad28a75e6fd0c4cc76a50ba859def4" in _COMPLEX_MAPPINGS  # Admin Areas + NI
        assert "09a3c5e1c9d0ac5fc1ac4cfaa4506e51" in _UNAVAILABLE_FILECODES

    def test_all_filecodes_are_hex_strings(self):
        """Test that all filecodes look like MD5 hashes (32 hex chars)."""
        for fc in _ALL_KNOWN_FILECODES:
            assert len(fc) == 32, f"Filecode '{fc}' is not 32 characters"
            assert all(c in "0123456789abcdef" for c in fc), (
                f"Filecode '{fc}' contains non-hex characters"
            )


# =============================================================================
# Test Copyright & Licence Data
# =============================================================================


class TestCopyrightDefaults:
    """Tests for the copyright and licence default constants."""

    def test_default_tailte_copyright_is_nonempty(self):
        """Test that the default Tailte copyright is non-empty."""
        assert isinstance(_DEFAULT_TAILTE_COPYRIGHT, str)
        assert _DEFAULT_TAILTE_COPYRIGHT.strip()
        assert "Tailte" in _DEFAULT_TAILTE_COPYRIGHT

    def test_default_tailte_licence_is_cc_by_4(self):
        """Test that the default Tailte licence references CC BY 4.0."""
        assert "CC BY 4.0" in _DEFAULT_TAILTE_LICENCE
        assert "creativecommons.org" in _DEFAULT_TAILTE_LICENCE

    def test_osni_copyright_text(self):
        """Test the OSNI copyright text."""
        assert "Ordnance Survey of Northern Ireland" in _OSNI_COPYRIGHT

    def test_osni_licence_is_ogl(self):
        """Test that the OSNI licence references the Open Government Licence."""
        assert "Open Government Licence" in _OSNI_LICENCE
        assert "nationalarchives.gov.uk" in _OSNI_LICENCE


class TestParseXml:
    """Tests for _parse_copyright_from_xml."""

    def test_parses_credit_and_use_limitation(self, tmp_path):
        """Test XML with <credit> and <useLimitation> elements."""
        xml = (
            '<?xml version="1.0"?>'
            "<metadata><dataIdInfo>"
            "<idCredit>Test Credit</idCredit>"
            "<resConst><Consts><useLimit>Test Licence</useLimit></Consts></resConst>"
            "</dataIdInfo></metadata>"
        )
        p = tmp_path / "metadata.xml"
        p.write_text(xml)
        copyright_, licence_ = _parse_copyright_from_xml(p)
        assert copyright_ == "Test Credit"
        assert licence_ == "Test Licence"

    def test_parses_gmd_namespace(self, tmp_path):
        """Test XML with ISO 19139 GMD namespace."""
        xml = (
            '<?xml version="1.0"?>'
            '<gmd:MD_Metadata xmlns:gmd="http://www.isotc211.org/2005/gmd"'
            ' xmlns:gco="http://www.isotc211.org/2005/gco">'
            "<gmd:identificationInfo><gmd:MD_DataIdentification>"
            "<gmd:credit><gco:CharacterString>GMD Credit</gco:CharacterString></gmd:credit>"
            "<gmd:resourceConstraints><gmd:MD_LegalConstraints>"
            "<gmd:useLimitation><gco:CharacterString>GMD Licence</gco:CharacterString>"
            "</gmd:useLimitation>"
            "</gmd:MD_LegalConstraints></gmd:resourceConstraints>"
            "</gmd:MD_DataIdentification></gmd:identificationInfo>"
            "</gmd:MD_Metadata>"
        )
        p = tmp_path / "metadata.xml"
        p.write_text(xml)
        copyright_, licence_ = _parse_copyright_from_xml(p)
        assert copyright_ == "GMD Credit"
        assert licence_ == "GMD Licence"

    def test_strips_html_tags(self, tmp_path):
        """Test that HTML tags are stripped from copyright text."""
        xml = (
            '<?xml version="1.0"?>'
            "<metadata><dataIdInfo>"
            "<idCredit>&lt;b&gt;Bold Credit&lt;/b&gt;</idCredit>"
            "</dataIdInfo></metadata>"
        )
        p = tmp_path / "metadata.xml"
        p.write_text(xml)
        copyright_, _ = _parse_copyright_from_xml(p)
        assert copyright_ == "Bold Credit"

    def test_returns_none_for_missing_elements(self, tmp_path):
        """Test that missing elements return None."""
        xml = '<?xml version="1.0"?><metadata><other>stuff</other></metadata>'
        p = tmp_path / "metadata.xml"
        p.write_text(xml)
        assert _parse_copyright_from_xml(p) == (None, None)

    def test_returns_none_for_invalid_xml(self, tmp_path):
        """Test that invalid XML returns None."""
        p = tmp_path / "metadata.xml"
        p.write_text("not xml at all")
        assert _parse_copyright_from_xml(p) == (None, None)

    def test_returns_none_for_missing_file(self, tmp_path):
        """Test that a missing file returns None."""
        assert _parse_copyright_from_xml(tmp_path / "nope.xml") == (None, None)


class TestParseJson:
    """Tests for _parse_copyright_from_json."""

    def test_parses_copyright_text(self, tmp_path):
        """Test parsing copyrightText from JSON properties."""
        p = tmp_path / "properties.json"
        p.write_text(json.dumps({"copyrightText": "JSON Copyright"}))
        copyright_, _ = _parse_copyright_from_json(p)
        assert copyright_ == "JSON Copyright"

    def test_returns_none_for_empty_copyright(self, tmp_path):
        """Test that empty copyrightText returns None."""
        p = tmp_path / "properties.json"
        p.write_text(json.dumps({"copyrightText": ""}))
        assert _parse_copyright_from_json(p) == (None, None)

    def test_returns_none_for_missing_key(self, tmp_path):
        """Test that missing copyrightText key returns None."""
        p = tmp_path / "properties.json"
        p.write_text(json.dumps({"name": "Test"}))
        assert _parse_copyright_from_json(p) == (None, None)

    def test_returns_none_for_invalid_json(self, tmp_path):
        """Test that invalid JSON returns None."""
        p = tmp_path / "properties.json"
        p.write_text("not json")
        assert _parse_copyright_from_json(p) == (None, None)

    def test_returns_none_for_missing_file(self, tmp_path):
        """Test that a missing file returns None."""
        assert _parse_copyright_from_json(tmp_path / "nope.json") == (None, None)


class TestParseTxt:
    """Tests for _parse_copyright_from_txt."""

    def test_parses_copyright_and_licence(self, tmp_path):
        """Test parsing Copyright and Licence lines from text."""
        p = tmp_path / "metadata.txt"
        p.write_text("URL: https://example.com\nCopyright: My Copyright\nLicence: My Licence\n")
        copyright_, licence_ = _parse_copyright_from_txt(p)
        assert copyright_ == "My Copyright"
        assert licence_ == "My Licence"

    def test_parses_license_spelling(self, tmp_path):
        """Test that 'License:' (US spelling) also works."""
        p = tmp_path / "metadata.txt"
        p.write_text("License: US Licence\n")
        _, licence_ = _parse_copyright_from_txt(p)
        assert licence_ == "US Licence"

    def test_returns_none_for_empty_file(self, tmp_path):
        """Test that an empty file returns None."""
        p = tmp_path / "metadata.txt"
        p.write_text("")
        assert _parse_copyright_from_txt(p) == (None, None)

    def test_returns_none_for_missing_file(self, tmp_path):
        """Test that a missing file returns None."""
        assert _parse_copyright_from_txt(tmp_path / "nope.txt") == (None, None)


class TestWriteMetadataTxt:
    """Tests for _write_metadata_txt."""

    def test_creates_metadata_txt(self, tmp_path):
        """Test that metadata.txt is created with expected content."""
        _write_metadata_txt(tmp_path, "https://example.com/FeatureServer/0")
        txt = (tmp_path / "metadata.txt").read_text()
        assert "URL: https://example.com/FeatureServer/0" in txt
        assert f"Copyright: {_DEFAULT_TAILTE_COPYRIGHT}" in txt
        assert "CC BY 4.0" in txt
        assert "Downloaded:" in txt

    def test_roundtrip_with_parse(self, tmp_path):
        """Test that _write_metadata_txt output is parseable by _parse_copyright_from_txt."""
        _write_metadata_txt(tmp_path, "https://example.com/FeatureServer/0")
        copyright_, licence_ = _parse_copyright_from_txt(tmp_path / "metadata.txt")
        assert copyright_ == _DEFAULT_TAILTE_COPYRIGHT
        assert type(licence_) is str
        assert "CC BY 4.0" in licence_


class TestReadCachedCopyright:
    """Tests for _read_cached_copyright."""

    def test_reads_xml_first(self, tmp_path):
        """Test that XML metadata is preferred over JSON and text."""
        xml = (
            '<?xml version="1.0"?>'
            "<metadata><dataIdInfo>"
            "<idCredit>XML Credit</idCredit>"
            "<resConst><Consts><useLimit>XML Licence</useLimit></Consts></resConst>"
            "</dataIdInfo></metadata>"
        )
        url = "https://example.com/FeatureServer/0"
        from pycsodata.ungeneralised import _url_cache_key

        cache_key = _url_cache_key(url)
        md_dir = tmp_path / f"tailte_{cache_key}" / "metadata"
        md_dir.mkdir(parents=True)
        (md_dir / "metadata.xml").write_text(xml)
        (md_dir / "properties.json").write_text(json.dumps({"copyrightText": "JSON Credit"}))

        with patch("pycsodata.ungeneralised._get_cache_dir", return_value=tmp_path):
            copyright_, licence_ = _read_cached_copyright(url, prefix="tailte")
        assert copyright_ == "XML Credit"
        assert licence_ == "XML Licence"

    def test_falls_back_to_json(self, tmp_path):
        """Test that JSON is used when XML is not available."""
        url = "https://example.com/FeatureServer/0"
        cache_key = _url_cache_key(url)
        md_dir = tmp_path / f"tailte_{cache_key}" / "metadata"
        md_dir.mkdir(parents=True)
        (md_dir / "properties.json").write_text(json.dumps({"copyrightText": "JSON Credit"}))

        with patch("pycsodata.ungeneralised._get_cache_dir", return_value=tmp_path):
            copyright_, _ = _read_cached_copyright(url, prefix="tailte")
        assert copyright_ == "JSON Credit"

    def test_falls_back_to_txt(self, tmp_path):
        """Test that text is used when XML and JSON are not available."""
        url = "https://example.com/FeatureServer/0"
        cache_key = _url_cache_key(url)
        md_dir = tmp_path / f"tailte_{cache_key}" / "metadata"
        md_dir.mkdir(parents=True)
        (md_dir / "metadata.txt").write_text("Copyright: TXT Credit\nLicence: TXT Licence\n")

        with patch("pycsodata.ungeneralised._get_cache_dir", return_value=tmp_path):
            copyright_, licence_ = _read_cached_copyright(url, prefix="tailte")
        assert copyright_ == "TXT Credit"
        assert licence_ == "TXT Licence"

    def test_returns_none_when_no_cache(self, tmp_path):
        """Test that None is returned when no cache directory exists."""
        with patch("pycsodata.ungeneralised._get_cache_dir", return_value=tmp_path):
            assert _read_cached_copyright("https://example.com/nope") == (None, None)


class TestLogCopyrightInfo:
    """Tests for the _log_copyright_info function."""

    def test_logs_tailte_copyright_for_simple_mapping(self):
        """Test that Tailte copyright is logged for a simple mapping filecode."""
        with (
            patch("builtins.print") as mock_print,
            patch(
                "pycsodata.ungeneralised._read_cached_copyright",
                return_value=(None, None),
            ),
        ):
            _log_copyright_info("440c36d3b86e067e97ffb2fabf55900e")

        calls = [str(c) for c in mock_print.call_args_list]
        assert any("Tailte" in c for c in calls)
        assert any("CC BY 4.0" in c for c in calls)

    def test_logs_dynamic_copyright_from_cache(self):
        """Test that dynamically parsed copyright text appears in the log."""
        with (
            patch("builtins.print") as mock_print,
            patch(
                "pycsodata.ungeneralised._read_cached_copyright",
                return_value=("HSE/CSO", "Custom Licence"),
            ),
        ):
            _log_copyright_info("295fa6b26cb0e26f75e316b64e4c22b4")

        calls = [str(c) for c in mock_print.call_args_list]
        assert any("HSE/CSO" in c for c in calls)
        assert any("Custom Licence" in c for c in calls)

    def test_logs_osni_copyright_for_ni_filecode(self):
        """Test that OSNI copyright is logged when NI data is included."""
        with (
            patch("builtins.print") as mock_print,
            patch(
                "pycsodata.ungeneralised._read_cached_copyright",
                return_value=(None, None),
            ),
        ):
            _log_copyright_info("c0ad28a75e6fd0c4cc76a50ba859def4")

        calls = [str(c) for c in mock_print.call_args_list]
        assert any("Ordnance Survey of Northern Ireland" in c for c in calls)
        assert any("Open Government Licence" in c for c in calls)

    def test_no_osni_log_when_no_ni_data(self):
        """Test that OSNI copyright is NOT logged when there is no NI data."""
        with (
            patch("builtins.print") as mock_print,
            patch(
                "pycsodata.ungeneralised._read_cached_copyright",
                return_value=(None, None),
            ),
        ):
            _log_copyright_info("9ae1df4db5df6639ed4724f3a1b314ee")

        calls = [str(c) for c in mock_print.call_args_list]
        assert not any("Ordnance Survey of Northern Ireland" in c for c in calls)

    def test_no_crash_for_unknown_filecode(self):
        """Test that an unknown filecode does not crash."""
        with patch("builtins.print") as mock_print:
            _log_copyright_info("ffffffffffffffffffffffffffffffff")

        # Should produce no log output but should not crash
        mock_print.assert_not_called()

    def test_both_tailte_and_osni_logged_for_ni_filecode(self):
        """Test that both Tailte and OSNI are logged for NI filecodes."""
        with (
            patch("builtins.print") as mock_print,
            patch(
                "pycsodata.ungeneralised._read_cached_copyright",
                return_value=(None, None),
            ),
        ):
            _log_copyright_info("526860fb25a6567dae4dbaff1e6d48d3")

        # Should have exactly 2 info calls: one for Tailte, one for OSNI
        assert mock_print.call_count == 2


# =============================================================================
# Test Filecode Extraction
# =============================================================================


class TestExtractFilecode:
    """Tests for the _extract_filecode function."""

    def test_extracts_from_standard_url(self):
        """Test extraction from a standard CSO GeoMap URL."""
        url = (
            "https://ws.cso.ie/public/api.static/"
            "PxStat.Data.GeoMap_API.Read/"
            "8618bd9a9b8b23c966fdd8a37a1b3204"
        )
        assert _extract_filecode(url) == "8618bd9a9b8b23c966fdd8a37a1b3204"

    def test_extracts_from_url_with_trailing_slash(self):
        """Test extraction from URL with trailing slash."""
        url = "https://ws.cso.ie/public/api.static/PxStat.Data.GeoMap_API.Read/abc123def456/"
        assert _extract_filecode(url) == "abc123def456"

    def test_extracts_from_simple_url(self):
        """Test extraction from a simple URL."""
        assert _extract_filecode("https://example.com/filecode123") == "filecode123"


# =============================================================================
# Test Cache Configuration
# =============================================================================


class TestCacheConfiguration:
    """Tests for cache directory configuration."""

    def test_default_cache_dir(self, tmp_path, monkeypatch):
        """Test that default cache dir is in home directory."""
        monkeypatch.delenv("PYCSODATA_CACHE_DIR", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        cache_dir = _get_cache_dir()
        assert cache_dir == Path.home() / ".pycsodata" / "cache" / "ungeneralised"

    def test_custom_cache_dir_from_env(self):
        """Test that PYCSODATA_CACHE_DIR environment variable is respected."""
        with patch.dict(os.environ, {"PYCSODATA_CACHE_DIR": "/tmp/test_cache"}):
            cache_dir = _get_cache_dir()
            assert cache_dir == Path("/tmp/test_cache") / "ungeneralised"

    def test_url_cache_key_is_deterministic(self):
        """Test that the same URL always produces the same cache key."""
        url = "https://example.com/feature/service"
        assert _url_cache_key(url) == _url_cache_key(url)

    def test_url_cache_key_differs_for_different_urls(self):
        """Test that different URLs produce different cache keys."""
        url1 = "https://example.com/service1"
        url2 = "https://example.com/service2"
        assert _url_cache_key(url1) != _url_cache_key(url2)

    def test_url_cache_key_is_16_chars(self):
        """Test that cache key is exactly 16 characters."""
        key = _url_cache_key("https://example.com")
        assert len(key) == 16


# =============================================================================
# Test Coordinate Counting
# =============================================================================


class TestCountCoordinates:
    """Tests for the _count_coordinates function."""

    def test_counts_point_coordinates(self):
        """Test counting coordinates for point geometries."""
        gdf = gpd.GeoDataFrame(
            {"a": [1, 2]},
            geometry=[Point(0, 0), Point(1, 1)],
        )
        assert _count_coordinates(gdf) == 2

    def test_counts_polygon_coordinates(self):
        """Test counting coordinates for polygon geometries."""
        poly = box(0, 0, 1, 1)  # Has 5 coordinates (closed ring)
        gdf = gpd.GeoDataFrame({"a": [1]}, geometry=[poly])
        assert _count_coordinates(gdf) == 5

    def test_handles_null_geometries(self):
        """Test that null geometries are excluded from count."""
        gdf = gpd.GeoDataFrame(
            {"a": [1, 2]},
            geometry=[Point(0, 0), None],  # type: ignore
        )
        result = _count_coordinates(gdf)
        assert result >= 1  # At least the non-null point

    def test_returns_zero_for_empty_gdf(self):
        """Test that empty GeoDataFrame returns 0."""
        gdf = gpd.GeoDataFrame({"a": []}, geometry=[])
        assert _count_coordinates(gdf) == 0

    def test_returns_zero_for_all_null_gdf(self):
        """Test that GDF with only null geometries returns 0."""
        gdf = gpd.GeoDataFrame({"a": [1]}, geometry=[None])  # type: ignore
        assert _count_coordinates(gdf) == 0


# =============================================================================
# Test Force 2D
# =============================================================================


class TestForce2D:
    """Tests for the _force_2d function."""

    def test_leaves_2d_geometry_unchanged(self):
        """Test that 2D geometries are not modified."""
        poly = box(0, 0, 1, 1)
        gdf = gpd.GeoDataFrame({"a": [1]}, geometry=[poly], crs="EPSG:4326")
        result = _force_2d(gdf)
        assert cast("BaseGeometry", result.geometry.iloc[0]).equals(poly)

    def test_strips_z_from_polygon_z(self):
        """Test that POLYGON Z is converted to 2D POLYGON."""
        from shapely.geometry import Polygon as ShapelyPolygon

        poly_z = ShapelyPolygon([(0, 0, 10), (1, 0, 20), (1, 1, 30), (0, 1, 40), (0, 0, 10)])
        assert poly_z.has_z
        gdf = gpd.GeoDataFrame({"a": [1]}, geometry=[poly_z], crs="EPSG:4326")
        result = _force_2d(gdf)
        assert not cast("BaseGeometry", result.geometry.iloc[0]).has_z

    def test_strips_z_from_multipolygon_z(self):
        """Test that MULTIPOLYGON Z is converted to 2D MULTIPOLYGON."""
        from shapely.geometry import MultiPolygon
        from shapely.geometry import Polygon as ShapelyPolygon

        poly1 = ShapelyPolygon([(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 0, 1)])
        poly2 = ShapelyPolygon([(2, 2, 5), (3, 2, 5), (3, 3, 5), (2, 2, 5)])
        multi_z = MultiPolygon([poly1, poly2])
        assert multi_z.has_z
        gdf = gpd.GeoDataFrame({"a": [1]}, geometry=[multi_z], crs="EPSG:4326")
        result = _force_2d(gdf)
        assert not cast("BaseGeometry", result.geometry.iloc[0]).has_z

    def test_handles_null_geometries(self):
        """Test that null geometries are preserved."""
        from shapely.geometry import Polygon as ShapelyPolygon

        poly_z = ShapelyPolygon([(0, 0, 10), (1, 0, 20), (1, 1, 30), (0, 0, 10)])
        gdf = gpd.GeoDataFrame(
            {"a": [1, 2]},
            geometry=[poly_z, None],  # type: ignore
            crs="EPSG:4326",
        )
        result = _force_2d(gdf)
        geom = result.geometry.iloc[0]
        assert geom is not None
        assert not cast("BaseGeometry", geom).has_z
        assert result.geometry.iloc[1] is None

    def test_handles_all_null_geometries(self):
        """Test that all-null GeoDataFrame is returned unchanged."""
        gdf = gpd.GeoDataFrame({"a": [1]}, geometry=[None], crs="EPSG:4326")  # type: ignore
        result = _force_2d(gdf)
        assert result.geometry.isna().all()


# =============================================================================
# Test README Update
# =============================================================================


class TestUpdateReadme:
    """Tests for the _update_readme function."""

    def test_creates_readme_with_header(self, tmp_path):
        """Test that README is created with header on first call."""
        _update_readme(tmp_path, "features.gpkg", "https://example.com")
        readme = tmp_path / "README.txt"
        assert readme.exists()
        content = readme.read_text()
        assert "pycsodata Ungeneralised Geometry Cache" in content
        assert "Download Log:" in content
        assert "features.gpkg" in content
        assert "https://example.com" in content

    def test_appends_to_existing_readme(self, tmp_path):
        """Test that subsequent writes append to the README."""
        _update_readme(tmp_path, "file1.gpkg", "https://example.com/1")
        _update_readme(tmp_path, "file2.gpkg", "https://example.com/2")
        readme = tmp_path / "README.txt"
        content = readme.read_text()
        assert "file1.gpkg" in content
        assert "file2.gpkg" in content
        # Header should appear only once
        assert content.count("pycsodata Ungeneralised Geometry Cache") == 1

    def test_includes_timestamp(self, tmp_path):
        """Test that timestamp is included in README entries."""
        _update_readme(tmp_path, "features.gpkg", "https://example.com")
        readme = tmp_path / "README.txt"
        content = readme.read_text()
        # Should contain an ISO format timestamp (contains 'T' and '+')
        lines = content.strip().split("\n")
        last_line = lines[-1]
        assert "T" in last_line  # ISO format timestamp

    def test_creates_directory_if_needed(self, tmp_path):
        """Test that cache directory is created if it doesn't exist."""
        subdir = tmp_path / "subdir" / "nested"
        _update_readme(subdir, "features.gpkg", "https://example.com")
        assert subdir.exists()
        assert (subdir / "README.txt").exists()


# =============================================================================
# Test Download Functions (Mocked)
# =============================================================================


def _make_mock_session(
    feature_properties: list[dict],
    wkid: int = 2157,
    max_record_count: int = 1000,
):
    """Create a mock requests.Session whose ``.get()`` returns appropriate
    responses for the ArcGIS REST API calls made by ``_download_feature_service``.

    Args:
        feature_properties: List of property dicts for each feature.
        wkid: WKID for the spatial reference.
        max_record_count: Maximum records per page.

    Returns:
        A mock Session instance.
    """

    def _mock_get(url, **kwargs):
        params = kwargs.get("params", {})
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()

        if isinstance(params, dict) and params.get("returnCountOnly") == "true":
            # Feature count request
            resp.json.return_value = {"count": len(feature_properties)}
        elif "/query" in url:
            # Feature page request - build GeoJSON features
            features = [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[i, i], [i + 1, i], [i + 1, i + 1], [i, i + 1], [i, i]]],
                    },
                    "properties": props,
                }
                for i, props in enumerate(feature_properties)
            ]
            body = json.dumps(
                {
                    "type": "FeatureCollection",
                    "features": features,
                }
            ).encode()
            resp.json.return_value = json.loads(body)
            # Support streaming via iter_content
            resp.iter_content = MagicMock(return_value=iter([body]))
        elif "/metadata" in url:
            # Metadata endpoint - return non-XML so fallback triggers
            resp.text = "not xml"
            resp.content = b"not xml"
            resp.headers = {"Content-Type": "text/html"}
            resp.json.return_value = {}
        else:
            # Service info request
            resp.json.return_value = {
                "maxRecordCount": max_record_count,
                "extent": {"spatialReference": {"wkid": wkid}},
            }
        return resp

    mock_session = MagicMock()
    mock_session.get.side_effect = _mock_get
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=False)
    return mock_session


class TestDownloadFeatureService:
    """Tests for _download_feature_service (mocked)."""

    def test_downloads_and_caches(self, tmp_path):
        """Test that download creates cache file."""
        mock_session = _make_mock_session([{"GUID": "abc123"}])

        with (
            patch("pycsodata.ungeneralised._create_session", return_value=mock_session),
            patch("pycsodata.ungeneralised._get_cache_dir", return_value=tmp_path),
        ):
            from pycsodata.ungeneralised import _download_feature_service

            gdf = _download_feature_service(
                "https://example.com/FeatureServer/0",
                out_fields="GUID",
            )

            assert isinstance(gdf, gpd.GeoDataFrame)
            assert "GUID" in gdf.columns
            assert "geometry" in gdf.columns

    def test_loads_from_cache(self, tmp_path):
        """Test that cached file is loaded on second call."""
        # Create a cached file
        cache_key_dir = tmp_path / "tailte_test12345678"
        cache_key_dir.mkdir(parents=True)
        cache_file = cache_key_dir / "features.gpkg"

        gdf = gpd.GeoDataFrame(
            {"GUID": ["abc"]},
            geometry=[Point(0, 0)],
            crs="EPSG:4326",
        )
        gdf.to_file(cache_file, driver="GPKG")

        with (
            patch("pycsodata.ungeneralised._get_cache_dir", return_value=tmp_path),
            patch("pycsodata.ungeneralised._url_cache_key", return_value="test12345678"),
        ):
            from pycsodata.ungeneralised import _download_feature_service

            result = _download_feature_service(
                "https://example.com/FeatureServer/0",
                out_fields="GUID",
            )
            assert isinstance(result, gpd.GeoDataFrame)

    def test_force_reload_bypasses_cache(self, tmp_path):
        """Test that force_reload re-downloads even when cached."""
        # Create a cached file
        cache_key_dir = tmp_path / "tailte_test12345678"
        cache_key_dir.mkdir(parents=True)
        cache_file = cache_key_dir / "features.gpkg"

        gdf = gpd.GeoDataFrame(
            {"GUID": ["old_value"]},
            geometry=[Point(0, 0)],
            crs="EPSG:4326",
        )
        gdf.to_file(cache_file, driver="GPKG")

        mock_session = _make_mock_session([{"GUID": "new_value"}])

        with (
            patch("pycsodata.ungeneralised._get_cache_dir", return_value=tmp_path),
            patch("pycsodata.ungeneralised._url_cache_key", return_value="test12345678"),
            patch("pycsodata.ungeneralised._create_session", return_value=mock_session),
        ):
            from pycsodata.ungeneralised import _download_feature_service

            result = _download_feature_service(
                "https://example.com/FeatureServer/0",
                out_fields="*",
                force_reload=True,
            )
            assert isinstance(result, gpd.GeoDataFrame)
            # Should have called the mock session (not loaded from cache)
            mock_session.get.assert_called()


class TestDownloadNIGeoJSON:
    """Tests for _download_ni_geojson (mocked)."""

    def test_downloads_geojson(self, tmp_path):
        """Test that NI GeoJSON is downloaded."""
        ni_gdf = gpd.GeoDataFrame(
            {"CountyName": ["Antrim"]},
            geometry=[box(0, 0, 1, 1)],
            crs="EPSG:4326",
        )

        with (
            patch("pycsodata.ungeneralised._get_cache_dir", return_value=tmp_path),
            patch("geopandas.read_file", return_value=ni_gdf),
        ):
            from pycsodata.ungeneralised import _download_ni_geojson

            result = _download_ni_geojson("https://example.com/ni.geojson")
            assert isinstance(result, gpd.GeoDataFrame)
            assert "CountyName" in result.columns

    def test_caches_result(self, tmp_path):
        """Test that NI download is cached to disk."""
        ni_gdf = gpd.GeoDataFrame(
            {"CountyName": ["Antrim"]},
            geometry=[box(0, 0, 1, 1)],
            crs="EPSG:4326",
        )

        with (
            patch("pycsodata.ungeneralised._get_cache_dir", return_value=tmp_path),
            patch(
                "pycsodata.ungeneralised._url_cache_key",
                return_value="ni_test12345678",
            ),
            patch("geopandas.read_file", return_value=ni_gdf),
        ):
            from pycsodata.ungeneralised import _download_ni_geojson

            _download_ni_geojson("https://example.com/ni.geojson")

            # Check that a file was created in the cache directory
            cache_files = list(tmp_path.rglob("features.gpkg"))
            assert len(cache_files) >= 1


# =============================================================================
# Test Merge Strategies
# =============================================================================


def _make_cso_attrs(**kwargs):
    """Helper to create a CSO attributes DataFrame."""
    defaults = {"code": ["A", "B", "C"], "en": ["Area A", "Area B", "Area C"]}
    defaults.update(kwargs)
    return pd.DataFrame(defaults)


def _make_tailte_gdf(
    key_col: str,
    key_values: list[str],
    crs: str = "EPSG:2157",
) -> gpd.GeoDataFrame:
    """Helper to create a mock Tailte GeoDataFrame."""
    return gpd.GeoDataFrame(
        {key_col: key_values},
        geometry=[box(i, i, i + 1, i + 1) for i in range(len(key_values))],
        crs=crs,
    )


class TestSimpleMerge:
    """Tests for the _simple_merge function."""

    def test_merges_on_code_column(self):
        """Test that simple merge correctly joins on code."""
        cso_attrs = _make_cso_attrs(code=["abc", "def", "ghi"])
        tailte_gdf = _make_tailte_gdf("GUID", ["abc", "def", "ghi"])

        with patch(
            "pycsodata.ungeneralised._download_feature_service",
            return_value=tailte_gdf,
        ):
            from pycsodata.ungeneralised import _simple_merge

            result = _simple_merge(
                "440c36d3b86e067e97ffb2fabf55900e",
                cso_attrs,
            )

            assert isinstance(result, gpd.GeoDataFrame)
            assert len(result) == 3
            assert "geometry" in result.columns
            assert not result.geometry.isna().any()

    def test_preserves_unmatched_rows(self):
        """Test that unmatched CSO rows get null geometry."""
        cso_attrs = _make_cso_attrs(code=["abc", "def", "MISSING"])
        tailte_gdf = _make_tailte_gdf("GUID", ["abc", "def"])

        with patch(
            "pycsodata.ungeneralised._download_feature_service",
            return_value=tailte_gdf,
        ):
            from pycsodata.ungeneralised import _simple_merge

            result = _simple_merge(
                "440c36d3b86e067e97ffb2fabf55900e",
                cso_attrs,
            )

            assert len(result) == 3
            missing_row = result[result["code"] == "MISSING"]
            assert missing_row.geometry.isna().all()


class TestComplexMerges:
    """Tests for complex merge strategy functions."""

    def test_counties_dissolve(self):
        """Test that counties are dissolved by ENG_NAME_VALUE."""
        cso_attrs = _make_cso_attrs(
            code=["a", "b"],
            en=["Dublin", "Cork"],
        )
        # Two Dublin features that should be dissolved into one
        tailte_gdf = gpd.GeoDataFrame(
            {"ENG_NAME_VALUE": ["Dublin", "Dublin", "Cork"]},
            geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1), box(3, 3, 4, 4)],
            crs="EPSG:2157",
        )

        with patch(
            "pycsodata.ungeneralised._download_feature_service",
            return_value=tailte_gdf,
        ):
            from pycsodata.ungeneralised import _merge_counties_dissolve

            result = _merge_counties_dissolve(
                "893a6eb4f4a6f907410396ec8d8b738b",
                cso_attrs,
            )

            assert isinstance(result, gpd.GeoDataFrame)
            assert len(result) == 2
            # Dublin geometry should be a union of two boxes
            dublin_row = result[result["en"] == "Dublin"]
            assert not dublin_row.geometry.isna().any()

    def test_gaeltacht_county_title(self):
        """Test COUNTY.title() mapping."""
        cso_attrs = _make_cso_attrs(
            code=["a", "b"],
            en=["Dublin", "Cork"],
        )
        tailte_gdf = gpd.GeoDataFrame(
            {"COUNTY": ["DUBLIN", "CORK"]},
            geometry=[box(0, 0, 1, 1), box(1, 1, 2, 2)],
            crs="EPSG:2157",
        )

        with patch(
            "pycsodata.ungeneralised._download_feature_service",
            return_value=tailte_gdf,
        ):
            from pycsodata.ungeneralised import _merge_gaeltacht_county_title

            result = _merge_gaeltacht_county_title(
                "af2d32358e02fff16dfe1f54ecc5225d",
                cso_attrs,
            )

            assert isinstance(result, gpd.GeoDataFrame)
            assert len(result) == 2
            assert not result.geometry.isna().any()

    def test_provinces_ulster_mapping(self):
        """Test that 'Ulster' is mapped to 'Ulster (part of)'."""
        cso_attrs = pd.DataFrame(
            {
                "code": ["a", "b"],
                "en": ["Connacht", "Ulster (part of)"],
            }
        )
        tailte_gdf = gpd.GeoDataFrame(
            {"PROVINCE": ["Connacht", "Ulster"]},
            geometry=[box(0, 0, 1, 1), box(1, 1, 2, 2)],
            crs="EPSG:2157",
        )

        with patch(
            "pycsodata.ungeneralised._download_feature_service",
            return_value=tailte_gdf,
        ):
            from pycsodata.ungeneralised import _merge_provinces_ulster

            result = _merge_provinces_ulster(
                "9ae1df4db5df6639ed4724f3a1b314ee",
                cso_attrs,
            )

            assert isinstance(result, gpd.GeoDataFrame)
            assert len(result) == 2
            ulster_row = result[result["en"] == "Ulster (part of)"]
            assert not ulster_row.geometry.isna().any()

    def test_provinces_guid_lower(self):
        """Test that GUID.lower() matches code."""
        cso_attrs = pd.DataFrame(
            {
                "code": ["abc-123", "def-456"],
                "en": ["Connacht", "Leinster"],
            }
        )
        tailte_gdf = gpd.GeoDataFrame(
            {"GUID": ["ABC-123", "DEF-456"]},
            geometry=[box(0, 0, 1, 1), box(1, 1, 2, 2)],
            crs="EPSG:2157",
        )

        with patch(
            "pycsodata.ungeneralised._download_feature_service",
            return_value=tailte_gdf,
        ):
            from pycsodata.ungeneralised import _merge_provinces_guid_lower

            result = _merge_provinces_guid_lower(
                "9f352336de5e2a0d42455237888478b7",
                cso_attrs,
            )

            assert isinstance(result, gpd.GeoDataFrame)
            assert len(result) == 2
            assert not result.geometry.isna().any()

    def test_admin_areas_with_ni(self):
        """Test that RoI and NI geometries are combined."""
        cso_attrs = pd.DataFrame(
            {
                "code": ["roi_001", "ni_001", "roi_002"],
                "en": ["Dublin", "Belfast", "Cork"],
            }
        )
        tailte_gdf = gpd.GeoDataFrame(
            {"GUID": ["roi_001", "roi_002"]},
            geometry=[box(0, 0, 1, 1), box(1, 1, 2, 2)],
            crs="EPSG:2157",
        )
        ni_gdf = gpd.GeoDataFrame(
            {"LGDFilecode": ["ni_001"]},
            geometry=[box(3, 3, 4, 4)],
            crs="EPSG:4326",
        )

        with (
            patch(
                "pycsodata.ungeneralised._download_feature_service",
                return_value=tailte_gdf,
            ),
            patch(
                "pycsodata.ungeneralised._download_ni_geojson",
                return_value=ni_gdf,
            ),
        ):
            from pycsodata.ungeneralised import _merge_admin_areas_with_ni_lgd

            result = _merge_admin_areas_with_ni_lgd(
                "c0ad28a75e6fd0c4cc76a50ba859def4",
                cso_attrs,
            )

            assert isinstance(result, gpd.GeoDataFrame)
            assert len(result) == 3
            # All rows should have geometry (all codes matched)
            assert not result.geometry.isna().any()

    def test_counties_with_ni_derry_mapping(self):
        """Test that LONDONDERRY is mapped to DERRY/LONDONDERRY."""
        cso_attrs = pd.DataFrame(
            {
                "code": ["a", "b"],
                "en": ["Dublin", "DERRY/LONDONDERRY"],
            }
        )
        tailte_gdf = gpd.GeoDataFrame(
            {"ENG_NAME_VALUE": ["Dublin"]},
            geometry=[box(0, 0, 1, 1)],
            crs="EPSG:2157",
        )
        ni_gdf = gpd.GeoDataFrame(
            {"CountyName": ["LONDONDERRY"]},
            geometry=[box(3, 3, 4, 4)],
            crs="EPSG:4326",
        )

        with (
            patch(
                "pycsodata.ungeneralised._download_feature_service",
                return_value=tailte_gdf,
            ),
            patch(
                "pycsodata.ungeneralised._download_ni_geojson",
                return_value=ni_gdf,
            ),
        ):
            from pycsodata.ungeneralised import _merge_counties_with_ni

            result = _merge_counties_with_ni(
                "526860fb25a6567dae4dbaff1e6d48d3",
                cso_attrs,
            )

            assert isinstance(result, gpd.GeoDataFrame)
            derry = result[result["en"] == "DERRY/LONDONDERRY"]
            assert not derry.geometry.isna().any()


# =============================================================================
# Test Main Public Function
# =============================================================================


class TestCreateUngeneralisedGeoDataFrame:
    """Tests for the create_ungeneralised_geodataframe function."""

    def test_raises_when_no_spatial_url(self):
        """Test that SpatialError is raised when no spatial URL."""
        df = pd.DataFrame({"County": ["Dublin"], "value": [100]})
        with pytest.raises(SpatialError, match="no spatial information"):
            create_ungeneralised_geodataframe(df, None, "County")

    def test_raises_when_no_spatial_key(self):
        """Test that SpatialError is raised when no spatial key."""
        df = pd.DataFrame({"County": ["Dublin"], "value": [100]})
        with pytest.raises(SpatialError, match="no spatial information"):
            create_ungeneralised_geodataframe(df, "http://example.com/abc123", None)

    def test_raises_for_unavailable_filecode(self):
        """Test that SpatialError is raised for unavailable filecodes."""
        df = pd.DataFrame({"County": ["Dublin"], "value": [100]})
        url = (
            "https://ws.cso.ie/public/api.static/"
            "PxStat.Data.GeoMap_API.Read/"
            "09a3c5e1c9d0ac5fc1ac4cfaa4506e51"
        )
        with pytest.raises(SpatialError, match="not available"):
            create_ungeneralised_geodataframe(df, url, "County")

    def test_raises_for_unknown_filecode(self):
        """Test that SpatialError is raised for unknown filecodes."""
        df = pd.DataFrame({"County": ["Dublin"], "value": [100]})
        url = (
            "https://ws.cso.ie/public/api.static/"
            "PxStat.Data.GeoMap_API.Read/"
            "ffffffffffffffffffffffffffffffff"
        )
        with pytest.raises(SpatialError, match="not recognised"):
            create_ungeneralised_geodataframe(df, url, "County")

    def test_suggests_default_for_unavailable(self):
        """Test that error message suggests using ungeneralised=False."""
        df = pd.DataFrame({"County": ["Dublin"], "value": [100]})
        url = (
            "https://ws.cso.ie/public/api.static/"
            "PxStat.Data.GeoMap_API.Read/"
            "09a3c5e1c9d0ac5fc1ac4cfaa4506e51"
        )
        with pytest.raises(SpatialError, match="ungeneralised=False"):
            create_ungeneralised_geodataframe(df, url, "County")

    def test_end_to_end_simple_merge(self):
        """Test full pipeline with a simple merge (mocked downloads)."""
        # Mock CSO GeoJSON
        cso_geojson = {
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [0, 0]},
                    "properties": {"code": "abc123", "en": "Dublin"},
                },
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [1, 1]},
                    "properties": {"code": "def456", "en": "Cork"},
                },
            ],
        }

        # Statistical data
        df = pd.DataFrame(
            {
                "County": ["Dublin", "Cork"],
                "County ID": ["abc123", "def456"],
                "value": [100, 200],
            }
        )

        # Mock Tailte geodata (more detailed geometry)
        tailte_gdf = gpd.GeoDataFrame(
            {"GUID": ["abc123", "def456"]},
            geometry=[
                Polygon([(0, 0), (1, 0), (1, 1), (0.5, 1.5), (0, 1), (0, 0)]),
                Polygon([(2, 2), (3, 2), (3, 3), (2.5, 3.5), (2, 3), (2, 2)]),
            ],
            crs="EPSG:2157",
        )

        # Known simple filecode
        url = (
            "https://ws.cso.ie/public/api.static/"
            "PxStat.Data.GeoMap_API.Read/"
            "440c36d3b86e067e97ffb2fabf55900e"
        )

        with (
            patch("pycsodata.ungeneralised.fetch_json", return_value=cso_geojson),
            patch(
                "pycsodata.ungeneralised._download_feature_service",
                return_value=tailte_gdf,
            ),
        ):
            result = create_ungeneralised_geodataframe(df, url, "County")

            assert isinstance(result, gpd.GeoDataFrame)
            assert len(result) == 2
            assert "geometry" in result.columns
            assert "value" in result.columns

    def test_end_to_end_logs_copyright(self):
        """Test that copyright info is logged during end-to-end pipeline."""
        cso_geojson = {
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [0, 0]},
                    "properties": {"code": "abc123", "en": "Dublin"},
                },
            ],
        }
        df = pd.DataFrame({"County": ["Dublin"], "County ID": ["abc123"], "value": [100]})
        tailte_gdf = gpd.GeoDataFrame(
            {"GUID": ["abc123"]},
            geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0.5, 1.5), (0, 1), (0, 0)])],
            crs="EPSG:2157",
        )
        url = (
            "https://ws.cso.ie/public/api.static/"
            "PxStat.Data.GeoMap_API.Read/"
            "440c36d3b86e067e97ffb2fabf55900e"
        )

        with (
            patch("builtins.print") as mock_print,
            patch("pycsodata.ungeneralised.fetch_json", return_value=cso_geojson),
            patch(
                "pycsodata.ungeneralised._download_feature_service",
                return_value=tailte_gdf,
            ),
            patch(
                "pycsodata.ungeneralised._read_cached_copyright",
                return_value=(None, None),
            ),
        ):
            create_ungeneralised_geodataframe(df, url, "County")

        calls = [str(c) for c in mock_print.call_args_list]
        assert any("Licence" in c for c in calls)
        assert any("CC BY 4.0" in c for c in calls)


# =============================================================================
# Test CSODataset Integration
# =============================================================================


class TestCSODatasetUngeneralised:
    """Tests for the ungeneralised parameter on CSODataset.gdf()."""

    def test_ungeneralised_default_uses_standard_gdf(self):
        """Test that ungeneralised=False uses create_geodataframe."""
        with (
            patch("pycsodata.dataset.load_metadata") as mock_meta,
            patch("pycsodata.dataset.extract_spatial_info") as mock_spatial,
            patch("pycsodata.dataset.create_geodataframe") as mock_create_gdf,
        ):
            mock_meta.return_value = {"dimension": {}}
            mock_spatial.return_value = MagicMock(
                url="http://example.com/abc",
                key="County",
                is_available=True,
            )
            mock_create_gdf.return_value = gpd.GeoDataFrame(
                {"County": ["Dublin"], "value": [1]},
                geometry=[Point(0, 0)],
            )

            from pycsodata.dataset import CSODataset

            dataset = CSODataset.__new__(CSODataset)
            dataset.table_code = "TEST01"
            dataset._include_ids = MagicMock()
            dataset._include_ids.configure_mock(**{"__eq__.return_value": False})
            dataset._spatial_info = mock_spatial.return_value
            dataset._is_met_dataset = False
            dataset._cached_gdf = None
            dataset._cached_gdf_ungeneralised = None
            dataset._filters = None
            dataset._drop_filtered_cols = False
            dataset._cache_enabled = True
            dataset._raw_metadata = {}
            dataset._sanitise = False
            dataset._cached_base_df = pd.DataFrame({"County": ["Dublin"], "value": [1]})
            dataset._cached_df = None
            dataset._convert_dates = False
            dataset._drop_national_data = False

            dataset.gdf(ungeneralised=False)
            assert mock_create_gdf.called

    def test_met_dataset_rejects_ungeneralised(self):
        """Test that Met Éireann datasets reject ungeneralised=True."""
        from pycsodata.dataset import CSODataset

        dataset = CSODataset.__new__(CSODataset)
        dataset.table_code = "MTM01"
        dataset._is_met_dataset = True
        dataset._spatial_info = MagicMock(is_available=True)
        dataset._cached_gdf = None
        dataset._cached_gdf_ungeneralised = None

        with pytest.raises(SpatialError, match="Met Éireann"):
            dataset.gdf(ungeneralised=True)

    def test_ungeneralised_uses_ungeneralised_geometries(self):
        """Test that ungeneralised=True uses create_ungeneralised_geodataframe."""
        with (
            patch("pycsodata.dataset.load_metadata") as mock_meta,
            patch("pycsodata.dataset.extract_spatial_info") as mock_spatial,
            patch("pycsodata.dataset.create_ungeneralised_geodataframe") as mock_create,
        ):
            mock_meta.return_value = {"dimension": {}}
            mock_spatial.return_value = MagicMock(
                url="http://example.com/abc",
                key="County",
                is_available=True,
            )
            mock_create.return_value = gpd.GeoDataFrame(
                {"County": ["Dublin"], "value": [1]},
                geometry=[Point(0, 0)],
            )

            from pycsodata.dataset import CSODataset

            dataset = CSODataset.__new__(CSODataset)
            dataset.table_code = "TEST01"
            dataset._include_ids = MagicMock()
            dataset._include_ids.configure_mock(**{"__eq__.return_value": False})
            dataset._spatial_info = mock_spatial.return_value
            dataset._is_met_dataset = False
            dataset._cached_gdf = None
            dataset._cached_gdf_ungeneralised = None
            dataset._filters = None
            dataset._drop_filtered_cols = False
            dataset._cache_enabled = True
            dataset._raw_metadata = {}
            dataset._sanitise = False
            dataset._cached_base_df = pd.DataFrame({"County": ["Dublin"], "value": [1]})
            dataset._cached_df = None
            dataset._convert_dates = False
            dataset._drop_national_data = False

            dataset.gdf(ungeneralised=True)
            assert mock_create.called
            # Should have passed force_reload=False
            call_kwargs = mock_create.call_args
            assert call_kwargs.kwargs.get("force_reload") is False

    def test_force_reload_clears_ungeneralised_cache(self):
        """Test that force_reload_geometries clears in-memory cache."""
        with (
            patch("pycsodata.dataset.load_metadata") as mock_meta,
            patch("pycsodata.dataset.extract_spatial_info") as mock_spatial,
            patch("pycsodata.dataset.create_ungeneralised_geodataframe") as mock_create,
        ):
            mock_meta.return_value = {"dimension": {}}
            mock_spatial.return_value = MagicMock(
                url="http://example.com/abc",
                key="County",
                is_available=True,
            )
            gdf1 = gpd.GeoDataFrame(
                {"County": ["Dublin"], "value": [1]},
                geometry=[Point(0, 0)],
            )
            mock_create.return_value = gdf1

            from pycsodata.dataset import CSODataset

            dataset = CSODataset.__new__(CSODataset)
            dataset.table_code = "TEST01"
            dataset._include_ids = MagicMock()
            dataset._include_ids.configure_mock(**{"__eq__.return_value": False})
            dataset._spatial_info = mock_spatial.return_value
            dataset._is_met_dataset = False
            dataset._cached_gdf = None
            dataset._cached_gdf_ungeneralised = gdf1  # Previously cached
            dataset._filters = None
            dataset._drop_filtered_cols = False
            dataset._cache_enabled = True
            dataset._raw_metadata = {}
            dataset._sanitise = False
            dataset._cached_base_df = pd.DataFrame({"County": ["Dublin"], "value": [1]})
            dataset._cached_df = None
            dataset._convert_dates = False
            dataset._drop_national_data = False

            dataset.gdf(ungeneralised=True, force_reload_geometries=True)

            assert mock_create.called  # force_reload=True was passed
            call_kwargs = mock_create.call_args
            assert call_kwargs.kwargs.get("force_reload") is True


# =============================================================================
# Test Build Ungeneralised GDF Dispatch
# =============================================================================


class TestBuildUngeneralisedGdf:
    """Tests for the _build_ungeneralised_gdf dispatch function."""

    def test_dispatches_to_simple_merge(self):
        """Test that simple filecodes dispatch to _simple_merge."""
        cso_attrs = _make_cso_attrs(code=["a"], en=["A"])

        with patch("pycsodata.ungeneralised._simple_merge") as mock_simple:
            mock_simple.return_value = gpd.GeoDataFrame({"code": ["a"]}, geometry=[Point(0, 0)])
            _build_ungeneralised_gdf("440c36d3b86e067e97ffb2fabf55900e", cso_attrs)
            assert mock_simple.called

    def test_dispatches_to_complex_merge(self):
        """Test that complex filecodes dispatch to _complex_merge."""
        cso_attrs = _make_cso_attrs()

        with patch("pycsodata.ungeneralised._complex_merge") as mock_complex:
            mock_complex.return_value = gpd.GeoDataFrame({"en": ["a"]}, geometry=[Point(0, 0)])
            _build_ungeneralised_gdf("c0ad28a75e6fd0c4cc76a50ba859def4", cso_attrs)
            assert mock_complex.called

    def test_raises_for_unmapped_filecode(self):
        """Test that unmapped filecodes raise SpatialError."""
        cso_attrs = _make_cso_attrs()

        with pytest.raises(SpatialError, match="No merge strategy"):
            _build_ungeneralised_gdf("0" * 32, cso_attrs)


# =============================================================================
# Test Metadata Caching
# =============================================================================


class TestCacheFeatureServiceMetadata:
    """Tests for _cache_feature_service_metadata."""

    def test_saves_json_and_txt_when_no_xml_and_no_copyright_in_json(self, tmp_path):
        """Test that JSON + text fallback are saved when XML unavailable
        and JSON lacks copyright."""

        # Mock: /metadata returns non-XML, service info returns JSON without copyrightText
        def _mock_get(url, **kwargs):
            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status = MagicMock()
            if "/metadata" in url:
                resp.text = "not xml"
                resp.content = b"not xml"
                resp.headers = {"Content-Type": "text/html"}
            else:
                resp.json.return_value = {"name": "Test Layer", "type": "Feature Layer"}
                resp.headers = {"Content-Type": "application/json"}
            return resp

        with patch("pycsodata.ungeneralised.requests.get", side_effect=_mock_get):
            _cache_feature_service_metadata("https://example.com/FeatureServer/0", tmp_path)

        # JSON is saved
        props_file = tmp_path / "metadata" / "properties.json"
        assert props_file.exists()
        with props_file.open() as f:
            data = json.load(f)
        assert data["name"] == "Test Layer"

        # Text fallback is also saved because copyrightText is missing
        txt_file = tmp_path / "metadata" / "metadata.txt"
        assert txt_file.exists()
        txt = txt_file.read_text()
        assert "Copyright:" in txt

    def test_saves_json_only_when_copyright_present(self, tmp_path):
        """Test that only JSON is saved when copyrightText is present."""
        from pycsodata.ungeneralised import _cache_feature_service_metadata

        def _mock_get(url, **kwargs):
            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status = MagicMock()
            if "/metadata" in url:
                resp.text = "not xml"
                resp.content = b"not xml"
                resp.headers = {"Content-Type": "text/html"}
            else:
                resp.json.return_value = {
                    "name": "Test",
                    "copyrightText": "Some Copyright",
                }
                resp.headers = {"Content-Type": "application/json"}
            return resp

        with patch("pycsodata.ungeneralised.requests.get", side_effect=_mock_get):
            _cache_feature_service_metadata("https://example.com/FeatureServer/0", tmp_path)

        # JSON is saved with copyright
        props_file = tmp_path / "metadata" / "properties.json"
        assert props_file.exists()

        # Text fallback is NOT saved because JSON has copyright
        txt_file = tmp_path / "metadata" / "metadata.txt"
        assert not txt_file.exists()

    def test_downloads_xml_metadata(self, tmp_path):
        """Test that XML metadata is saved when available."""
        from pycsodata.ungeneralised import _cache_feature_service_metadata

        xml_content = b'<?xml version="1.0"?><metadata><title>Test</title></metadata>'

        def _mock_get(url, **kwargs):
            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status = MagicMock()
            if "/metadata" in url:
                resp.text = xml_content.decode()
                resp.content = xml_content
                resp.headers = {"Content-Type": "text/xml"}
            else:
                resp.json.return_value = {}
            return resp

        with patch("pycsodata.ungeneralised.requests.get", side_effect=_mock_get):
            _cache_feature_service_metadata("https://example.com/FeatureServer/0", tmp_path)

        xml_file = tmp_path / "metadata" / "metadata.xml"
        assert xml_file.exists()
        assert xml_content in xml_file.read_bytes()

    def test_txt_fallback_when_both_xml_and_json_fail(self, tmp_path):
        """Test that text fallback is created when both XML and JSON fail."""
        from pycsodata.ungeneralised import _cache_feature_service_metadata

        def _mock_get(url, **kwargs):
            raise ConnectionError("Network down")

        with patch("pycsodata.ungeneralised.requests.get", side_effect=_mock_get):
            _cache_feature_service_metadata("https://example.com/FeatureServer/0", tmp_path)

        txt_file = tmp_path / "metadata" / "metadata.txt"
        assert txt_file.exists()
        txt = txt_file.read_text()
        assert "URL: https://example.com/FeatureServer/0" in txt
        assert "Copyright:" in txt
        assert "Licence:" in txt


# =============================================================================
# Test Gaeltacht Language Planning Area dissolve merge
# =============================================================================


class TestGaeltachtLpDissolve:
    """Tests for _merge_gaeltacht_lp_dissolve."""

    def test_dissolves_by_eng_name_value(self):
        """Test that features are dissolved by ENG_NAME_VALUE."""
        cso_attrs = pd.DataFrame(
            {
                "code": ["a", "b"],
                "en": ["Gaeltacht A", "Gaeltacht B"],
            }
        )
        # Two rows with same ENG_NAME_VALUE should be dissolved
        tailte_gdf = gpd.GeoDataFrame(
            {"ENG_NAME_VALUE": ["Gaeltacht A", "Gaeltacht A", "Gaeltacht B"]},
            geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1), box(3, 3, 4, 4)],
            crs="EPSG:2157",
        )

        with patch(
            "pycsodata.ungeneralised._download_feature_service",
            return_value=tailte_gdf,
        ):
            result = _merge_gaeltacht_lp_dissolve(
                "9b504eb50b10e0087c2b4913ade4d10d",
                cso_attrs,
            )

        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == 2
        assert "en" in result.columns
        gaeltacht_a = result[result["en"] == "Gaeltacht A"]
        assert not gaeltacht_a.geometry.isna().any()

    def test_unmatched_rows_get_null_geometry(self):
        """Test that unmatched rows get null geometry."""
        cso_attrs = pd.DataFrame(
            {
                "code": ["a", "b"],
                "en": ["Gaeltacht A", "Gaeltacht Missing"],
            }
        )
        tailte_gdf = gpd.GeoDataFrame(
            {"ENG_NAME_VALUE": ["Gaeltacht A"]},
            geometry=[box(0, 0, 1, 1)],
            crs="EPSG:2157",
        )

        with patch(
            "pycsodata.ungeneralised._download_feature_service",
            return_value=tailte_gdf,
        ):
            result = _merge_gaeltacht_lp_dissolve(
                "9b504eb50b10e0087c2b4913ade4d10d",
                cso_attrs,
            )

        assert len(result) == 2
        missing = result[result["en"] == "Gaeltacht Missing"]
        assert missing.geometry.isna().all()


# =============================================================================
# Test _download_feature_service error paths
# =============================================================================


class TestDownloadFeatureServiceErrorPaths:
    """Tests for error handling in _download_feature_service."""

    def test_non_spatial_error_is_wrapped_in_spatial_error(self, tmp_path):
        """Test that unexpected exceptions are wrapped in SpatialError."""
        from pycsodata.ungeneralised import _download_feature_service

        def _broken_get(url, **kwargs):
            raise ValueError("Unexpected connection error")

        mock_session = MagicMock()
        mock_session.get.side_effect = _broken_get
        mock_session.close = MagicMock()

        with (
            patch("pycsodata.ungeneralised._create_session", return_value=mock_session),
            patch("pycsodata.ungeneralised._get_cache_dir", return_value=tmp_path),
            pytest.raises(SpatialError, match="Failed to download Feature Service"),
        ):
            _download_feature_service("https://example.com/FeatureServer/0")

    def test_spatial_error_is_reraised_directly(self, tmp_path):
        """Test that SpatialError is re-raised without wrapping."""
        from pycsodata.ungeneralised import _download_feature_service

        def _broken_get(url, **kwargs):
            raise SpatialError("Inner spatial error")

        mock_session = MagicMock()
        mock_session.get.side_effect = _broken_get
        mock_session.close = MagicMock()

        with (
            patch("pycsodata.ungeneralised._create_session", return_value=mock_session),
            patch("pycsodata.ungeneralised._get_cache_dir", return_value=tmp_path),
            pytest.raises(SpatialError, match="Inner spatial error"),
        ):
            _download_feature_service("https://example.com/FeatureServer/0")

    def test_out_fields_not_in_gdf_returns_full_gdf(self, tmp_path):
        """Test that when requested field is absent, full gdf is returned."""
        mock_session = _make_mock_session([{"OTHER_FIELD": "abc"}])

        with (
            patch("pycsodata.ungeneralised._create_session", return_value=mock_session),
            patch("pycsodata.ungeneralised._get_cache_dir", return_value=tmp_path),
        ):
            from pycsodata.ungeneralised import _download_feature_service

            gdf = _download_feature_service(
                "https://example.com/FeatureServer/0",
                out_fields="MISSING_FIELD",  # Field not in response
            )

        assert isinstance(gdf, gpd.GeoDataFrame)
        # Full gdf is returned because MISSING_FIELD is not in gdf.columns
        assert "geometry" in gdf.columns


# =============================================================================
# Test _download_ni_geojson error paths
# =============================================================================


class TestDownloadNIGeoJSONErrorPaths:
    """Tests for error handling in _download_ni_geojson."""

    def test_raises_spatial_error_on_failure(self, tmp_path):
        """Test that download failures are wrapped in SpatialError."""
        from pycsodata.ungeneralised import _download_ni_geojson

        with (
            patch("pycsodata.ungeneralised._get_cache_dir", return_value=tmp_path),
            patch(
                "geopandas.read_file",
                side_effect=ConnectionError("Network unreachable"),
            ),
            pytest.raises(SpatialError, match="Failed to download NI geometry"),
        ):
            _download_ni_geojson("https://example.com/ni.geojson")

    def test_loads_from_cache_when_exists(self, tmp_path):
        """Test that NI geometry is loaded from cache on subsequent calls."""
        from pycsodata.ungeneralised import _download_ni_geojson

        ni_gdf = gpd.GeoDataFrame(
            {"CountyName": ["Down"]},
            geometry=[box(0, 0, 1, 1)],
            crs="EPSG:4326",
        )

        # Pre-populate the cache
        cache_key = _url_cache_key("https://example.com/ni.geojson")
        cache_subdir = tmp_path / f"osni_{cache_key}"
        cache_subdir.mkdir(parents=True)
        ni_gdf.to_file(cache_subdir / "features.gpkg", driver="GPKG")

        with patch("pycsodata.ungeneralised._get_cache_dir", return_value=tmp_path):
            result = _download_ni_geojson("https://example.com/ni.geojson")

        assert isinstance(result, gpd.GeoDataFrame)
        assert "CountyName" in result.columns


# =============================================================================
# Test create_ungeneralised_geodataframe additional error paths
# =============================================================================


class TestCreateUngeneralisedAdditionalCases:
    """Tests for additional error/edge cases in create_ungeneralised_geodataframe."""

    def test_raises_when_cso_geojson_has_no_features(self):
        """Test that SpatialError is raised when CSO GeoJSON has no features."""
        df = pd.DataFrame({"County": ["Dublin"], "value": [100]})
        url = (
            "https://ws.cso.ie/public/api.static/"
            "PxStat.Data.GeoMap_API.Read/"
            "440c36d3b86e067e97ffb2fabf55900e"
        )

        with (
            patch("pycsodata.ungeneralised.fetch_json", return_value={"features": []}),
            pytest.raises(SpatialError, match="No features found"),
        ):
            create_ungeneralised_geodataframe(df, url, "County")

    def test_raises_when_merge_returns_none(self):
        """Test that SpatialError is raised when _merge_dataframes returns None."""
        df = pd.DataFrame({"County": ["Dublin"], "County ID": ["abc123"], "value": [100]})
        url = (
            "https://ws.cso.ie/public/api.static/"
            "PxStat.Data.GeoMap_API.Read/"
            "440c36d3b86e067e97ffb2fabf55900e"
        )
        cso_geojson = {
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [0, 0]},
                    "properties": {"code": "abc123"},
                }
            ]
        }
        tailte_gdf = gpd.GeoDataFrame(
            {"GUID": ["abc123"]},
            geometry=[box(0, 0, 1, 1)],
            crs="EPSG:2157",
        )

        with (
            patch("pycsodata.ungeneralised.fetch_json", return_value=cso_geojson),
            patch(
                "pycsodata.ungeneralised._download_feature_service",
                return_value=tailte_gdf,
            ),
            patch("pycsodata.ungeneralised._merge_dataframes", return_value=None),
            pytest.raises(SpatialError, match="Spatial merge with ungeneralised geometry failed"),
        ):
            create_ungeneralised_geodataframe(df, url, "County")

    def test_converts_key_error_to_spatial_error(self):
        """Test that KeyError during merge is converted to SpatialError."""
        df = pd.DataFrame({"County": ["Dublin"], "value": [100]})
        url = (
            "https://ws.cso.ie/public/api.static/"
            "PxStat.Data.GeoMap_API.Read/"
            "440c36d3b86e067e97ffb2fabf55900e"
        )
        cso_geojson = {
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [0, 0]},
                    "properties": {"code": "abc123"},
                }
            ]
        }
        tailte_gdf = gpd.GeoDataFrame(
            {"GUID": ["abc123"]},
            geometry=[box(0, 0, 1, 1)],
            crs="EPSG:2157",
        )

        with (
            patch("pycsodata.ungeneralised.fetch_json", return_value=cso_geojson),
            patch(
                "pycsodata.ungeneralised._download_feature_service",
                return_value=tailte_gdf,
            ),
            patch(
                "pycsodata.ungeneralised._merge_dataframes",
                side_effect=KeyError("missing_key"),
            ),
            pytest.raises(SpatialError, match="Error creating ungeneralised GeoDataFrame"),
        ):
            create_ungeneralised_geodataframe(df, url, "County")

    def test_logs_warning_when_ungeneralised_less_detailed(self):
        """Test warning is logged when ungeneralised coord count is not larger."""
        df = pd.DataFrame({"County": ["Dublin"], "County ID": ["abc123"], "value": [100]})
        url = (
            "https://ws.cso.ie/public/api.static/"
            "PxStat.Data.GeoMap_API.Read/"
            "440c36d3b86e067e97ffb2fabf55900e"
        )
        # CSO geojson with a more detailed polygon than the ungeneralised one
        cso_geojson = {
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [[0, 0], [1, 0], [2, 0], [3, 0], [3, 1], [2, 1], [1, 1], [0, 1], [0, 0]]
                        ],
                    },
                    "properties": {"code": "abc123"},
                }
            ]
        }
        # Simple point geometry for ungeneralised (1 coord, less than CSO default)
        tailte_gdf = gpd.GeoDataFrame(
            {"GUID": ["abc123"]},
            geometry=[box(0, 0, 0.001, 0.001)],  # triangle = 4 coords
            crs="EPSG:2157",
        )
        merged_gdf = gpd.GeoDataFrame(
            {"County": ["Dublin"], "value": [100]},
            geometry=[box(0, 0, 0.001, 0.001)],
            crs="EPSG:2157",
        )

        with (
            patch("pycsodata.ungeneralised.logger") as mock_logger,
            patch("pycsodata.ungeneralised.fetch_json", return_value=cso_geojson),
            patch(
                "pycsodata.ungeneralised._download_feature_service",
                return_value=tailte_gdf,
            ),
            patch("pycsodata.ungeneralised._merge_dataframes", return_value=merged_gdf),
            # Patch _count_coordinates to make ungeneralised appear LESS detailed
            patch(
                "pycsodata.ungeneralised._count_coordinates",
                side_effect=[100, 50],  # default=100, ungeneralised=50
            ),
        ):
            create_ungeneralised_geodataframe(df, url, "County")

        warning_calls = [str(c) for c in mock_logger.warning.call_args_list]
        assert any("not more than the default" in c for c in warning_calls)

    def test_logs_info_when_more_detailed(self):
        """Test info is logged when ungeneralised is more detailed than default."""
        df = pd.DataFrame({"County": ["Dublin"], "County ID": ["abc123"], "value": [100]})
        url = (
            "https://ws.cso.ie/public/api.static/"
            "PxStat.Data.GeoMap_API.Read/"
            "440c36d3b86e067e97ffb2fabf55900e"
        )
        cso_geojson = {
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [0, 0]},
                    "properties": {"code": "abc123"},
                }
            ]
        }
        tailte_gdf = gpd.GeoDataFrame(
            {"GUID": ["abc123"]},
            geometry=[box(0, 0, 1, 1)],
            crs="EPSG:2157",
        )
        merged_gdf = gpd.GeoDataFrame(
            {"County": ["Dublin"], "value": [100]},
            geometry=[box(0, 0, 1, 1)],
            crs="EPSG:2157",
        )

        with (
            patch("builtins.print") as mock_print,
            patch("pycsodata.ungeneralised.fetch_json", return_value=cso_geojson),
            patch(
                "pycsodata.ungeneralised._download_feature_service",
                return_value=tailte_gdf,
            ),
            patch("pycsodata.ungeneralised._merge_dataframes", return_value=merged_gdf),
            patch(
                "pycsodata.ungeneralised._count_coordinates",
                side_effect=[50, 500],  # default=50, ungeneralised=500
            ),
        ):
            create_ungeneralised_geodataframe(df, url, "County")

        info_calls = [str(c) for c in mock_print.call_args_list]
        assert any("coordinates vs" in c for c in info_calls)


# =============================================================================
# Test _complex_merge dispatch: none case
# =============================================================================


class TestComplexMergeDispatchNone:
    """Tests for _complex_merge when no dispatch function exists."""

    def test_raises_for_unknown_filecode_in_complex(self):
        """Test that SpatialError is raised when filecode has no dispatch fn."""
        # Patch _COMPLEX_MAPPINGS to include an unknown filecode
        cso_attrs = pd.DataFrame({"code": ["a"], "en": ["A"]})

        with (
            patch.dict(
                "pycsodata.ungeneralised._COMPLEX_MAPPINGS",
                {"unknownfilecode1234567890123456": ("https://example.com/FS/0", None)},
            ),
            pytest.raises(SpatialError, match="No complex merge function"),
        ):
            _complex_merge("unknownfilecode1234567890123456", cso_attrs)


# =============================================================================
# Test _count_coordinates fallback (shapely path)
# =============================================================================


class TestCountCoordinatesFallbackPath:
    """Tests for the shapely-based fallback in _count_coordinates."""

    def test_fallback_path_via_shapely(self):
        """Test that shapely.get_num_coordinates is used as fallback."""
        gdf = gpd.GeoDataFrame(
            {"a": [1]},
            geometry=[box(0, 0, 1, 1)],
        )
        # Patch GeoSeries.count_coordinates to raise AttributeError (simulating old geopandas)
        with patch.object(
            type(gdf.geometry),
            "count_coordinates",
            new_callable=lambda: property(
                lambda self: (_ for _ in ()).throw(AttributeError("no count_coordinates"))
            ),
        ):
            result = _count_coordinates(gdf)
        # Result should be > 0 (5 for a box)
        assert result > 0

    def test_returns_zero_when_all_null(self):
        """Test that empty geodataframe returns 0."""
        gdf = gpd.GeoDataFrame({"a": []}, geometry=gpd.GeoSeries([], dtype="geometry"))
        result = _count_coordinates(gdf)
        assert result == 0


# =============================================================================
# Test _read_cached_copyright: XML present but parses to (None, None)
# =============================================================================


class TestReadCachedCopyrightFallthrough:
    """Tests for _read_cached_copyright fallthrough behaviour."""

    def test_falls_through_xml_to_json_when_xml_parses_empty(self, tmp_path):
        """Test that JSON is tried when XML parses to (None, None)."""
        url = "https://example.com/FeatureServer/0"
        cache_key = _url_cache_key(url)
        md_dir = tmp_path / f"tailte_{cache_key}" / "metadata"
        md_dir.mkdir(parents=True)

        # XML that parses to (None, None) - no credit/useLimit elements
        (md_dir / "metadata.xml").write_text(
            '<?xml version="1.0"?><metadata><other>stuff</other></metadata>'
        )
        (md_dir / "properties.json").write_text(
            json.dumps({"copyrightText": "JSON Fallback Copyright"})
        )

        with patch("pycsodata.ungeneralised._get_cache_dir", return_value=tmp_path):
            copyright_, _ = _read_cached_copyright(url, prefix="tailte")

        assert copyright_ == "JSON Fallback Copyright"

    def test_falls_through_json_to_txt_when_json_empty(self, tmp_path):
        """Test that text is tried when JSON parses to (None, None)."""
        url = "https://example.com/FeatureServer/0"
        cache_key = _url_cache_key(url)
        md_dir = tmp_path / f"tailte_{cache_key}" / "metadata"
        md_dir.mkdir(parents=True)

        # JSON with no copyrightText
        (md_dir / "properties.json").write_text(json.dumps({"name": "Test"}))
        (md_dir / "metadata.txt").write_text("Copyright: TXT Copyright\nLicence: TXT Licence\n")

        with patch("pycsodata.ungeneralised._get_cache_dir", return_value=tmp_path):
            copyright_, licence_ = _read_cached_copyright(url, prefix="tailte")

        assert copyright_ == "TXT Copyright"
        assert licence_ == "TXT Licence"

    def test_returns_none_none_when_all_sources_empty(self, tmp_path):
        """Test that (None, None) is returned when all sources parse empty."""
        url = "https://example.com/FeatureServer/0"
        cache_key = _url_cache_key(url)
        md_dir = tmp_path / f"tailte_{cache_key}" / "metadata"
        md_dir.mkdir(parents=True)

        # All sources parse to (None, None)
        (md_dir / "metadata.xml").write_text(
            '<?xml version="1.0"?><metadata><other>stuff</other></metadata>'
        )
        (md_dir / "properties.json").write_text(json.dumps({"name": "No Copyright"}))
        (md_dir / "metadata.txt").write_text("")  # Empty text file

        with patch("pycsodata.ungeneralised._get_cache_dir", return_value=tmp_path):
            result = _read_cached_copyright(url, prefix="tailte")

        assert result == (None, None)


# =============================================================================
# Additional _download_feature_service caching tests
# =============================================================================


class TestDownloadFeatureServiceCachingEdgeCases:
    """Additional caching edge cases for _download_feature_service."""

    def test_star_out_fields_returns_all_columns(self, tmp_path):
        """Test that out_fields='*' returns all columns in the downloaded gdf."""
        mock_session = _make_mock_session([{"GUID": "abc123", "NAME": "Dublin"}])

        with (
            patch("pycsodata.ungeneralised._create_session", return_value=mock_session),
            patch("pycsodata.ungeneralised._get_cache_dir", return_value=tmp_path),
        ):
            from pycsodata.ungeneralised import _download_feature_service

            gdf = _download_feature_service(
                "https://example.com/FeatureServer/0",
                out_fields="*",
            )
            assert isinstance(gdf, gpd.GeoDataFrame)
            assert "geometry" in gdf.columns
