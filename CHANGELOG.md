# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-04-08

### Added

- Geocoding of weather stations used in Met Éireann datasets
- Ability to download and merge ungeneralised geometries from Tailte Éireann/OSNI
- NOTICES.md containing third-party copyright notices and license information

### Fixed

- Prevent pivot methods from dropping statistic columns that become all-NaN after filtering
- Added alternative filecode for ungeneralised 2017 constituency geometries

## [0.1.0] - 2026-02-03

### Added

- Functions for reading CSO datasets into pandas DataFrames
- Optional spatial merges into geopandas GeoDataFrames
- Functions for browsing the catalogue of CSO datasets
- Support for filtering, pivoting, and caching
