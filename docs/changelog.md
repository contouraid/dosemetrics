# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-08-08

### Changed
- **BREAKING**: Restructured repository to use src/ layout
- **BREAKING**: Reorganized modules into logical packages:
  - `dosemetrics.metrics`: Core dose calculation functions (DVH, scores, etc.)
  - `dosemetrics.io`: File I/O utilities for reading dose and mask data
  - `dosemetrics.utils`: Utility functions for compliance, plotting, etc.
- **BREAKING**: Moved Streamlit app code to separate `dosemetrics_app` package
- **BREAKING**: App-specific code (variations_tab.py) moved to `dosemetrics_app.tabs`
- Updated package configuration to exclude app code from PyPI distribution
- Added CLI entry point via `dosemetrics` command

### Added
- Command-line interface for basic operations
- Clear separation between library and application code
- Improved package structure following Python best practices
- Updated `pyproject.toml` to support src/ layout

### Removed
- App-specific modules are no longer part of the core library API
- Removed stray files from the root package namespace

## [0.1.x] - Previous versions
- Initial releases with flat package structure
