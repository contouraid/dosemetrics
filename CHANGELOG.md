# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2025-12-28

### Added
- **Dose Class**: New `Dose` class for managing 3D dose distributions separately from structures
- **StructureSet Class**: New `StructureSet` aggregation object for managing collections of radiotherapy structures
- **Structure Type Classification**: Proper OAR/Target/Avoidance distinction using enums
- **Bulk Operations**: Calculate statistics and DVH for all structures simultaneously via StructureSet
- **Enhanced I/O Functions**: New functions for loading data directly into StructureSet objects
  - `get_dose_and_structures_as_structure_set()`: Load from folder to StructureSet
  - `read_dose_and_mask_files_as_structure_set()`: Convert file lists to StructureSet
  - `create_structure_set_from_existing_data()`: Upgrade existing data to StructureSet
- **Factory Functions**: Convenient creation methods (`create_structure_set_from_folder()`, `create_structure_set_from_masks()`)
- **Comprehensive Test Suite**: 25+ tests for Dose class, 30+ tests for StructureSet
- **Documentation**: New tutorials for NIfTI I/O, DICOM I/O, and StructureSet usage

### Changed
- **BREAKING**: Separated dose data from Structure class - structures no longer directly contain dose data
- **BREAKING**: StructureSet objects now manage dose data for all structures
- **BREAKING**: Removed `set_dose_data()` and related methods from Structure class
- Improved multi-dose file handling - can now work with multiple RT-DOSE files per patient
- Enhanced spatial compatibility checking between dose grids and structures
- Updated all I/O functions to support the new architecture

### Fixed
- DICOM SliceThickness None handling in dose loading
- Multiple RT-DOSE file loading and management
- Spatial compatibility validation between dose and mask data

### Notes
- **Development Status**: Still in alpha - API may change in future versions
- Maintained backward compatibility for basic Structure operations
- All existing tests passing with new architecture

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
