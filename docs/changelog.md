# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-07-20

### Added

**Dose quality metrics** drawn from the radiotherapy literature:

- `comparison.compare_dvh_score(reference, evaluated, targets, oars)` — Complete DVH-criterion mean absolute error over target D1/D95/D99 and OAR mean-dose/D0.1cc criteria.
- `compute_dvh_auc(dose, structure, num_bins, normalize, dose_range)` — Area under the DVH curve via trapezoidal integration. Single-distribution metric; higher value = more volume at higher dose. *(metrics/dvh.py)*
- `compute_rtog_conformity_index(dose, target, prescription_dose)` — RTOG CI = V_Rx / V_target. The ICRU/RTOG 90-05 standard conformity definition, distinct from the existing ICRU CI (V_target_rx / V_rx). *(metrics/conformity.py)*
- `compute_prescription_mae(dose, target, prescription_dose)` — Mean absolute error between actual dose and prescription dose within the target; directly measures under/overdosing. *(metrics/conformity.py)*
- `compute_variance_of_laplacian(dose, structure=None)` — Variance of the 2D Laplacian applied to the dose volume, measuring dose-gradient sharpness. *(metrics/dose_comparison.py)*
- `compare_normalized_mae(reference, evaluated, structure, normalization_value, dose_threshold_gy)` — MAE normalized by a reference value with optional high-dose threshold masking. *(metrics/dose_comparison.py)*

**Two new visualization functions** in `utils/plot.py`:

- `plot_dvh_score_breakdown` — DVH comparison overlaid with D1/D95/D99 markers and gap annotations, with the DVH Score in the title.
- `plot_dvh_auc` — DVH with filled area-under-curve and AUC value annotated.

**Documentation**:

- Expanded `docs/user-guide/quality-metrics.md` from a placeholder into a full guide covering all conformity and homogeneity indices with formulas, named-after context (Paddick, van't Riet, RTOG, ICRU), interpretation guidance, and code examples.
- Updated `docs/api/metrics.md` to document the domain-module and comparison API.
- Updated `docs/architecture/metrics-architecture.md` with correct function signatures and the two new modules.

### Changed

- **BREAKING**: Consolidated `advanced_dvh.py` into `dvh.py`.
- **BREAKING**: Replaced the flat metrics function namespace with domain
  modules. One-plan quantities use `compute_*`; reference-based plan metrics
  use `comparison.compare_*` and accept `reference` before `evaluated`.
- **BREAKING**: Standardized image and gamma comparisons on the same
  `compare_*` naming and `(reference, evaluated, ...)` parameter order.
- **BREAKING**: Removed the ambiguous target-only `compute_dvh_score`.
  `comparison.compare_dvh_score` is the single complete definition and accepts
  a target-only invocation when that subset is deliberately required.

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
