"""
Dosemetrics: A library for measuring and analyzing radiotherapy doses.

This library provides tools for:
- Reading dose and mask data from various formats
- Computing dose-volume histograms (DVH)
- Calculating dose metrics and scores
- Compliance checking against dose constraints
- Visualization utilities

Public API:
"""

# Core metrics and calculations
from .metrics import (
    Structure,
    OAR,
    Target,
    mean_dose,
    max_dose,
    volume,
    compute_dvh,
    dvh_by_structure,
    dvh_by_dose,
    dose_summary,
    dose_score,
    dvh_score,
)

# I/O utilities
from .io import (
    read_file,
    read_byte_data,
    read_from_eclipse,
    read_dose_and_mask_files,
    read_from_nifti,
    find_all_files,
)

# Utility functions
from .utils import (
    get_default_constraints,
    check_compliance,
    quality_index,
    from_dataframe,
    compare_dvh,
    plot_dvh,
)

# Version information
__version__ = "0.2.0"

# Public API
__all__ = [
    # Version
    "__version__",
    # Core metrics
    "Structure",
    "OAR",
    "Target",
    "mean_dose",
    "max_dose",
    "volume",
    "compute_dvh",
    "dvh_by_structure",
    "dvh_by_dose",
    "dose_summary",
    "dose_score",
    "dvh_score",
    # I/O functions
    "read_file",
    "read_byte_data",
    "read_from_eclipse",
    "read_dose_and_mask_files",
    "read_from_nifti",
    "find_all_files",
    # Utilities
    "get_default_constraints",
    "check_compliance",
    "quality_index",
    "from_dataframe",
    "compare_dvh",
    "plot_dvh",
]
