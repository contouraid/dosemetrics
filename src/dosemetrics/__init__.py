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

# Core structure classes
from .io import (
    Structure,
    OAR,
    Target,
    StructureType,
    AvoidanceStructure,
)

# Structure set management
from .io import (
    StructureSet,
    create_structure_set_from_folder,
    create_structure_set_from_masks,
)

# Core metrics and calculations
from .metrics import (
    mean_dose,
    max_dose,
    volume,
    compute_dvh,
    dvh_by_structure,
    dvh_by_dose,
    dose_summary,
    dose_score,
    dvh_score,
    compare_predicted_doses,
    compare_quality_indices,
    compute_geometric_metrics,
    batch_dvh_analysis,
    process_subject_folder,
)

# I/O utilities
from .io import (
    read_file,
    read_byte_data,
    read_from_eclipse,
    read_dose_and_mask_files,
    read_from_nifti,
    find_all_files,
    read_dose_and_mask_files_as_structure_set,
    get_dose_and_structures_as_structure_set,
    create_structure_set_from_existing_data,
    get_dose,
    get_structures,
)

# Utility functions
from .utils import (
    get_default_constraints,
    check_compliance,
    quality_index,
    from_dataframe,
    compare_dvh,
    variability,
    generate_dvh_variations,
    plot_dvh_variations,
    plot_dvh,
    plot_dose_differences,
    plot_frequency_analysis,
    generate_dvh_family_plot,
    interactive_dvh_plotter,
    get_structures_from_folder,
    read_dose_and_mask_files_from_folder,
    create_standard_contents_csv,
    validate_folder_structure,
    batch_folder_validation,
    find_subject_folders,
    setup_output_structure,
)

# Version information
__version__ = "0.2.0"

# Public API
__all__ = [
    # Version
    "__version__",
    # Structure classes
    "Structure",
    "OAR",
    "Target",
    "StructureType",
    "AvoidanceStructure",
    # Structure set management
    "StructureSet",
    "create_structure_set_from_folder",
    "create_structure_set_from_masks",
    # Core metrics
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
    "read_dose_and_mask_files_as_structure_set",
    "get_dose_and_structures_as_structure_set",
    "create_structure_set_from_existing_data",
    "find_all_files",
    "get_dose",
    "get_structures",
    # Utilities
    "get_default_constraints",
    "check_compliance",
    "quality_index",
    "from_dataframe",
    "compare_dvh",
    "variability",
    "generate_dvh_variations",
    "plot_dvh_variations",
    "plot_dvh",
    "plot_dose_differences",
    "plot_frequency_analysis",
    "generate_dvh_family_plot",
    "interactive_dvh_plotter",
    "get_structures_from_folder",
    "read_dose_and_mask_files_from_folder",
    "create_standard_contents_csv",
    "validate_folder_structure",
    "batch_folder_validation",
    "find_subject_folders",
    "setup_output_structure",
]
