"""
Utilities for compliance checking, comparison, and plotting.
"""

from .compliance import (
    get_custom_constraints,
    get_default_constraints,
    check_compliance,
    quality_index,
    compute_mirage_compliance,
)
from .plot import (
    from_dataframe,
    compare_dvh,
    variability,
    plot_dvh,
    plot_dose_differences,
    plot_frequency_analysis,
    generate_dvh_family_plot,
    interactive_dvh_plotter,
)
from .batch import (
    get_structures_from_folder,
    read_dose_and_mask_files_from_folder,
    create_standard_contents_csv,
    validate_folder_structure,
    batch_folder_validation,
    find_subject_folders,
    setup_output_structure,
)

__all__ = [
    "get_custom_constraints",
    "get_default_constraints",
    "check_compliance",
    "quality_index",
    "compute_mirage_compliance",
    "from_dataframe",
    "compare_dvh",
    "variability",
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
