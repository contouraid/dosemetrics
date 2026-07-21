"""
Utilities for batch processing, multi-level analysis, and publication-quality plotting.
"""

# Compliance checking
from .compliance import (
    get_custom_constraints,
    get_default_constraints,
    check_compliance,
)

# Batch processing
from .batch import (
    load_dataset,
    load_multiple_doses,
    process_dataset_with_metric,
    batch_compute_dvh,
    compare_doses_batch,
    aggregate_results,
    export_batch_results,
)

# Multi-level analysis
from .analysis import (
    dose_statistics_table,
    compare_dose_statistics,
    compare_structure_geometry,
    analyze_by_structure,
    analyze_by_subject,
    analyze_by_dataset,
    analyze_subset,
    compute_cohort_statistics,
    compare_cohorts,
)

# Publication-quality plotting
from .plot import (
    plot_dvh,
    plot_subject_dvhs,
    plot_dvh_comparison,
    plot_dvh_band,
    plot_metric_boxplot,
    plot_metric_comparison,
    plot_dose_slice,
    plot_metric_values,
    plot_dose_difference,
    save_figure,
)

from .data import (
    download_example_data,
    load_example_study,
    load_dicom_study,
    load_dicom_ct_on_dose_grid,
)

__all__ = [
    # Compliance
    "get_custom_constraints",
    "get_default_constraints",
    "check_compliance",
    # Batch processing
    "load_dataset",
    "load_multiple_doses",
    "process_dataset_with_metric",
    "batch_compute_dvh",
    "compare_doses_batch",
    "aggregate_results",
    "export_batch_results",
    # Multi-level analysis
    "dose_statistics_table",
    "compare_dose_statistics",
    "compare_structure_geometry",
    "analyze_by_structure",
    "analyze_by_subject",
    "analyze_by_dataset",
    "analyze_subset",
    "compute_cohort_statistics",
    "compare_cohorts",
    # Plotting
    "plot_dvh",
    "plot_subject_dvhs",
    "plot_dvh_comparison",
    "plot_dvh_band",
    "plot_metric_boxplot",
    "plot_metric_comparison",
    "plot_dose_slice",
    "plot_metric_values",
    "plot_dose_difference",
    "save_figure",
    # Data workflows
    "download_example_data",
    "load_example_study",
    "load_dicom_study",
    "load_dicom_ct_on_dose_grid",
]
