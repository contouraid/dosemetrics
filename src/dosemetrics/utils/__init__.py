"""
Utilities for batch processing, multi-level analysis, and publication-quality plotting.
"""

# Compliance checking
from .compliance import (
    get_custom_constraints,
    get_default_constraints,
    check_compliance,
    quality_index,
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
    save_figure,
)

__all__ = [
    # Compliance
    "get_custom_constraints",
    "get_default_constraints",
    "check_compliance",
    "quality_index",
    # Batch processing
    "load_dataset",
    "load_multiple_doses",
    "process_dataset_with_metric",
    "batch_compute_dvh",
    "compare_doses_batch",
    "aggregate_results",
    "export_batch_results",
    # Multi-level analysis
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
    "save_figure",
]
