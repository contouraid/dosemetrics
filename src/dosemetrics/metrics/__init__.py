"""
Core dose metrics and calculations.

This package provides metrics for radiotherapy dose analysis, including:
- DVH computation and analysis
- Dose statistics  
- Conformity indices
- Homogeneity indices
- Geometric metrics for structure comparison
- Gamma analysis
- Advanced DVH metrics
- Dose comparison metrics
"""

# Import all metrics modules
from . import dvh
from . import conformity
from . import homogeneity
from . import geometric
from . import gamma
from . import advanced_dvh
from . import dose_comparison

# Import commonly used functions
from .dvh import (
    compute_dvh,
    compute_volume_at_dose,
    compute_dose_at_volume,
    compute_dose_at_volume_cc,
    compute_equivalent_uniform_dose,
    create_dvh_table,
    extract_dvh_metrics,
    # Dose statistics (moved from statistics.py)
    compute_dose_statistics,
    compute_mean_dose,
    compute_max_dose,
    compute_min_dose,
    compute_median_dose,
    compute_dose_percentile,
)

from .conformity import (
    compute_conformity_index,
    compute_conformity_number,
    compute_paddick_conformity_index,
    compute_coverage,
    compute_spillage,
)

from .homogeneity import (
    compute_homogeneity_index,
    compute_gradient_index,
    compute_dose_homogeneity,
    compute_uniformity_index,
)

from .geometric import (
    compute_dice_coefficient,
    compute_jaccard_index,
    compute_volume_difference,
    compute_volume_ratio,
    compute_sensitivity,
    compute_specificity,
    compare_structure_sets,
)

__all__ = [
    # Submodules
    "dvh",
    "conformity",
    "homogeneity",
    "geometric",
    "gamma",
    "advanced_dvh",
    "dose_comparison",
    # DVH and statistics functions
    "compute_dvh",
    "compute_volume_at_dose",
    "compute_dose_at_volume",
    "compute_dose_at_volume_cc",
    "compute_equivalent_uniform_dose",
    "create_dvh_table",
    "extract_dvh_metrics",
    "compute_dose_statistics",
    "compute_mean_dose",
    "compute_max_dose",
    "compute_min_dose",
    "compute_median_dose",
    "compute_dose_percentile",
    # Conformity functions
    "compute_conformity_index",
    "compute_conformity_number",
    "compute_paddick_conformity_index",
    "compute_coverage",
    "compute_spillage",
    # Homogeneity functions
    "compute_homogeneity_index",
    "compute_gradient_index",
    "compute_dose_homogeneity",
    "compute_uniformity_index",
    # Geometric functions
    "compute_dice_coefficient",
    "compute_jaccard_index",
    "compute_volume_difference",
    "compute_volume_ratio",
    "compute_sensitivity",
    "compute_specificity",
    "compare_structure_sets",
]

