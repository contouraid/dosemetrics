"""Radiotherapy dose metrics organized by domain and comparison semantics.

Single-plan quantities are computed in their clinical domain modules:
``dvh``, ``conformity``, and ``homogeneity``. Named reference-based plan
metrics are exported directly from this package and consistently accept
``reference`` before ``evaluated``. Lower-level image comparisons remain in
``dose_comparison``.

Domain modules remain public for single-plan computations. Named clinical
``compare_*`` functions are also available directly so callers do not need a
redundant module prefix.
"""

from . import comparison
from . import conformity
from . import dose_comparison
from . import dvh
from . import gamma
from . import geometric
from . import homogeneity
from .comparison import (
    compare_body_rmse,
    compare_dvh_score,
    compare_gamma,
    compare_homogeneity_index,
    compare_mean_oar_dvh_auc,
    compare_oar_constraints,
    compare_oar_dvh_auc,
    compare_paddick_conformity_index,
    compare_paddick_gradient_index,
    compare_ptv_dose,
)

__all__ = [
    "comparison",
    "conformity",
    "dose_comparison",
    "dvh",
    "gamma",
    "geometric",
    "homogeneity",
    "compare_body_rmse",
    "compare_dvh_score",
    "compare_gamma",
    "compare_homogeneity_index",
    "compare_mean_oar_dvh_auc",
    "compare_oar_constraints",
    "compare_oar_dvh_auc",
    "compare_paddick_conformity_index",
    "compare_paddick_gradient_index",
    "compare_ptv_dose",
]
