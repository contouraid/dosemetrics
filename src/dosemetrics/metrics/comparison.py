"""Metrics that compare an evaluated radiotherapy plan with a reference plan.

Every public function in this module accepts ``reference`` before
``evaluated``. Single-plan quantities remain in their clinical domain modules
(``dvh``, ``conformity``, and ``homogeneity``); this module contains only
reference-based comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from ..dose import Dose
from ..structures import Structure
from .conformity import compute_paddick_conformity_index
from .dvh import (
    compare_dvh_score,
    compare_mean_oar_dvh_auc,
    compare_oar_dvh_auc,
)
from .gamma import compare_gamma_index, compute_gamma_passing_rate
from .homogeneity import compute_homogeneity_index

__all__ = [
    "COMPARISON_METRICS",
    "EvaluationTask",
    "MetricCategory",
    "MetricDefinition",
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


class MetricCategory(str, Enum):
    """Clinical category used by the plan-comparison metric catalogue."""

    GLOBAL = "global"
    VOXEL_BASED = "voxel-based"
    PTV_COVERAGE = "PTV coverage"
    PTV_HOMOGENEITY = "PTV Homogeneity"
    OAR_SPARING = "OAR sparing"


class EvaluationTask(str, Enum):
    """Dose-estimation task for which a metric is intended."""

    DOSE_PREDICTION = "dose prediction"
    DOSE_CALCULATION = "dose calculation"


@dataclass(frozen=True)
class MetricDefinition:
    """Metadata for one implemented plan-comparison metric."""

    name: str
    function: str
    category: MetricCategory
    tasks: Tuple[EvaluationTask, ...]
    unit: str
    lower_is_better: bool = True


_BOTH_TASKS = (
    EvaluationTask.DOSE_PREDICTION,
    EvaluationTask.DOSE_CALCULATION,
)


COMPARISON_METRICS: Tuple[MetricDefinition, ...] = (
    MetricDefinition(
        "DVH Score",
        "compare_dvh_score",
        MetricCategory.GLOBAL,
        (EvaluationTask.DOSE_PREDICTION,),
        "Gy",
    ),
    MetricDefinition(
        "Root Mean Squared Error",
        "compare_body_rmse",
        MetricCategory.VOXEL_BASED,
        (EvaluationTask.DOSE_CALCULATION,),
        "Gy",
    ),
    MetricDefinition(
        "Gamma Index Passing Rate",
        "compare_gamma",
        MetricCategory.VOXEL_BASED,
        (EvaluationTask.DOSE_CALCULATION,),
        "%",
        lower_is_better=False,
    ),
    MetricDefinition(
        "PTV Dose Distance",
        "compare_ptv_dose",
        MetricCategory.PTV_COVERAGE,
        _BOTH_TASKS,
        "Gy",
    ),
    MetricDefinition(
        "Paddick Conformity Index Distance",
        "compare_paddick_conformity_index",
        MetricCategory.PTV_COVERAGE,
        _BOTH_TASKS,
        "dimensionless",
    ),
    MetricDefinition(
        "Homogeneity Index Distance",
        "compare_homogeneity_index",
        MetricCategory.PTV_HOMOGENEITY,
        (EvaluationTask.DOSE_CALCULATION,),
        "dimensionless",
    ),
    MetricDefinition(
        "Paddick Gradient Index Distance",
        "compare_paddick_gradient_index",
        MetricCategory.OAR_SPARING,
        _BOTH_TASKS,
        "dimensionless",
    ),
    MetricDefinition(
        "OAR Constraint Disagreement",
        "compare_oar_constraints",
        MetricCategory.OAR_SPARING,
        _BOTH_TASKS,
        "fraction",
    ),
    MetricDefinition(
        "OAR DVH Area Between Curves",
        "compare_oar_dvh_auc",
        MetricCategory.OAR_SPARING,
        _BOTH_TASKS,
        "Gy",
    ),
)


def _validate_dose_geometry(reference: Dose, evaluated: Dose) -> None:
    if reference.shape != evaluated.shape:
        raise ValueError(
            f"Dose shapes must match: {reference.shape} vs {evaluated.shape}"
        )
    if not np.allclose(reference.spacing, evaluated.spacing):
        raise ValueError(
            f"Dose spacings must match: {reference.spacing} vs {evaluated.spacing}"
        )
    if not np.allclose(reference.origin, evaluated.origin, rtol=1e-5, atol=1e-3):
        raise ValueError(
            f"Dose origins must match: {reference.origin} vs {evaluated.origin}"
        )


def compare_ptv_dose(
    reference: Dose,
    evaluated: Dose,
    ptv: Structure,
) -> float:
    """Compute the absolute difference in mean PTV dose.

    .. math::

        |\\bar D_{\\mathrm{evaluated}} - \\bar D_{\\mathrm{reference}}|
    """

    _validate_dose_geometry(reference, evaluated)
    reference_values = reference.get_dose_in_structure(ptv)
    evaluated_values = evaluated.get_dose_in_structure(ptv)
    if len(reference_values) == 0:
        return float("nan")
    return float(abs(np.mean(evaluated_values) - np.mean(reference_values)))


def compare_paddick_conformity_index(
    reference: Dose,
    evaluated: Dose,
    ptv: Structure,
    prescription_dose: float,
) -> float:
    """Compute absolute distance between two Paddick conformity indices.

    The underlying index is

    .. math::

        CI = \\frac{V_{PTV,PIV}}{V_{PTV}}
             \\frac{V_{PTV,PIV}}{V_{PIV}}.
    """

    _validate_dose_geometry(reference, evaluated)
    reference_ci = compute_paddick_conformity_index(reference, ptv, prescription_dose)
    evaluated_ci = compute_paddick_conformity_index(evaluated, ptv, prescription_dose)
    return float(abs(evaluated_ci - reference_ci))


def compare_paddick_gradient_index(
    reference: Dose,
    evaluated: Dose,
    prescription_dose: float,
) -> float:
    """Compute absolute distance between Paddick gradient indices.

    The target argument accepted by :func:`compute_gradient_index` is not used
    by its volume-ratio definition, so this implementation calculates the
    specified ``V_50% / V_100%`` ratio directly.
    """

    _validate_dose_geometry(reference, evaluated)

    def gradient_index(dose: Dose) -> float:
        v_100 = np.count_nonzero(dose.dose_array >= prescription_dose)
        v_50 = np.count_nonzero(dose.dose_array >= 0.5 * prescription_dose)
        if v_100 == 0:
            return float("inf")
        return float(v_50 / v_100)

    reference_gi = gradient_index(reference)
    evaluated_gi = gradient_index(evaluated)
    return float(abs(evaluated_gi - reference_gi))


def compare_homogeneity_index(
    reference: Dose,
    evaluated: Dose,
    ptv: Structure,
) -> float:
    """Compute distance between ``HI = (D2 - D98) / D50`` values."""

    _validate_dose_geometry(reference, evaluated)
    reference_hi = compute_homogeneity_index(reference, ptv)
    evaluated_hi = compute_homogeneity_index(evaluated, ptv)
    return float(abs(evaluated_hi - reference_hi))


def compare_body_rmse(
    reference: Dose,
    evaluated: Dose,
    body: Structure,
) -> float:
    """Compute root mean squared error over body-mask voxels, in Gy."""

    _validate_dose_geometry(reference, evaluated)
    reference_values = reference.get_dose_in_structure(body)
    evaluated_values = evaluated.get_dose_in_structure(body)
    if len(reference_values) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((evaluated_values - reference_values) ** 2)))


def compare_gamma(
    reference: Dose,
    evaluated: Dose,
    body: Optional[Structure] = None,
    dose_criterion_percent: float = 3.0,
    distance_criterion_mm: float = 3.0,
    dose_threshold_percent: float = 0.0,
    global_normalization: bool = True,
    max_search_distance_mm: Optional[float] = None,
) -> float:
    """Compute the percentage of evaluated voxels with ``gamma <= 1``.

    Defaults implement 3%/3 mm global gamma and evaluate every reference
    voxel. Pass ``body`` to restrict the reported passing rate to the body
    mask. A low-dose threshold can be requested explicitly, but is disabled by
    default because it is not part of this metric definition.
    """

    _validate_dose_geometry(reference, evaluated)
    gamma = compare_gamma_index(
        reference,
        evaluated,
        dose_criterion_percent=dose_criterion_percent,
        distance_criterion_mm=distance_criterion_mm,
        dose_threshold_percent=dose_threshold_percent,
        global_normalization=global_normalization,
        max_search_distance_mm=max_search_distance_mm,
    )
    if body is not None:
        # This also validates compatibility between the body and dose grids.
        reference.get_dose_in_structure(body)
        gamma = gamma[body.mask]
    return compute_gamma_passing_rate(gamma, threshold=1.0)


def compare_oar_constraints(
    reference_satisfaction: Union[Sequence[bool], Mapping[str, bool]],
    evaluated_satisfaction: Union[Sequence[bool], Mapping[str, bool]],
    expected_count: Optional[int] = 38,
) -> float:
    """Compare OAR-constraint states and return their disagreement fraction.

    Mappings must contain identical keys. Sequences are compared by position.
    By default, the inputs must each contain the 38 CORSAIR-derived states in
    the head-and-neck protocol. Set ``expected_count=None`` for a deliberately
    different constraint protocol.
    """

    if isinstance(reference_satisfaction, Mapping):
        if not isinstance(evaluated_satisfaction, Mapping):
            raise TypeError("Both satisfaction inputs must use the same container type")
        if set(reference_satisfaction) != set(evaluated_satisfaction):
            raise ValueError("Constraint mappings must contain identical keys")
        keys = tuple(reference_satisfaction)
        reference = np.asarray(
            [reference_satisfaction[key] for key in keys], dtype=bool
        )
        evaluated = np.asarray(
            [evaluated_satisfaction[key] for key in keys], dtype=bool
        )
    else:
        if isinstance(evaluated_satisfaction, Mapping):
            raise TypeError("Both satisfaction inputs must use the same container type")
        reference = np.asarray(reference_satisfaction, dtype=bool)
        evaluated = np.asarray(evaluated_satisfaction, dtype=bool)

    if reference.ndim != 1 or evaluated.ndim != 1:
        raise ValueError("Constraint satisfaction inputs must be one-dimensional")
    if len(reference) == 0:
        raise ValueError("At least one constraint is required")
    if len(reference) != len(evaluated):
        raise ValueError(
            "Constraint satisfaction inputs must have the same number of entries"
        )
    if expected_count is not None and len(reference) != expected_count:
        raise ValueError(
            f"Expected {expected_count} constraint states, got {len(reference)}"
        )
    return float(np.mean(reference != evaluated))
