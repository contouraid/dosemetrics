"""Dose-volume histogram computation, metrics, and comparisons.

``compute_*`` functions describe one dose plan or a collection of plans.
``compare_*`` functions always accept ``reference`` before ``evaluated``.
Keeping both in this module makes the DVH data model and all operations on it
discoverable in one place.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, TYPE_CHECKING, Union
import numpy as np
import pandas as pd

from ..dose import Dose
from ..structures import Structure

if TYPE_CHECKING:
    from ..structure_set import StructureSet


def compute_dvh(
    dose: Dose,
    structure: Structure,
    max_dose: Optional[float] = None,
    step_size: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute dose-volume histogram for a structure.

    A DVH shows the percentage of structure volume that receives at least
    a given dose level.

    Args:
        dose: Dose distribution object
        structure: Structure to compute DVH for
        max_dose: Maximum dose for histogram bins (auto-detect if None)
        step_size: Bin width in Gy

    Returns:
        Tuple of (dose_bins, volume_percentages)
        - dose_bins: Array of dose levels (Gy)
        - volume_percentages: Percentage of volume receiving >= each dose (0-100)

    Examples:
        >>> from dosemetrics.dose import Dose
        >>> from dosemetrics.metrics import dvh
        >>>
        >>> dose = Dose.from_dicom("rtdose.dcm")
        >>> ptv = structures.get_structure("PTV")
        >>>
        >>> dose_bins, volumes = dvh.compute_dvh(dose, ptv)
        >>>
        >>> # Plot DVH
        >>> import matplotlib.pyplot as plt
        >>> plt.plot(dose_bins, volumes)
        >>> plt.xlabel("Dose (Gy)")
        >>> plt.ylabel("Volume (%)")
    """
    dose_values = dose.get_dose_in_structure(structure)

    if len(dose_values) == 0:
        bins = np.array([0.0])
        volumes = np.array([0.0])
        return bins, volumes

    if max_dose is None:
        max_dose = float(np.max(dose_values))

    bins = np.arange(0, max_dose + step_size, step_size)
    volumes = np.array(
        [
            100.0 * np.sum(dose_values >= dose_bin) / len(dose_values)
            for dose_bin in bins
        ]
    )

    return bins, volumes


def compute_volume_at_dose(
    dose: Dose, structure: Structure, dose_threshold: float
) -> float:
    """
    Compute percentage of structure receiving at least the dose threshold.

    This computes VX where X is the dose threshold (e.g., V20 = % volume >= 20 Gy).

    Args:
        dose: Dose distribution object
        structure: Structure to analyze
        dose_threshold: Dose threshold in Gy

    Returns:
        Percentage of volume (0-100) receiving >= dose_threshold

    Examples:
        >>> # V20: percentage of lung receiving >= 20 Gy
        >>> v20 = compute_volume_at_dose(dose, lung, 20.0)
        >>> print(f"V20: {v20:.1f}%")
        >>>
        >>> # V5: percentage of heart receiving >= 5 Gy
        >>> v5 = compute_volume_at_dose(dose, heart, 5.0)
    """
    dose_values = dose.get_dose_in_structure(structure)

    if len(dose_values) == 0:
        return 0.0

    return float(100.0 * np.sum(dose_values >= dose_threshold) / len(dose_values))


def compute_dose_at_volume(
    dose: Dose, structure: Structure, volume_percent: float
) -> float:
    """
    Compute dose received by a given percentage of structure volume.

    This computes DX where X is the volume percentage (e.g., D95 = dose to 95% of volume).

    Args:
        dose: Dose distribution object
        structure: Structure to analyze
        volume_percent: Volume percentage (0-100)

    Returns:
        Dose in Gy that the specified volume percentage receives

    Raises:
        ValueError: If volume_percent is not in range 0-100

    Examples:
        >>> # D95: dose covering 95% of PTV
        >>> d95 = compute_dose_at_volume(dose, ptv, 95)
        >>> print(f"D95: {d95:.2f} Gy")
        >>>
        >>> # D_0.1cc for OAR (requires volume in cc conversion)
        >>> # For now, use percentile approximation
        >>> d_max = compute_dose_at_volume(dose, brainstem, 0.1)
    """
    if not 0 <= volume_percent <= 100:
        raise ValueError(f"Volume percent must be 0-100, got {volume_percent}")

    dose_values = dose.get_dose_in_structure(structure)

    if len(dose_values) == 0:
        return 0.0

    # DX means X% of volume receives AT LEAST this dose
    # This is the (100-X)th percentile of dose distribution
    percentile = 100 - volume_percent
    return float(np.percentile(dose_values, percentile))


def compute_dose_at_volume_cc(
    dose: Dose, structure: Structure, volume_cc: float
) -> float:
    """
    Compute dose received by a given absolute volume in cc.

    This computes D_Xcc (e.g., D_0.1cc = dose to hottest 0.1 cc).

    Args:
        dose: Dose distribution object
        structure: Structure to analyze
        volume_cc: Absolute volume in cubic centimeters

    Returns:
        Dose in Gy received by the specified volume

    Examples:
        >>> # D_0.1cc: dose to hottest 0.1 cc (common OAR metric)
        >>> d_0_1cc = compute_dose_at_volume_cc(dose, brainstem, 0.1)
        >>> print(f"D_0.1cc: {d_0_1cc:.2f} Gy")
    """
    dose_values = dose.get_dose_in_structure(structure)

    if len(dose_values) == 0:
        return 0.0

    # Convert cc to number of voxels
    voxel_volume_cc = np.prod(structure.spacing) / 1000.0  # mm³ to cc
    num_voxels = int(np.round(volume_cc / voxel_volume_cc))

    if num_voxels >= len(dose_values):
        # Requested volume exceeds structure volume
        return float(np.min(dose_values))

    if num_voxels <= 0:
        return float(np.max(dose_values))

    # Sort dose values in descending order and take the dose at num_voxels
    sorted_doses = np.sort(dose_values)[::-1]
    return float(sorted_doses[num_voxels - 1])


def compute_equivalent_uniform_dose(
    dose: Dose, structure: Structure, a_parameter: float
) -> float:
    """
    Compute Equivalent Uniform Dose (EUD).

    EUD = (mean(D_i^a))^(1/a)

    The a-parameter depends on tissue type:
    - a < 0 for tumors (emphasizes cold spots)
    - a > 0 for normal tissues (emphasizes hot spots)

    Args:
        dose: Dose distribution object
        structure: Structure to analyze
        a_parameter: Tissue-specific parameter

    Returns:
        Equivalent uniform dose in Gy

    References:
        Niemierko, Med Phys 1997

    Examples:
        >>> # For tumor (emphasize underdosage)
        >>> eud_tumor = compute_equivalent_uniform_dose(dose, ptv, a_parameter=-10)
        >>>
        >>> # For OAR (emphasize overdosage)
        >>> eud_oar = compute_equivalent_uniform_dose(dose, brainstem, a_parameter=5)
    """
    dose_values = dose.get_dose_in_structure(structure)

    if len(dose_values) == 0:
        return 0.0

    if a_parameter == 0:
        # Limit case: geometric mean
        return float(np.exp(np.mean(np.log(dose_values + 1e-10))))

    powered_doses = np.power(dose_values, a_parameter)
    mean_powered = np.mean(powered_doses)
    eud = np.power(mean_powered, 1.0 / a_parameter)

    return float(eud)


def create_dvh_table(
    dose: Dose,
    structure_set: StructureSet,
    structure_names: Optional[list] = None,
    max_dose: Optional[float] = None,
    step_size: float = 0.1,
) -> pd.DataFrame:
    """
    Create DVH table for multiple structures in long format.

    Args:
        dose: Dose distribution object
        structure_set: StructureSet containing structures
        structure_names: List of structure names to include (optional)
        max_dose: Maximum dose for bins
        step_size: Dose bin width in Gy

    Returns:
        DataFrame with columns [Dose, Structure, Volume]

    Examples:
        >>> dvh_df = create_dvh_table(dose, structures,
        ...                           structure_names=["PTV", "Brainstem", "SpinalCord"])
        >>> dvh_df.to_csv("dvh_data.csv")
    """
    if structure_names is None:
        structure_names = structure_set.structure_names

    dvh_data = []

    for name in structure_names:
        try:
            structure = structure_set.get_structure(name)
            dose_bins, volumes = compute_dvh(dose, structure, max_dose, step_size)

            for dose_val, vol_val in zip(dose_bins, volumes):
                dvh_data.append(
                    {"Dose": dose_val, "Structure": name, "Volume": vol_val}
                )
        except ValueError:
            # Structure not found
            continue

    return pd.DataFrame(dvh_data)


def extract_dvh_metrics(
    dose: Dose,
    structure: Structure,
    dose_thresholds: Optional[list] = None,
    volume_percentages: Optional[list] = None,
) -> Dict[str, float]:
    """
    Extract common DVH metrics for a structure.

    Args:
        dose: Dose distribution object
        structure: Structure to analyze
        dose_thresholds: List of dose levels for VX metrics (Gy)
        volume_percentages: List of volume percentages for DX metrics

    Returns:
        Dictionary with DVH metrics

    Examples:
        >>> metrics = extract_dvh_metrics(
        ...     dose, ptv,
        ...     dose_thresholds=[20, 40, 60],
        ...     volume_percentages=[2, 50, 95, 98]
        ... )
        >>> print(metrics)
        {'V20': 98.5, 'V40': 97.2, 'V60': 95.8, 'D2': 63.5, 'D50': 60.2, ...}
    """
    metrics = {}

    # Volume at dose metrics (VX)
    if dose_thresholds:
        for threshold in dose_thresholds:
            v_x = compute_volume_at_dose(dose, structure, threshold)
            metrics[f"V{threshold}"] = v_x

    # Dose at volume metrics (DX)
    if volume_percentages:
        for vol_pct in volume_percentages:
            d_x = compute_dose_at_volume(dose, structure, vol_pct)
            metrics[f"D{vol_pct}"] = d_x

    return metrics


# Dose statistics functions (formerly in statistics.py)


def compute_dose_statistics(dose: Dose, structure: Structure) -> Dict[str, float]:
    """
    Compute comprehensive dose statistics for a structure.

    Args:
        dose: Dose distribution object
        structure: Structure to analyze

    Returns:
        Dictionary with statistics including:
        - mean_dose, max_dose, min_dose, median_dose, std_dose
        - D95, D50, D05, D02, D98 (dose percentiles)

    Examples:
        >>> from dosemetrics.dose import Dose
        >>> from dosemetrics.structure_set import StructureSet
        >>> from dosemetrics.metrics import dvh
        >>>
        >>> dose = Dose.from_dicom("rtdose.dcm")
        >>> structures = StructureSet(...)
        >>> ptv = structures.get_structure("PTV")
        >>>
        >>> stats = dvh.compute_dose_statistics(dose, ptv)
        >>> print(f"Mean dose: {stats['mean_dose']:.2f} Gy")
        >>> print(f"D95: {stats['D95']:.2f} Gy")
    """
    dose_values = dose.get_dose_in_structure(structure)

    if len(dose_values) == 0:
        return {
            "mean_dose": 0.0,
            "max_dose": 0.0,
            "min_dose": 0.0,
            "median_dose": 0.0,
            "std_dose": 0.0,
            "D95": 0.0,
            "D50": 0.0,
            "D05": 0.0,
            "D02": 0.0,
            "D98": 0.0,
        }

    return {
        "mean_dose": float(np.mean(dose_values)),
        "max_dose": float(np.max(dose_values)),
        "min_dose": float(np.min(dose_values)),
        "median_dose": float(np.median(dose_values)),
        "std_dose": float(np.std(dose_values)),
        "D95": float(np.percentile(dose_values, 5)),  # 95% receives at least this
        "D50": float(np.percentile(dose_values, 50)),
        "D05": float(np.percentile(dose_values, 95)),  # 5% receives at least this
        "D02": float(np.percentile(dose_values, 98)),  # 2% receives at least this
        "D98": float(np.percentile(dose_values, 2)),  # 98% receives at least this
    }


def compute_mean_dose(dose: Dose, structure: Structure) -> float:
    """
    Compute mean dose in structure.

    Args:
        dose: Dose distribution object
        structure: Structure to analyze

    Returns:
        Mean dose in Gy
    """
    dose_values = dose.get_dose_in_structure(structure)
    return float(np.mean(dose_values)) if len(dose_values) > 0 else 0.0


def compute_max_dose(dose: Dose, structure: Structure) -> float:
    """
    Compute maximum dose in structure.

    Args:
        dose: Dose distribution object
        structure: Structure to analyze

    Returns:
        Maximum dose in Gy
    """
    dose_values = dose.get_dose_in_structure(structure)
    return float(np.max(dose_values)) if len(dose_values) > 0 else 0.0


def compute_min_dose(dose: Dose, structure: Structure) -> float:
    """
    Compute minimum dose in structure.

    Args:
        dose: Dose distribution object
        structure: Structure to analyze

    Returns:
        Minimum dose in Gy
    """
    dose_values = dose.get_dose_in_structure(structure)
    return float(np.min(dose_values)) if len(dose_values) > 0 else 0.0


def compute_median_dose(dose: Dose, structure: Structure) -> float:
    """
    Compute median dose in structure.

    Args:
        dose: Dose distribution object
        structure: Structure to analyze

    Returns:
        Median dose in Gy
    """
    dose_values = dose.get_dose_in_structure(structure)
    return float(np.median(dose_values)) if len(dose_values) > 0 else 0.0


def compare_dvh_score(
    reference: Dose,
    evaluated: Dose,
    targets: Union[Structure, Sequence[Structure]],
    oars: Sequence[Structure] = (),
) -> float:
    """Compare plans using the complete OpenKBP DVH score.

    The score is the unweighted mean absolute error over D1, D95, and D99 for
    every target and mean dose and D0.1cc for every OAR. It is reported in Gy.
    """
    _validate_comparable_doses(reference, evaluated)
    target_structures = (targets,) if isinstance(targets, Structure) else tuple(targets)
    errors = []

    for target in target_structures:
        for volume_percent in (1.0, 95.0, 99.0):
            reference_value = compute_dose_at_volume(reference, target, volume_percent)
            evaluated_value = compute_dose_at_volume(evaluated, target, volume_percent)
            errors.append(abs(evaluated_value - reference_value))

    for oar in oars:
        errors.append(
            abs(compute_mean_dose(evaluated, oar) - compute_mean_dose(reference, oar))
        )
        errors.append(
            abs(
                compute_dose_at_volume_cc(evaluated, oar, 0.1)
                - compute_dose_at_volume_cc(reference, oar, 0.1)
            )
        )

    if not errors:
        raise ValueError("At least one target or OAR criterion is required")
    return float(np.mean(errors))


def compute_dvh_auc(
    dose: Dose,
    structure: Structure,
    num_bins: int = 100,
    normalize: bool = True,
    dose_range: Optional[Tuple[float, float]] = None,
) -> float:
    """
    Compute the Area Under the DVH Curve (DVH-AUC) using the trapezoidal rule.

    The DVH-AUC is the integral of volume percentage over the dose range.
    A higher AUC indicates that more of the structure volume receives higher doses.
    This is a single-distribution metric (unlike area-between-curves which compares two).

    Args:
        dose: Dose distribution object
        structure: Structure to compute DVH AUC for
        num_bins: Number of dose bins for DVH computation (default: 100)
        normalize: If True, normalize AUC to [0, 1] by dividing by the maximum
            possible area (100% volume × dose range). Default: True.
        dose_range: Fixed (min_dose, max_dose) in Gy for binning. Uses
            per-structure min/max if None.

    Returns:
        DVH AUC value. If normalize=True, returns value in [0, 1].
        If normalize=False, returns value in Gy (dose × volume units).

    References:
        Adapted from DVHAUC metric in GDP-HMM AAPM Challenge.

    Examples:
        >>> auc = compute_dvh_auc(dose, ptv, normalize=True)
        >>> print(f"DVH AUC (normalized): {auc:.3f}")
        >>>
        >>> # Compare two structures
        >>> ptv_auc = compute_dvh_auc(dose, ptv)
        >>> oar_auc = compute_dvh_auc(dose, brainstem)
    """
    dose_values = dose.get_dose_in_structure(structure)

    if len(dose_values) == 0:
        return 0.0

    if dose_range is not None:
        min_dose, max_dose = float(dose_range[0]), float(dose_range[1])
    else:
        min_dose = float(np.min(dose_values))
        max_dose = float(np.max(dose_values))

    if max_dose - min_dose < 1e-10:
        # All voxels at same dose: AUC = 1.0 normalized, or max_dose unnormalized
        return 1.0 if normalize else float(max_dose)

    dose_bins = np.linspace(min_dose, max_dose, num_bins)
    dvh_values = np.array(
        [100.0 * np.sum(dose_values >= d) / len(dose_values) for d in dose_bins]
    )

    auc = float(np.trapz(dvh_values, dose_bins))

    if normalize:
        max_area = (max_dose - min_dose) * 100.0
        auc = auc / max_area if max_area > 1e-10 else 0.0

    return auc


def compute_dose_percentile(
    dose: Dose, structure: Structure, percentile: float
) -> float:
    """
    Compute dose percentile (DX).

    D95 means 95% of the volume receives at least this dose.
    This corresponds to the 5th percentile of the dose distribution.

    Args:
        dose: Dose distribution object
        structure: Structure to analyze
        percentile: Volume percentage (0-100). For D95, use percentile=95

    Returns:
        Dose in Gy that the specified percentage of volume receives

    Raises:
        ValueError: If percentile is not in range 0-100

    Examples:
        >>> # D95: dose received by 95% of volume
        >>> d95 = compute_dose_percentile(dose, ptv, 95)
        >>>
        >>> # D50: median dose
        >>> d50 = compute_dose_percentile(dose, ptv, 50)
        >>>
        >>> # D05: near-maximum dose (hot spot)
        >>> d05 = compute_dose_percentile(dose, ptv, 5)
    """
    if not 0 <= percentile <= 100:
        raise ValueError(f"Percentile must be 0-100, got {percentile}")

    dose_values = dose.get_dose_in_structure(structure)

    if len(dose_values) == 0:
        return 0.0

    # DX means X% receives AT LEAST this dose
    # This is the (100-X)th percentile of the dose array
    return float(np.percentile(dose_values, 100 - percentile))


# Plan comparisons


def _validate_comparable_doses(reference: Dose, evaluated: Dose) -> None:
    """Validate geometry shared by all reference-based DVH comparisons."""
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


def _common_dvh_grid(
    reference: Dose,
    evaluated: Dose,
    structure: Structure,
    step_size: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return two cumulative DVHs interpolated onto one dose grid."""
    if step_size <= 0:
        raise ValueError("step_size must be positive")
    _validate_comparable_doses(reference, evaluated)
    reference_bins, reference_volume = compute_dvh(
        reference, structure, step_size=step_size
    )
    evaluated_bins, evaluated_volume = compute_dvh(
        evaluated, structure, step_size=step_size
    )
    max_dose = max(float(reference_bins[-1]), float(evaluated_bins[-1]))
    bins = np.arange(0.0, max_dose + step_size, step_size)
    return (
        bins,
        np.interp(bins, reference_bins, reference_volume),
        np.interp(bins, evaluated_bins, evaluated_volume),
    )


def compare_dvh_wasserstein(
    reference: Dose,
    evaluated: Dose,
    structure: Structure,
) -> float:
    """Return the Wasserstein distance between structure dose samples, in Gy."""
    from scipy.stats import wasserstein_distance

    _validate_comparable_doses(reference, evaluated)
    reference_values = reference.get_dose_in_structure(structure)
    evaluated_values = evaluated.get_dose_in_structure(structure)
    if len(reference_values) == 0:
        return float("nan")
    return float(wasserstein_distance(reference_values, evaluated_values))


def compare_dvh_area(
    reference: Dose,
    evaluated: Dose,
    structure: Structure,
    norm: str = "l1",
    step_size: float = 0.1,
) -> float:
    """Compare two cumulative DVHs using an integrated L1 or L2 distance.

    ``norm="l1"`` integrates the pointwise absolute separation. ``norm="l2"``
    returns the square root of the integrated squared separation.
    """
    norm = norm.lower()
    if norm not in {"l1", "l2"}:
        raise ValueError("norm must be 'l1' or 'l2'")
    bins, reference_volume, evaluated_volume = _common_dvh_grid(
        reference, evaluated, structure, step_size
    )
    difference = evaluated_volume - reference_volume
    integrand = np.abs(difference) if norm == "l1" else difference**2
    integral = float(np.trapz(integrand, bins))
    return integral if norm == "l1" else float(np.sqrt(integral))


def compare_dvh_chi_square(
    reference: Dose,
    evaluated: Dose,
    structure: Structure,
    step_size: float = 0.1,
) -> Tuple[float, float]:
    """Compare differential DVHs with a chi-square goodness-of-fit test."""
    from scipy.stats import chisquare

    _, reference_volume, evaluated_volume = _common_dvh_grid(
        reference, evaluated, structure, step_size
    )
    expected = np.maximum(-np.diff(np.append(reference_volume, 0.0)), 0.0)
    observed = np.maximum(-np.diff(np.append(evaluated_volume, 0.0)), 0.0)
    expected += np.finfo(float).eps
    observed *= np.sum(expected) / max(np.sum(observed), np.finfo(float).eps)
    statistic, p_value = chisquare(observed, expected)
    return float(statistic), float(p_value)


def compare_dvh_ks(
    reference: Dose,
    evaluated: Dose,
    structure: Structure,
) -> Tuple[float, float]:
    """Compare structure dose samples with a two-sample KS test."""
    from scipy.stats import ks_2samp

    _validate_comparable_doses(reference, evaluated)
    reference_values = reference.get_dose_in_structure(structure)
    evaluated_values = evaluated.get_dose_in_structure(structure)
    if len(reference_values) == 0:
        return float("nan"), float("nan")
    statistic, p_value = ks_2samp(reference_values, evaluated_values)
    return float(statistic), float(p_value)


def _aligned_dvhs(
    doses: Sequence[Dose],
    structure: Structure,
    step_size: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if not doses:
        raise ValueError("At least one dose is required")
    if step_size <= 0:
        raise ValueError("step_size must be positive")
    dvhs = [compute_dvh(dose, structure, step_size=step_size) for dose in doses]
    max_dose = max(float(bins[-1]) for bins, _ in dvhs)
    common_bins = np.arange(0.0, max_dose + step_size, step_size)
    aligned = np.asarray(
        [np.interp(common_bins, bins, volume) for bins, volume in dvhs]
    )
    return common_bins, aligned


def compute_dvh_confidence_interval(
    doses: Sequence[Dose],
    structure: Structure,
    confidence: float = 0.95,
    step_size: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Summarize a collection of plans with mean DVH and percentile interval."""
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    bins, dvhs = _aligned_dvhs(doses, structure, step_size)
    tail = (1.0 - confidence) * 50.0
    return (
        bins,
        np.mean(dvhs, axis=0),
        np.percentile(dvhs, tail, axis=0),
        np.percentile(dvhs, 100.0 - tail, axis=0),
    )


def compute_dvh_bandwidth(
    doses: Sequence[Dose],
    structure: Structure,
    step_size: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return dose bins and the pointwise max-minus-min DVH bandwidth."""
    bins, dvhs = _aligned_dvhs(doses, structure, step_size)
    return bins, np.max(dvhs, axis=0) - np.min(dvhs, axis=0)


def compare_dvh_similarity(
    reference: Dose,
    evaluated: Dose,
    structure: Structure,
    method: str = "dice",
    step_size: float = 0.1,
) -> float:
    """Compare two DVHs using Dice, Jaccard, correlation, or cosine similarity."""
    method = method.lower()
    if method not in {"dice", "jaccard", "correlation", "cosine"}:
        raise ValueError("method must be 'dice', 'jaccard', 'correlation', or 'cosine'")
    _, reference_volume, evaluated_volume = _common_dvh_grid(
        reference, evaluated, structure, step_size
    )
    intersection = np.minimum(reference_volume, evaluated_volume)
    if method == "dice":
        denominator = np.sum(reference_volume + evaluated_volume)
        return float(2.0 * np.sum(intersection) / denominator) if denominator else 0.0
    if method == "jaccard":
        union = np.sum(np.maximum(reference_volume, evaluated_volume))
        return float(np.sum(intersection) / union) if union else 0.0
    if method == "correlation":
        correlation = np.corrcoef(reference_volume, evaluated_volume)[0, 1]
        return float(correlation) if np.isfinite(correlation) else 0.0
    denominator = np.linalg.norm(reference_volume) * np.linalg.norm(evaluated_volume)
    return (
        float(np.dot(reference_volume, evaluated_volume) / denominator)
        if denominator
        else 0.0
    )


def compare_oar_dvh_auc(
    reference: Dose,
    evaluated: Dose,
    oar: Structure,
    num_bins: int = 100,
) -> float:
    """Return ``|AUC_evaluated - AUC_reference|`` for one OAR, in Gy."""
    _validate_comparable_doses(reference, evaluated)
    if num_bins < 2:
        raise ValueError("num_bins must be at least 2")
    reference_values = reference.get_dose_in_structure(oar)
    evaluated_values = evaluated.get_dose_in_structure(oar)
    if len(reference_values) == 0:
        return float("nan")
    min_dose = float(min(np.min(reference_values), np.min(evaluated_values)))
    max_dose = float(max(np.max(reference_values), np.max(evaluated_values)))
    if np.isclose(min_dose, max_dose):
        return 0.0
    bins = np.linspace(min_dose, max_dose, num_bins)
    reference_dvh = np.asarray([np.mean(reference_values >= d) for d in bins])
    evaluated_dvh = np.asarray([np.mean(evaluated_values >= d) for d in bins])
    return float(abs(np.trapz(evaluated_dvh, bins) - np.trapz(reference_dvh, bins)))


def compare_mean_oar_dvh_auc(
    reference: Dose,
    evaluated: Dose,
    oars: Sequence[Structure],
    num_bins: int = 100,
) -> float:
    """Average the per-OAR absolute AUC differences."""
    if not oars:
        raise ValueError("At least one OAR is required")
    return float(
        np.mean(
            [
                compare_oar_dvh_auc(reference, evaluated, oar, num_bins=num_bins)
                for oar in oars
            ]
        )
    )
