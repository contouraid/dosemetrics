"""
Conformity indices for target coverage evaluation.

This module provides various conformity indices used to evaluate how well
the prescription isodose conforms to the target volume. These metrics are
critical for assessing treatment plan quality.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from ..dose import Dose
    from ..structures import Structure


def compute_conformity_index(
    dose: Dose, target: Structure, prescription_dose: float
) -> float:
    """
    Compute Conformity Index (CI).

    CI = V_target_rx / V_rx

    Where:
    - V_target_rx = volume of target receiving >= prescription dose
    - V_rx = total volume receiving >= prescription dose

    Measures how well the prescription isodose conforms to the target.
    Ideal value is 1.0. Values < 1.0 indicate dose spillage outside target.

    Args:
        dose: Dose distribution object
        target: Target structure (PTV, CTV, etc.)
        prescription_dose: Prescription dose in Gy

    Returns:
        Conformity index (dimensionless, typically 0-1)

    References:
        ICRU Report 62 (1999)

    Examples:
        >>> ci = compute_conformity_index(dose, ptv, prescription_dose=60.0)
        >>> print(f"Conformity Index: {ci:.3f}")
    """
    # Volume of target receiving >= prescription dose
    target_dose_values = dose.get_dose_in_structure(target)
    v_target_rx = np.sum(target_dose_values >= prescription_dose)

    # Total volume receiving >= prescription dose
    v_rx = np.sum(dose.dose_array >= prescription_dose)

    if v_rx == 0:
        return 0.0

    return float(v_target_rx / v_rx)


def compute_conformity_number(
    dose: Dose, target: Structure, prescription_dose: float
) -> float:
    """
    Compute Conformity Number (CN) or Conformation Number.

    CN = (V_target_rx / V_target) * (V_target_rx / V_rx)

    Combines target coverage and dose spillage into a single metric.
    Ideal value is 1.0.

    The first factor (V_target_rx / V_target) represents target coverage.
    The second factor (V_target_rx / V_rx) represents conformity.

    Args:
        dose: Dose distribution object
        target: Target structure
        prescription_dose: Prescription dose in Gy

    Returns:
        Conformity number (0-1)

    References:
        van't Riet et al., Int J Radiat Oncol Biol Phys 1997

    Examples:
        >>> cn = compute_conformity_number(dose, ptv, prescription_dose=60.0)
        >>> print(f"Conformity Number: {cn:.3f}")
    """
    target_dose_values = dose.get_dose_in_structure(target)

    v_target = len(target_dose_values)
    if v_target == 0:
        return 0.0

    v_target_rx = np.sum(target_dose_values >= prescription_dose)
    v_rx = np.sum(dose.dose_array >= prescription_dose)

    if v_rx == 0:
        return 0.0

    coverage = v_target_rx / v_target
    conformity = v_target_rx / v_rx

    return float(coverage * conformity)


def compute_paddick_conformity_index(
    dose: Dose, target: Structure, prescription_dose: float
) -> float:
    """
    Compute Paddick Conformity Index (CI_Paddick).

    CI_Paddick = (V_target_rx)^2 / (V_target * V_rx)

    This index is commonly used for radiosurgery and SBRT plans.
    Ideal value is 1.0.

    Args:
        dose: Dose distribution object
        target: Target structure
        prescription_dose: Prescription dose in Gy

    Returns:
        Paddick conformity index (0-1)

    References:
        Paddick, J Neurosurg 2000

    Examples:
        >>> # Often used for stereotactic radiosurgery
        >>> ci_paddick = compute_paddick_conformity_index(dose, gtv, prescription_dose=18.0)
        >>> print(f"Paddick CI: {ci_paddick:.3f}")
    """
    target_dose_values = dose.get_dose_in_structure(target)

    v_target = len(target_dose_values)
    if v_target == 0:
        return 0.0

    v_target_rx = np.sum(target_dose_values >= prescription_dose)
    v_rx = np.sum(dose.dose_array >= prescription_dose)

    if v_rx == 0 or v_target == 0:
        return 0.0

    return float((v_target_rx**2) / (v_target * v_rx))


def compute_coverage(dose: Dose, target: Structure, prescription_dose: float) -> float:
    """
    Compute target coverage.

    Coverage = V_target_rx / V_target

    Percentage of target volume receiving at least the prescription dose.

    Args:
        dose: Dose distribution object
        target: Target structure
        prescription_dose: Prescription dose in Gy

    Returns:
        Coverage as fraction (0-1) or percentage if multiplied by 100

    Examples:
        >>> coverage = compute_coverage(dose, ptv, prescription_dose=60.0)
        >>> print(f"Target coverage: {coverage*100:.1f}%")
    """
    target_dose_values = dose.get_dose_in_structure(target)

    v_target = len(target_dose_values)
    if v_target == 0:
        return 0.0

    v_target_rx = np.sum(target_dose_values >= prescription_dose)

    return float(v_target_rx / v_target)


def compute_spillage(dose: Dose, target: Structure, prescription_dose: float) -> float:
    """
    Compute dose spillage outside target.

    Spillage = (V_rx - V_target_rx) / V_rx

    Fraction of prescription isodose volume that is outside the target.
    Lower values indicate better conformity.

    Args:
        dose: Dose distribution object
        target: Target structure
        prescription_dose: Prescription dose in Gy

    Returns:
        Spillage as fraction (0-1)

    Examples:
        >>> spillage = compute_spillage(dose, ptv, prescription_dose=60.0)
        >>> print(f"Dose spillage: {spillage*100:.1f}%")
    """
    target_dose_values = dose.get_dose_in_structure(target)
    v_target_rx = np.sum(target_dose_values >= prescription_dose)
    v_rx = np.sum(dose.dose_array >= prescription_dose)

    if v_rx == 0:
        return 0.0

    return float((v_rx - v_target_rx) / v_rx)


def compute_rtog_conformity_index(
    dose: Dose,
    target: Structure,
    prescription_dose: float,
) -> float:
    """
    Compute the RTOG Conformity Index (RTOG CI).

    RTOG CI = V_Rx / V_target

    Where:
    - V_Rx = total volume receiving >= prescription dose (prescription isodose volume)
    - V_target = target structure volume

    The RTOG CI measures how well the prescription isodose conforms to the target.
    Values close to 1.0 are ideal. Values > 1.0 indicate over-coverage (dose spillage);
    values < 1.0 indicate under-coverage.

    This differs from the ICRU-based CI in this library (V_target_rx / V_rx), which
    measures how much of the prescription isodose overlaps the target.

    Args:
        dose: Dose distribution object
        target: Target structure (PTV, CTV, etc.)
        prescription_dose: Prescription dose in Gy

    Returns:
        RTOG Conformity Index (dimensionless). Ideal value: 1.0.

    References:
        Shaw E, et al. Int J Radiat Oncol Biol Phys. 1993;27(5):1231-9.
        RTOG 90-05 stereotactic radiosurgery protocol.

    Examples:
        >>> rtog_ci = compute_rtog_conformity_index(dose, ptv, prescription_dose=60.0)
        >>> if 0.9 <= rtog_ci <= 1.1:
        ...     print("Excellent conformity (RTOG criteria)")
        >>> elif 0.7 <= rtog_ci <= 1.5:
        ...     print("Acceptable conformity (RTOG criteria)")
    """
    v_rx = int(np.sum(dose.dose_array >= prescription_dose))
    v_target = int(np.sum(target.mask))

    if v_target == 0:
        return float("nan")

    return float(v_rx / v_target)


def compute_prescription_mae(
    dose: Dose,
    target: Structure,
    prescription_dose: float,
) -> float:
    """
    Compute the Mean Absolute Error (MAE) between actual dose and prescription dose within target.

    Prescription MAE = mean(|dose_in_target - prescription_dose|)

    This metric measures how well the dose within the target matches the prescription.
    A value of 0.0 means every voxel in the target received exactly the prescription dose.
    Useful for quantifying underdosing and overdosing within the target volume.

    Args:
        dose: Dose distribution object
        target: Target structure (PTV, CTV, etc.)
        prescription_dose: Prescription dose in Gy

    Returns:
        Mean absolute error from prescription dose in Gy

    References:
        Adapted from PTVPrescriptionMAE in GDP-HMM AAPM Challenge evaluation.

    Examples:
        >>> mae = compute_prescription_mae(dose, ptv, prescription_dose=60.0)
        >>> print(f"Prescription MAE: {mae:.2f} Gy ({mae/60.0*100:.1f}% of prescription)")
    """
    dose_values = dose.get_dose_in_structure(target)

    if len(dose_values) == 0:
        return float("nan")

    return float(np.mean(np.abs(dose_values - prescription_dose)))
