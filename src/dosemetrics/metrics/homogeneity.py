"""
Homogeneity indices for target dose uniformity.

This module provides metrics to assess the uniformity of dose distribution
within target volumes. More homogeneous dose distributions are generally
preferred for tumor control.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from ..dose import Dose
    from ..structures import Structure


def compute_homogeneity_index(
    dose: Dose,
    target: Structure,
    d2_percentile: float = 2.0,
    d98_percentile: float = 98.0
) -> float:
    """
    Compute Homogeneity Index (HI).
    
    HI = (D2 - D98) / D50
    
    Where:
    - D2 = dose received by 2% of volume (near-maximum)
    - D98 = dose received by 98% of volume (near-minimum)
    - D50 = median dose
    
    Measures dose uniformity within target. Lower values indicate more
    homogeneous dose distribution.
    
    Typical acceptable range: 0.05 - 0.20
    
    Args:
        dose: Dose distribution object
        target: Target structure (PTV, CTV, etc.)
        d2_percentile: Upper percentile for near-max (typically 2%)
        d98_percentile: Lower percentile for near-min (typically 98%)
        
    Returns:
        Homogeneity index (dimensionless)
        
    References:
        ICRU Report 83 (2010)
        
    Examples:
        >>> hi = compute_homogeneity_index(dose, ptv)
        >>> print(f"Homogeneity Index: {hi:.3f}")
        >>> if hi < 0.15:
        ...     print("Excellent dose homogeneity")
    """
    dose_values = dose.get_dose_in_structure(target)
    
    if len(dose_values) == 0:
        return 0.0
    
    # Note: D2 means 2% of volume receives at least this dose
    # This corresponds to 98th percentile of dose array
    d2 = np.percentile(dose_values, 100 - d2_percentile)
    d98 = np.percentile(dose_values, 100 - d98_percentile)
    d50 = np.percentile(dose_values, 50)
    
    if d50 == 0:
        return float('inf')
    
    return float((d2 - d98) / d50)


def compute_gradient_index(
    dose: Dose,
    target: Structure,
    prescription_dose: float,
    half_prescription_volume_method: bool = True
) -> float:
    """
    Compute Gradient Index (GI) for dose fall-off outside target.
    
    Two calculation methods:
    1. Half-prescription volume: GI = V_50% / V_100%
    2. Distance-based: Ratio of volumes at specific distances
    
    Where:
    - V_100% = volume receiving >= prescription dose
    - V_50% = volume receiving >= 50% prescription dose
    
    Lower values indicate steeper dose fall-off (better for sparing OARs).
    
    Args:
        dose: Dose distribution object
        target: Target structure
        prescription_dose: Prescription dose in Gy
        half_prescription_volume_method: Use V_50%/V_100% method (default True)
        
    Returns:
        Gradient index (dimensionless, typically 2-8)
        
    References:
        Paddick and Lippitz, J Neurosurg 2006
        
    Examples:
        >>> gi = compute_gradient_index(dose, ptv, prescription_dose=60.0)
        >>> print(f"Gradient Index: {gi:.2f}")
        >>> if gi < 3.0:
        ...     print("Excellent dose fall-off")
    """
    v_100 = np.sum(dose.dose_array >= prescription_dose)
    v_50 = np.sum(dose.dose_array >= 0.5 * prescription_dose)
    
    if v_100 == 0:
        return float('inf')
    
    return float(v_50 / v_100)


def compute_dose_homogeneity(
    dose: Dose,
    target: Structure
) -> float:
    """
    Compute coefficient of variation (CV) of dose within target.
    
    CV = std_dose / mean_dose
    
    Alternative measure of dose homogeneity. Lower values indicate
    more uniform dose distribution.
    
    Args:
        dose: Dose distribution object
        target: Target structure
        
    Returns:
        Coefficient of variation (dimensionless)
        
    Examples:
        >>> cv = compute_dose_homogeneity(dose, ptv)
        >>> print(f"Dose CV: {cv:.3f}")
    """
    dose_values = dose.get_dose_in_structure(target)
    
    if len(dose_values) == 0:
        return 0.0
    
    mean = np.mean(dose_values)
    if mean == 0:
        return float('inf')
    
    std = np.std(dose_values)
    return float(std / mean)


def compute_uniformity_index(
    dose: Dose,
    target: Structure
) -> float:
    """
    Compute uniformity index.
    
    UI = 1 - (D_max - D_min) / D_prescription
    
    Values closer to 1.0 indicate better uniformity.
    
    Args:
        dose: Dose distribution object
        target: Target structure
        
    Returns:
        Uniformity index (0-1)
        
    Note:
        Requires prescription dose in target metadata or as parameter.
        Currently uses median dose as approximation.
        
    Examples:
        >>> ui = compute_uniformity_index(dose, ptv)
        >>> print(f"Uniformity Index: {ui:.3f}")
    """
    dose_values = dose.get_dose_in_structure(target)
    
    if len(dose_values) == 0:
        return 0.0
    
    d_max = np.max(dose_values)
    d_min = np.min(dose_values)
    d_ref = np.median(dose_values)  # Use median as reference
    
    if d_ref == 0:
        return 0.0
    
    return float(1.0 - (d_max - d_min) / d_ref)
