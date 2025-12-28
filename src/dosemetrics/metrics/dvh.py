"""
Dose-Volume Histogram (DVH) computation and analysis.

This module provides functions for computing DVHs and extracting DVH-based
metrics such as volume at dose (VX) and dose at volume (DX).
"""

from __future__ import annotations

from typing import Tuple, Optional, Dict, TYPE_CHECKING
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ..dose import Dose
    from ..structures import Structure
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
    volumes = np.array([
        100.0 * np.sum(dose_values >= dose_bin) / len(dose_values)
        for dose_bin in bins
    ])
    
    return bins, volumes


def compute_volume_at_dose(
    dose: Dose,
    structure: Structure,
    dose_threshold: float
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
    dose: Dose,
    structure: Structure,
    volume_percent: float
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
    dose: Dose,
    structure: Structure,
    volume_cc: float
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
    dose: Dose,
    structure: Structure,
    a_parameter: float
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
    step_size: float = 0.1
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
                dvh_data.append({
                    'Dose': dose_val,
                    'Structure': name,
                    'Volume': vol_val
                })
        except ValueError:
            # Structure not found
            continue
    
    return pd.DataFrame(dvh_data)


def extract_dvh_metrics(
    dose: Dose,
    structure: Structure,
    dose_thresholds: Optional[list] = None,
    volume_percentages: Optional[list] = None
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
            metrics[f'V{threshold}'] = v_x
    
    # Dose at volume metrics (DX)
    if volume_percentages:
        for vol_pct in volume_percentages:
            d_x = compute_dose_at_volume(dose, structure, vol_pct)
            metrics[f'D{vol_pct}'] = d_x
    
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
            'mean_dose': 0.0,
            'max_dose': 0.0,
            'min_dose': 0.0,
            'median_dose': 0.0,
            'std_dose': 0.0,
            'D95': 0.0,
            'D50': 0.0,
            'D05': 0.0,
            'D02': 0.0,
            'D98': 0.0,
        }
    
    return {
        'mean_dose': float(np.mean(dose_values)),
        'max_dose': float(np.max(dose_values)),
        'min_dose': float(np.min(dose_values)),
        'median_dose': float(np.median(dose_values)),
        'std_dose': float(np.std(dose_values)),
        'D95': float(np.percentile(dose_values, 5)),   # 95% receives at least this
        'D50': float(np.percentile(dose_values, 50)),
        'D05': float(np.percentile(dose_values, 95)),  # 5% receives at least this
        'D02': float(np.percentile(dose_values, 98)),  # 2% receives at least this
        'D98': float(np.percentile(dose_values, 2)),   # 98% receives at least this
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


def compute_dose_percentile(
    dose: Dose, 
    structure: Structure, 
    percentile: float
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
