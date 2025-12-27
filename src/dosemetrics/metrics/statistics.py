"""
Dose statistical metrics.

This module provides functions to compute dose statistics for structures,
including mean, max, min, median, standard deviation, and dose percentiles.
"""

from typing import Dict
import numpy as np


def compute_dose_statistics(dose: 'Dose', structure: 'Structure') -> Dict[str, float]:
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
        >>> from dosemetrics.metrics import statistics
        >>> 
        >>> dose = Dose.from_dicom("rtdose.dcm")
        >>> structures = StructureSet(...)
        >>> ptv = structures.get_structure("PTV")
        >>> 
        >>> stats = statistics.compute_dose_statistics(dose, ptv)
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


def compute_mean_dose(dose: 'Dose', structure: 'Structure') -> float:
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


def compute_max_dose(dose: 'Dose', structure: 'Structure') -> float:
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


def compute_min_dose(dose: 'Dose', structure: 'Structure') -> float:
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


def compute_median_dose(dose: 'Dose', structure: 'Structure') -> float:
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
    dose: 'Dose', 
    structure: 'Structure', 
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
