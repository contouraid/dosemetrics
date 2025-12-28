"""
Advanced DVH metrics and comparison tools.

This module provides advanced DVH-based metrics for comparing dose distributions
including Wasserstein distance, area between curves, and other statistical measures.

Future Implementation TODOs:
    - Wasserstein distance (Earth Mover's Distance) between DVHs
    - Area between DVH curves (L1/L2 norms)
    - DVH bandwidth and confidence intervals
    - Chi-square and Kolmogorov-Smirnov tests for DVH comparison
    - DVH-based TCP/NTCP models
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
import warnings
from scipy.stats import wasserstein_distance, kstest, chisquare

from ..dose import Dose
from ..structures import Structure
from .dvh import compute_dvh


def compute_dvh_wasserstein_distance(
    dose1: Dose,
    dose2: Dose,
    structure: Structure
) -> float:
    """
    Compute Wasserstein distance (Earth Mover's Distance) between two DVHs.
    
    The Wasserstein distance quantifies the minimum "work" required to transform
    one DVH into another, providing a meaningful metric for DVH similarity.
    
    Parameters
    ----------
    dose1 : Dose
        First dose distribution.
    dose2 : Dose
        Second dose distribution.
    structure : Structure
        Structure for which to compute DVHs.
    
    Returns
    -------
    distance : float
        Wasserstein distance between the two DVHs.
    
    Notes
    -----
    The Wasserstein distance is also known as:
        - Earth Mover's Distance (EMD)
        - Kantorovich-Rubinstein metric
        - Mallows distance
    
    It satisfies the triangle inequality and is a true metric, unlike
    simple area-between-curves measures.
    
    References
    ----------
    - Rubner Y, Tomasi C, Guibas LJ. "The Earth Mover's Distance as a Metric
      for Image Retrieval." Int J Comput Vision. 2000;40(2):99-121.
    
    Examples
    --------
    >>> from dosemetrics.metrics import advanced_dvh
    >>> distance = advanced_dvh.compute_dvh_wasserstein_distance(
    ...     planned_dose, delivered_dose, ptv
    ... )
    >>> print(f"DVH Wasserstein distance: {distance:.2f} Gy")
    
    Raises
    ------
    NotImplementedError
        This function is a stub for future implementation.
    """
    # Get dose values within structure for both doses
    dose_values1 = dose1.get_dose_in_structure(structure)
    dose_values2 = dose2.get_dose_in_structure(structure)
    
    if len(dose_values1) == 0 or len(dose_values2) == 0:
        return 0.0
    
    # Compute Wasserstein distance directly on dose values
    distance = wasserstein_distance(dose_values1, dose_values2)
    
    return float(distance)


def compute_area_between_dvh_curves(
    dose1: Dose,
    dose2: Dose,
    structure: Structure,
    norm: str = 'L2'
) -> float:
    """
    Compute area between two DVH curves.
    
    Parameters
    ----------
    dose1 : Dose
        First dose distribution.
    dose2 : Dose
        Second dose distribution.
    structure : Structure
        Structure for which to compute DVHs.
    norm : {'L1', 'L2'}, optional
        Norm to use for area calculation:
            - 'L1': Sum of absolute differences
            - 'L2': Sum of squared differences (default)
    
    Returns
    -------
    area : float
        Area between the two DVH curves.
    
    Notes
    -----
    The L1 norm gives the Manhattan distance, while L2 gives Euclidean distance.
    For DVH comparison, L1 is often more interpretable.
    
    Raises
    ------
    ValueError
        If norm is not 'L1' or 'L2'.
    """
    if norm not in ['L1', 'L2']:
        raise ValueError(f"norm must be 'L1' or 'L2', got '{norm}'")
    
    # Compute DVHs
    dose_bins1, volumes1 = compute_dvh(dose1, structure)
    dose_bins2, volumes2 = compute_dvh(dose2, structure)
    
    # Create common dose bins
    max_dose = max(np.max(dose_bins1), np.max(dose_bins2))
    step_size = min(
        dose_bins1[1] - dose_bins1[0] if len(dose_bins1) > 1 else 0.1,
        dose_bins2[1] - dose_bins2[0] if len(dose_bins2) > 1 else 0.1
    )
    common_bins = np.arange(0, max_dose + step_size, step_size)
    
    # Interpolate to common bins
    volumes1_interp = np.interp(common_bins, dose_bins1, volumes1)
    volumes2_interp = np.interp(common_bins, dose_bins2, volumes2)
    
    # Compute area based on norm
    if norm == 'L1':
        area = np.sum(np.abs(volumes1_interp - volumes2_interp)) * step_size
    else:  # L2
        area = np.sqrt(np.sum((volumes1_interp - volumes2_interp) ** 2)) * step_size
    
    return float(area)


def compute_dvh_chi_square(
    dose1: Dose,
    dose2: Dose,
    structure: Structure
) -> Tuple[float, float]:
    """
    Perform chi-square test comparing two DVHs.
    
    Parameters
    ----------
    dose1 : Dose
        First (expected) dose distribution.
    dose2 : Dose
        Second (observed) dose distribution.
    structure : Structure
        Structure for DVH computation.
    
    Returns
    -------
    chi2_statistic : float
        Chi-square test statistic.
    p_value : float
        P-value for the test.
    
    Notes
    -----
    Tests the null hypothesis that the two DVHs come from the same distribution.
    """
    # Compute DVHs
    dose_bins1, volumes1 = compute_dvh(dose1, structure)
    dose_bins2, volumes2 = compute_dvh(dose2, structure)
    
    # Create common bins
    max_dose = max(np.max(dose_bins1), np.max(dose_bins2))
    step_size = min(
        dose_bins1[1] - dose_bins1[0] if len(dose_bins1) > 1 else 0.1,
        dose_bins2[1] - dose_bins2[0] if len(dose_bins2) > 1 else 0.1
    )
    common_bins = np.arange(0, max_dose + step_size, step_size)
    
    # Interpolate
    volumes1_interp = np.interp(common_bins, dose_bins1, volumes1)
    volumes2_interp = np.interp(common_bins, dose_bins2, volumes2)
    
    # Convert cumulative DVH to differential (histogram)
    diff_volumes1 = -np.diff(np.append(volumes1_interp, 0))
    diff_volumes2 = -np.diff(np.append(volumes2_interp, 0))
    
    # Ensure non-negative
    diff_volumes1 = np.maximum(diff_volumes1, 0)
    diff_volumes2 = np.maximum(diff_volumes2, 0)
    
    # Avoid zeros in expected values
    diff_volumes1 = diff_volumes1 + 1e-10
    
    # Compute chi-square
    chi2_stat, p_value = chisquare(diff_volumes2, diff_volumes1)
    
    return float(chi2_stat), float(p_value)


def compute_dvh_ks_test(
    dose1: Dose,
    dose2: Dose,
    structure: Structure
) -> Tuple[float, float]:
    """
    Perform Kolmogorov-Smirnov test comparing two DVHs.
    
    Parameters
    ----------
    dose1 : Dose
        First dose distribution.
    dose2 : Dose
        Second dose distribution.
    structure : Structure
        Structure for DVH computation.
    
    Returns
    -------
    ks_statistic : float
        KS test statistic (maximum difference between CDFs).
    p_value : float
        P-value for the test.
    
    Notes
    -----
    The KS test is non-parametric and tests whether two samples come from
    the same distribution.
    """
    from scipy.stats import ks_2samp
    
    # Get dose values in structure for both doses
    dose_values1 = dose1.get_dose_in_structure(structure)
    dose_values2 = dose2.get_dose_in_structure(structure)
    
    if len(dose_values1) == 0 or len(dose_values2) == 0:
        return np.nan, np.nan
    
    # Perform two-sample KS test
    ks_stat, p_value = ks_2samp(dose_values1, dose_values2)
    
    return float(ks_stat), float(p_value)


def compute_dvh_confidence_interval(
    doses: List[Dose],
    structure: Structure,
    confidence: float = 0.95
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute DVH confidence intervals from multiple dose distributions.
    
    Useful for uncertainty quantification from multiple treatment plans
    or Monte Carlo dose simulations.
    
    Parameters
    ----------
    doses : list of Dose
        Multiple dose distributions (e.g., from robust optimization).
    structure : Structure
        Structure for DVH computation.
    confidence : float, optional
        Confidence level (default: 0.95 for 95% CI).
    
    Returns
    -------
    dose_bins : np.ndarray
        Dose bin values.
    mean_dvh : np.ndarray
        Mean DVH curve.
    ci_lower : np.ndarray
        Lower confidence interval.
    ci_upper : np.ndarray
        Upper confidence interval.
    
    Examples
    --------
    >>> dose_bins, mean, lower, upper = compute_dvh_confidence_interval(
    ...     [dose1, dose2, dose3], ptv
    ... )
    >>> plt.fill_between(dose_bins, lower, upper, alpha=0.3)
    >>> plt.plot(dose_bins, mean, 'k-', linewidth=2)
    """
    if len(doses) == 0:
        raise ValueError("At least one dose is required")
    
    # Compute DVH for each dose
    all_dvhs = []
    max_dose = 0
    min_step = float('inf')
    
    for dose in doses:
        dose_bins, volumes = compute_dvh(dose, structure)
        all_dvhs.append((dose_bins, volumes))
        max_dose = max(max_dose, np.max(dose_bins))
        if len(dose_bins) > 1:
            min_step = min(min_step, dose_bins[1] - dose_bins[0])
    
    if min_step == float('inf'):
        min_step = 0.1
    
    # Create common dose bins
    common_bins = np.arange(0, max_dose + min_step, min_step)
    
    # Interpolate all DVHs to common bins
    interpolated_dvhs = []
    for dose_bins, volumes in all_dvhs:
        volumes_interp = np.interp(common_bins, dose_bins, volumes)
        interpolated_dvhs.append(volumes_interp)
    
    # Stack into array (n_doses x n_bins)
    dvh_array = np.array(interpolated_dvhs)
    
    # Compute mean and confidence intervals
    mean_dvh = np.mean(dvh_array, axis=0)
    
    # Compute percentiles for confidence interval
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    ci_lower = np.percentile(dvh_array, lower_percentile, axis=0)
    ci_upper = np.percentile(dvh_array, upper_percentile, axis=0)
    
    return common_bins, mean_dvh, ci_lower, ci_upper


def compute_dvh_bandwidth(
    doses: List[Dose],
    structure: Structure
) -> np.ndarray:
    """
    Compute DVH bandwidth (maximum difference at each dose level).
    
    Parameters
    ----------
    doses : list of Dose
        Multiple dose distributions.
    structure : Structure
        Structure for DVH computation.
    
    Returns
    -------
    bandwidth : np.ndarray
        Maximum difference between DVHs at each dose bin.
    
    Notes
    -----
    Useful for robust plan evaluation - smaller bandwidth indicates
    more robust plan.
    """
    if len(doses) == 0:
        raise ValueError("At least one dose is required")
    
    # Compute DVH for each dose
    all_dvhs = []
    max_dose = 0
    min_step = float('inf')
    
    for dose in doses:
        dose_bins, volumes = compute_dvh(dose, structure)
        all_dvhs.append((dose_bins, volumes))
        max_dose = max(max_dose, np.max(dose_bins))
        if len(dose_bins) > 1:
            min_step = min(min_step, dose_bins[1] - dose_bins[0])
    
    if min_step == float('inf'):
        min_step = 0.1
    
    # Create common dose bins
    common_bins = np.arange(0, max_dose + min_step, min_step)
    
    # Interpolate all DVHs to common bins
    interpolated_dvhs = []
    for dose_bins, volumes in all_dvhs:
        volumes_interp = np.interp(common_bins, dose_bins, volumes)
        interpolated_dvhs.append(volumes_interp)
    
    # Stack into array
    dvh_array = np.array(interpolated_dvhs)
    
    # Compute bandwidth (max - min at each dose)
    bandwidth = np.max(dvh_array, axis=0) - np.min(dvh_array, axis=0)
    
    return bandwidth


def compute_dvh_similarity_index(
    dose1: Dose,
    dose2: Dose,
    structure: Structure,
    method: str = 'dice'
) -> float:
    """
    Compute DVH similarity index using various methods.
    
    Parameters
    ----------
    dose1 : Dose
        First dose distribution.
    dose2 : Dose
        Second dose distribution.
    structure : Structure
        Structure for DVH computation.
    method : {'dice', 'jaccard', 'correlation', 'cosine'}, optional
        Similarity metric to use (default: 'dice').
    
    Returns
    -------
    similarity : float
        Similarity score (0-1, higher is more similar).
    
    Raises
    ------
    ValueError
        If method is not recognized.
    """
    if method not in ['dice', 'jaccard', 'correlation', 'cosine']:
        raise ValueError(f"Unknown method: {method}. Use 'dice', 'jaccard', 'correlation', or 'cosine'.")
    
    # Compute DVHs
    dose_bins1, volumes1 = compute_dvh(dose1, structure)
    dose_bins2, volumes2 = compute_dvh(dose2, structure)
    
    # Create common bins and interpolate
    max_dose = max(np.max(dose_bins1), np.max(dose_bins2))
    step_size = min(
        dose_bins1[1] - dose_bins1[0] if len(dose_bins1) > 1 else 0.1,
        dose_bins2[1] - dose_bins2[0] if len(dose_bins2) > 1 else 0.1
    )
    common_bins = np.arange(0, max_dose + step_size, step_size)
    
    volumes1_interp = np.interp(common_bins, dose_bins1, volumes1)
    volumes2_interp = np.interp(common_bins, dose_bins2, volumes2)
    
    if method == 'dice':
        # Treat DVH curves as binary masks at each dose level
        intersection = np.minimum(volumes1_interp, volumes2_interp)
        union = volumes1_interp + volumes2_interp
        if np.sum(union) == 0:
            return 0.0
        return float(2.0 * np.sum(intersection) / np.sum(union))
    
    elif method == 'jaccard':
        # Jaccard index (IoU)
        intersection = np.minimum(volumes1_interp, volumes2_interp)
        union = np.maximum(volumes1_interp, volumes2_interp)
        if np.sum(union) == 0:
            return 0.0
        return float(np.sum(intersection) / np.sum(union))
    
    elif method == 'correlation':
        # Pearson correlation
        if len(volumes1_interp) < 2:
            return 0.0
        corr = np.corrcoef(volumes1_interp, volumes2_interp)[0, 1]
        return float(corr) if not np.isnan(corr) else 0.0
    
    elif method == 'cosine':
        # Cosine similarity
        dot_product = np.dot(volumes1_interp, volumes2_interp)
        norm1 = np.linalg.norm(volumes1_interp)
        norm2 = np.linalg.norm(volumes2_interp)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(dot_product / (norm1 * norm2))
    
    return 0.0


__all__ = [
    'compute_dvh_wasserstein_distance',
    'compute_area_between_dvh_curves',
    'compute_dvh_chi_square',
    'compute_dvh_ks_test',
    'compute_dvh_confidence_interval',
    'compute_dvh_bandwidth',
    'compute_dvh_similarity_index',
]
