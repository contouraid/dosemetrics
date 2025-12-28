"""
Gamma analysis for dose distribution comparison.

This module provides gamma index calculation following the methodology of
Low et al. (1998) and subsequent refinements.

References:
    - Low DA, Harms WB, Mutic S, Purdy JA. "A technique for the quantitative
      evaluation of dose distributions." Med Phys. 1998;25(5):656-61.
    - Depuydt T, Van Esch A, Huyskens DP. "A quantitative evaluation of IMRT
      dose distributions: refinement and clinical assessment of the gamma
      evaluation." Radiother Oncol. 2002;62(3):309-19.

Future Implementation TODOs:
    - Global vs. local gamma normalization
    - 2D and 3D gamma analysis
    - GPU-accelerated computation
    - Passing rate statistics
    - Gamma histograms and visualization
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
import warnings

try:
    from pymedphys import gamma as pymedphys_gamma

    PYMEDPHYS_AVAILABLE = True
except (ImportError, ModuleNotFoundError, FileNotFoundError) as e:
    PYMEDPHYS_AVAILABLE = False
    # Silently fail - we'll handle this gracefully in the functions that use it

from ..dose import Dose


def compute_gamma_index(
    dose_reference: Dose,
    dose_evaluated: Dose,
    dose_criterion_percent: float = 3.0,
    distance_criterion_mm: float = 3.0,
    dose_threshold_percent: float = 10.0,
    global_normalization: bool = True,
    max_search_distance_mm: Optional[float] = None,
) -> np.ndarray:
    """
    Compute 3D gamma index between reference and evaluated dose distributions.

    The gamma index quantifies the agreement between two dose distributions by
    combining dose difference and distance-to-agreement criteria.

    Parameters
    ----------
    dose_reference : Dose
        Reference (planned) dose distribution.
    dose_evaluated : Dose
        Evaluated (measured/calculated) dose distribution to compare.
    dose_criterion_percent : float, optional
        Dose difference criterion as percentage (default: 3.0 for 3%).
    distance_criterion_mm : float, optional
        Distance-to-agreement criterion in mm (default: 3.0 for 3mm).
    dose_threshold_percent : float, optional
        Low dose threshold below which gamma is not calculated (default: 10%).
    global_normalization : bool, optional
        If True, normalize to global maximum dose. If False, use local dose
        (default: True).
    max_search_distance_mm : float, optional
        Maximum search distance for gamma calculation. If None, uses
        3 * distance_criterion_mm (default: None).

    Returns
    -------
    gamma : np.ndarray
        3D array of gamma values. Values < 1 indicate passing points,
        values >= 1 indicate failing points. NaN for points below threshold.

    Notes
    -----
    Common gamma criteria:
        - Clinical QA: 3%/3mm (dose_criterion=3.0, distance_criterion=3.0)
        - Stricter QA: 2%/2mm
        - Research: 1%/1mm

    The gamma passing rate is typically calculated as the percentage of
    points with gamma <= 1.0.

    Examples
    --------
    >>> gamma = compute_gamma_index(planned_dose, measured_dose)
    >>> passing_rate = np.sum(gamma <= 1.0) / np.sum(~np.isnan(gamma)) * 100
    >>> print(f"Gamma passing rate: {passing_rate:.1f}%")

    Raises
    ------
    NotImplementedError
        This function is a stub for future implementation.
    ValueError
        If dose distributions have incompatible geometry.
    """
    if not PYMEDPHYS_AVAILABLE:
        raise ImportError(
            "pymedphys is required for gamma analysis. "
            "Install with: pip install pymedphys"
        )

    # Validate spatial compatibility
    if dose_reference.dose_array.shape != dose_evaluated.dose_array.shape:
        raise ValueError(
            f"Dose shapes must match: {dose_reference.dose_array.shape} vs "
            f"{dose_evaluated.dose_array.shape}"
        )

    # Get dose arrays
    ref_dose = dose_reference.dose_array
    eval_dose = dose_evaluated.dose_array

    # Get coordinate arrays from dose properties
    origin = dose_reference.origin
    spacing = dose_reference.spacing
    shape = dose_reference.shape

    axes = [origin[i] + np.arange(shape[i]) * spacing[i] for i in range(3)]

    # Determine normalization
    if global_normalization:
        dose_ref_value = np.max(ref_dose)
    else:
        dose_ref_value = None  # pymedphys will use local normalization

    # Calculate dose threshold
    dose_threshold = dose_threshold_percent / 100.0 * np.max(ref_dose)

    # Set max search distance
    if max_search_distance_mm is None:
        max_search_distance_mm = 3 * distance_criterion_mm

    try:
        # Use pymedphys gamma function
        # Note: pymedphys expects (axes_reference, dose_reference, axes_evaluation, dose_evaluation, ...)
        # where axes can be a tuple of coordinate arrays
        gamma_result = pymedphys_gamma(
            (axes[0], axes[1], axes[2]),  # reference axes (x, y, z)
            ref_dose,  # reference dose
            (axes[0], axes[1], axes[2]),  # evaluation axes (x, y, z)
            eval_dose,  # evaluation dose
            dose_criterion_percent,
            distance_criterion_mm,
            lower_percent_dose_cutoff=dose_threshold_percent,
            interp_fraction=10,  # interpolation factor
            max_gamma=2.0,  # cap gamma at 2 for performance
            local_gamma=not global_normalization,
            global_normalisation=dose_ref_value if global_normalization else None,
            quiet=True,
        )

        return gamma_result

    except FileNotFoundError as e:
        # pymedphys has missing dependency files issue - raise with clear message
        raise RuntimeError(
            f"pymedphys has an environment issue: {e}. "
            "This is a known issue with certain Python environments. "
            "Consider using a compatible pymedphys installation."
        ) from e
    except Exception as e:
        warnings.warn(f"Gamma calculation failed: {e}")
        raise


def compute_gamma_passing_rate(gamma: np.ndarray, threshold: float = 1.0) -> float:
    """
    Compute gamma passing rate from gamma index array.

    Parameters
    ----------
    gamma : np.ndarray
        Gamma index values from compute_gamma_index().
    threshold : float, optional
        Gamma threshold for passing (default: 1.0).

    Returns
    -------
    passing_rate : float
        Percentage of points with gamma <= threshold (0-100).
    """
    # Remove NaN values (below threshold points)
    valid_gamma = gamma[~np.isnan(gamma)]

    if len(valid_gamma) == 0:
        return 0.0

    # Calculate passing rate
    passing = np.sum(valid_gamma <= threshold)
    total = len(valid_gamma)
    passing_rate = (passing / total) * 100.0

    return float(passing_rate)


def compute_gamma_statistics(gamma: np.ndarray) -> Dict[str, float]:
    """
    Compute comprehensive statistics from gamma index array.

    Parameters
    ----------
    gamma : np.ndarray
        Gamma index values.

    Returns
    -------
    stats : dict
        Dictionary containing:
            - 'passing_rate_1_0': Passing rate at gamma=1.0
            - 'mean_gamma': Mean gamma value
            - 'max_gamma': Maximum gamma value
            - 'gamma_50': Median gamma value
            - 'gamma_95': 95th percentile gamma
    """
    # Remove NaN values
    valid_gamma = gamma[~np.isnan(gamma)]

    if len(valid_gamma) == 0:
        return {
            "passing_rate_1_0": 0.0,
            "mean_gamma": np.nan,
            "max_gamma": np.nan,
            "gamma_50": np.nan,
            "gamma_95": np.nan,
        }

    stats = {
        "passing_rate_1_0": compute_gamma_passing_rate(gamma, threshold=1.0),
        "mean_gamma": float(np.mean(valid_gamma)),
        "max_gamma": float(np.max(valid_gamma)),
        "gamma_50": float(np.percentile(valid_gamma, 50)),
        "gamma_95": float(np.percentile(valid_gamma, 95)),
    }

    return stats


def compute_2d_gamma(
    dose_reference_slice: np.ndarray,
    dose_evaluated_slice: np.ndarray,
    dose_criterion_percent: float = 3.0,
    distance_criterion_mm: float = 3.0,
    pixel_spacing: Tuple[float, float] = (1.0, 1.0),
) -> np.ndarray:
    """
    Compute 2D gamma index for a single slice (faster than 3D).

    Parameters
    ----------
    dose_reference_slice : np.ndarray
        2D reference dose slice.
    dose_evaluated_slice : np.ndarray
        2D evaluated dose slice.
    dose_criterion_percent : float
        Dose criterion (%).
    distance_criterion_mm : float
        Distance criterion (mm).
    pixel_spacing : tuple of float
        Pixel spacing in mm (row_spacing, col_spacing).

    Returns
    -------
    gamma : np.ndarray
        2D gamma index array.

    Raises
    ------
    ImportError
        If pymedphys is not available.
    """
    if not PYMEDPHYS_AVAILABLE:
        raise ImportError(
            "pymedphys is required for gamma analysis. "
            "Install with: pip install pymedphys"
        )

    # Create coordinate arrays
    rows = np.arange(dose_reference_slice.shape[0]) * pixel_spacing[0]
    cols = np.arange(dose_reference_slice.shape[1]) * pixel_spacing[1]

    try:
        # pymedphys expects (axes_reference, dose_reference, axes_evaluation, dose_evaluation, ...)
        # For 2D, pass as tuple of 2D coordinate arrays
        gamma_result = pymedphys_gamma(
            (rows, cols),  # reference axes
            dose_reference_slice,  # reference dose
            (rows, cols),  # evaluation axes
            dose_evaluated_slice,  # evaluation dose
            dose_criterion_percent,
            distance_criterion_mm,
            quiet=True,
        )
        return gamma_result
    except FileNotFoundError as e:
        # pymedphys has missing dependency files issue - raise with clear message
        raise RuntimeError(
            f"pymedphys has an environment issue: {e}. "
            "This is a known issue with certain Python environments. "
            "Consider using a compatible pymedphys installation."
        ) from e
    except Exception as e:
        warnings.warn(f"2D Gamma calculation failed: {e}")
        raise


# Placeholder for GPU-accelerated gamma
def compute_gamma_index_gpu(
    dose_reference: Dose,
    dose_evaluated: Dose,
    dose_criterion_percent: float = 3.0,
    distance_criterion_mm: float = 3.0,
) -> np.ndarray:
    """
    GPU-accelerated gamma index calculation (requires CuPy or similar).

    Note: This is a placeholder. For GPU acceleration, use pymedphys
    with GPU backend or implement using CuPy.

    Parameters
    ----------
    dose_reference : Dose
        Reference dose.
    dose_evaluated : Dose
        Evaluated dose.
    dose_criterion_percent : float
        Dose criterion (%).
    distance_criterion_mm : float
        Distance criterion (mm).

    Returns
    -------
    gamma : np.ndarray
        Gamma index array.

    Raises
    ------
    NotImplementedError
        GPU acceleration not implemented. Use pymedphys with GPU backend.
    """
    warnings.warn(
        "GPU-accelerated gamma is not implemented. "
        "Use compute_gamma_index() which leverages pymedphys, "
        "or configure pymedphys with GPU backend for acceleration.",
        FutureWarning,
    )
    raise NotImplementedError(
        "GPU-accelerated gamma not implemented. "
        "Use compute_gamma_index() or configure pymedphys with GPU backend."
    )


__all__ = [
    "compute_gamma_index",
    "compute_gamma_passing_rate",
    "compute_gamma_statistics",
    "compute_2d_gamma",
    "compute_gamma_index_gpu",
]
