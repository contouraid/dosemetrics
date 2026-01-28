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
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
import warnings
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt

from ..dose import Dose


def _compute_gamma_at_point(
    eval_coords: np.ndarray,
    ref_dose_value: float,
    eval_interpolator: RegularGridInterpolator,
    dose_criterion: float,
    distance_criterion_mm: float,
    normalization_dose: float,
) -> float:
    """
    Compute gamma at a single reference point.

    Parameters
    ----------
    eval_coords : np.ndarray
        Coordinates to evaluate (Nx3 array of search positions)
    ref_dose_value : float
        Reference dose value at the point
    eval_interpolator : RegularGridInterpolator
        Interpolator for evaluated dose
    dose_criterion : float
        Dose criterion in absolute units
    distance_criterion_mm : float
        Distance criterion in mm
    normalization_dose : float
        Normalization dose value

    Returns
    -------
    float
        Minimum gamma value at this point
    """
    # Interpolate evaluated dose at search positions
    eval_dose_values = eval_interpolator(eval_coords)

    # Compute dose differences (normalized)
    dose_diff = np.abs(eval_dose_values - ref_dose_value) / normalization_dose

    # Compute distances (already in physical units)
    # eval_coords contains the search grid, we need distance from origin
    distances = np.sqrt(np.sum(eval_coords**2, axis=1))

    # Compute gamma values
    gamma_values = np.sqrt(
        (distances / distance_criterion_mm) ** 2
        + (dose_diff / (dose_criterion / 100.0)) ** 2
    )

    return np.min(gamma_values)


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
    ValueError
        If dose distributions have incompatible geometry.
    """
    # Validate spatial compatibility
    if dose_reference.dose_array.shape != dose_evaluated.dose_array.shape:
        raise ValueError(
            f"Dose shapes must match: {dose_reference.dose_array.shape} vs "
            f"{dose_evaluated.dose_array.shape}"
        )

    if not np.allclose(dose_reference.spacing, dose_evaluated.spacing):
        raise ValueError(
            f"Dose spacings must match: {dose_reference.spacing} vs "
            f"{dose_evaluated.spacing}"
        )

    # Get dose arrays and spatial information
    ref_dose = dose_reference.dose_array
    eval_dose = dose_evaluated.dose_array
    spacing = np.array(dose_reference.spacing)
    origin = np.array(dose_reference.origin)
    shape = dose_reference.shape

    # Set max search distance
    if max_search_distance_mm is None:
        max_search_distance_mm = 3 * distance_criterion_mm

    # Determine normalization dose
    if global_normalization:
        normalization_dose = np.max(ref_dose)
    else:
        normalization_dose = None  # Will use local normalization

    # Calculate absolute dose threshold
    dose_threshold = dose_threshold_percent / 100.0 * np.max(ref_dose)

    # Create coordinate grids for both distributions
    x = origin[0] + np.arange(shape[0]) * spacing[0]
    y = origin[1] + np.arange(shape[1]) * spacing[1]
    z = origin[2] + np.arange(shape[2]) * spacing[2]

    # Create interpolator for evaluated dose
    eval_interpolator = RegularGridInterpolator(
        (x, y, z), eval_dose, method="linear", bounds_error=False, fill_value=0.0
    )

    # Initialize gamma array
    gamma_result = np.full(shape, np.nan, dtype=np.float32)

    # Create search grid offsets (in voxel indices)
    search_radius_voxels = np.ceil(max_search_distance_mm / spacing).astype(int)

    # For efficiency, create a search template
    i_range = np.arange(-search_radius_voxels[0], search_radius_voxels[0] + 1)
    j_range = np.arange(-search_radius_voxels[1], search_radius_voxels[1] + 1)
    k_range = np.arange(-search_radius_voxels[2], search_radius_voxels[2] + 1)

    di, dj, dk = np.meshgrid(i_range, j_range, k_range, indexing="ij")

    # Physical distances for search template
    dx = di * spacing[0]
    dy = dj * spacing[1]
    dz = dk * spacing[2]
    distances_template = np.sqrt(dx**2 + dy**2 + dz**2)

    # Only keep points within search distance
    valid_search = distances_template <= max_search_distance_mm
    di_valid = di[valid_search]
    dj_valid = dj[valid_search]
    dk_valid = dk[valid_search]
    distances_valid = distances_template[valid_search]

    # Iterate through reference dose grid
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                ref_value = ref_dose[i, j, k]

                # Skip if below threshold
                if ref_value < dose_threshold:
                    continue

                # Use local normalization if requested
                if not global_normalization:
                    normalization_dose = ref_value if ref_value > 0 else 1.0

                # Get search positions in index space
                i_search = i + di_valid
                j_search = j + dj_valid
                k_search = k + dk_valid

                # Filter for valid indices
                valid_mask = (
                    (i_search >= 0)
                    & (i_search < shape[0])
                    & (j_search >= 0)
                    & (j_search < shape[1])
                    & (k_search >= 0)
                    & (k_search < shape[2])
                )

                i_search = i_search[valid_mask]
                j_search = j_search[valid_mask]
                k_search = k_search[valid_mask]
                local_distances = distances_valid[valid_mask]

                if len(i_search) == 0:
                    continue

                # Get evaluated dose values at search positions
                eval_values = eval_dose[i_search, j_search, k_search]

                # Compute dose differences (normalized)
                dose_diff = np.abs(eval_values - ref_value) / normalization_dose

                # Compute gamma values using the Low et al. formula
                gamma_values = np.sqrt(
                    (local_distances / distance_criterion_mm) ** 2
                    + (dose_diff / (dose_criterion_percent / 100.0)) ** 2
                )

                # Store minimum gamma value
                gamma_result[i, j, k] = np.min(gamma_values)

    return gamma_result


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
    ValueError
        If slice shapes don't match or are not 2D.
    """
    # Validate input
    if dose_reference_slice.ndim != 2:
        raise ValueError(
            f"Reference slice must be 2D, got shape {dose_reference_slice.shape}"
        )
    if dose_evaluated_slice.ndim != 2:
        raise ValueError(
            f"Evaluated slice must be 2D, got shape {dose_evaluated_slice.shape}"
        )
    if dose_reference_slice.shape != dose_evaluated_slice.shape:
        raise ValueError(
            f"Slice shapes must match: {dose_reference_slice.shape} vs "
            f"{dose_evaluated_slice.shape}"
        )

    # Get shape and spacing
    shape = dose_reference_slice.shape
    spacing = np.array(pixel_spacing)

    # Determine normalization dose (global max)
    normalization_dose = np.max(dose_reference_slice)
    if normalization_dose == 0:
        normalization_dose = 1.0

    # Create interpolator for evaluated dose
    x = np.arange(shape[0]) * spacing[0]
    y = np.arange(shape[1]) * spacing[1]
    eval_interpolator = RegularGridInterpolator(
        (x, y),
        dose_evaluated_slice,
        method="linear",
        bounds_error=False,
        fill_value=0.0,
    )

    # Initialize gamma array
    gamma_result = np.full(shape, np.nan, dtype=np.float32)

    # Create search grid offsets (in voxel indices)
    max_search_distance_mm = 3 * distance_criterion_mm
    search_radius_voxels = np.ceil(max_search_distance_mm / spacing).astype(int)

    # Create search template
    i_range = np.arange(-search_radius_voxels[0], search_radius_voxels[0] + 1)
    j_range = np.arange(-search_radius_voxels[1], search_radius_voxels[1] + 1)

    di, dj = np.meshgrid(i_range, j_range, indexing="ij")

    # Physical distances for search template
    dx = di * spacing[0]
    dy = dj * spacing[1]
    distances_template = np.sqrt(dx**2 + dy**2)

    # Only keep points within search distance
    valid_search = distances_template <= max_search_distance_mm
    di_valid = di[valid_search]
    dj_valid = dj[valid_search]
    distances_valid = distances_template[valid_search]

    # Iterate through reference dose grid
    for i in range(shape[0]):
        for j in range(shape[1]):
            ref_value = dose_reference_slice[i, j]

            # Get search positions in index space
            i_search = i + di_valid
            j_search = j + dj_valid

            # Filter for valid indices
            valid_mask = (
                (i_search >= 0)
                & (i_search < shape[0])
                & (j_search >= 0)
                & (j_search < shape[1])
            )

            i_search = i_search[valid_mask]
            j_search = j_search[valid_mask]
            local_distances = distances_valid[valid_mask]

            if len(i_search) == 0:
                continue

            # Get evaluated dose values at search positions
            eval_values = dose_evaluated_slice[i_search, j_search]

            # Compute dose differences (normalized)
            dose_diff = np.abs(eval_values - ref_value) / normalization_dose

            # Compute gamma values
            gamma_values = np.sqrt(
                (local_distances / distance_criterion_mm) ** 2
                + (dose_diff / (dose_criterion_percent / 100.0)) ** 2
            )

            # Store minimum gamma value
            gamma_result[i, j] = np.min(gamma_values)

    return gamma_result


# Placeholder for GPU-accelerated gamma
def compute_gamma_index_gpu(
    dose_reference: Dose,
    dose_evaluated: Dose,
    dose_criterion_percent: float = 3.0,
    distance_criterion_mm: float = 3.0,
) -> np.ndarray:
    """
    GPU-accelerated gamma index calculation (requires CuPy or similar).

    Note: This is a placeholder for future GPU acceleration using CuPy or similar.

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
        GPU acceleration not implemented yet.
    """
    warnings.warn(
        "GPU-accelerated gamma is not implemented. "
        "Use compute_gamma_index() for CPU-based calculation.",
        FutureWarning,
    )
    raise NotImplementedError(
        "GPU-accelerated gamma not implemented. Use compute_gamma_index()."
    )


__all__ = [
    "compute_gamma_index",
    "compute_gamma_passing_rate",
    "compute_gamma_statistics",
    "compute_2d_gamma",
    "compute_gamma_index_gpu",
]
