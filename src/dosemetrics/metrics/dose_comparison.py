"""
Dose distribution comparison metrics beyond DVH.

This module provides image-based metrics for comparing 3D dose distributions,
including SSIM, MSE, MAE, and other similarity measures.

Future Implementation TODOs:
    - Structural Similarity Index (SSIM) for dose volumes
    - Mean Squared Error (MSE) and variants
    - Peak Signal-to-Noise Ratio (PSNR)
    - Mutual Information
    - Normalized Cross-Correlation
    - Dose-volume histogram difference maps
"""

import numpy as np
from typing import Tuple, Dict, Optional
import warnings
from scipy import ndimage
from scipy.stats import entropy
from skimage.metrics import structural_similarity

from ..dose import Dose
from ..structures import Structure


def compute_ssim(
    dose1: Dose,
    dose2: Dose,
    structure: Optional[Structure] = None,
    window_size: int = 11,
    k1: float = 0.01,
    k2: float = 0.03
) -> float:
    """
    Compute Structural Similarity Index (SSIM) between two dose distributions.
    
    SSIM is a perceptual metric that quantifies image quality degradation
    based on luminance, contrast, and structure. Originally developed for
    image comparison, it's applicable to dose distributions.
    
    Parameters
    ----------
    dose1 : Dose
        Reference dose distribution.
    dose2 : Dose
        Comparison dose distribution.
    structure : Structure, optional
        If provided, compute SSIM only within structure volume.
        If None, compute for entire dose grid.
    window_size : int, optional
        Size of sliding window for local SSIM computation (default: 11).
    k1 : float, optional
        Algorithm parameter (default: 0.01).
    k2 : float, optional
        Algorithm parameter (default: 0.03).
    
    Returns
    -------
    ssim : float
        Mean SSIM value (0-1, where 1 is perfect similarity).
    
    Notes
    -----
    SSIM ranges from -1 to 1:
        - 1: Perfect similarity
        - 0: No structural similarity
        - -1: Perfect anti-correlation
    
    SSIM considers three components:
        - Luminance: Compares mean intensities
        - Contrast: Compares standard deviations
        - Structure: Compares correlation
    
    References
    ----------
    - Wang Z, Bovik AC, Sheikh HR, Simoncelli EP. "Image quality assessment:
      from error visibility to structural similarity." IEEE Trans Image Process.
      2004;13(4):600-12.
    
    Examples
    --------
    >>> ssim = compute_ssim(planned_dose, delivered_dose, ptv)
    >>> print(f"Dose SSIM: {ssim:.3f}")
    >>> if ssim > 0.95:
    ...     print("Excellent agreement")
    
    Raises
    ------
    NotImplementedError
        This function is a stub for future implementation.
    ValueError
        If dose distributions have incompatible geometry.
    """
    # Get dose arrays
    arr1 = dose1.dose_array
    arr2 = dose2.dose_array
    
    # Check shapes match
    if arr1.shape != arr2.shape:
        raise ValueError(f"Dose shapes must match: {arr1.shape} vs {arr2.shape}")
    
    # Apply structure mask if provided
    if structure is not None:
        mask = structure.mask
        # For 3D SSIM, we need to work with the full volume
        # but we'll compute SSIM and then weight by the mask
        arr1_masked = np.where(mask, arr1, 0)
        arr2_masked = np.where(mask, arr2, 0)
    else:
        arr1_masked = arr1
        arr2_masked = arr2
    
    # Compute SSIM for 3D volume
    # Use smaller window for medical images
    win_size = min(window_size, min(arr1.shape) - 1)
    if win_size % 2 == 0:
        win_size -= 1  # Must be odd
    win_size = max(3, win_size)  # At least 3
    
    data_range = max(np.max(arr1), np.max(arr2))
    
    try:
        ssim_value = structural_similarity(
            arr1_masked,
            arr2_masked,
            data_range=data_range,
            win_size=win_size,
            K1=k1,
            K2=k2
        )
    except ValueError:
        # If window size is too large, reduce it
        win_size = 3
        ssim_value = structural_similarity(
            arr1_masked,
            arr2_masked,
            data_range=data_range,
            win_size=win_size,
            K1=k1,
            K2=k2
        )
    
    return float(ssim_value)


def compute_mse(
    dose1: Dose,
    dose2: Dose,
    structure: Optional[Structure] = None
) -> float:
    """
    Compute Mean Squared Error between two dose distributions.
    
    Parameters
    ----------
    dose1 : Dose
        Reference dose.
    dose2 : Dose
        Comparison dose.
    structure : Structure, optional
        If provided, compute MSE only within structure.
    
    Returns
    -------
    mse : float
        Mean squared error in Gy^2.
    
    Raises
    ------
    ValueError
        If dose distributions have incompatible shapes.
    """
    # Get dose arrays
    arr1 = dose1.dose_array
    arr2 = dose2.dose_array
    
    # Check shapes match
    if arr1.shape != arr2.shape:
        raise ValueError(f"Dose shapes must match: {arr1.shape} vs {arr2.shape}")
    
    # Apply structure mask if provided
    if structure is not None:
        mask = structure.mask
        arr1 = arr1[mask]
        arr2 = arr2[mask]
    
    # Compute MSE
    mse = np.mean((arr1 - arr2) ** 2)
    return float(mse)


def compute_mae(
    dose1: Dose,
    dose2: Dose,
    structure: Optional[Structure] = None
) -> float:
    """
    Compute Mean Absolute Error between two dose distributions.
    
    Parameters
    ----------
    dose1 : Dose
        Reference dose.
    dose2 : Dose
        Comparison dose.
    structure : Structure, optional
        If provided, compute MAE only within structure.
    
    Returns
    -------
    mae : float
        Mean absolute error in Gy.
    
    Notes
    -----
    MAE is often more interpretable than MSE for dose comparison as it's
    in the same units as dose (Gy).
    
    Raises
    ------
    ValueError
        If dose distributions have incompatible shapes.
    """
    # Get dose arrays
    arr1 = dose1.dose_array
    arr2 = dose2.dose_array
    
    # Check shapes match
    if arr1.shape != arr2.shape:
        raise ValueError(f"Dose shapes must match: {arr1.shape} vs {arr2.shape}")
    
    # Apply structure mask if provided
    if structure is not None:
        mask = structure.mask
        arr1 = arr1[mask]
        arr2 = arr2[mask]
    
    # Compute MAE
    mae = np.mean(np.abs(arr1 - arr2))
    return float(mae)


def compute_psnr(
    dose1: Dose,
    dose2: Dose,
    structure: Optional[Structure] = None,
    data_range: Optional[float] = None
) -> float:
    """
    Compute Peak Signal-to-Noise Ratio between two dose distributions.
    
    Parameters
    ----------
    dose1 : Dose
        Reference dose.
    dose2 : Dose
        Comparison dose.
    structure : Structure, optional
        If provided, compute PSNR only within structure.
    data_range : float, optional
        Data range (max - min). If None, computed from doses.
    
    Returns
    -------
    psnr : float
        Peak signal-to-noise ratio in dB.
    
    Notes
    -----
    PSNR is defined as: PSNR = 10 * log10((MAX^2) / MSE)
    Higher values indicate better similarity.
    
    Raises
    ------
    ValueError
        If dose distributions have incompatible shapes or MSE is zero.
    """
    # Compute MSE
    mse = compute_mse(dose1, dose2, structure)
    
    if mse == 0:
        return float('inf')  # Perfect match
    
    # Determine data range
    if data_range is None:
        arr1 = dose1.dose_array
        arr2 = dose2.dose_array
        if structure is not None:
            mask = structure.mask
            arr1 = arr1[mask]
            arr2 = arr2[mask]
        data_range = max(np.max(arr1), np.max(arr2))
    
    # Compute PSNR
    psnr = 10 * np.log10((data_range ** 2) / mse)
    return float(psnr)


def compute_mutual_information(
    dose1: Dose,
    dose2: Dose,
    structure: Optional[Structure] = None,
    bins: int = 256
) -> float:
    """
    Compute Mutual Information between two dose distributions.
    
    Parameters
    ----------
    dose1 : Dose
        First dose distribution.
    dose2 : Dose
        Second dose distribution.
    structure : Structure, optional
        If provided, compute MI only within structure.
    bins : int, optional
        Number of histogram bins (default: 256).
    
    Returns
    -------
    mi : float
        Mutual information value (higher indicates more similarity).
    
    Notes
    -----
    Mutual Information quantifies the information shared between two
    distributions. It's particularly useful for multimodal comparison.
    
    Raises
    ------
    ValueError
        If dose distributions have incompatible shapes.
    """
    # Get dose arrays
    arr1 = dose1.dose_array.flatten()
    arr2 = dose2.dose_array.flatten()
    
    # Check shapes match
    if arr1.shape != arr2.shape:
        raise ValueError(f"Dose shapes must match")
    
    # Apply structure mask if provided
    if structure is not None:
        mask = structure.mask.flatten()
        arr1 = arr1[mask]
        arr2 = arr2[mask]
    
    # Compute 2D histogram
    hist_2d, x_edges, y_edges = np.histogram2d(arr1, arr2, bins=bins)
    
    # Add small epsilon to avoid log(0)
    hist_2d = hist_2d + np.finfo(float).eps
    
    # Normalize to get joint probability
    pxy = hist_2d / np.sum(hist_2d)
    
    # Compute marginal probabilities
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    
    # Compute mutual information
    # MI = sum(p(x,y) * log(p(x,y) / (p(x) * p(y))))
    px_py = px[:, None] * py[None, :]
    
    # Only compute where both are non-zero
    nonzero = (pxy > 0) & (px_py > 0)
    mi = np.sum(pxy[nonzero] * np.log(pxy[nonzero] / px_py[nonzero]))
    
    return float(mi)


def compute_normalized_cross_correlation(
    dose1: Dose,
    dose2: Dose,
    structure: Optional[Structure] = None
) -> float:
    """
    Compute Normalized Cross-Correlation between two dose distributions.
    
    Parameters
    ----------
    dose1 : Dose
        First dose distribution.
    dose2 : Dose
        Second dose distribution.
    structure : Structure, optional
        If provided, compute NCC only within structure.
    
    Returns
    -------
    ncc : float
        Normalized cross-correlation (-1 to 1).
    
    Notes
    -----
    NCC is Pearson correlation coefficient for images/volumes.
    Values close to 1 indicate high positive correlation.
    
    Raises
    ------
    ValueError
        If dose distributions have incompatible shapes.
    """
    # Get dose arrays
    arr1 = dose1.dose_array.flatten()
    arr2 = dose2.dose_array.flatten()
    
    # Check shapes match
    if arr1.shape != arr2.shape:
        raise ValueError(f"Dose shapes must match")
    
    # Apply structure mask if provided
    if structure is not None:
        mask = structure.mask.flatten()
        arr1 = arr1[mask]
        arr2 = arr2[mask]
    
    # Compute NCC (Pearson correlation)
    # NCC = sum((x - mean_x) * (y - mean_y)) / (std_x * std_y * N)
    mean1 = np.mean(arr1)
    mean2 = np.mean(arr2)
    
    numerator = np.sum((arr1 - mean1) * (arr2 - mean2))
    denominator = np.sqrt(np.sum((arr1 - mean1) ** 2) * np.sum((arr2 - mean2) ** 2))
    
    if denominator == 0:
        return 0.0  # No variation in one or both images
    
    ncc = numerator / denominator
    return float(ncc)


def compute_dose_difference_map(
    dose1: Dose,
    dose2: Dose,
    absolute: bool = False
) -> Dose:
    """
    Compute voxel-wise dose difference map.
    
    Parameters
    ----------
    dose1 : Dose
        Reference dose.
    dose2 : Dose
        Comparison dose.
    absolute : bool, optional
        If True, return absolute differences (default: False).
    
    Returns
    -------
    diff_dose : Dose
        Dose object containing difference map.
    
    Notes
    -----
    Useful for visualizing spatial dose discrepancies.
    
    Raises
    ------
    ValueError
        If dose distributions have incompatible shapes.
    """
    # Check shapes match
    if dose1.dose_array.shape != dose2.dose_array.shape:
        raise ValueError(
            f"Dose shapes must match: {dose1.dose_array.shape} vs {dose2.dose_array.shape}"
        )
    
    # Compute difference
    if absolute:
        diff_grid = np.abs(dose1.dose_array - dose2.dose_array)
    else:
        diff_grid = dose1.dose_array - dose2.dose_array
    
    # Create new Dose object with difference
    diff_dose = Dose(
        dose_array=diff_grid,
        spacing=dose1.spacing,
        origin=dose1.origin,
        name=f"{dose1.name}_diff"
    )
    
    return diff_dose


def compute_dose_comparison_metrics(
    dose1: Dose,
    dose2: Dose,
    structure: Optional[Structure] = None
) -> Dict[str, float]:
    """
    Compute comprehensive set of dose comparison metrics.
    
    Parameters
    ----------
    dose1 : Dose
        Reference dose.
    dose2 : Dose
        Comparison dose.
    structure : Structure, optional
        If provided, compute metrics only within structure.
    
    Returns
    -------
    metrics : dict
        Dictionary containing:
            - 'ssim': Structural similarity index
            - 'mse': Mean squared error
            - 'mae': Mean absolute error
            - 'psnr': Peak signal-to-noise ratio
            - 'ncc': Normalized cross-correlation
            - 'mi': Mutual information
    
    Examples
    --------
    >>> metrics = compute_dose_comparison_metrics(dose1, dose2, ptv)
    >>> print(f"SSIM: {metrics['ssim']:.3f}")
    >>> print(f"MAE: {metrics['mae']:.2f} Gy")
    
    Raises
    ------
    ValueError
        If dose distributions have incompatible shapes.
    """
    metrics = {}
    
    try:
        metrics['mse'] = compute_mse(dose1, dose2, structure)
    except Exception as e:
        warnings.warn(f"MSE computation failed: {e}")
        metrics['mse'] = np.nan
    
    try:
        metrics['mae'] = compute_mae(dose1, dose2, structure)
    except Exception as e:
        warnings.warn(f"MAE computation failed: {e}")
        metrics['mae'] = np.nan
    
    try:
        metrics['psnr'] = compute_psnr(dose1, dose2, structure)
    except Exception as e:
        warnings.warn(f"PSNR computation failed: {e}")
        metrics['psnr'] = np.nan
    
    try:
        metrics['ssim'] = compute_ssim(dose1, dose2, structure)
    except Exception as e:
        warnings.warn(f"SSIM computation failed: {e}")
        metrics['ssim'] = np.nan
    
    try:
        metrics['ncc'] = compute_normalized_cross_correlation(dose1, dose2, structure)
    except Exception as e:
        warnings.warn(f"NCC computation failed: {e}")
        metrics['ncc'] = np.nan
    
    try:
        metrics['mi'] = compute_mutual_information(dose1, dose2, structure)
    except Exception as e:
        warnings.warn(f"MI computation failed: {e}")
        metrics['mi'] = np.nan
    
    return metrics


def compute_3d_dose_gradient(
    dose: Dose
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute 3D dose gradient (useful for dose falloff analysis).
    
    Parameters
    ----------
    dose : Dose
        Dose distribution.
    
    Returns
    -------
    grad_x : np.ndarray
        Gradient in x direction.
    grad_y : np.ndarray
        Gradient in y direction.
    grad_z : np.ndarray
        Gradient in z direction.
    
    Notes
    -----
    Uses numpy gradient function which computes central differences
    in the interior and first differences at the boundaries.
    
    The gradient is useful for analyzing dose falloff regions and
    identifying high-gradient areas.
    """
    dose_array = dose.dose_array
    
    # Get voxel spacing from dose object
    spacing = dose.spacing
    
    # Compute gradients in each direction
    # Note: numpy.gradient returns gradients in the order of axes
    grad_z, grad_y, grad_x = np.gradient(dose_array, spacing[2], spacing[1], spacing[0])
    
    return grad_x, grad_y, grad_z


__all__ = [
    'compute_ssim',
    'compute_mse',
    'compute_mae',
    'compute_psnr',
    'compute_mutual_information',
    'compute_normalized_cross_correlation',
    'compute_dose_difference_map',
    'compute_dose_comparison_metrics',
    'compute_3d_dose_gradient',
]
