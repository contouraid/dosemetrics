"""
Geometric similarity and overlap metrics for structure comparison.

This module provides metrics to compare two structure sets, typically used
for evaluating auto-segmentation algorithms or inter-observer variability.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Optional, TYPE_CHECKING
import pandas as pd
from scipy.spatial.distance import directed_hausdorff
from scipy import ndimage

if TYPE_CHECKING:
    from ..structures import Structure
    from ..structure_set import StructureSet


def compute_dice_coefficient(structure1: Structure, structure2: Structure) -> float:
    """
    Compute Dice coefficient (Sørensen-Dice index).
    
    Dice = 2 * |A ∩ B| / (|A| + |B|)
    
    Measures overlap between two structures. Range [0, 1], where 1 is perfect overlap.
    
    Args:
        structure1: First structure
        structure2: Second structure
        
    Returns:
        Dice coefficient (0-1)
        
    References:
        Dice, Ecology 1945; Sørensen, Biologiske Skrifter 1948
        
    Examples:
        >>> auto_ptv = structures_auto.get_structure("PTV")
        >>> manual_ptv = structures_manual.get_structure("PTV")
        >>> dice = compute_dice_coefficient(auto_ptv, manual_ptv)
        >>> print(f"Dice: {dice:.3f}")
    """
    if structure1.mask is None or structure2.mask is None:
        return 0.0
    
    intersection = np.logical_and(structure1.mask, structure2.mask)
    sum_volumes = structure1.volume_voxels() + structure2.volume_voxels()
    
    if sum_volumes == 0:
        return 0.0
    
    return float(2.0 * np.sum(intersection) / sum_volumes)


def compute_jaccard_index(structure1: Structure, structure2: Structure) -> float:
    """
    Compute Jaccard index (Intersection over Union, IoU).
    
    Jaccard = |A ∩ B| / |A ∪ B|
    
    Measures overlap between two structures. Range [0, 1], where 1 is perfect overlap.
    More conservative than Dice coefficient.
    
    Args:
        structure1: First structure
        structure2: Second structure
        
    Returns:
        Jaccard index (0-1)
        
    References:
        Jaccard, New Phytologist 1912
        
    Examples:
        >>> jaccard = compute_jaccard_index(auto_ptv, manual_ptv)
        >>> print(f"IoU: {jaccard:.3f}")
    """
    if structure1.mask is None or structure2.mask is None:
        return 0.0
    
    intersection = np.logical_and(structure1.mask, structure2.mask)
    union = np.logical_or(structure1.mask, structure2.mask)
    
    union_sum = np.sum(union)
    if union_sum == 0:
        return 0.0
    
    return float(np.sum(intersection) / union_sum)


def compute_volume_difference(structure1: Structure, structure2: Structure) -> float:
    """
    Compute absolute volume difference.
    
    Args:
        structure1: First structure
        structure2: Second structure
        
    Returns:
        Absolute volume difference in cubic centimeters
        
    Examples:
        >>> vol_diff = compute_volume_difference(auto_ptv, manual_ptv)
        >>> print(f"Volume difference: {vol_diff:.2f} cc")
    """
    return abs(structure1.volume_cc() - structure2.volume_cc())


def compute_volume_ratio(structure1: Structure, structure2: Structure) -> float:
    """
    Compute volume ratio V1/V2.
    
    Args:
        structure1: First structure (numerator)
        structure2: Second structure (denominator)
        
    Returns:
        Volume ratio (dimensionless)
        
    Examples:
        >>> ratio = compute_volume_ratio(auto_ptv, manual_ptv)
        >>> print(f"Volume ratio: {ratio:.3f}")
    """
    v2 = structure2.volume_cc()
    if v2 == 0:
        return float('inf') if structure1.volume_cc() > 0 else 1.0
    
    return structure1.volume_cc() / v2


def compute_sensitivity(structure1: Structure, structure2: Structure) -> float:
    """
    Compute sensitivity (recall, true positive rate).
    
    Sensitivity = TP / (TP + FN) = |A ∩ B| / |B|
    
    Measures how much of structure2 is covered by structure1.
    
    Args:
        structure1: Predicted/test structure
        structure2: Reference/ground truth structure
        
    Returns:
        Sensitivity (0-1)
        
    Examples:
        >>> sens = compute_sensitivity(auto_structure, manual_structure)
        >>> print(f"Sensitivity: {sens:.3f}")
    """
    if structure1.mask is None or structure2.mask is None:
        return 0.0
    
    intersection = np.logical_and(structure1.mask, structure2.mask)
    v2 = structure2.volume_voxels()
    
    if v2 == 0:
        return 0.0
    
    return float(np.sum(intersection) / v2)


def compute_specificity(
    structure1: Structure, 
    structure2: Structure,
    background_mask: Optional[np.ndarray] = None
) -> float:
    """
    Compute specificity (true negative rate).
    
    Specificity = TN / (TN + FP)
    
    Requires definition of background/universe. If not provided, uses
    the bounding box union of both structures.
    
    Args:
        structure1: Predicted/test structure
        structure2: Reference/ground truth structure
        background_mask: Mask defining the universe (optional)
        
    Returns:
        Specificity (0-1)
        
    Examples:
        >>> spec = compute_specificity(auto_structure, manual_structure)
        >>> print(f"Specificity: {spec:.3f}")
    """
    if structure1.mask is None or structure2.mask is None:
        return 0.0
    
    # True negatives: voxels outside both structures
    # False positives: in structure1 but not in structure2
    not_s1 = ~structure1.mask
    not_s2 = ~structure2.mask
    
    true_negatives = np.logical_and(not_s1, not_s2)
    false_positives = np.logical_and(structure1.mask, not_s2)
    
    denominator = np.sum(true_negatives) + np.sum(false_positives)
    
    if denominator == 0:
        return 0.0
    
    return float(np.sum(true_negatives) / denominator)


def compute_hausdorff_distance(
    structure1: Structure,
    structure2: Structure,
    percentile: Optional[float] = None
) -> float:
    """
    Compute Hausdorff distance between two structures.
    
    If percentile is specified, computes the percentile Hausdorff distance
    (e.g., 95th percentile HD95), which is more robust to outliers.
    
    Args:
        structure1: First structure
        structure2: Second structure
        percentile: If specified, compute percentile HD (e.g., 95 for HD95)
        
    Returns:
        Hausdorff distance in mm
        
    Examples:
        >>> hd = compute_hausdorff_distance(auto_structure, manual_structure)
        >>> hd95 = compute_hausdorff_distance(auto_structure, manual_structure, percentile=95)
    """
    # Validate percentile
    if percentile is not None:
        if not (0 < percentile <= 100):
            raise ValueError(f"Percentile must be between 0 and 100, got {percentile}")
    
    if structure1.mask is None or structure2.mask is None:
        return float('inf')
    
    # Get surface points (boundary voxels)
    # Use binary erosion to get boundary
    eroded1 = ndimage.binary_erosion(structure1.mask)
    eroded2 = ndimage.binary_erosion(structure2.mask)
    surface1 = structure1.mask & ~eroded1
    surface2 = structure2.mask & ~eroded2
    
    # Get coordinates of surface points
    points1 = np.argwhere(surface1)
    points2 = np.argwhere(surface2)
    
    if len(points1) == 0 or len(points2) == 0:
        return float('inf')
    
    # Scale by voxel spacing to get mm
    spacing = np.array(structure1.spacing)
    points1_mm = points1 * spacing
    points2_mm = points2 * spacing
    
    if percentile is not None:
        # Compute percentile Hausdorff distance
        # Calculate distances from points1 to points2
        from scipy.spatial.distance import cdist
        distances = cdist(points1_mm, points2_mm)
        
        # For each point in set 1, find min distance to set 2
        min_distances_1_to_2 = np.min(distances, axis=1)
        # For each point in set 2, find min distance to set 1
        min_distances_2_to_1 = np.min(distances, axis=0)
        
        # Compute percentile
        hd_1_to_2 = np.percentile(min_distances_1_to_2, percentile)
        hd_2_to_1 = np.percentile(min_distances_2_to_1, percentile)
        
        return float(max(hd_1_to_2, hd_2_to_1))
    else:
        # Standard Hausdorff distance
        hd_1_to_2, _, _ = directed_hausdorff(points1_mm, points2_mm)
        hd_2_to_1, _, _ = directed_hausdorff(points2_mm, points1_mm)
        
        return float(max(hd_1_to_2, hd_2_to_1))


def compute_mean_surface_distance(
    structure1: Structure,
    structure2: Structure
) -> float:
    """
    Compute mean surface distance between two structures.
    
    Average of all point-to-surface distances (symmetric).
    
    Args:
        structure1: First structure
        structure2: Second structure
        
    Returns:
        Mean surface distance in mm
    """
    if structure1.mask is None or structure2.mask is None:
        return float('inf')
    
    # Get surface points (boundary voxels)
    eroded1 = ndimage.binary_erosion(structure1.mask)
    eroded2 = ndimage.binary_erosion(structure2.mask)
    surface1 = structure1.mask & ~eroded1
    surface2 = structure2.mask & ~eroded2
    
    # Get coordinates of surface points
    points1 = np.argwhere(surface1)
    points2 = np.argwhere(surface2)
    
    if len(points1) == 0 or len(points2) == 0:
        return float('inf')
    
    # Scale by voxel spacing to get mm
    spacing = np.array(structure1.spacing)
    points1_mm = points1 * spacing
    points2_mm = points2 * spacing
    
    # Compute pairwise distances
    from scipy.spatial.distance import cdist
    distances = cdist(points1_mm, points2_mm)
    
    # Mean of minimum distances from each point to other surface
    mean_1_to_2 = np.mean(np.min(distances, axis=1))
    mean_2_to_1 = np.mean(np.min(distances, axis=0))
    
    # Return symmetric average
    return float((mean_1_to_2 + mean_2_to_1) / 2.0)


def compare_structure_sets(
    structure_set1: StructureSet,
    structure_set2: StructureSet,
    structure_names: Optional[list] = None
) -> pd.DataFrame:
    """
    Compute geometric metrics between two structure sets.
    
    Args:
        structure_set1: First structure set (e.g., auto-segmentation)
        structure_set2: Second structure set (e.g., manual segmentation)
        structure_names: List of structure names to compare (optional)
        
    Returns:
        DataFrame with geometric metrics for each structure
        
    Examples:
        >>> auto_structures = load_structure_set("auto/")
        >>> manual_structures = load_structure_set("manual/")
        >>> comparison = compare_structure_sets(auto_structures, manual_structures)
        >>> print(comparison)
    """
    if structure_names is None:
        # Use common structures
        names1 = set(structure_set1.structure_names)
        names2 = set(structure_set2.structure_names)
        structure_names = list(names1.intersection(names2))
    
    results = []
    
    for name in structure_names:
        try:
            struct1 = structure_set1.get_structure(name)
            struct2 = structure_set2.get_structure(name)
            
            dice = compute_dice_coefficient(struct1, struct2)
            jaccard = compute_jaccard_index(struct1, struct2)
            vol_diff = compute_volume_difference(struct1, struct2)
            vol_ratio = compute_volume_ratio(struct1, struct2)
            sensitivity = compute_sensitivity(struct1, struct2)
            
            results.append({
                'Structure': name,
                'Dice': dice,
                'Jaccard': jaccard,
                'Volume_Difference_cc': vol_diff,
                'Volume_Ratio': vol_ratio,
                'Sensitivity': sensitivity,
            })
        except ValueError:
            # Structure not found in one of the sets
            continue
    
    return pd.DataFrame(results)
