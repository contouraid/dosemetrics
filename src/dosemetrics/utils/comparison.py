"""
High-level comparison workflows for dose and structure comparison.

This module provides convenient functions for comparing dose distributions,
quality indices, and generating comparison visualizations.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt


def compare_dose_distributions(
    dose1: Dose,
    dose2: Dose,
    structure_set: StructureSet,
    structure_names: Optional[List[str]] = None,
    output_file: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compare two dose distributions across multiple structures.
    
    Args:
        dose1: First dose distribution (e.g., reference)
        dose2: Second dose distribution (e.g., predicted)
        structure_set: StructureSet containing structures
        structure_names: List of structure names to compare (optional)
        output_file: Output PDF file path for DVH plots (optional)
        
    Returns:
        DataFrame with dose comparison metrics
        
    Examples:
        >>> from dosemetrics.dose import Dose
        >>> from dosemetrics.io import load_structure_set
        >>> 
        >>> dose_ref = Dose.from_dicom("plan1.dcm")
        >>> dose_pred = Dose.from_nifti("predicted.nii.gz")
        >>> structures = load_structure_set("structures/")
        >>> 
        >>> comparison = compare_dose_distributions(
        ...     dose_ref, dose_pred, structures, output_file="comparison.pdf"
        ... )
    """
    from ..metrics import statistics, dvh
    from ..utils.plot import compare_dvh
    
    if structure_names is None:
        structure_names = structure_set.structure_names
    
    # Create PDF for plots if requested
    pp = None
    if output_file:
        pp = PdfPages(output_file)
    
    comparison_data = []
    
    for name in structure_names:
        if name in ["CT", "Dose_Mask"]:
            continue
        
        try:
            structure = structure_set.get_structure(name)
            
            # Compute statistics for both doses
            stats1 = statistics.compute_dose_statistics(dose1, structure)
            stats2 = statistics.compute_dose_statistics(dose2, structure)
            
            # Compute differences
            row = {
                'Structure': name,
                'Mean_Dose_1': stats1['mean_dose'],
                'Mean_Dose_2': stats2['mean_dose'],
                'Mean_Diff': stats2['mean_dose'] - stats1['mean_dose'],
                'Mean_Diff_Pct': 100 * (stats2['mean_dose'] - stats1['mean_dose']) / (stats1['mean_dose'] + 1e-10),
                'Max_Dose_1': stats1['max_dose'],
                'Max_Dose_2': stats2['max_dose'],
                'Max_Diff': stats2['max_dose'] - stats1['max_dose'],
                'D95_1': stats1['D95'],
                'D95_2': stats2['D95'],
                'D95_Diff': stats2['D95'] - stats1['D95'],
            }
            comparison_data.append(row)
            
            # Plot DVH comparison if PDF output requested
            if pp is not None:
                try:
                    fig = compare_dvh(dose1, dose2, structure, 
                                      labels=[dose1.name, dose2.name])
                    plt.title(f"{name} - DVH Comparison")
                    plt.grid(True)
                    pp.savefig(fig)
                    plt.close()
                except Exception as plot_error:
                    print(f"Error plotting DVH for {name}: {plot_error}")
        
        except Exception as e:
            print(f"Error processing {name}: {e}")
            continue
    
    if pp is not None:
        pp.close()
    
    return pd.DataFrame(comparison_data)


def compare_quality_indices(
    doses: List[Dose],
    structure_set: StructureSet,
    constraints: Optional[pd.DataFrame] = None,
    labels: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compare quality indices between multiple dose distributions.
    
    Args:
        doses: List of Dose objects to compare
        structure_set: StructureSet containing structures
        constraints: Dose constraints DataFrame (optional)
        labels: Labels for each dose (optional)
        
    Returns:
        DataFrame with quality index comparison
        
    Examples:
        >>> doses = [dose_plan1, dose_plan2, dose_plan3]
        >>> labels = ["Plan A", "Plan B", "Plan C"]
        >>> comparison = compare_quality_indices(doses, structures, labels=labels)
    """
    from ..utils.compliance import get_default_constraints, quality_index
    
    if constraints is None:
        constraints = get_default_constraints()
    
    if labels is None:
        labels = [f"Dose_{i+1}" for i in range(len(doses))]
    
    results = []
    
    for name in structure_set.structure_names:
        if name not in constraints.index:
            continue
        
        try:
            structure = structure_set.get_structure(name)
            
            constraint_type = str(constraints.loc[name, "Constraint Type"])
            constraint_limit = float(constraints.loc[name, "Level"])
            
            row = {
                'Structure': name,
                'Type': constraint_type,
                'Limit': constraint_limit,
            }
            
            for i, dose in enumerate(doses):
                qi = quality_index(dose, structure, constraint_type, constraint_limit)
                row[f'QI_{labels[i]}'] = qi
            
            # Calculate difference if comparing two plans
            if len(doses) == 2:
                row['QI_Difference'] = row[f'QI_{labels[1]}'] - row[f'QI_{labels[0]}']
            
            results.append(row)
        
        except Exception as e:
            print(f"Error processing {name}: {e}")
            continue
    
    return pd.DataFrame(results)


def compare_treatment_plans(
    plan1_folder: str,
    plan2_folder: str,
    output_folder: str,
    plan1_name: str = "Plan1",
    plan2_name: str = "Plan2",
) -> Dict[str, pd.DataFrame]:
    """
    Comprehensive comparison of two treatment plans.
    
    Args:
        plan1_folder: Folder containing first plan data
        plan2_folder: Folder containing second plan data
        output_folder: Output folder for results
        plan1_name: Name for first plan (default: "Plan1")
        plan2_name: Name for second plan (default: "Plan2")
        
    Returns:
        Dictionary with comparison results
        
    Examples:
        >>> results = compare_treatment_plans(
        ...     "plan_v1/",
        ...     "plan_v2/",
        ...     "comparison_output/"
        ... )
    """
    from ..dose import Dose
    from ..io.data_io import load_structure_set
    from ..metrics import statistics, geometric
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Load dose distributions
    dose1 = Dose.from_nifti(os.path.join(plan1_folder, "Dose.nii.gz"), name=plan1_name)
    dose2 = Dose.from_nifti(os.path.join(plan2_folder, "Dose.nii.gz"), name=plan2_name)
    
    # Load structure sets
    structures1 = load_structure_set(plan1_folder)
    structures2 = load_structure_set(plan2_folder)
    
    results = {}
    
    # 1. Dose comparison
    dose_comparison = compare_dose_distributions(
        dose1, dose2, structures1,
        output_file=os.path.join(output_folder, "dvh_comparison.pdf")
    )
    dose_comparison.to_csv(os.path.join(output_folder, "dose_comparison.csv"), index=False)
    results['dose_comparison'] = dose_comparison
    
    # 2. Structure comparison (if both plans have structures)
    structure_comparison = geometric.compare_structure_sets(structures1, structures2)
    structure_comparison.to_csv(os.path.join(output_folder, "structure_comparison.csv"), index=False)
    results['structure_comparison'] = structure_comparison
    
    # 3. Quality indices
    qi_comparison = compare_quality_indices([dose1, dose2], structures1, labels=[plan1_name, plan2_name])
    qi_comparison.to_csv(os.path.join(output_folder, "quality_index_comparison.csv"), index=False)
    results['quality_index'] = qi_comparison
    
    return results


def dose_difference_analysis(
    dose1: Dose,
    dose2: Dose,
    output_folder: str,
    threshold_gy: float = 3.0,
) -> Dict[str, any]:
    """
    Analyze point-by-point dose differences between two distributions.
    
    Args:
        dose1: First dose distribution (reference)
        dose2: Second dose distribution (evaluated)
        output_folder: Output folder for results
        threshold_gy: Threshold for significant differences in Gy
        
    Returns:
        Dictionary with difference statistics
        
    Examples:
        >>> diff_analysis = dose_difference_analysis(
        ...     dose_reference, dose_predicted,
        ...     "diff_analysis/",
        ...     threshold_gy=3.0
        ... )
    """
    import nibabel as nib
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Compute dose difference
    dose_diff = dose2.dose_array - dose1.dose_array
    
    # Compute statistics
    stats = {
        'mean_difference': float(np.mean(dose_diff)),
        'std_difference': float(np.std(dose_diff)),
        'max_positive_diff': float(np.max(dose_diff)),
        'max_negative_diff': float(np.min(dose_diff)),
        'mean_absolute_difference': float(np.mean(np.abs(dose_diff))),
        'rmse': float(np.sqrt(np.mean(dose_diff ** 2))),
    }
    
    # Count voxels exceeding threshold
    exceeds_threshold = np.abs(dose_diff) > threshold_gy
    stats['voxels_exceeding_threshold'] = int(np.sum(exceeds_threshold))
    stats['percent_exceeding_threshold'] = 100 * np.sum(exceeds_threshold) / dose_diff.size
    
    # Save difference volume as NIfTI
    diff_img = nib.Nifti1Image(dose_diff, affine=np.eye(4))
    nib.save(diff_img, os.path.join(output_folder, "dose_difference.nii.gz"))
    
    # Save statistics
    stats_df = pd.DataFrame([stats])
    stats_df.to_csv(os.path.join(output_folder, "difference_statistics.csv"), index=False)
    
    return stats