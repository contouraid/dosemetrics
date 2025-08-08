"""
Comparison and analysis functions for dose metrics.

This module provides functions for comparing predicted vs actual doses,
quality indices, and geometric metrics between different plans or predictions.
"""

import os
import glob
import numpy as np
import pandas as pd
import SimpleITK as sitk
from typing import Dict, List, Optional
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

from .dvh import compute_dvh
from .scores import dose_summary
from ..utils.compliance import quality_index, get_default_constraints
from ..utils.plot import compare_dvh
from ..io.data_io import read_from_nifti


def compare_predicted_doses(
    dose_array: np.ndarray,
    predicted_array: np.ndarray,
    structure_masks: Dict[str, np.ndarray],
    output_file: Optional[str] = None,
) -> None:
    """
    Compare actual vs predicted doses for multiple structures.

    Parameters:
    -----------
    dose_array : np.ndarray
        Ground truth dose array
    predicted_array : np.ndarray
        Predicted dose array
    structure_masks : Dict[str, np.ndarray]
        Dictionary mapping structure names to mask arrays
    output_file : str, optional
        Output PDF file path for plots
    """
    pp = None
    if output_file:
        pp = PdfPages(output_file)

    for struct_name, mask in structure_masks.items():
        if struct_name in ["CT", "Dose_Mask"]:
            continue

        fig = compare_dvh(dose_array, predicted_array, mask)
        plt.title(struct_name)
        plt.grid()

        if pp is not None:
            pp.savefig(fig)
        else:
            plt.show()

        plt.close()

    if pp is not None:
        pp.close()


def compare_quality_indices(
    dose_arrays: List[np.ndarray],
    structure_masks: Dict[str, np.ndarray],
    constraints: Optional[pd.DataFrame] = None,
    labels: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compare quality indices between multiple dose distributions.

    Parameters:
    -----------
    dose_arrays : List[np.ndarray]
        List of dose arrays to compare
    structure_masks : Dict[str, np.ndarray]
        Dictionary mapping structure names to mask arrays
    constraints : pd.DataFrame, optional
        Dose constraints DataFrame
    labels : List[str], optional
        Labels for each dose array

    Returns:
    --------
    pd.DataFrame
        Comparison results with quality indices
    """
    if constraints is None:
        constraints = get_default_constraints()

    if labels is None:
        labels = [f"Plan_{i+1}" for i in range(len(dose_arrays))]

    results = []

    for struct_name, mask in structure_masks.items():
        if struct_name not in constraints.index:
            continue

        constraint_type = str(constraints.loc[struct_name, "Constraint Type"])
        constraint_limit = float(constraints.loc[struct_name, "Level"])

        row = {
            "Structure": struct_name,
            "Type": constraint_type,
            "Limit": constraint_limit,
        }

        for i, dose_array in enumerate(dose_arrays):
            qi = quality_index(dose_array, mask, constraint_type, constraint_limit)
            row[f"QI_{labels[i]}"] = qi

        # Calculate difference if comparing two plans
        if len(dose_arrays) == 2:
            row["QI_Difference"] = row[f"QI_{labels[1]}"] - row[f"QI_{labels[0]}"]

        results.append(row)

    return pd.DataFrame(results)


def compute_geometric_metrics(
    structure_masks_1: Dict[str, np.ndarray],
    structure_masks_2: Dict[str, np.ndarray],
) -> pd.DataFrame:
    """
    Compute geometric metrics (Dice coefficient, etc.) between two sets of structure masks.

    Parameters:
    -----------
    structure_masks_1 : Dict[str, np.ndarray]
        First set of structure masks
    structure_masks_2 : Dict[str, np.ndarray]
        Second set of structure masks

    Returns:
    --------
    pd.DataFrame
        Geometric metrics for each structure
    """
    results = []

    for struct_name in structure_masks_1.keys():
        if struct_name not in structure_masks_2:
            continue

        mask1 = structure_masks_1[struct_name]
        mask2 = structure_masks_2[struct_name]

        # Dice coefficient
        intersection = np.logical_and(mask1, mask2)
        dice = (
            2 * intersection.sum() / (mask1.sum() + mask2.sum())
            if (mask1.sum() + mask2.sum()) > 0
            else 0
        )

        # Jaccard index
        union = np.logical_or(mask1, mask2)
        jaccard = intersection.sum() / union.sum() if union.sum() > 0 else 0

        # Volume difference
        vol_diff = abs(mask1.sum() - mask2.sum())

        results.append(
            {
                "Structure": struct_name,
                "Dice": dice,
                "Jaccard": jaccard,
                "Volume_Difference": vol_diff,
            }
        )

    return pd.DataFrame(results)


def batch_dvh_analysis(
    input_folders: List[str],
    output_folder: str,
    structure_list: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Perform DVH analysis on multiple subjects/folders.

    Parameters:
    -----------
    input_folders : List[str]
        List of input folder paths
    output_folder : str
        Output folder for results
    structure_list : List[str], optional
        List of structure names to analyze

    Returns:
    --------
    Dict[str, pd.DataFrame]
        Dictionary mapping folder names to DVH DataFrames
    """
    os.makedirs(output_folder, exist_ok=True)
    results = {}

    for folder in input_folders:
        folder_name = os.path.basename(folder)

        try:
            # Read dose file
            dose_file = os.path.join(folder, "Dose.nii.gz")
            if not os.path.exists(dose_file):
                print(f"Warning: No dose file found in {folder}")
                continue

            dose_array = read_from_nifti(dose_file)

            # Read structure masks
            if structure_list is None:
                # Auto-detect structures
                structure_files = glob.glob(os.path.join(folder, "*[!Dose*].nii.gz"))
                structure_list = [
                    os.path.basename(f).replace(".nii.gz", "")
                    for f in structure_files
                    if "CT" not in f and "Dose_Mask" not in f
                ]

            structure_masks = {}
            for struct_name in structure_list:
                struct_file = os.path.join(folder, f"{struct_name}.nii.gz")
                if os.path.exists(struct_file):
                    structure_masks[struct_name] = read_from_nifti(struct_file)

            # Compute dose summary
            df = dose_summary(dose_array, structure_masks)
            results[folder_name] = df

            # Save individual results
            df.to_csv(os.path.join(output_folder, f"{folder_name}_dvh.csv"))

        except Exception as e:
            print(f"Error processing {folder}: {e}")
            continue

    return results


def process_subject_folder(
    input_folder: str,
    output_folder: str,
    analysis_types: Optional[List[str]] = None,
    constraints: Optional[pd.DataFrame] = None,
) -> Dict[str, any]:
    """
    Comprehensive analysis of a single subject folder.

    Parameters:
    -----------
    input_folder : str
        Input folder path containing dose and structure files
    output_folder : str
        Output folder for results
    analysis_types : List[str]
        Types of analysis to perform
    constraints : pd.DataFrame, optional
        Dose constraints

    Returns:
    --------
    Dict[str, any]
        Dictionary containing analysis results
    """
    if analysis_types is None:
        analysis_types = ["dvh", "quality_index", "compliance"]

    if constraints is None:
        constraints = get_default_constraints()

    # Read dose file
    dose_file = os.path.join(input_folder, "Dose.nii.gz")
    dose_array = read_from_nifti(dose_file)

    # Auto-detect structures
    structure_files = glob.glob(os.path.join(input_folder, "*[!Dose*].nii.gz"))
    structure_masks = {}

    for struct_file in structure_files:
        struct_name = os.path.basename(struct_file).replace(".nii.gz", "")
        if struct_name not in ["CT", "Dose_Mask"]:
            structure_masks[struct_name] = read_from_nifti(struct_file)

    results = {}

    # DVH analysis
    if "dvh" in analysis_types:
        dvh_df = dose_summary(dose_array, structure_masks)
        dvh_df.to_csv(os.path.join(output_folder, "dvh_summary.csv"))
        results["dvh"] = dvh_df

    # Quality index analysis
    if "quality_index" in analysis_types:
        qi_results = []
        for struct_name, mask in structure_masks.items():
            if struct_name in constraints.index:
                try:
                    constraint_type = constraints.loc[struct_name, "Constraint Type"]
                    constraint_limit = constraints.loc[struct_name, "Level"]
                    qi = quality_index(dose_array, mask, str(constraint_type), float(constraint_limit))  # type: ignore
                except (ValueError, TypeError):
                    continue
                qi_results.append(
                    {
                        "Structure": struct_name,
                        "Quality_Index": qi,
                        "Constraint_Type": constraint_type,
                        "Constraint_Limit": constraint_limit,
                    }
                )

        qi_df = pd.DataFrame(qi_results)
        qi_df.to_csv(os.path.join(output_folder, "quality_index.csv"))
        results["quality_index"] = qi_df

    # Compliance checking
    if "compliance" in analysis_types:
        from ..utils.compliance import check_compliance

        compliance_df = check_compliance(
            dose_summary(dose_array, structure_masks), constraints
        )
        compliance_df.to_csv(os.path.join(output_folder, "compliance.csv"))
        results["compliance"] = compliance_df

    return results
