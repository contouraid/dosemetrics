"""
Batch processing and workflow utilities for dosemetrics.

This module provides high-level functions for processing multiple subjects,
directories, and performing common dosimetric workflows.
"""

import os
import glob
from typing import Dict, List, Optional
import pandas as pd
import SimpleITK as sitk
from ..io.data_io import read_from_nifti


def get_structures_from_folder(
    input_folder: str, exclude_patterns: Optional[List[str]] = None
) -> List[str]:
    """
    Auto-detect structure files in a folder.

    Parameters:
    -----------
    input_folder : str
        Path to folder containing structure files
    exclude_patterns : List[str], optional
        Patterns to exclude (default: ["CT", "Dose"])

    Returns:
    --------
    List[str]
        List of structure names found
    """
    if exclude_patterns is None:
        exclude_patterns = ["CT", "Dose_Mask", "Dose"]

    structure_files = glob.glob(os.path.join(input_folder, "*.nii.gz"))
    structure_names = []

    for struct_file in structure_files:
        struct_name = os.path.basename(struct_file).replace(".nii.gz", "")

        # Check if structure should be excluded
        exclude = False
        for pattern in exclude_patterns:
            if pattern in struct_name:
                exclude = True
                break

        if not exclude:
            structure_names.append(struct_name)

    return sorted(structure_names)


def read_dose_and_mask_files_from_folder(
    input_folder: str,
    dose_filename: str = "Dose.nii.gz",
    structure_list: Optional[List[str]] = None,
) -> tuple:
    """
    Read dose and structure mask files from a folder.

    Parameters:
    -----------
    input_folder : str
        Path to folder containing files
    dose_filename : str
        Name of dose file
    structure_list : List[str], optional
        List of structure names to read

    Returns:
    --------
    tuple
        (dose_array, structure_masks_dict)
    """
    # Read dose file
    dose_file = os.path.join(input_folder, dose_filename)
    if not os.path.exists(dose_file):
        raise FileNotFoundError(f"Dose file not found: {dose_file}")

    dose_array = read_from_nifti(dose_file)

    # Auto-detect structures if not provided
    if structure_list is None:
        structure_list = get_structures_from_folder(input_folder)

    # Read structure masks
    structure_masks = {}
    for struct_name in structure_list:
        struct_file = os.path.join(input_folder, f"{struct_name}.nii.gz")
        if os.path.exists(struct_file):
            structure_masks[struct_name] = read_from_nifti(struct_file)
        else:
            print(f"Warning: Structure file not found: {struct_file}")

    return dose_array, structure_masks


def create_standard_contents_csv(
    input_folder: str,
    output_file: Optional[str] = None,
    target_structures: Optional[List[str]] = None,
    oar_structures: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Create a standard contents CSV file for a subject folder.

    Parameters:
    -----------
    input_folder : str
        Path to folder containing structure files
    output_file : str, optional
        Path to save CSV file
    target_structures : List[str], optional
        List of target structure names
    oar_structures : List[str], optional
        List of OAR structure names

    Returns:
    --------
    pd.DataFrame
        DataFrame with structure information
    """
    # Auto-detect all structures
    all_structures = get_structures_from_folder(input_folder)

    # Default classification
    if target_structures is None:
        target_structures = [
            s
            for s in all_structures
            if "Target" in s or "GTV" in s or "CTV" in s or "PTV" in s
        ]

    if oar_structures is None:
        oar_structures = [s for s in all_structures if s not in target_structures]

    # Create DataFrame
    contents = []

    # Add dose entry
    dose_file = os.path.join(input_folder, "Dose.nii.gz")
    if os.path.exists(dose_file):
        contents.append({"Structure": "Dose", "Type": "Dose"})

    # Add target structures
    for target in target_structures:
        if target in all_structures:
            contents.append({"Structure": target, "Type": "Target"})

    # Add OAR structures
    for oar in oar_structures:
        if oar in all_structures:
            contents.append({"Structure": oar, "Type": "OAR"})

    df = pd.DataFrame(contents)

    # Save if output file specified
    if output_file is None:
        output_file = os.path.join(input_folder, "standard_contents.csv")

    df.to_csv(output_file, index=False)
    return df


def validate_folder_structure(
    input_folder: str, required_files: Optional[List[str]] = None
) -> Dict[str, bool]:
    """
    Validate that a folder contains required files for dosimetric analysis.

    Parameters:
    -----------
    input_folder : str
        Path to folder to validate
    required_files : List[str], optional
        List of required files (default: ["Dose.nii.gz"])

    Returns:
    --------
    Dict[str, bool]
        Dictionary mapping file names to presence status
    """
    if required_files is None:
        required_files = ["Dose.nii.gz"]

    validation_results = {}

    for required_file in required_files:
        file_path = os.path.join(input_folder, required_file)
        validation_results[required_file] = os.path.exists(file_path)

    # Check for at least one structure file
    structure_files = get_structures_from_folder(input_folder)
    validation_results["has_structures"] = len(structure_files) > 0
    validation_results["structure_count"] = len(structure_files)
    validation_results["structure_names"] = structure_files

    return validation_results


def batch_folder_validation(
    input_folders: List[str],
    output_file: Optional[str] = None,
) -> pd.DataFrame:
    """
    Validate multiple folders and create a summary report.

    Parameters:
    -----------
    input_folders : List[str]
        List of folder paths to validate
    output_file : str, optional
        Path to save validation report CSV

    Returns:
    --------
    pd.DataFrame
        Validation summary DataFrame
    """
    validation_results = []

    for folder in input_folders:
        folder_name = os.path.basename(folder)
        validation = validate_folder_structure(folder)

        result = {
            "Folder": folder_name,
            "Path": folder,
            "Has_Dose": validation.get("Dose.nii.gz", False),
            "Structure_Count": validation.get("structure_count", 0),
            "Has_Structures": validation.get("has_structures", False),
            "Valid": validation.get("Dose.nii.gz", False)
            and validation.get("has_structures", False),
        }

        validation_results.append(result)

    df = pd.DataFrame(validation_results)

    if output_file:
        df.to_csv(output_file, index=False)

    return df


def find_subject_folders(
    root_path: str,
    pattern: str = "*",
    must_contain_dose: bool = True,
) -> List[str]:
    """
    Find subject folders matching a pattern.

    Parameters:
    -----------
    root_path : str
        Root directory to search
    pattern : str
        Pattern to match folder names
    must_contain_dose : bool
        Whether folders must contain a dose file

    Returns:
    --------
    List[str]
        List of matching folder paths
    """
    candidate_folders = glob.glob(os.path.join(root_path, pattern))
    valid_folders = []

    for folder in candidate_folders:
        if os.path.isdir(folder):
            if must_contain_dose:
                dose_file = os.path.join(folder, "Dose.nii.gz")
                if os.path.exists(dose_file):
                    valid_folders.append(folder)
            else:
                valid_folders.append(folder)

    return sorted(valid_folders)


def setup_output_structure(
    output_root: str,
    subject_names: List[str],
    analysis_types: List[str] = ["dvh", "quality_index", "compliance", "plots"],
) -> Dict[str, str]:
    """
    Create standardized output directory structure.

    Parameters:
    -----------
    output_root : str
        Root output directory
    subject_names : List[str]
        List of subject names
    analysis_types : List[str]
        Types of analysis to create folders for

    Returns:
    --------
    Dict[str, str]
        Dictionary mapping folder purposes to paths
    """
    os.makedirs(output_root, exist_ok=True)

    folder_structure = {
        "root": output_root,
        "summary": os.path.join(output_root, "summary"),
        "individual": os.path.join(output_root, "individual"),
    }

    # Create summary folders
    os.makedirs(folder_structure["summary"], exist_ok=True)

    # Create individual subject folders
    os.makedirs(folder_structure["individual"], exist_ok=True)

    for subject in subject_names:
        subject_folder = os.path.join(folder_structure["individual"], subject)
        os.makedirs(subject_folder, exist_ok=True)

        for analysis_type in analysis_types:
            analysis_folder = os.path.join(subject_folder, analysis_type)
            os.makedirs(analysis_folder, exist_ok=True)

    return folder_structure
