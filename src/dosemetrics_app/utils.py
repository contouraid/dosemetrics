"""
Utility functions for loading example data from HuggingFace in the Streamlit app.
"""

import streamlit as st
from pathlib import Path
from huggingface_hub import snapshot_download
import tempfile
import shutil
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional

import dosemetrics
from dosemetrics import Dose, Target, OAR, Structure
from dosemetrics.metrics import dvh


def infer_structure_type(name: str) -> str:
    """
    Infer if a structure is a target or OAR based on its name.

    Parameters
    ----------
    name : str
        Structure name

    Returns
    -------
    str
        'target' or 'oar'
    """
    name_upper = name.upper()
    target_keywords = ["PTV", "CTV", "GTV", "TARGET", "TUMOR", "TUMOUR"]

    for keyword in target_keywords:
        if keyword in name_upper:
            return "target"

    return "oar"


@st.cache_resource
def download_example_data():
    """
    Download example data from HuggingFace and cache it.

    Returns:
        Path: Path to the downloaded data directory
    """
    try:
        data_path = snapshot_download(
            repo_id="contouraid/dosemetrics-data", repo_type="dataset"
        )
        return Path(data_path)
    except Exception as e:
        st.error(f"Error downloading example data: {e}")
        return None


def get_example_datasets():
    """
    Get list of available example datasets from HuggingFace.

    Returns:
        dict: Dictionary mapping dataset names to paths, with test_subject as default
    """
    datasets = {}

    # Get HuggingFace data
    data_path = download_example_data()
    if data_path is None:
        return {}

    # Add test_subject (default option)
    test_subject_path = data_path / "test_subject"
    if test_subject_path.exists() and (test_subject_path / "Dose.nii.gz").exists():
        datasets["test_subject"] = test_subject_path

    # Add longitudinal timepoints
    longitudinal_path = data_path / "longitudinal"
    if longitudinal_path.exists():
        for time_point in sorted(longitudinal_path.iterdir()):
            if time_point.is_dir() and (time_point / "Dose.nii.gz").exists():
                datasets[time_point.name] = time_point

    return datasets


def load_example_files(dataset_path):
    """
    Load dose and mask files from an example dataset.

    Args:
        dataset_path: Path to the dataset directory

    Returns:
        tuple: (dose_file_path, list of mask_file_paths)
    """
    dataset_path = Path(dataset_path)

    # Find dose file
    dose_file = None
    for f in dataset_path.glob("Dose*.nii.gz"):
        dose_file = f
        break

    # Find mask files (everything except dose and CT)
    mask_files = []
    for f in dataset_path.glob("*.nii.gz"):
        if "Dose" not in f.name and "CT" not in f.name:
            mask_files.append(f)

    return dose_file, sorted(mask_files)


def read_byte_data(
    dose_file,
    mask_files,
) -> Tuple[Dose, Dict[str, Structure]]:
    """
    Read dose and mask data from Streamlit uploaded files or example data paths.

    This function handles multiple input types:
    - Uploaded files (BytesIO objects with .read() method)
    - Raw bytes
    - File paths (Path objects)

    Parameters
    ----------
    dose_file : BytesIO, bytes, Path, or str
        Dose NIfTI file content or path
    mask_files : list of BytesIO, bytes, Path, or dict
        List of mask files or dict mapping names to files

    Returns
    -------
    tuple
        (dose_object, structures_dict) where:
        - dose_object: Dose object with dose distribution
        - structures_dict: Dictionary mapping structure names to Structure objects
    """
    # Create temporary directory for file operations
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Handle dose file - convert to bytes if needed
        if isinstance(dose_file, (str, Path)):
            # Direct path - just load it
            dose_array, spacing, origin = dosemetrics.load_volume(str(dose_file))
            dose = Dose(dose_array, spacing, origin)
        else:
            # Handle BytesIO or bytes
            if hasattr(dose_file, "read"):
                dose_bytes = dose_file.read()
                dose_filename = getattr(dose_file, "name", "dose.nii.gz")
            else:
                dose_bytes = dose_file
                dose_filename = "dose.nii.gz"

            # Write dose file
            dose_path = temp_path / dose_filename
            dose_path.write_bytes(dose_bytes)

            # Load dose using dosemetrics
            dose_array, spacing, origin = dosemetrics.load_volume(str(dose_path))
            dose = Dose(dose_array, spacing, origin)

        # Handle mask files
        structures = {}

        # Convert list to dict if needed
        if isinstance(mask_files, list):
            mask_dict = {}
            for mf in mask_files:
                if hasattr(mf, "name"):
                    name = Path(mf.name).stem.replace(".nii", "")
                else:
                    name = f"Structure_{len(mask_dict)}"
                mask_dict[name] = mf
            mask_files = mask_dict

        for struct_name, mask_file in mask_files.items():
            if isinstance(mask_file, (str, Path)):
                # Direct path - just load it
                mask_array, mask_spacing, mask_origin = dosemetrics.load_volume(
                    str(mask_file)
                )
            else:
                # Handle BytesIO or bytes
                if hasattr(mask_file, "read"):
                    mask_bytes = mask_file.read()
                    mask_filename = getattr(mask_file, "name", f"{struct_name}.nii.gz")
                else:
                    mask_bytes = mask_file
                    mask_filename = f"{struct_name}.nii.gz"

                # Write mask file
                safe_name = struct_name.replace(" ", "_").replace("/", "_")
                mask_path = temp_path / f"{safe_name}.nii.gz"
                mask_path.write_bytes(mask_bytes)

                # Load mask using dosemetrics
                mask_array, mask_spacing, mask_origin = dosemetrics.load_volume(
                    str(mask_path)
                )

            # Create Structure object (Target or OAR based on name)
            structure_type = infer_structure_type(struct_name)
            if structure_type == "target":
                structure = Target(
                    name=struct_name,
                    mask=mask_array > 0.5,  # Binarize if needed
                    spacing=mask_spacing if "mask_spacing" in locals() else spacing,
                    origin=mask_origin if "mask_origin" in locals() else origin,
                )
            else:
                structure = OAR(
                    name=struct_name,
                    mask=mask_array > 0.5,  # Binarize if needed
                    spacing=mask_spacing if "mask_spacing" in locals() else spacing,
                    origin=mask_origin if "mask_origin" in locals() else origin,
                )
            structures[struct_name] = structure

    return dose, structures


def dvh_by_structure(dose: Dose, structures: Dict[str, Structure]) -> pd.DataFrame:
    """
    Compute DVH for multiple structures and return as a DataFrame.

    Parameters
    ----------
    dose : Dose
        Dose distribution object
    structures : dict
        Dictionary mapping structure names to Structure objects

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Dose, Volume, Structure
    """
    results = []

    for struct_name, struct in structures.items():
        # Compute DVH with adaptive step size
        step_size = dose.max_dose / 100  # 100 bins
        dose_bins, volumes = dvh.compute_dvh(dose, struct, step_size=step_size)

        for dose_val, volume_val in zip(dose_bins, volumes):
            results.append(
                {"Dose": dose_val, "Volume": volume_val, "Structure": struct_name}
            )

    return pd.DataFrame(results)
