"""
Utility functions for loading example data from HuggingFace in the Streamlit app.
"""

import streamlit as st
from pathlib import Path
from huggingface_hub import snapshot_download
import tempfile
import shutil


@st.cache_resource
def download_example_data():
    """
    Download example data from HuggingFace and cache it.
    
    Returns:
        Path: Path to the downloaded data directory
    """
    try:
        data_path = snapshot_download(
            repo_id="contouraid/dosemetrics-data",
            repo_type="dataset"
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
