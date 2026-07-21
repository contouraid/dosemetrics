"""
Utility functions for loading example data from HuggingFace in the Streamlit app.
"""

import streamlit as st
from pathlib import Path
from huggingface_hub import snapshot_download
from io import BytesIO
import tempfile
import numpy as np
import pandas as pd
import pydicom
from typing import Tuple, Dict, Optional, Union

import dosemetrics
from dosemetrics import Dose, Target, OAR, Structure
from dosemetrics.io import dicom_io
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


def get_example_dicom_dataset():
    """Return the DICOM example folder from the hosted dataset, if available."""
    data_path = download_example_data()
    if data_path is None:
        return None

    dicom_path = data_path / "dicom"
    required_folders = (dicom_path / "RTDOSE", dicom_path / "RTSTRUCT")
    if all(folder.exists() for folder in required_folders):
        return dicom_path
    return None


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
    if isinstance(dose_file, Dose):
        return dose_file, dict(mask_files)

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


def load_dicom_analysis_data(
    dicom_path: Union[Path, str],
    dose_file: Optional[Union[Path, str]] = None,
) -> Tuple[Dose, Dict[str, Structure]]:
    """Load an RTDOSE and rasterize its RTSTRUCT directly on the dose grid.

    Rasterizing on the selected dose grid makes the returned structures immediately
    usable by the metric functions, even when the accompanying CT has a different
    resolution.
    """
    dicom_path = Path(dicom_path)
    dose_files = sorted((dicom_path / "RTDOSE").glob("*.dcm"))
    structure_files = sorted((dicom_path / "RTSTRUCT").glob("*.dcm"))

    if dose_file is None:
        if not dose_files:
            raise ValueError("No RTDOSE file was found.")
        dose_path = dose_files[0]
    else:
        dose_path = Path(dose_file)
    if not structure_files:
        raise ValueError("No RTSTRUCT file was found.")

    dose = Dose.from_dicom(dose_path)
    reference_grid = (dose.shape, dose.spacing, dose.origin)
    raw_structures = dicom_io.read_dicom_rtstruct(
        structure_files[0], reference_image=reference_grid
    )

    structures: Dict[str, Structure] = {}
    for name, values in raw_structures.items():
        mask = values.get("mask")
        if mask is None:
            continue
        structure_class = Target if infer_structure_type(name) == "target" else OAR
        structures[name] = structure_class(
            name=name,
            mask=mask,
            spacing=dose.spacing,
            origin=dose.origin,
        )

    if not structures:
        raise ValueError("The RTSTRUCT did not contain any rasterizable structures.")
    return dose, structures


def load_uploaded_dicom_data(uploaded_files) -> Tuple[Dose, Dict[str, Structure]]:
    """Load uploaded DICOM files after rebuilding modality subfolders."""
    if not uploaded_files:
        raise ValueError("Upload at least one RTDOSE and one RTSTRUCT file.")

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        modality_counts: Dict[str, int] = {}

        for uploaded_file in uploaded_files:
            if hasattr(uploaded_file, "getvalue"):
                contents = uploaded_file.getvalue()
            else:
                contents = uploaded_file.read()
            header = pydicom.dcmread(
                BytesIO(contents), stop_before_pixels=True, force=True
            )
            modality = str(getattr(header, "Modality", "UNKNOWN")).upper()
            if modality not in {"CT", "RTDOSE", "RTSTRUCT", "RTPLAN"}:
                continue

            modality_dir = root / modality
            modality_dir.mkdir(exist_ok=True)
            count = modality_counts.get(modality, 0)
            modality_counts[modality] = count + 1
            original_name = Path(getattr(uploaded_file, "name", "upload.dcm")).name
            if not original_name.lower().endswith(".dcm"):
                original_name = f"{original_name}.dcm"
            destination = modality_dir / f"{count:04d}_{original_name}"
            destination.write_bytes(contents)

        if modality_counts.get("RTDOSE", 0) == 0:
            raise ValueError("The upload does not contain an RTDOSE object.")
        if modality_counts.get("RTSTRUCT", 0) == 0:
            raise ValueError("The upload does not contain an RTSTRUCT object.")

        return load_dicom_analysis_data(root)


def request_analysis_data(
    instruction_text: str, key: str = "analysis"
) -> Tuple[Optional[Dose], Dict[str, Structure]]:
    """Render the shared hosted/NIfTI/DICOM selector used by the live demo."""
    st.markdown(instruction_text)
    st.caption(
        "Start with a hosted study from contouraid/dosemetrics-data, or upload "
        "your own NIfTI or DICOM-RT files."
    )
    source = st.radio(
        "Data source",
        ["Hosted example", "Upload NIfTI", "Upload DICOM"],
        horizontal=True,
        key=f"{key}_source",
    )

    try:
        if source == "Hosted example":
            data_format = st.radio(
                "Example format",
                ["NIfTI", "DICOM"],
                horizontal=True,
                key=f"{key}_example_format",
            )
            if data_format == "NIfTI":
                datasets = get_example_datasets()
                if not datasets:
                    return None, {}
                names = list(datasets)
                selected = st.selectbox(
                    "Study",
                    names,
                    index=names.index("test_subject") if "test_subject" in names else 0,
                    key=f"{key}_nifti_study",
                )
                dose_path, mask_paths = load_example_files(datasets[selected])
                if dose_path is None:
                    raise ValueError(f"No Dose.nii.gz found in {selected}.")
                with st.spinner(f"Loading hosted NIfTI study “{selected}”…"):
                    dose, structures = read_byte_data(dose_path, mask_paths)
                st.success(f"Loaded {selected}: {len(structures)} structures")
                return dose, structures

            dicom_path = get_example_dicom_dataset()
            if dicom_path is None:
                raise ValueError("The hosted DICOM example is unavailable.")
            dose_files = sorted((dicom_path / "RTDOSE").glob("*.dcm"))
            selected_name = st.selectbox(
                "RTDOSE object",
                [path.name for path in dose_files],
                key=f"{key}_dicom_dose",
            )
            selected_dose = next(path for path in dose_files if path.name == selected_name)
            with st.spinner(f"Loading hosted DICOM study with {selected_name}…"):
                dose, structures = load_dicom_analysis_data(dicom_path, selected_dose)
            st.success(
                f"Loaded {selected_name}: {len(structures)} RTSTRUCT regions on the dose grid"
            )
            return dose, structures

        if source == "Upload NIfTI":
            dose_file = st.file_uploader(
                "Dose volume", type=["nii", "gz"], key=f"{key}_nifti_dose"
            )
            mask_files = st.file_uploader(
                "Structure masks",
                type=["nii", "gz"],
                accept_multiple_files=True,
                key=f"{key}_nifti_masks",
            )
            if dose_file is None or not mask_files:
                st.info("Upload one dose volume and at least one binary structure mask.")
                return None, {}
            with st.spinner("Loading uploaded NIfTI data…"):
                return read_byte_data(dose_file, mask_files)

        uploaded_files = st.file_uploader(
            "DICOM-RT files",
            accept_multiple_files=True,
            help=(
                "Select at least one RTDOSE and one RTSTRUCT file. CT and RTPLAN "
                "files may be included; files are identified by their DICOM modality, "
                "so extensionless DICOM files are supported."
            ),
            key=f"{key}_dicom_files",
        )
        if not uploaded_files:
            st.info("Upload an RTDOSE and RTSTRUCT to begin. CT images are optional.")
            return None, {}
        with st.spinner("Loading uploaded DICOM-RT data…"):
            return load_uploaded_dicom_data(uploaded_files)
    except Exception as exc:
        st.error(f"Could not load the selected data: {exc}")
        return None, {}


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
