"""Small data-loading workflows used in tutorials and interactive analyses."""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Optional, Tuple, Union

from ..dose import Dose
from ..io import dicom_io, load_structure_set
from ..structure_set import StructureSet
from ..structures import StructureType


EXAMPLE_DATASET = "contouraid/dosemetrics-data"


def download_example_data(
    relative_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Download the public example dataset and return a cached local path.

    Parameters
    ----------
    relative_path : str or Path, optional
        A path inside the dataset, such as ``"test_subject"``.
    """
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="IProgress not found.*")
        from huggingface_hub import snapshot_download

        root = Path(snapshot_download(repo_id=EXAMPLE_DATASET, repo_type="dataset"))
    return root / relative_path if relative_path is not None else root


def load_example_study(
    relative_path: Union[str, Path] = "test_subject",
    dose_filename: str = "Dose.nii.gz",
) -> Tuple[Dose, StructureSet]:
    """Load an aligned NIfTI dose and structure set from the example dataset."""
    study_path = download_example_data(relative_path)
    dose = Dose.from_nifti(study_path / dose_filename, name=study_path.name)
    structures = load_structure_set(study_path, format="nifti")
    return dose, structures


def load_dicom_study(
    directory: Union[str, Path],
    dose_file: Optional[Union[str, Path]] = None,
) -> Tuple[Dose, StructureSet]:
    """Load RTDOSE and rasterize RTSTRUCT contours on its native dose grid.

    This high-level loader ensures that every returned structure is directly
    compatible with the returned dose distribution.
    """
    directory = Path(directory)
    if dose_file is None:
        dose_candidates = sorted((directory / "RTDOSE").glob("*.dcm"))
        if not dose_candidates:
            raise FileNotFoundError(f"No RTDOSE file found in {directory}")
        dose_path = dose_candidates[0]
    else:
        dose_path = Path(dose_file)

    rtstruct_candidates = sorted((directory / "RTSTRUCT").glob("*.dcm"))
    if not rtstruct_candidates:
        raise FileNotFoundError(f"No RTSTRUCT file found in {directory}")

    dose = Dose.from_dicom(dose_path, name=dose_path.stem)
    rtstruct_path = rtstruct_candidates[0]
    raw_structures = dicom_io.read_dicom_rtstruct(
        rtstruct_path,
        reference_image=(dose.shape, dose.spacing, dose.origin),
    )
    structures = StructureSet(
        spacing=dose.spacing,
        origin=dose.origin,
        name=rtstruct_path.stem,
    )
    for name, values in raw_structures.items():
        if "mask" not in values:
            continue
        upper_name = name.upper()
        structure_type = (
            StructureType.TARGET
            if any(token in upper_name for token in ("PTV", "CTV", "GTV"))
            else StructureType.OAR
        )
        structures.add_structure(name, values["mask"], structure_type)

    return dose, structures


def load_dicom_ct_on_dose_grid(
    directory: Union[str, Path], dose: Dose
) -> np.ndarray:
    """Load a DICOM CT series and resample it onto a dose grid for display."""
    import numpy as np
    import pydicom
    import SimpleITK as sitk

    directory = Path(directory)
    ct_directory = directory / "CT" if (directory / "CT").is_dir() else directory
    ct_volume, ct_spacing, ct_origin = dicom_io.read_dicom_ct_volume(ct_directory)
    ct_display = ct_volume.astype(np.float32)

    # Some series store signed CT values with unsigned pixel metadata.
    if np.percentile(ct_display, 99) > 10_000:
        first_slice = sorted(ct_directory.glob("*.dcm"))[0]
        metadata = pydicom.dcmread(first_slice, stop_before_pixels=True)
        slope = float(getattr(metadata, "RescaleSlope", 1.0))
        intercept = float(getattr(metadata, "RescaleIntercept", 0.0))
        stored = np.rint((ct_display - intercept) / slope).astype(np.uint16)
        ct_display = stored.view(np.int16).astype(np.float32)

    ct_image = sitk.GetImageFromArray(ct_display)
    ct_image.SetSpacing(ct_spacing)
    ct_image.SetOrigin(ct_origin)
    dose_image = sitk.GetImageFromArray(np.zeros(dose.shape, dtype=np.float32))
    dose_image.SetSpacing(dose.spacing)
    dose_image.SetOrigin(dose.origin)
    return sitk.GetArrayFromImage(
        sitk.Resample(
            ct_image,
            dose_image,
            sitk.Transform(),
            sitk.sitkLinear,
            -1000.0,
        )
    )
