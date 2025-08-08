import os
import numpy as np
import pandas as pd
from glob import glob
from gzip import GzipFile
import SimpleITK as sitk
from nibabel.fileholders import FileHolder
from nibabel.nifti1 import Nifti1Image
from typing import Tuple, Dict, Optional, List


def find_all_files(name, path):
    result = []
    for root, dirs, files in os.walk(path):
        if name in files:
            result.append(os.path.join(root, name))
    result = sorted(result)
    return result


def read_file(byte_file):
    """
    READ_FILE: Read the file from the byte data.
    This is exclusively meant for what st.file_uploader returns.
    :param byte_file:
    :return: arr, img.header - numpy array and header information.
    """
    # See https://stackoverflow.com/questions/62579425/simpleitk-read-io-byteio
    fh = FileHolder(fileobj=GzipFile(fileobj=byte_file))
    img = Nifti1Image.from_file_map({"header": fh, "image": fh})
    arr = np.array(img.dataobj)
    return arr, img.header


def read_dose(dose_file):
    dose_volume, dose_header = read_file(dose_file)
    return dose_volume, dose_header


def read_masks(mask_files):
    structure_masks = {}
    for mask_file in mask_files:
        mask_volume, mask_header = read_file(mask_file)
        struct_name = mask_file.name.split(".")[0]
        structure_masks[struct_name] = mask_volume
    return structure_masks


def read_byte_data(dose_file, mask_files):
    """
    READ_BYTE_DATA: Read the dose and mask files from the byte data.
    This is exclusively meant for what st.file_uploader returns.
    Do not use it for reading files from the filesystem.

    :param dose_file: byte array for dose data.
    :param mask_files: byte array for multiple masks.
    :return: dose_volume, structure_masks - numpy array and dictionary of numpy arrays.
    """
    dose_volume, _ = read_file(dose_file)

    structure_masks = {}
    struct_identifiers = []
    for mask_file in mask_files:
        mask_volume, mask_header = read_file(mask_file)
        struct_name = mask_file.name.split(".")[0]
        struct_identifiers.append(struct_name)
        structure_masks[struct_name] = mask_volume
    return dose_volume, structure_masks


def read_from_eclipse(file_name):
    df = pd.DataFrame()
    f = open(file_name, "r")
    for line in f:
        if "Structure:" in line:
            name = line.split(" ")[-1]
            for line in f:
                if "Relative dose [%]" in line:
                    row_cnt = 0
                    for line in f:
                        if len(line.split()) > 2:
                            df.loc[row_cnt, name + "_dose"] = (
                                float(line.split()[1]) / 100.0
                            )
                            df.loc[row_cnt, name + "_vol"] = float(line.split()[2])
                            row_cnt += 1
                        else:
                            break
                    break
    f.close()
    return df


def read_from_nifti(nifti_filename):
    """
    READ_FROM_NIFTI: Read the nifti file and return the numpy array.
    :param nifti_filename: file name of the nifti file.
    :return: numpy array.
    """
    img = sitk.ReadImage(nifti_filename)
    return sitk.GetArrayFromImage(img)


def read_dose_and_mask_files(dose_file, mask_files):
    dose_volume = read_from_nifti(dose_file)
    structure_masks = {}
    for mask_file in mask_files:
        mask_volume = read_from_nifti(mask_file)
        struct_name = mask_file.split("/")[-1].split(".")[0]
        structure_masks[struct_name] = mask_volume
    return dose_volume, structure_masks


def get_dose(data_path: str):
    """
    GET_DOSE:
    Read the dose volume from the specified data root directory.
    :param data_root: Path to the directory containing the dose file.
    :return: Dose volume as a numpy array.
    """
    dose_file = os.path.join(data_path, "Dose.nii.gz")
    dose_volume = read_from_nifti(dose_file)
    return dose_volume


def get_structures(data_path: str):
    """
    GET_STRUCTURES:
    Read the structure masks from the specified data root directory.
    :param data_root: Path to the directory containing the structure files.
    :return: A dictionary of structure masks and a list of mask files.
    """
    contents_file = glob(os.path.join(data_path, "*.csv"))

    mask_structures = {}
    mask_files = []
    if len(contents_file) == 1:
        cf = pd.read_csv(contents_file[0])
        info = cf[["Structure", "Type"]].copy()

        for i in range(info.shape[0]):
            if info.loc[i, "Type"] == "Target" or info.loc[i, "Type"] == "OAR":
                mask_file = os.path.join(
                    data_path, str(info.loc[i, "Structure"]) + ".nii.gz"
                )
                mask_structures[info.loc[i, "Structure"]] = read_from_nifti(mask_file)
                mask_files.append(mask_file)
    return mask_structures, mask_files


def read_dose_and_mask_files_as_structure_set(
    dose_file: str,
    mask_files: List[str],
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    structure_types: Optional[Dict[str, str]] = None,
    name: str = "StructureSet",
):
    """
    Read dose and mask files and return as a StructureSet object.

    Args:
        dose_file: Path to dose NIfTI file
        mask_files: List of paths to mask NIfTI files
        spacing: Voxel spacing in (x, y, z) mm
        origin: Origin coordinates in mm
        structure_types: Optional mapping of structure names to types ("OAR", "Target", "Avoidance")
        name: Name for the structure set

    Returns:
        StructureSet object with loaded dose and structures
    """
    from .structure_set import create_structure_set_from_masks, StructureType

    # Read dose volume
    dose_volume = read_from_nifti(dose_file)

    # Read structure masks
    structure_masks = {}
    for mask_file in mask_files:
        mask_volume = read_from_nifti(mask_file)
        struct_name = mask_file.split("/")[-1].split(".")[0]
        structure_masks[struct_name] = mask_volume

    # Convert string types to StructureType enum if provided
    structure_type_mapping = None
    if structure_types:
        structure_type_mapping = {}
        for name, type_str in structure_types.items():
            if type_str.lower() == "target":
                structure_type_mapping[name] = StructureType.TARGET
            elif type_str.lower() == "oar":
                structure_type_mapping[name] = StructureType.OAR
            elif type_str.lower() == "avoidance":
                structure_type_mapping[name] = StructureType.AVOIDANCE

    # Create and return StructureSet
    return create_structure_set_from_masks(
        structure_masks=structure_masks,
        dose_volume=dose_volume,
        spacing=spacing,
        origin=origin,
        structure_types=structure_type_mapping,
        name=name,
    )


def get_dose_and_structures_as_structure_set(
    data_path: str,
    dose_filename: str = "Dose.nii.gz",
    contents_filename: Optional[str] = None,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    name: Optional[str] = None,
):
    """
    Read dose and structures from a directory and return as a StructureSet.

    Enhanced version of get_dose() and get_structures() that returns a unified StructureSet object.

    Args:
        data_path: Path to directory containing dose and structure files
        dose_filename: Name of the dose file (default: "Dose.nii.gz")
        contents_filename: Optional name of CSV file with structure type information
        spacing: Voxel spacing in (x, y, z) mm
        origin: Origin coordinates in mm
        name: Name for the structure set (defaults to directory name)

    Returns:
        StructureSet object with loaded dose and structures
    """
    from .structure_set import create_structure_set_from_folder

    if name is None:
        name = os.path.basename(os.path.normpath(data_path))

    contents_file_path = None
    if contents_filename:
        contents_file_path = os.path.join(data_path, contents_filename)

    return create_structure_set_from_folder(
        data_path=data_path,
        dose_file=dose_filename,
        contents_file=contents_file_path,
        spacing=spacing,
        origin=origin,
        name=name,
    )


def create_structure_set_from_existing_data(
    dose_volume: np.ndarray,
    structure_masks: Dict[str, np.ndarray],
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    structure_types: Optional[Dict[str, str]] = None,
    name: str = "StructureSet",
):
    """
    Create a StructureSet from existing dose and mask data arrays.

    Utility function to convert existing data (from read_dose_and_mask_files)
    into a StructureSet object.

    Args:
        dose_volume: 3D dose array
        structure_masks: Dictionary mapping structure names to 3D mask arrays
        spacing: Voxel spacing in (x, y, z) mm
        origin: Origin coordinates in mm
        structure_types: Optional mapping of structure names to types ("OAR", "Target", "Avoidance")
        name: Name for the structure set

    Returns:
        StructureSet object
    """
    from .structure_set import create_structure_set_from_masks, StructureType

    # Convert string types to StructureType enum if provided
    structure_type_mapping = None
    if structure_types:
        structure_type_mapping = {}
        for struct_name, type_str in structure_types.items():
            if type_str.lower() == "target":
                structure_type_mapping[struct_name] = StructureType.TARGET
            elif type_str.lower() == "oar":
                structure_type_mapping[struct_name] = StructureType.OAR
            elif type_str.lower() == "avoidance":
                structure_type_mapping[struct_name] = StructureType.AVOIDANCE

    return create_structure_set_from_masks(
        structure_masks=structure_masks,
        dose_volume=dose_volume,
        spacing=spacing,
        origin=origin,
        structure_types=structure_type_mapping,
        name=name,
    )
