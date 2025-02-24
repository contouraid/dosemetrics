import os
import numpy as np
import pandas as pd
from gzip import GzipFile
import SimpleITK as sitk
from nibabel import FileHolder, Nifti1Image


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
    img = Nifti1Image.from_file_map({'header': fh, 'image': fh})
    arr = np.array(img.dataobj)
    return arr, img.header


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
