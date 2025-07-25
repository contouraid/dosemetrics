import numpy as np
import pandas as pd
from numpy import ndarray


def mean_dose(_dose: np.ndarray, _struct_mask: np.ndarray):
    dose_in_struct = _dose[_struct_mask > 0]
    return np.mean(dose_in_struct)


def max_dose(_dose: np.ndarray, _struct_mask: np.ndarray):
    dose_in_struct = _dose[_struct_mask > 0]
    return np.max(dose_in_struct)


def volume(_struct_mask: np.ndarray, _vox_dims: tuple):
    num_voxels = np.count_nonzero(_struct_mask)
    return num_voxels * np.prod(_vox_dims) / 1000.0  # in centimeter cube.


def get_volumes(file_name):
    volumes = {}

    df = pd.DataFrame()
    with open(file_name, "r") as f:
        for line in f:
            if "Structure:" in line:
                idx = line.find(" ") + 1
                struct = line[idx:]
                name = struct.split("\n")[0]
                # print("parsing: " + name)
                for line in f:
                    if "Volume [cm" in line:
                        idy = line.find(":") + 2
                        vol = line[idy:]
                        volume = vol.split("\n")[0]
                        # print(name + ": " + volume)
                        volumes[name] = [volume]
                        break
    return volumes


def compute_dvh(
    _dose: np.ndarray,
    _struct_mask: np.ndarray,
    max_dose=65,
    step_size=0.1,
) -> tuple[ndarray, ndarray]:

    dose_in_oar = _dose[_struct_mask > 0]
    bins = np.arange(0, max_dose, step_size)
    total_voxels = len(dose_in_oar)
    values = []

    if total_voxels == 0:
        # There's no voxels in the mask
        values = np.zeros(len(bins))
    else:
        for bin in bins:
            number = (dose_in_oar >= bin).sum()
            value = (number / total_voxels) * 100
            values.append(value)
        values = np.asarray(values)

    return bins, values


def dvh_by_structure(dose_volume, structure_masks):

    dvh_data = {}
    max_dose = 70
    step_size = 0.1
    dvh_data["Dose"] = np.arange(0, max_dose, step_size)

    for structure in structure_masks.keys():
        bins, values = compute_dvh(
            dose_volume, structure_masks[structure], max_dose, step_size
        )
        dvh_data[structure] = values

    df = pd.DataFrame.from_dict(dvh_data)
    df = pd.melt(
        df,
        id_vars=["Dose"],
        value_vars=structure_masks.keys(),
        var_name="Structure",
        value_name="Volume",
    )
    return df


def dvh_by_dose(dose_volumes, structure_mask, structure_name):
    dvh_data = {}
    max_dose = 70
    step_size = 0.1
    dvh_data["Dose"] = np.arange(0, max_dose, step_size)

    dose_id = []
    for id in dose_volumes.keys():
        bins, values = compute_dvh(
            dose_volumes[id], structure_mask, max_dose, step_size
        )
        dose_id.append(structure_name + "_" + str(id))
        dvh_data[structure_name + "_" + str(id)] = values

    df = pd.DataFrame.from_dict(dvh_data)
    df = pd.melt(
        df,
        id_vars=["Dose"],
        value_vars=dose_id,
        var_name="Structure",
        value_name="Volume",
    )
    return df
