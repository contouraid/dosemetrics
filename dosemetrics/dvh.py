from typing import Tuple, Union, List, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import ndarray


def read_from_eclipse(file_name):
    df = pd.DataFrame()
    with open(file_name, "r") as f:
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
                                f.close()
                                break
                        break
        return df


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


# function that calculates and plots the DVHs based on the dose array of a specific structure
def compare_dvh(dose_array_gt, dose_array_pred, case_nr, name):
    bins = np.arange(0, np.ceil(np.max(dose_array_gt)), 0.1)
    total_voxels = len(dose_array_gt)
    values_gt = []
    values_pred = []
    for bin in bins:
        number_gt = (dose_array_gt >= bin).sum()
        number_pred = (dose_array_pred >= bin).sum()

        value_gt = number_gt / total_voxels * 100
        value_pred = number_pred / total_voxels * 100

        values_gt.append(value_gt)
        values_pred.append(value_pred)

    fig = plt.figure()
    plt.plot(bins, values_gt, color="b", label="ground truth")
    plt.plot(bins, values_pred, color="r", label="prediction")

    plt.xlabel("Dose [Gy]")
    plt.ylabel("Ratio of Total Structure Volume [%]")
    plt.title(case_nr + " " + name)
    plt.legend(loc="best")

    return fig


def compute_dvh(
    dose_array: np.ndarray, structure_mask: np.ndarray, max_dose=65, step_size=0.1,
) -> tuple[ndarray, ndarray]:
    dose_in_oar = dose_array[structure_mask > 0]
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
