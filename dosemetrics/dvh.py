import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_dvh_file(file_name):
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
                                f.close
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


def get_cmap(n, name="gist_ncar"):
    """Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name."""
    return plt.cm.get_cmap(name, n)


def plot_dvh(dataframe: pd.DataFrame, plot_title: str) -> None:
    col_names = dataframe.columns
    cmap = get_cmap(40)

    plt.style.use("dark_background")
    fig, ax = plt.subplots()

    for i in range(len(col_names)):
        if i % 2 == 0:
            name = col_names[i].split("\n")[0]
            line_color = cmap(i)
            x = dataframe[col_names[i]]
            y = dataframe[col_names[i + 1]]
            plt.plot(x, y, color=line_color, label=name)

    plt.xlabel("Dose [Gy]")
    plt.xlim([0, 65])
    plt.grid()
    plt.ylabel("Ratio of Total Structure Volume [%]")
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    plt.title(plot_title)
    filename = plot_title + ".png"
    plt.savefig(filename)
    # plt.show()
    plt.close(fig)


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
    dose_array: np.ndarray, structure_mask: np.ndarray
) -> tuple[list, list]:
    dose_in_oar = dose_array[structure_mask > 0]
    bins = np.arange(0, 65, 0.1)
    total_voxels = len(dose_in_oar)
    values = []

    if total_voxels == 0:
        # There's no voxels in the mask
        values = [0] * len(bins)
    else:
        for bin in bins:
            number = (dose_in_oar >= bin).sum()
            value = (number / total_voxels) * 100
            values.append(value)

    return (bins, values)
