import matplotlib.pyplot as plt
import numpy as np


def dvh(dose_array: np.ndarray, structure_name: str = None):
    """
    plot_dvh: function that calculates and plots the DVHs based on the dose array of a specific structure
    """
    bins = np.arange(0, np.ceil(np.max(dose_array)), 0.1)
    total_voxels = len(dose_array)
    values = []

    for bin in bins:
        number = (dose_array >= bin).sum()
        value = number / total_voxels * 100

        values.append(value)

    fig = plt.figure()
    if structure_name is not None:
        plt.plot(bins, values, color="b", label=structure_name)
        plt.legend(loc="best")
    else:
        plt.plot(bins, values, color="b")

    plt.xlabel("Dose [Gy]")
    plt.ylabel("Ratio of Total Structure Volume [%]")

    return fig
