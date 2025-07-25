import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import dosemetrics.dvh as dvh
from matplotlib.transforms import Bbox


def _get_cmap(n, name="gist_ncar"):
    """Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name."""
    return plt.cm.get_cmap(name, n)


def from_dataframe(dataframe: pd.DataFrame, plot_title: str, output_path: str) -> None:
    col_names = dataframe.columns
    cmap = _get_cmap(40)

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
    plt.savefig(output_path)
    plt.close(fig)


# function that calculates and plots the DVHs based on the dose array of a specific structure
def compare_dvh(
    _gt: np.ndarray,
    _pred: np.ndarray,
    _struct_mask: np.ndarray,
    max_dose=65,
    step_size=0.1,
):
    bins_gt, values_gt = dvh.compute_dvh(
        _gt, _struct_mask, max_dose=max_dose, step_size=step_size
    )
    bins_pred, values_pred = dvh.compute_dvh(
        _pred, _struct_mask, max_dose=max_dose, step_size=step_size
    )

    fig = plt.figure()
    plt.plot(bins_gt, values_gt, color="b", label="ground truth")
    plt.plot(bins_pred, values_pred, color="r", label="prediction")

    plt.xlabel("Dose [Gy]")
    plt.ylabel("Ratio of Total Structure Volume [%]")
    plt.legend(loc="best")

    return fig


def variability(dose_volume, structure_mask, constraint_limit, structure_of_interest):
    fig = plt.figure()
    n_lines = 100
    n_est = 5

    max_dsc = 0.0
    min_dsc = 1.0

    cmap = mpl.colormaps["viridis"]
    colors = cmap(np.linspace(0, 1, n_lines + 1))
    sc = None  # Initialize sc to None
    for x_range in range(-n_est, n_est + 1):
        for y_range in range(-n_est, n_est + 1):
            for z_range in range(-n_est, n_est + 1):
                new_structure_mask = structure_mask.copy()
                new_structure_mask = np.roll(new_structure_mask, x_range, axis=0)
                new_structure_mask = np.roll(new_structure_mask, y_range, axis=1)
                new_structure_mask = np.roll(new_structure_mask, z_range, axis=2)
                bins, values = dvh.compute_dvh(dose_volume, new_structure_mask)

                intersection = np.logical_and(structure_mask, new_structure_mask)
                dice = (
                    2
                    * intersection.sum()
                    / (structure_mask.sum() + new_structure_mask.sum())
                )
                if dice > max_dsc:
                    max_dsc = dice
                if dice < min_dsc:
                    min_dsc = dice
                color = colors[int(dice * n_lines)]
                sc = plt.scatter(bins, values, s=0.5, c=color, alpha=0.25)
    bins, values = dvh.compute_dvh(dose_volume, structure_mask)
    plt.plot(bins, values, color="r", label=structure_of_interest)
    plt.axvline(x=constraint_limit, color="g", label="Constraint Limit")
    plt.xlabel("Dose [Gy]")
    plt.ylabel("Ratio of Total Structure Volume [%]")
    plt.title(f"DVH Family for {structure_of_interest}")
    if sc is not None:
        color_bar = plt.colorbar(sc)
        color_bar.set_alpha(1)
    plt.grid()
    return fig, (max_dsc, min_dsc)


def plot_dvh(dose_volume: np.ndarray, structure_masks: dict, output_file: str):
    """
    PLOT_DVH:
    Plot the dose-volume histogram (DVH) for the given dose volume and structure masks.
    :param dose_volume: Dose volume data as a numpy array.
    :param structure_masks: Dictionary of structure masks.
    :param output_file: Path to save the DVH plot.
    """
    df = dvh.dvh_by_structure(dose_volume, structure_masks)
    _, ax = plt.subplots()
    df.set_index("Dose", inplace=True)
    df.groupby("Structure")["Volume"].plot(legend=True, ax=ax)

    # Shrink current axis by 20%
    box = ax.get_position()
    new_box = Bbox.from_bounds(box.x0, box.y0, box.width * 0.8, box.height)
    ax.set_position(new_box)

    # Put a legend to the right of the current axis
    ax.legend(loc="center left", bbox_to_anchor=(0.9, 0.5))

    plt.xlabel("Dose [Gy]")
    plt.ylabel("Ratio of Total Structure Volume [%]")
    plt.grid()
    plt.savefig(output_file)
    plt.close()
