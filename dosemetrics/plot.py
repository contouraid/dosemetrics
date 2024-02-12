import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import dosemetrics.dvh as dvh


def _get_cmap(n, name="gist_ncar"):
    """Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name."""
    return plt.cm.get_cmap(name, n)


def from_dataframe(dataframe: pd.DataFrame, plot_title: str) -> None:
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
    filename = plot_title + ".png"
    plt.savefig(filename)
    # plt.show()
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
