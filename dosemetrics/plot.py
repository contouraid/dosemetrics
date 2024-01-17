import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
