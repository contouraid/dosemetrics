import os
from dosemetrics import dvh

import SimpleITK as sitk
import numpy as np

import tkinter as tk
from tkinter.filedialog import askopenfilename, asksaveasfilename
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

plt.style.use("dark_background")


def compute_stats(_file_name: str, _dose_array: np.ndarray) -> dict:
    stats = {}
    stats["name"] = _file_name.split("/")[-1].split(".")[0]
    struct_image = sitk.ReadImage(_file_name)
    struct_array = sitk.GetArrayFromImage(struct_image)
    stats["bins"], stats["values"] = dvh.compute_dvh(_dose_array, struct_array)
    stats["max"] = dvh.max_dose(_dose_array, struct_array)
    stats["mean"] = dvh.mean_dose(_dose_array, struct_array)
    stats["volume"] = dvh.volume(struct_array, struct_image.GetSpacing())
    stats["color"] = "b"
    return stats


def plot_stats(_stats):
    fig = plt.figure()
    plt.plot(
        _stats["bins"], _stats["values"], color=_stats["color"], label=_stats["name"]
    )
    plt.legend(loc="best")
    plt.xlabel("Dose [Gy]")
    plt.ylabel("Ratio of Total Structure Volume [%]")
    plt.title(
        f"Volume: {_stats['volume']:4.3f} (cc); Max Dose: {_stats['max']:2.3f}; Mean Dose: {_stats['mean']:2.3f}"
    )
    plt.axvline(x=_stats["mean"], color="y")
    plt.axvline(x=_stats["max"], color="r")
    plt.grid()
    return fig


def main():
    root = tk.Tk()
    root.withdraw()

    output_file = asksaveasfilename()
    print(f"File path entered is: {output_file}")
    pp = PdfPages(output_file)

    dose_file_name = askopenfilename(
        title="Choose Dose file", filetypes=[("Image files", ".gz .nii")]
    )
    print(f"Dose file name entered is: {dose_file_name}")
    file_path = os.path.dirname(dose_file_name)
    dose_image = sitk.ReadImage(dose_file_name)
    dose_array = sitk.GetArrayFromImage(dose_image)

    oar_file_names = askopenfilename(
        initialdir=file_path,
        title="Choose OAR files",
        filetypes=[("Image files", ".gz .nii")],
        multiple=True,
    )
    oar_files = list(oar_file_names)
    oar_files = sorted(oar_files)

    for file in oar_files:
        stats = compute_stats(file, dose_array)
        fig = plot_stats(stats)
        pp.savefig(fig)

    target_file_name = askopenfilename(
        initialdir=file_path,
        title="Choose Target file",
        filetypes=[("Image files", ".gz .nii")],
    )
    stats = compute_stats(target_file_name, dose_array)
    stats["color"] = "r"
    fig = plot_stats(stats)

    pp.savefig(fig)
    pp.close()


if __name__ == "__main__":
    main()
