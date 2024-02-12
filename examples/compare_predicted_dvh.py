import os
import glob

import SimpleITK as sitk
import numpy as np


from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

plt.style.use("dark_background")

from dosemetrics import dvh
from dosemetrics import plot


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
    repo_root = os.path.abspath("..")
    data_folder = os.path.join(repo_root, "data/test_subject")
    structures = glob.glob(data_folder + "/*[!Dose*].nii.gz")

    dose_image = sitk.ReadImage(data_folder + "/Dose.nii.gz")
    dose_array = sitk.GetArrayFromImage(dose_image)

    prediction_image = sitk.ReadImage(data_folder + "/Predicted_Dose.nii.gz")
    prediction_array = sitk.GetArrayFromImage(prediction_image)

    pp = PdfPages(os.path.join(data_folder, "compare_prediction.pdf"))

    structures = sorted(structures)
    for structure in structures:
        struct_name = structure.split("/")[-1].split(".")[0]
        if struct_name == "CT" or struct_name == "Dose_Mask":
            continue
        else:
            oar_image = sitk.ReadImage(structure)
            oar_mask = sitk.GetArrayFromImage(oar_image)

            fig = plot.compare_dvh(dose_array, prediction_array, oar_mask)
            plt.title(struct_name)
            plt.grid()
            pp.savefig(fig)
    pp.close()


if __name__ == "__main__":
    main()
