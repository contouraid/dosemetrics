import os
import glob

import SimpleITK as sitk
import numpy as np


from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

plt.style.use("dark_background")

from dosemetrics import dvh
from dosemetrics import compliance


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

    pp = PdfPages(os.path.join(data_folder, "..", "quality_index.pdf"))

    constraints = compliance.get_default_constraints()

    structures = sorted(structures)
    for structure in structures:
        struct_name = structure.split("/")[-1].split(".")[0]
        if struct_name not in constraints.index:
            continue
        else:
            oar_image = sitk.ReadImage(structure)
            oar_mask = sitk.GetArrayFromImage(oar_image)

            constraint_type = constraints.loc[struct_name, "Constraint Type"]
            constraint_limit = constraints.loc[struct_name, "Level"]

            bins, values = dvh.compute_dvh(dose_array, oar_mask)
            qi = compliance.quality_index(
                dose_array, oar_mask, constraint_type, constraint_limit
            )
            fig = plt.figure()
            plt.plot(
                bins, values, color="b", label=struct_name,
            )
            plt.axvline(x=constraint_limit, color="r", label="constraint limit")
            if constraint_type == "max":
                max_dose = dvh.max_dose(dose_array, oar_mask)
                plt.axvline(x=max_dose, color="g", label="max_dose")
            elif constraint_type == "mean":
                mean_dose = dvh.mean_dose(dose_array, oar_mask)
                plt.axvline(x=mean_dose, color="g", label="mean_dose")
            print(
                f"{struct_name}, Type: {constraint_type}, Limit: {constraint_limit}, QI: {qi:.2f}"
            )
            plt.title(
                f"{struct_name}, Type: {constraint_type}, Limit: {constraint_limit}, QI: {qi:.2f}"
            )
            plt.grid()
            pp.savefig(fig)
    pp.close()


if __name__ == "__main__":
    main()
