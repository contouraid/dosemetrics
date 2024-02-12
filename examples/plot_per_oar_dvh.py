import os
import glob

import SimpleITK as sitk
import matplotlib.pyplot as plt

plt.style.use("dark_background")

from matplotlib.backends.backend_pdf import PdfPages

import dosemetrics.dvh as dvh


def plot_dvh_from_eclipse(data_folder):
    pp = PdfPages(os.path.join(data_folder, "..", "reference_dvh_per_oar.pdf"))

    dose_image = sitk.ReadImage(data_folder + "/Dose.nii.gz")
    dose_array = sitk.GetArrayFromImage(dose_image)

    structures = glob.glob(data_folder + "/*[!Dose*].nii.gz")

    structures = sorted(structures)
    for structure in structures:
        struct_name = structure.split("/")[-1].split(".")[0]
        if struct_name == "CT" or struct_name == "Dose_Mask":
            continue
        else:
            oar_image = sitk.ReadImage(structure)
            oar_mask = sitk.GetArrayFromImage(oar_image)

            if struct_name == "Target":
                color = "r"
            else:
                color = "b"

            results = dvh.compute_dvh(dose_array, oar_mask)
            max = dvh.max_dose(dose_array, oar_mask)
            mean = dvh.mean_dose(dose_array, oar_mask)
            volume = dvh.volume(oar_mask, oar_image.GetSpacing())

            fig = plt.figure()
            plt.plot(
                results[0], results[1], color=color, label=struct_name,
            )
            plt.xlabel("Dose [Gy]")
            plt.ylabel("Ratio of Total Structure Volume [%]")
            plt.legend(loc="best")
            plt.title(
                f"Volume: {volume:4.3f} (cc); Max Dose: {max:2.3f}; Mean Dose: {mean:2.3f}"
            )
            plt.grid()
            pp.savefig(fig)
            plt.close()
    pp.close()


if __name__ == "__main__":
    repo_root = os.path.abspath("..")
    data_folder = os.path.join(repo_root, "data/test_subject")
    plot_dvh_from_eclipse(data_folder)
