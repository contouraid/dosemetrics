import os
import glob

import SimpleITK as sitk
import matplotlib.pyplot as plt

plt.style.use("dark_background")

from matplotlib.backends.backend_pdf import PdfPages

from dosemetrics.dvh import compute_dvh


def plot_dvh_from_eclipse(root_folder):
    pp = PdfPages(os.path.join(data_folder, "reference_dvh_per_oar.pdf"))

    dose_image = sitk.ReadImage(root_folder + "/Dose.nii.gz")
    dose_array = sitk.GetArrayFromImage(dose_image)

    structures = glob.glob(root_folder + "/*[!Dose*].nii.gz")
    for structure in structures:
        struct_name = structure.split("/")[-1].split(".")[0]
        oar_image = sitk.ReadImage(structure)
        oar_mask = sitk.GetArrayFromImage(oar_image)
        results = compute_dvh(dose_array, oar_mask)

        fig = plt.figure()
        plt.plot(
            results[0], results[1], color="b", label=struct_name,
        )
        plt.xlabel("Dose [Gy]")
        plt.ylabel("Ratio of Total Structure Volume [%]")
        plt.legend(loc="best")
        plt.title(struct_name)
        plt.grid()
        pp.savefig(fig)
        plt.close()
    pp.close()


if __name__ == "__main__":
    repo_root = os.path.abspath("..")
    data_folder = os.path.join(repo_root, "data/test_subject")
    plot_dvh_from_eclipse(data_folder)
