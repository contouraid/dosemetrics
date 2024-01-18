import os
import glob

import SimpleITK as sitk
import matplotlib.pyplot as plt

from dosemetrics.dvh import compute_dvh

plt.rcParams["figure.figsize"] = [20, 12]
plt.style.use("dark_background")


def plot_dvh_from_eclipse(root_folder):
    structures = glob.glob(root_folder + "/*[!Dose*].nii.gz")
    dose_image = sitk.ReadImage(root_folder + "/Dose.nii.gz")

    os.makedirs(root_folder + "/DVH/", exist_ok=True)
    dose_array = sitk.GetArrayFromImage(dose_image)
    for structure in structures:
        struct_name = structure.split("/")[-1].split(".")[0]
        print(struct_name)
        oar_image = sitk.ReadImage(structure)
        oar_mask = sitk.GetArrayFromImage(oar_image)
        results = compute_dvh(dose_array, oar_mask)

        plt.figure()
        plt.plot(
            results[0], results[1], color="b", label=struct_name,
        )
        plt.xlabel("Dose [Gy]")
        plt.ylabel("Ratio of Total Structure Volume [%]")
        plt.legend(loc="best")
        plt.grid()
        plt.savefig(root_folder + "/DVH/" + struct_name + ".png")
        plt.close()


if __name__ == "__main__":
    repo_root = os.path.abspath("..")
    data_folder = os.path.join(repo_root, "data/test_subject")
    plot_dvh_from_eclipse(data_folder)
