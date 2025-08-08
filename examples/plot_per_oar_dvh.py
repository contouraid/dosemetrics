import os
import glob

import SimpleITK as sitk
import matplotlib.pyplot as plt

plt.style.use("dark_background")

from matplotlib.backends.backend_pdf import PdfPages

import dosemetrics


def plot_dvh_to_pdf(data_folder, output_file="dvh.pdf"):
    pp = PdfPages(output_file)
    dose_image = sitk.ReadImage(data_folder + "/Dose.nii.gz")
    dose_image.SetOrigin((0, 0, 0))
    dose_array = sitk.GetArrayFromImage(dose_image)

    structures = glob.glob(data_folder + "/*[!Dose*].nii.gz")

    structures = sorted(structures)
    for structure in structures:
        struct_name = structure.split("/")[-1].split(".")[0]
        if struct_name == "CT" or struct_name == "Dose_Mask":
            continue
        else:
            oar_image = sitk.ReadImage(structure)
            oar_image.SetOrigin((0, 0, 0))
            oar_mask = sitk.GetArrayFromImage(oar_image)

            results = dosemetrics.compute_dvh(dose_array, oar_mask)
            max_val = dosemetrics.max_dose(dose_array, oar_mask)
            mean = dosemetrics.mean_dose(dose_array, oar_mask)
            volume = dosemetrics.volume(oar_mask, oar_image.GetSpacing())

            color = "r"
            fig = plt.figure()
            plt.plot(
                results[0],
                results[1],
                color=color,
                label=struct_name,
            )
            plt.xlabel("Dose [Gy]")
            plt.ylabel("Ratio of Total Structure Volume [%]")
            plt.legend(loc="best")
            plt.title(
                f"Volume: {volume:4.3f} (cc); Max Dose: {max_val:2.3f}; Mean Dose: {mean:2.3f}"
            )
            plt.grid()
            pp.savefig(fig)
            plt.close()
    pp.close()


if __name__ == "__main__":

    repo_root = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(repo_root, "..", "data", "compare_plans", "first")
    output_file = os.path.join(repo_root, "..", "results", "first_per_oar_dvh.pdf")
    plot_dvh_to_pdf(data_folder, output_file)
