import os
import glob
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = [20, 12]


def compute_dvh(dose_array, oar_mask):
    dose_in_oar = dose_array[oar_mask > 0]
    bins = np.arange(0, 65, 0.1)
    total_voxels = len(oar_mask)
    values = []
    for bin in bins:
        number = (dose_in_oar >= bin).sum()
        value = number / total_voxels * 100
        values.append(value)

    return (bins, values)


def plot_predicted_dvh_for_case(root_folder, case):
    for subject in [
        str(case).zfill(3) + str(i) for i in range(0, 4)
    ]:  # alternatives from 0 to 4
        structures = glob.glob(
            root_folder + "ISAS_GBM_" + str(subject) + "/*[!Dose*].nii.gz"
        )
        dose_image = sitk.ReadImage(
            root_folder + "ISAS_GBM_" + str(subject) + "/Dose.nii.gz"
        )

        os.makedirs(root_folder + "ISAS_GBM_" + str(subject) + "/DVH/", exist_ok=True)
        dose_array = sitk.GetArrayFromImage(dose_image)
        for structure in structures:
            print(structure)
            oar_image = sitk.ReadImage(structure)
            oar_mask = sitk.GetArrayFromImage(oar_image)
            results = compute_dvh(dose_array, oar_mask)

            plt.figure()
            plt.plot(
                results[0],
                results[1],
                color="b",
                label=structure.split("/")[-1].split(".")[0],
            )
            plt.xlabel("Dose [Gy]")
            plt.ylabel("Ratio of Total Structure Volume [%]")
            plt.legend(loc="best")
            plt.savefig(
                root_folder
                + "ISAS_GBM_"
                + str(subject)
                + "/DVH/"
                + structure.split("/")[-1].split(".")[0]
                + ".png"
            )
            plt.close()


if __name__ == "__main__":
    root_folder = "/Users/amithkamath/data/DLDP/astute-experiments/"
    case = 70
    plot_predicted_dvh_for_case(root_folder, case)
