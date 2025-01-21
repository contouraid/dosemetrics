import os
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("dark_background")
from matplotlib.backends.backend_pdf import PdfPages

import dosemetrics.dvh as dvh
import dosemetrics.data_utils as data_utils


def plot_dvh_to_pdf(data_folder, output_file="dvh.pdf"):

    pp = PdfPages(output_file)
    contents_files = [f for f in os.listdir(data_folder) if f.endswith(".csv")]
    if len(contents_files) > 0:
        contents_file = contents_files[0]
        contents = pd.read_csv(os.path.join(data_folder, contents_file))

        dose_file = os.path.join(data_folder, "Dose.nii.gz")
        dose_volume = data_utils.read_from_nifti(dose_file)

        structure_masks = {}
        oar_list = contents[contents["Type"] == "OAR"]["Structure"].values
        for oar in oar_list:
            oar_file = os.path.join(data_folder, oar + ".nii.gz")
            oar_volume = data_utils.read_from_nifti(oar_file)
            structure_masks[oar] = oar_volume

        target_list = contents[contents["Type"] == "Target"]["Structure"].values
        for target in target_list:
            target_file = os.path.join(data_folder, target + ".nii.gz")
            target_volume = data_utils.read_from_nifti(target_file)
            structure_masks[target] = target_volume

            fig, ax = plt.subplots()
            fig.set_figheight(8)
            fig.set_figwidth(12)
            for structure in structure_masks.keys():
                bins, values = dvh.compute_dvh(dose_volume, structure_masks[structure])
                ax.plot(bins, values, label=structure)

            plt.legend()
            plt.grid()
            ax.legend(loc="center left", bbox_to_anchor=(0.95, 0.5))
            pp.savefig(fig, dpi=300)
            pp.close()


if __name__ == "__main__":

    dataset_root = "/Users/amithkamath/data/Insel/training-data"
    output_folder = os.path.join(dataset_root, "..", "quality-check")
    os.makedirs(output_folder, exist_ok=True)

    subfolders = [f.path for f in os.scandir(dataset_root) if f.is_dir()]
    subfolders = sorted(subfolders)

    for subject_folder in subfolders:
        subject_name = subject_folder.split("/")[-1]
        dvh_file = os.path.join(dataset_root, "..", subject_name + "_dvh.pdf")
        try:
            plot_dvh_to_pdf(subject_folder, dvh_file)
        except Exception as e:
            print(f"Error processing {subject_folder}: {e}")
            continue