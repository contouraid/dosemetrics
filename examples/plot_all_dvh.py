import os
from glob import glob

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

import dosemetrics.dvh as dvh
import dosemetrics.data_utils as data_utils

plt.style.use("dark_background")
figure(figsize=(12, 8), dpi=100)

def plot_dvh(data_root: str, output_file="dvh.png"):

    contents_file = glob(os.path.join(data_root, "*.csv"))
    if len(contents_file) == 1:
        cf = pd.read_csv(contents_file[0])
        info = cf[['Structure', 'Type']].copy()

        if any(info['Type'] == "Dose"):
            dose_file = os.path.join(data_root, "Dose.nii.gz")

            mask_files = []
            for i in range(info.shape[0]):
                if info.loc[i, "Type"] == "Target":
                    mask_files.append(os.path.join(data_root, info.loc[i, "Structure"] + ".nii.gz"))

            dose_volume, structure_masks = data_utils.read_dose_and_mask_files(dose_file, mask_files)
            df = dvh.dvh_by_structure(dose_volume, structure_masks)
            fig, ax = plt.subplots()
            df.set_index('Dose', inplace=True)
            df.groupby('Structure')['Volume'].plot(legend=True, ax=ax)

            # Shrink current axis by 20%
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

            # Put a legend to the right of the current axis
            ax.legend(loc='center left', bbox_to_anchor=(0.9, 0.5))

            plt.xlabel("Dose [Gy]")
            plt.ylabel("Ratio of Total Structure Volume [%]")
            plt.grid()
            plt.savefig(output_file)
            plt.close()


if __name__ == "__main__":

    repo_root = os.path.dirname(os.path.abspath(__file__))
    """
    data_root = os.path.join(repo_root, "..", "data", "compare_plans", "first")
    output_file = os.path.join(repo_root, "..", "results", "first_overall_dvh.png")
    plot_dvh(data_root, output_file)
    """

    test_folder = "/home/akamath/Documents/data/ICR/output"
    subfolder = [f.path for f in os.scandir(test_folder) if f.is_dir()]
    for sub in subfolder:
        sub_name = sub.split("/")[-1]
        planfolders = [k.path for k in os.scandir(sub) if k.is_dir()]
        for plan in planfolders:
            plan_name = plan.split("/")[-1]
            output_file = os.path.join(plan, "..", plan_name + "_overall_dvh.png")
            plot_dvh(plan, output_file)
