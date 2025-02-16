from dosemetrics import dvh
from dosemetrics import compliance
from dosemetrics import data_utils
from dosemetrics import scores

import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import matplotlib as mpl

plt.style.use("dark_background")


def read_dose_and_mask_files(dose_file, mask_files):
    dose_volume = data_utils.read_from_nifti(dose_file)
    structure_masks = {}
    for mask_file in mask_files:
        mask_volume = data_utils.read_from_nifti(mask_file)
        struct_name = mask_file.split("/")[-1].split(".")[0]
        structure_masks[struct_name] = mask_volume
    return dose_volume, structure_masks

def generate_dvh_family(input_folder: str, structure_of_interest: str, output_folder: str):
    dose_file = os.path.join(input_folder, "Dose.nii.gz")
    structure_file = os.path.join(input_folder, structure_of_interest + ".nii.gz")
    dose_volume = data_utils.read_from_nifti(dose_file)
    structure_image = sitk.ReadImage(structure_file)
    structure_mask = sitk.GetArrayFromImage(structure_image)
    spacing = structure_image.GetSpacing()
    print(f"Spacing: {spacing}")

    dose_df = scores.dose_summary(dose_volume, {structure_of_interest: structure_mask})
    constraints = compliance.get_custom_constraints()
    dose_compliance = compliance.check_compliance(dose_df, constraints)
    print(dose_compliance)

    constraint_limit = constraints.loc[structure_of_interest, "Level"]

    fig = plt.figure()
    n_lines = 100
    cmap = mpl.colormaps['jet']
    colors = cmap(np.linspace(0, 1, n_lines + 1))
    for x_range in range(-1, 1):
        for y_range in range(-5, 6):
            for z_range in range(-5, 6):
                new_structure_mask = structure_mask.copy()
                new_structure_mask = np.roll(new_structure_mask, x_range, axis=0)
                new_structure_mask = np.roll(new_structure_mask, y_range, axis=1)
                new_structure_mask = np.roll(new_structure_mask, z_range, axis=2)
                bins, values = dvh.compute_dvh(dose_volume, new_structure_mask)

                intersection = np.logical_and(structure_mask, new_structure_mask)
                dice = 2 * intersection.sum() / (structure_mask.sum() + new_structure_mask.sum())
                print(f"Dice: {dice}")
                color = colors[int(dice * n_lines)]
                plt.scatter(
                    bins, values, s=0.5, c=color, alpha=0.25
                )
    bins, values = dvh.compute_dvh(dose_volume, structure_mask)
    plt.plot(
        bins, values, color='r', label=structure_of_interest
    )
    plt.axvline(x=constraint_limit, color="g", label="Constraint Limit")
    plt.xlabel("Dose [Gy]")
    plt.ylabel("Ratio of Total Structure Volume [%]")
    plt.title(f"DVH Family for {structure_of_interest}")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    repo_root = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(repo_root, "..", "data", "compare_plans", "first")
    structure_of_interest = "LacrimalGland_R"
    results_folder = os.path.join(repo_root, "..", "results")
    generate_dvh_family(data_folder, structure_of_interest, results_folder)
