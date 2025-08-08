import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

import dosemetrics

plt.style.use("dark_background")


def read_dose_and_mask_files(dose_file, mask_files):
    dose_volume = dosemetrics.read_from_nifti(dose_file)
    structure_masks = {}
    for mask_file in mask_files:
        mask_volume = dosemetrics.read_from_nifti(mask_file)
        struct_name = mask_file.split("/")[-1].split(".")[0]
        structure_masks[struct_name] = mask_volume
    return dose_volume, structure_masks


def generate_dvh_family(
    input_folder: str,
    constraints: pd.DataFrame,
    structure_of_interest: str,
    output_folder: str,
):
    dose_file = os.path.join(input_folder, "Dose.nii.gz")
    structure_file = os.path.join(input_folder, structure_of_interest + ".nii.gz")
    dose_volume = dosemetrics.read_from_nifti(dose_file)
    structure_image = sitk.ReadImage(structure_file)
    structure_mask = sitk.GetArrayFromImage(structure_image)
    spacing = structure_image.GetSpacing()
    print(f"Spacing: {spacing}")

    dose_df = dosemetrics.dose_summary(
        dose_volume, {structure_of_interest: structure_mask}
    )
    dose_compliance = dosemetrics.check_compliance(dose_df, constraints)
    print(dose_compliance)

    constraint_limit = constraints.loc[structure_of_interest, "Level"]

    fig = plt.figure()
    n_lines = 100
    cmap = mpl.colormaps["jet"]
    colors = cmap(np.linspace(0, 1, n_lines + 1))
    x_range = 4
    y_range = 0
    z_range = 0
    new_structure_mask = structure_mask.copy()
    new_structure_mask = np.roll(new_structure_mask, x_range, axis=0)
    new_structure_mask = np.roll(new_structure_mask, y_range, axis=1)
    new_structure_mask = np.roll(new_structure_mask, z_range, axis=2)
    bins, values = dosemetrics.compute_dvh(dose_volume, new_structure_mask)
    plt.plot(
        bins,
        values,
        color="g",
    )
    intersection = np.logical_and(structure_mask, new_structure_mask)
    dice = 2 * intersection.sum() / (structure_mask.sum() + new_structure_mask.sum())
    bins, values = dosemetrics.compute_dvh(dose_volume, structure_mask)
    plt.plot(bins, values, color="r", label=structure_of_interest)
    # plt.axvline(x=constraint_limit, color="g", label="Constraint Limit")
    plt.xlabel("Dose [Gy]")
    plt.ylabel("Ratio of Total Structure Volume [%]")
    plt.title(f"DVH with DSC: {dice:.2f}, on {structure_of_interest}")
    plt.grid()
    plt.savefig(os.path.join(output_folder, f"{structure_of_interest}_dvh_alt.png"))


if __name__ == "__main__":
    repo_root = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(repo_root, "..", "data", "test_subject")
    structures = [
        "BrainStem",
        "Chiasm",
        "OpticNerve_L",
        "OpticNerve_R",
        "Cochlea_L",
        "Cochlea_R",
        "LacrimalGland_L",
        "LacrimalGland_R",
        "Target",
    ]
    constraints = dosemetrics.get_default_constraints()

    for structure_of_interest in structures:
        results_folder = os.path.join(repo_root, "..", "results", "test_subject_TPS-X")
        os.makedirs(results_folder, exist_ok=True)
        generate_dvh_family(
            data_folder, constraints, structure_of_interest, results_folder
        )
