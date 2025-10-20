import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
import pandas as pd

import dosemetrics
from dosemetrics.utils.plot import generate_dvh_variations, plot_dvh_variations

plt.style.use("dark_background")


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

    # Extract constraint limit as float
    constraint_limit = constraints.loc[structure_of_interest, "Level"]
    if hasattr(constraint_limit, "values"):
        constraint_limit = float(constraint_limit.values[0])
    else:
        constraint_limit = float(constraint_limit)

    # Generate DVH variations with configurable parameters
    dvh_data, dice_coefficients, original_dvh = generate_dvh_variations(
        dose_volume,
        structure_mask,
        n_variations=100,
        dice_range=(0.65, 1.0),  # Target Dice range
        volume_variation=0.15,  # ±15% volume variation
        max_dose=65,
        step_size=0.1,
    )

    # Plot the variations
    fig, (min_dice, max_dice) = plot_dvh_variations(
        dvh_data,
        dice_coefficients,
        original_dvh,
        constraint_limit,
        structure_of_interest,
    )

    print(f"Generated {len(dice_coefficients)} variations")
    print(f"Dice coefficient range: {min_dice:.3f} - {max_dice:.3f}")

    plt.savefig(os.path.join(output_folder, f"{structure_of_interest}_dvh_family.png"))


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

    # For EORTC data
    # constraints = compliance.get_custom_constraints()

    # For Insel data
    constraints = dosemetrics.get_default_constraints()

    for structure_of_interest in structures:
        results_folder = os.path.join(repo_root, "..", "results", "test_subject_TPS")
        os.makedirs(results_folder, exist_ok=True)
        generate_dvh_family(
            data_folder, constraints, structure_of_interest, results_folder
        )
