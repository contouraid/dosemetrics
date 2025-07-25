import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
import pandas as pd

from dosemetrics import compliance
from dosemetrics import data_utils
from dosemetrics import scores
from dosemetrics import plot

plt.style.use("dark_background")


def generate_dvh_family(
    input_folder: str,
    constraints: pd.DataFrame,
    structure_of_interest: str,
    output_folder: str,
):
    dose_file = os.path.join(input_folder, "Dose.nii.gz")
    structure_file = os.path.join(input_folder, structure_of_interest + ".nii.gz")
    dose_volume = data_utils.read_from_nifti(dose_file)
    structure_image = sitk.ReadImage(structure_file)
    structure_mask = sitk.GetArrayFromImage(structure_image)
    spacing = structure_image.GetSpacing()
    print(f"Spacing: {spacing}")

    dose_df = scores.dose_summary(dose_volume, {structure_of_interest: structure_mask})
    dose_compliance = compliance.check_compliance(dose_df, constraints)
    print(dose_compliance)

    constraint_limit = constraints.loc[structure_of_interest, "Level"]
    plot.variability(
        dose_volume, structure_mask, constraint_limit, structure_of_interest
    )
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
    constraints = compliance.get_default_constraints()

    for structure_of_interest in structures:
        results_folder = os.path.join(repo_root, "..", "results", "test_subject_TPS")
        os.makedirs(results_folder, exist_ok=True)
        generate_dvh_family(
            data_folder, constraints, structure_of_interest, results_folder
        )
