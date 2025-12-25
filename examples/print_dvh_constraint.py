import os
import glob
import SimpleITK as sitk
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

import dosemetrics

plt.style.use("dark_background")


def print_dvh_constraint(input_folder: str, output_folder: str):
    contents = os.path.join(input_folder, "standard_contents.csv")
    df = pd.read_csv(contents)
    masks = list(df[df["Type"] == "OAR"]["Structure"])
    masks += list(df[df["Type"] == "PTV"]["Structure"])
    dose_file = os.path.join(input_folder, "Dose.nii.gz")
    mask_files = [os.path.join(input_folder, f + ".nii.gz") for f in masks]

    dose_volume, structure_masks = dosemetrics.read_dose_and_mask_files(
        dose_file, mask_files
    )
    dose_df = dosemetrics.dose_summary(dose_volume, structure_masks)
    constraints = dosemetrics.get_default_constraints()
    dose_compliance = dosemetrics.check_compliance(dose_df, constraints)
    dose_compliance.to_csv(os.path.join(output_folder, "dvh_with_constraints.csv"))
    pp = PdfPages(os.path.join(output_folder, "dvh_with_constraints.pdf"))

    structures = glob.glob(input_folder + "/*[!Dose*].nii.gz")
    structures = sorted(structures)
    for structure in structures:
        struct_name = structure.split("/")[-1].split(".")[0]
        if struct_name not in constraints.index:
            continue
        else:
            oar_image = sitk.ReadImage(structure)
            oar_mask = sitk.GetArrayFromImage(oar_image)

            constraint_type = constraints.loc[struct_name, "Constraint Type"]
            constraint_limit = constraints.loc[struct_name, "Level"]

            bins, values = dosemetrics.compute_dvh(dose_volume, oar_mask)
            fig = plt.figure()
            plt.plot(
                bins,
                values,
                color="b",
                label=struct_name,
            )
            plt.xlabel("Dose [Gy]")
            plt.ylabel("Ratio of Total Structure Volume [%]")
            plt.legend(loc="best")
            plt.axvline(x=constraint_limit, color="r", label="constraint limit")
            if constraint_type == "max":
                max_dose = dosemetrics.max_dose(dose_volume, oar_mask)
                plt.axvline(x=max_dose, color="g", label="max_dose")
            elif constraint_type == "mean":
                mean_dose = dosemetrics.mean_dose(dose_volume, oar_mask)
                plt.axvline(x=mean_dose, color="g", label="mean_dose")
            print(
                f"{struct_name}, Type: {constraint_type}, Limit: {constraint_limit}, "
                f"Compliance: {dose_compliance.loc[struct_name, 'Compliance']}"
            )
            plt.title(
                f"{struct_name}, Type: {constraint_type}, Limit: {constraint_limit}, "
                f"Compliance: {'Yes' in dose_compliance.loc[struct_name, 'Compliance']}"
            )
            plt.grid()
            pp.savefig(fig)
    pp.close()


if __name__ == "__main__":
    repo_root = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(repo_root, "..", "data", "compare_plans", "first")
    results_folder = os.path.join(repo_root, "..", "results")
    print_dvh_constraint(data_folder, results_folder)

    """
    dataset_folder = "/Users/amithkamath/data/EORTC-ICR/ICR-unacceptable-variation-output"
    subject_folders = [f.path for f in os.scandir(dataset_folder) if f.is_dir()]
    for subject_folder in subject_folders:
        plan_folders = [f.path for f in os.scandir(subject_folder) if f.is_dir()]
        for plan_folder in plan_folders:
            print(f"Processing {plan_folder} ...\n")
            try:
                print_dvh_constraint(plan_folder, plan_folder)
            except:
                print(f"Failed to process {plan_folder}.\n")
            print(f"Completed {plan_folder}.\n")
    """
