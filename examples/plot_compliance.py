import os
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("dark_background")

import dosemetrics


def compute_dose_compliance(data_folder):

    compliance_dict = {}
    contents_files = [f for f in os.listdir(data_folder) if f.endswith(".csv")]
    if len(contents_files) > 0:
        contents_file = contents_files[0]
        contents = pd.read_csv(os.path.join(data_folder, contents_file))
        dose_file = os.path.join(data_folder, "Dose.nii.gz")
        dose_volume = dosemetrics.read_from_nifti(dose_file)

        structure_masks = {}
        oar_list = contents[contents["Type"] == "OAR"]["Structure"].values
        for oar in oar_list:
            oar_file = os.path.join(data_folder, oar + ".nii.gz")
            oar_volume = dosemetrics.read_from_nifti(oar_file)
            structure_masks[oar] = oar_volume

        target_list = contents[contents["Type"] == "Target"]["Structure"].values
        for target in target_list:
            target_file = os.path.join(data_folder, target + ".nii.gz")
            target_volume = dosemetrics.read_from_nifti(target_file)
            structure_masks[target] = target_volume

        constraints = dosemetrics.get_default_constraints()
        for structure in structure_masks.keys():
            if structure not in constraints.index:
                continue
            else:
                constraint_type = constraints.loc[structure, "Constraint Type"]
                constraint_limit = constraints.loc[structure, "Level"]
                if constraint_type == "max":
                    max_dose = dosemetrics.max_dose(
                        dose_volume, structure_masks[structure]
                    )
                    if max_dose > constraint_limit:
                        reason = f"Max dose constraint: {constraint_limit}, exceeded: {max_dose:.2f}"
                        compliance_dict[structure] = [
                            constraint_type,
                            constraint_limit,
                            "❌ No",
                            reason,
                        ]
                    else:
                        compliance_dict[structure] = [
                            constraint_type,
                            constraint_limit,
                            "✅ Yes",
                            "NA",
                        ]
                elif constraint_type == "mean":
                    mean_dose = dosemetrics.mean_dose(
                        dose_volume, structure_masks[structure]
                    )
                    if mean_dose > constraint_limit:
                        reason = f"Mean dose constraint: {constraint_limit}, exceeded: {mean_dose:.2f}"
                        compliance_dict[structure] = [
                            constraint_type,
                            constraint_limit,
                            "❌ No",
                            reason,
                        ]
                    else:
                        compliance_dict[structure] = [
                            constraint_type,
                            constraint_limit,
                            "✅ Yes",
                            "NA",
                        ]

        compliance_df = pd.DataFrame.from_dict(
            compliance_dict,
            orient="index",
            columns=["Type", "Limit", "Compliance", "Reason"],
        )
        return compliance_df


if __name__ == "__main__":

    dataset_root = "/mnt/5b9b7229-4179-4263-babd-004c30510079/data/USZ/nifti-output"
    output_folder = os.path.join(dataset_root, "..", "compliance-data")
    os.makedirs(output_folder, exist_ok=True)

    subfolders = [f.path for f in os.scandir(dataset_root) if f.is_dir()]
    subfolders = sorted(subfolders)

    for subject_folder in subfolders:
        subject_name = subject_folder.split("/")[-1]
        try:
            compliance_data = compute_dose_compliance(subject_folder)
            compliance_data.to_csv(
                os.path.join(output_folder, f"{subject_name}_compliance.csv")
            )
        except Exception as e:
            print(f"Error processing {subject_name}: {e}")
            continue
