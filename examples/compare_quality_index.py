import dosemetrics

import os
import glob
import SimpleITK as sitk
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("dark_background")


def compare_quality_index(input_folder: str, output_folder: str):
    dose_image = sitk.ReadImage(os.path.join(input_folder, "Dose.nii.gz"))
    dose_array = sitk.GetArrayFromImage(dose_image)

    predicted_dose_image = sitk.ReadImage(
        os.path.join(input_folder, "Predicted_Dose.nii.gz")
    )
    predicted_dose_array = sitk.GetArrayFromImage(predicted_dose_image)

    df = pd.DataFrame()

    constraints = dosemetrics.get_default_constraints()

    structures = glob.glob(data_folder + "/*[!Dose*].nii.gz")
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

            qi = dosemetrics.quality_index(
                dose_array, oar_mask, constraint_type, constraint_limit
            )
            predicted_qi = dosemetrics.quality_index(
                predicted_dose_array,
                oar_mask,
                constraint_type,
                constraint_limit,
            )
            print(
                f"{struct_name}, Type: {constraint_type}, Limit: {constraint_limit}, "
                f"QI: {qi:.2f}, "
                f"Predicted QI: {predicted_qi:.2f}, "
            )

            df.loc[struct_name, "Type"] = constraint_type
            df.loc[struct_name, "Limit"] = constraint_limit
            df.loc[struct_name, "QI"] = qi
            df.loc[struct_name, "Predicted QI"] = predicted_qi
    df.loc["Correlation", "Predicted QI"] = df["QI"].corr(df["Predicted QI"])
    df.to_csv(
        os.path.join(
            output_folder, f"compare_quality_index_{data_folder.split('/')[-1]}.csv"
        )
    )


if __name__ == "__main__":
    repo_root = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(repo_root, "..", "data", "test_subject")

    result_folder = os.path.join(repo_root, "..", "results")
    os.makedirs(result_folder, exist_ok=True)
    compare_quality_index(data_folder, result_folder)
