from dosemetrics import compliance

import os
import glob
import SimpleITK as sitk
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("dark_background")


def main():

    repo_root = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(repo_root, "..", "data", "test_subject")

    dose_image = sitk.ReadImage(data_folder + "/Dose.nii.gz")
    dose_array = sitk.GetArrayFromImage(dose_image)

    predicted_dose_image = sitk.ReadImage(
        data_folder + "/Predicted_Dose.nii.gz"
    )
    predicted_dose_array = sitk.GetArrayFromImage(predicted_dose_image)

    df = pd.DataFrame()

    constraints = compliance.get_default_constraints()

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

            qi = compliance.quality_index(
                dose_array, oar_mask, constraint_type, constraint_limit
            )
            predicted_qi = compliance.quality_index(
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
            data_folder,
            "..",
            f"compare_quality_index_{data_folder.split('/')[-1]}.csv",
        )
    )


if __name__ == "__main__":
    main()
