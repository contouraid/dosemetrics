from dosemetrics import dvh
from dosemetrics import compliance

import os
import glob
import SimpleITK as sitk
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

plt.style.use("dark_background")


def print_quality_index(input_folder: str, output_folder: str):
    dose_image = sitk.ReadImage(input_folder + "/Dose.nii.gz")
    dose_array = sitk.GetArrayFromImage(dose_image)

    df = pd.DataFrame()

    pp = PdfPages(os.path.join(output_folder, "quality_index.pdf"))

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

            bins, values = dvh.compute_dvh(dose_array, oar_mask)
            qi = compliance.quality_index(
                dose_array, oar_mask, constraint_type, constraint_limit
            )
            fig = plt.figure()
            plt.plot(
                bins, values, color="b", label=struct_name,
            )
            plt.axvline(x=constraint_limit, color="r", label="constraint limit")
            if constraint_type == "max":
                max_dose = dvh.max_dose(dose_array, oar_mask)
                plt.axvline(x=max_dose, color="g", label="max_dose")
            elif constraint_type == "mean":
                mean_dose = dvh.mean_dose(dose_array, oar_mask)
                plt.axvline(x=mean_dose, color="g", label="mean_dose")
            print(
                f"{struct_name}, Type: {constraint_type}, Limit: {constraint_limit}, QI: {qi:.2f}"
            )
            plt.title(
                f"{struct_name}, Type: {constraint_type}, Limit: {constraint_limit}, QI: {qi:.2f}"
            )
            plt.grid()
            pp.savefig(fig)

            df.loc[struct_name, "Type"] = constraint_type
            df.loc[struct_name, "Limit"] = constraint_limit
            df.loc[struct_name, "QI"] = qi
    pp.close()
    df.to_csv(os.path.join(output_folder, "quality_index.csv"))


if __name__ == "__main__":
    repo_root = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(repo_root, "..", "data", "test_subject")
    results_folder = os.path.join(repo_root, "..", "results")
    print_quality_index(data_folder, results_folder)
