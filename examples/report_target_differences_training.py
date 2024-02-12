# -*- encoding: utf-8 -*-
import os
import numpy as np
import SimpleITK as sitk
import pandas as pd

import dosemetrics.scores as metrics


def compute_scores(data_dir):
    """
    compute_scores runs dose_score, dvh_score and plots DVH curves for each OAR
    stored in .nii files in data_dir.
    """

    dose_nii = sitk.ReadImage(os.path.join(data_dir, "Dose.nii.gz"))
    dose_array = sitk.GetArrayFromImage(dose_nii)
    spacing = dose_nii.GetSpacing()

    predicted_dose_nii = sitk.ReadImage(os.path.join(data_dir, "Predicted_Dose.nii.gz"))
    predicted_dose_array = sitk.GetArrayFromImage(predicted_dose_nii)

    list_oar_names = [
        "BrainStem",
        "Chiasm",
        "Cochlea_L",
        "Cochlea_R",
        "Eye_L",
        "Eye_R",
        "Hippocampus_L",
        "Hippocampus_R",
        "LacrimalGland_L",
        "LacrimalGland_R",
        "OpticNerve_L",
        "OpticNerve_R",
        "Pituitary",
        "Target",
    ]

    scores = {}
    for oar in list_oar_names:
        oar_nii = sitk.ReadImage(os.path.join(data_dir, oar + ".nii.gz"))
        oar_array = sitk.GetArrayFromImage(oar_nii)

        # Dose score computation
        dose_result = metrics.dose_score(predicted_dose_array, dose_array, oar_array)

        # DVH score computation
        if oar == "Target":
            actual_dvh_results = metrics.dvh_score(
                dose_array, oar_array, "target", spacing
            )
            predicted_dvh_results = metrics.dvh_score(
                predicted_dose_array, oar_array, "target", spacing
            )
        else:
            actual_dvh_results = metrics.dvh_score(
                dose_array, oar_array, "OAR", spacing
            )
            predicted_dvh_results = metrics.dvh_score(
                predicted_dose_array, oar_array, "OAR", spacing
            )

        dvh_value = 0.0
        for metric in actual_dvh_results.keys():
            dvh_value += abs(actual_dvh_results[metric] - predicted_dvh_results[metric])

        scores[oar] = {"Dose": dose_result, "DVH": dvh_value}

    subject_df = pd.DataFrame(scores).transpose()
    subject_df.columns = ["Dose", "DVH"]

    return subject_df


if __name__ == "__main__":
    data_dir = os.path.join("/Users/amithkamath", "data", "DLDP", "processed-to-train")

    subject_prefix = "ISAS_GBM_"
    for subject_id in range(20, 31):

        subject_name = subject_prefix + str(subject_id).zfill(3)
        input_dir = os.path.join(data_dir, subject_name)
        output_dir = input_dir

        scores = compute_scores(input_dir)
        scores.to_csv(os.path.join(output_dir, subject_name + ".csv"))
        print(f"Completed: {subject_name}")
