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
    data_dir = os.path.join("/Users/amithkamath", "data", "DLDP", "astute-experiments")
    output_dir = os.path.join(data_dir, "DVH")
    os.makedirs(output_dir, exist_ok=True)

    subject_prefix = "ISAS_GBM_074"

    # First report dose and DVH scores between predicted dose and planned dose.
    # This is sufficient validation that the predicted dose is a reasonably good
    # proxy to the actual planned dose. If the differences are large, this is a problem.
    # TODO after Robert shares the ground truth dose.
    # print(f"Dose/DVH scores for: {subject_prefix}")
    # scores = compute_scores(data_dir)
    # scores.to_csv(os.path.join(output_dir, subject_prefix + ".csv"))
    # print(f"Completed: {subject_prefix}")

    # Then look at the dose in the difference between the original target and the modified target
    # If the difference region already has a large enough dose, adding it into the target is
    # less risky as the dose there is already high. How this impacts the OARs is where the analysis helps!
    default_dose_folder = os.path.join(data_dir, subject_prefix + "0")
    default_predicted_dose = sitk.GetArrayFromImage(
        sitk.ReadImage(os.path.join(default_dose_folder, "Predicted_Dose.nii.gz"))
    )

    default_target_mask = sitk.GetArrayFromImage(
        sitk.ReadImage(os.path.join(default_dose_folder, "Target.nii.gz"))
    )

    for variant in range(1, 5):
        variant_folder = os.path.join(data_dir, subject_prefix + str(variant))
        variant_target_mask = sitk.GetArrayFromImage(
            sitk.ReadImage(os.path.join(variant_folder, "Target.nii.gz"))
        )

        difference_mask = np.logical_xor(default_target_mask, variant_target_mask)
        mean_dose_in_difference = np.mean(default_predicted_dose[difference_mask > 0])
        variance_dose_in_difference = np.std(
            default_predicted_dose[difference_mask > 0]
        )
        print(
            f"For variant {variant}, the mean dose in the difference target region is: mean (var): {mean_dose_in_difference} ({variance_dose_in_difference})"
        )

    # Then look at the mean dose difference within each OAR - to get an OAR specific score (maybe like
    # the OAR specific dose score) and link it to the constraints - if these are violated, the ranking should
    # get worse. Aggregate these to all the OARs.
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
    ]

    default_dose_folder = os.path.join(data_dir, subject_prefix + "0")
    default_predicted_dose = sitk.GetArrayFromImage(
        sitk.ReadImage(os.path.join(default_dose_folder, "Predicted_Dose.nii.gz"))
    )

    oar_mean_df = pd.DataFrame(index=list_oar_names, columns=range(5))
    oar_max_df = pd.DataFrame(index=list_oar_names, columns=range(5))

    for oar in list_oar_names:

        oar_mask = sitk.GetArrayFromImage(
            sitk.ReadImage(os.path.join(default_dose_folder, oar + ".nii.gz"))
        )

        mean_dose_in_oar_orig = np.mean(default_predicted_dose[oar_mask > 0])
        variance_dose_in_orig = np.std(default_predicted_dose[oar_mask > 0])
        oar_mean_df.loc[oar, 0] = mean_dose_in_oar_orig

        print(
            f"The mean original dose in {oar} region is: mean (var): {mean_dose_in_oar_orig} ({variance_dose_in_orig})"
        )

        max_dose_in_oar_orig = np.max(default_predicted_dose[oar_mask > 0])
        oar_max_df.loc[oar, 0] = max_dose_in_oar_orig

        print(f"The max original dose in {oar} region is: {max_dose_in_oar_orig})")

        for variant in range(1, 5):
            variant_folder = os.path.join(data_dir, subject_prefix + str(variant))
            variant_predicted_dose = sitk.GetArrayFromImage(
                sitk.ReadImage(os.path.join(variant_folder, "Predicted_Dose.nii.gz"))
            )

            mean_dose_in_variant_oar = np.mean(variant_predicted_dose[oar_mask > 0])
            variance_dose_in_variant_oar = np.std(variant_predicted_dose[oar_mask > 0])
            oar_mean_df.loc[oar, variant] = mean_dose_in_variant_oar

            print(
                f"For variant {variant}, the mean variant dose in {oar} region is: mean (var): {mean_dose_in_variant_oar} ({variance_dose_in_variant_oar})"
            )

            max_dose_in_variant_oar = np.max(variant_predicted_dose[oar_mask > 0])
            oar_max_df.loc[oar, variant] = max_dose_in_variant_oar

            print(
                f"For variant {variant}, the max dose in {oar} region is: {max_dose_in_variant_oar})"
            )

    oar_mean_df.to_csv(os.path.join(output_dir, subject_prefix + "_mean.csv"))
    oar_max_df.to_csv(os.path.join(output_dir, subject_prefix + "_max.csv"))

    # Based on this, rank the four alternatives versus 0: do this separately for the dose plans;
    # then also do this for the predicted dose - compare against predicted doses 0 to 1 - 4.

    # End up with a list of rankings for predicted doses - a list of four numbers for each case.

    # This can then be correlated to the radio-oncologists' rankings - whoever has better correlation wins!
