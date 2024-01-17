# -*- encoding: utf-8 -*-
import os
import sys
import numpy as np
import SimpleITK as sitk
import pandas as pd

import dosemetrics.metrics as metrics

if os.path.abspath("..") not in sys.path:
    sys.path.insert(0, os.path.abspath(".."))


def copy_sitk_imageinfo(image1, image2):
    image2.SetSpacing(image1.GetSpacing())
    image2.SetDirection(image1.GetDirection())
    image2.SetOrigin(image1.GetOrigin())

    return image2


def read_data(patient_dir):
    dict_images = {}
    list_structures = [
        "CT",
        "Dose_Mask",
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
        "Dose",
        "Target",
    ]

    for structure_name in list_structures:
        structure_file = patient_dir + "/" + structure_name + ".nii.gz"

        if structure_name == "CT":
            dtype = sitk.sitkInt16
        elif structure_name == "Dose":
            dtype = sitk.sitkFloat32
        else:
            dtype = sitk.sitkUInt8

        if os.path.exists(structure_file):
            dict_images[structure_name] = sitk.ReadImage(structure_file, dtype)
            # To numpy array (C * Z * H * W)
            dict_images[structure_name] = sitk.GetArrayFromImage(
                dict_images[structure_name]
            )[np.newaxis, :, :, :]
        else:
            dict_images[structure_name] = np.zeros((1, 128, 128, 128), np.uint8)

    return dict_images


def compute_dosemetrics(data_dir, compare_dir):
    """
    compute_dosemetrics runs dose_score, dvh_score and plots DVH curves for each OAR
    stored in .nii files in data_dir.
    """

    dose_nii = sitk.ReadImage(data_dir + "/Dose.nii.gz")
    dose_array = sitk.GetArrayFromImage(dose_nii)
    spacing = dose_nii.GetSpacing()

    compare_dose_nii = sitk.ReadImage(compare_dir + "/Dose.nii.gz")
    compare_dose_array = sitk.GetArrayFromImage(compare_dose_nii)

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
        # print("Working on: ", oar.split("_")[0])
        oar_nii = sitk.ReadImage(data_dir + "/" + oar + ".nii.gz")
        oar_array = sitk.GetArrayFromImage(oar_nii)

        # Dose score computation
        dose_result = metrics.dose_score(compare_dose_array, dose_array, oar_array)
        # print(f"Dose score: {dose_result}")

        # DVH score computation
        if oar == "Target":
            dvh_results = metrics.dvh_score(dose_array, oar_array, "target", spacing)
            compare_dvh_results = metrics.dvh_score(
                compare_dose_array, oar_array, "target", spacing
            )
        else:
            dvh_results = metrics.dvh_score(dose_array, oar_array, "OAR", spacing)
            compare_dvh_results = metrics.dvh_score(
                compare_dose_array, oar_array, "OAR", spacing
            )

        dvh_value = 0.0
        for metric in dvh_results.keys():
            dvh_value += abs(dvh_results[metric] - compare_dvh_results[metric])
        # print(f"DVH score: {dvh_value}")

        scores[oar] = {"Dose": dose_result, "DVH": dvh_value}

    return scores


if __name__ == "__main__":
    root_dir = os.path.abspath("..")
    data_dir = "/Users/amithkamath/data/DLDP"
    output_dir = os.path.join(data_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    gt_dir = os.path.join(data_dir, "ground_truth")
    pred_noTTA_dir = os.path.join(data_dir, "output_perturb_noTTA", "Prediction")
    pred_TTA_dir = os.path.join(data_dir, "output_perturb_withTTA", "Prediction")

    subjects = range(81, 100)
    for subject_id in subjects:
        gt_data_dir = os.path.join(gt_dir, "DLDP_" + str(subject_id).zfill(3))
        noTTA_dir = os.path.join(pred_noTTA_dir, "DLDP_" + str(subject_id).zfill(3))
        noTTA_scores = compute_dosemetrics(gt_data_dir, noTTA_dir)

        TTA_dir = os.path.join(pred_TTA_dir, "DLDP_" + str(subject_id).zfill(3))
        TTA_scores = compute_dosemetrics(gt_data_dir, TTA_dir)

        noTTA_df = pd.DataFrame(noTTA_scores)
        TTA_df = pd.DataFrame(TTA_scores)
        subject_df = pd.concat([noTTA_df, TTA_df]).transpose()
        subject_df.columns = ["Dose_noTTA", "DVH_noTTA", "Dose_TTA", "DVH_TTA"]
        subject_df.to_csv(os.path.join(output_dir, str(subject_id) + ".csv"))
        print(f"Done with subject {subject_id}")
