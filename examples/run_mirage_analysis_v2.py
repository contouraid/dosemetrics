import os
from glob import glob

import pandas as pd
import numpy as np
import SimpleITK as sitk
import pymia.evaluation.metric as metric
import pymia.evaluation.evaluator as eval_
import pymia.evaluation.writer as writer
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

import dosemetrics.dvh as dvh
import dosemetrics.data_utils as data_utils
import dosemetrics.scores as scores
import dosemetrics.compliance as compliance

plt.style.use("dark_background")
figure(figsize=(12, 8), dpi=100)


def get_dose(data_root: str):
    contents_file = glob(os.path.join(data_root, "*.csv"))
    if len(contents_file) == 1:
        cf = pd.read_csv(contents_file[0])
        info = cf[['Structure', 'Type']].copy()
    dose_file = os.path.join(data_root, "Dose.nii.gz")
    dose_volume = data_utils.read_from_nifti(dose_file)
    return dose_volume


def get_structures(data_root: str):
    contents_file = glob(os.path.join(data_root, "*.csv"))

    mask_structures = {}
    mask_files = []
    if len(contents_file) == 1:
        cf = pd.read_csv(contents_file[0])
        info = cf[['Structure', 'Type']].copy()

        for i in range(info.shape[0]):
            if info.loc[i, "Type"] == "Target" or info.loc[i, "Type"] == "OAR":
                mask_file = os.path.join(data_root, info.loc[i, "Structure"] + ".nii.gz")
                mask_structures[info.loc[i, "Structure"]] = data_utils.read_from_nifti(mask_file)
                mask_files.append(mask_file)
    return mask_structures, mask_files


def compute_geometric_scores(a_mask_files, b_mask_files):
    a_masks = {}
    for a_file in a_mask_files:
        struct_name = a_file.split('/')[-1].split('.')[0]
        a_masks[struct_name] = a_file
    b_masks = {}
    for b_file in b_mask_files:
        struct_name = b_file.split('/')[-1].split('.')[0]
        b_masks[struct_name] = b_file

    metric_list = ["DSC",
                   "HausdorffDistance95 (mm)",
                   "HausdorffDistance100 (mm)",
                   "VolumeSimilarity",
                   "SurfaceDice",
                   "FalseNegative (cc)"
                   ]
    metrics = [metric.DiceCoefficient(),
               metric.HausdorffDistance(percentile=95, metric='HDRFDST95'),
               metric.HausdorffDistance(percentile=100, metric='HDRFDST'),
               metric.VolumeSimilarity(),
               metric.SurfaceDiceOverlap(),
               metric.FalseNegative()]
    labels = {1: 'FG'}

    stats = {}
    for struct_name in b_masks:
        if struct_name in a_masks:
            first_mask = sitk.ReadImage(a_masks[struct_name])
            first_mask.SetOrigin((0, 0, 0))
            last_mask = sitk.ReadImage(b_masks[struct_name])
            last_mask.SetOrigin((0, 0, 0))

            evaluator = eval_.SegmentationEvaluator(metrics, labels)
            evaluator.evaluate(first_mask, last_mask, struct_name)
            writer.ConsoleWriter().write(evaluator.results)
            stats[struct_name] = [f"{evaluator.results[0].value:.3f}",
                                  f"{evaluator.results[1].value:.3f}",
                                  f"{evaluator.results[2].value:.3f}",
                                  f"{evaluator.results[3].value:.3f}",
                                  f"{evaluator.results[4].value:.3f}",
                                  f"{evaluator.results[5].value * 0.008:.3f}",
                                  ]

    geom_df = pd.DataFrame.from_dict(stats, orient="index")
    geom_df.columns = metric_list
    return geom_df


def plot_dvh(dose_volume: np.ndarray, structure_masks: dict, output_file: str):

    df = dvh.dvh_by_structure(dose_volume, structure_masks)
    fig, ax = plt.subplots()
    df.set_index('Dose', inplace=True)
    df.groupby('Structure')['Volume'].plot(legend=True, ax=ax)

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(0.9, 0.5))

    plt.xlabel("Dose [Gy]")
    plt.ylabel("Ratio of Total Structure Volume [%]")
    plt.grid()
    plt.savefig(output_file)
    plt.close()


if __name__ == "__main__":

    repo_root = os.path.dirname(os.path.abspath(__file__))
    """
    data_root = os.path.join(repo_root, "..", "data", "compare_plans", "first")
    output_file = os.path.join(repo_root, "..", "results", "first_overall_dvh.png")
    plot_dvh(data_root, output_file)
    """
    data_root = "/home/akamath/Documents/data/ICR/output"

    data_struct = pd.read_csv(os.path.join(data_root, "first_last_data.csv"))

    for index, row in data_struct.iterrows():
        subject_name = row["case"]
        first_plan = row["first"]
        last_plan = row["last"]

        if (type(first_plan) is not str) or (type(last_plan) is       not str):
            continue
        else:
            print(f"Analyzing subject: {subject_name} with first plan: {first_plan} and last plan: {last_plan}")
            first_folder = os.path.join(data_root, subject_name, first_plan)
            last_folder = os.path.join(data_root, subject_name, last_plan)

            first_dose = get_dose(first_folder)
            first_structures, first_mask_files = get_structures(first_folder)

            last_dose = get_dose(last_folder)
            last_structures, last_mask_files = get_structures(last_folder)

            # First compute geometric evaluations
            geometric_scores = compute_geometric_scores(first_mask_files, last_mask_files)
            output_csv = os.path.join(data_root, subject_name, "geometric_differences.csv")
            print(f"Geometric Scores between: first structures, last structures: ")
            print(geometric_scores)
            geometric_scores.to_csv(output_csv)

            # Then create the A, B, and C conditions for dosimetric evaluations
            # and generate the corresponding DVHs and dose metrics
            output_csv = os.path.join(data_root, subject_name, "a_first_dose_first_structures_dvh.csv")
            output_image = os.path.join(data_root, subject_name, "a_first_dose_first_structures_dvh.png")
            a_df = scores.dose_summary(first_dose, first_structures)
            print(f"Dose Summary for A: first dose, first structures: ")
            print(a_df)
            a_df.to_csv(output_csv)
            plot_dvh(first_dose, first_structures, output_image)

            output_csv = os.path.join(data_root, subject_name, "b_first_dose_last_structures_dvh.csv")
            output_image = os.path.join(data_root, subject_name, "b_first_dose_last_structures_dvh.png")
            b_df = scores.dose_summary(first_dose, last_structures)
            print(f"Dose Summary for B: first dose, last structures: ")
            print(b_df)
            b_df.to_csv(output_csv)
            plot_dvh(first_dose, last_structures, output_image)

            output_csv = os.path.join(data_root, subject_name, "c_last_dose_last_structures_dvh.csv")
            output_image = os.path.join(data_root, subject_name, "c_last_dose_last_structures_dvh.png")
            c_df = scores.dose_summary(last_dose, last_structures)
            print(f"Dose Summary for C: last dose, last structures: ")
            print(c_df)
            c_df.to_csv(output_csv)
            plot_dvh(last_dose, last_structures, output_image)

            # Finally check what the clinical constraint violations look like.

            output_csv = os.path.join(data_root, subject_name, "a_first_dose_first_structures_compliance.csv")
            compliance_df = compliance.compute_mirage_compliance(first_dose, first_structures)
            compliance_df.to_csv(output_csv)

            output_csv = os.path.join(data_root, subject_name, "b_first_dose_last_structures_compliance.csv")
            compliance_df = compliance.compute_mirage_compliance(first_dose, last_structures)
            compliance_df.to_csv(output_csv)

            output_csv = os.path.join(data_root, subject_name, "c_last_dose_last_structures_compliance.csv")
            compliance_df = compliance.compute_mirage_compliance(last_dose, last_structures)
            compliance_df.to_csv(output_csv)