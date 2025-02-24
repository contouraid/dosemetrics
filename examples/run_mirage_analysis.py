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


def compute_compliance(dose_volume: np.ndarray, structure_masks: dict, output_csv: str):
    compliance_stats = {}

    for struct_name in sorted(structure_masks.keys()):
        struct_mask = structure_masks[struct_name]
        dose_in_struct = dose_volume[struct_mask > 0]
        if struct_name == "Chiasm":
            # Optic Chiasm: ≤55 Gy. to 0.03cc, Optic Chiasm PRV ≤55Gy to 0.03cc
            # Checking for 4 because our voxel grid is 2mmx2mmx2mm, meaning each voxel is 8mm3.
            # 0.03cc is 30mm3, which is between 3 and 4 voxels - 24mm3 and 32mm3.
            if struct_mask.sum() > 4:
                sorted_dose = np.sort(dose_in_struct)[::-1]
                calculated_dose = sorted_dose[3]
                limit_dose = 55
                if calculated_dose >= limit_dose:
                    reason = f"{struct_name} <= {limit_dose} Gy to 0.03cc violated. Dose in 0.03cc is {calculated_dose:.3f}"
                    compliance_stats[struct_name] = ["Fail", reason]
                else:
                    reason = f"{struct_name} <= {limit_dose} Gy to 0.03cc achieved. Dose in 0.03cc is {calculated_dose:.3f}"
                    compliance_stats[struct_name] = ["Pass", reason]
            else:
                reason = f"{struct_name} volume is smaller than 0.03cc."
                compliance_stats[struct_name] = ["NA", reason]
            print(compliance_stats[struct_name])

        elif struct_name == "Brainstem":
            # Brainstem: ≤56 Gy. to 0.03cc
            if struct_mask.sum() > 4:
                sorted_dose = np.sort(dose_in_struct)[::-1]
                calculated_dose = sorted_dose[3]
                limit_dose = 56
                if calculated_dose >= limit_dose:
                    reason = f"{struct_name} <= {limit_dose} Gy to 0.03cc violated. Dose in 0.03cc is {calculated_dose:.3f}"
                    compliance_stats[struct_name] = ["Fail", reason]
                else:
                    reason = f"{struct_name} <= {limit_dose} Gy to 0.03cc achieved. Dose in 0.03cc is {calculated_dose:.3f}"
                    compliance_stats[struct_name] = ["Pass", reason]
            else:
                reason = f"{struct_name} volume is smaller than 0.03cc."
                compliance_stats[struct_name] = ["NA", reason]
            print(compliance_stats[struct_name])

        elif "Cochlea" in struct_name:
            # Cochlea: ≤45 Gy if both sides are involved; otherwise ≤60 Gy. (Low priority OaR) to 0.03cc
            if struct_mask.sum() > 4:
                sorted_dose = np.sort(dose_in_struct)[::-1]
                calculated_dose = sorted_dose[3]
                limit_dose = 45
                if calculated_dose >= limit_dose:
                    reason = f"{struct_name} <= {limit_dose} Gy to 0.03cc violated. Dose in 0.03cc is {calculated_dose:.3f}"
                    compliance_stats[struct_name] = ["Fail", reason]
                else:
                    reason = f"{struct_name} <= {limit_dose} Gy to 0.03cc achieved. Dose in 0.03cc is {calculated_dose:.3f}"
                    compliance_stats[struct_name] = ["Pass", reason]
            else:
                reason = f"{struct_name} volume is smaller than 0.03cc."
                compliance_stats[struct_name] = ["NA", reason]
            print(compliance_stats[struct_name])

        elif "LacrimalGland" in struct_name:
            # Lacrimal glands: <40 Gy to 0.03cc
            if struct_mask.sum() > 4:
                sorted_dose = np.sort(dose_in_struct)[::-1]
                calculated_dose = sorted_dose[3]
                limit_dose = 40
                if calculated_dose >= limit_dose:
                    reason = f"{struct_name} <= {limit_dose} Gy to 0.03cc violated. Dose in 0.03cc is {calculated_dose:.3f}"
                    compliance_stats[struct_name] = ["Fail", reason]
                else:
                    reason = f"{struct_name} <= {limit_dose} Gy to 0.03cc achieved. Dose in 0.03cc is {calculated_dose:.3f}"
                    compliance_stats[struct_name] = ["Pass", reason]
            else:
                reason = f"{struct_name} volume is smaller than 0.03cc."
                compliance_stats[struct_name] = ["NA", reason]
            print(compliance_stats[struct_name])

        elif "OpticNerve" in struct_name:
            # Optic Nerves ≤ 56 Gy to 0.03cc, Optic Nerves PRV: ≤56 Gy. to 0.03cc            
            if struct_mask.sum() > 4:
                sorted_dose = np.sort(dose_in_struct)[::-1]
                calculated_dose = sorted_dose[3]
                limit_dose = 56
                if calculated_dose >= limit_dose:
                    reason = f"{struct_name} <= {limit_dose} Gy to 0.03cc violated. Dose in 0.03cc is {calculated_dose:.3f}"
                    compliance_stats[struct_name] = ["Fail", reason]
                else:
                    reason = f"{struct_name} <= {limit_dose} Gy to 0.03cc achieved. Dose in 0.03cc is {calculated_dose:.3f}"
                    compliance_stats[struct_name] = ["Pass", reason]
            else:
                reason = f"{struct_name} volume is smaller than 0.03cc."
                compliance_stats[struct_name] = ["NA", reason]
            print(compliance_stats[struct_name])

        elif struct_name == "Brain":
            # The dose to the normal brain minus the PTV should be kept as low as possible. The Dmean is to be ≤ 30 Gy
            limit_dose = 30
            calculated_dose = dose_in_struct.mean()
            if calculated_dose >= limit_dose:
                reason = f"{struct_name} Dmean <= {limit_dose} Gy is violated. Dmean is {calculated_dose:.3f}"
                compliance_stats[struct_name] = ["Fail", reason]
            else:
                reason = f"{struct_name} Dmean <= {limit_dose} Gy is achieved. Dmean is {calculated_dose:.3f}"
                compliance_stats[struct_name] = ["Pass", reason]
            print(compliance_stats[struct_name])

        elif "Eye" in struct_name:
            # Eye balls, retina <= 40 G to Dmax
            limit_dose = 40
            calculated_dose = dose_in_struct.max()
            if calculated_dose >= limit_dose:
                reason = f"{struct_name} Dmax <= {limit_dose} Gy is violated. Dmean is {calculated_dose:.3f}"
                compliance_stats[struct_name] = ["Fail", reason]
            else:
                reason = f"{struct_name} Dmax <= {limit_dose} Gy is achieved. Dmean is {calculated_dose:.3f}"
                compliance_stats[struct_name] = ["Pass", reason]
            print(compliance_stats[struct_name])

        elif "Lens" in struct_name:
            # Lens: 10 Gy to 0.03cc
            if struct_mask.sum() > 4:
                sorted_dose = np.sort(dose_in_struct)[::-1]
                calculated_dose = sorted_dose[3]
                limit_dose = 10
                if calculated_dose >= limit_dose:
                    reason = f"{struct_name} <= {limit_dose} Gy to 0.03cc violated. Dose in 0.03cc is {calculated_dose:.3f}"
                    compliance_stats[struct_name] = ["Fail", reason]
                else:
                    reason = f"{struct_name} <= {limit_dose} Gy to 0.03cc achieved. Dose in 0.03cc is {calculated_dose:.3f}"
                    compliance_stats[struct_name] = ["Pass", reason]
            else:
                reason = f"{struct_name} volume is smaller than 0.03cc."
                compliance_stats[struct_name] = ["NA", reason]
            print(compliance_stats[struct_name])

        else:
            compliance_stats[struct_name] = ["NA", f"{struct_name} either has no constraints; or is not defined for both versions."]

    compliance_df = pd.DataFrame.from_dict(compliance_stats, orient="index")
    compliance_df.columns = ["Status", "Reason"]
    compliance_df.to_csv(output_csv)

if __name__ == "__main__":

    repo_root = os.path.dirname(os.path.abspath(__file__))
    """
    data_root = os.path.join(repo_root, "..", "data", "compare_plans", "first")
    output_file = os.path.join(repo_root, "..", "results", "first_overall_dvh.png")
    plot_dvh(data_root, output_file)
    """
    data_root = "/Users/amithkamath/data/EORTC-ICR/output/"
    subject_name = "002_site_306"
    first_plan = "eortc_0001_002_site_306_delineation_1_corrected"
    last_plan = "eortc_0003_002_site_306_delineation_2_dose_1_corrected"

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
    compute_compliance(first_dose, first_structures, output_csv)

    output_csv = os.path.join(data_root, subject_name, "b_first_dose_last_structures_compliance.csv")
    compute_compliance(first_dose, last_structures, output_csv)

    output_csv = os.path.join(data_root, subject_name, "c_last_dose_last_structures_compliance.csv")
    compute_compliance(last_dose, last_structures, output_csv)