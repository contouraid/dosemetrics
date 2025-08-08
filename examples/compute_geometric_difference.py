import numpy as np

import dosemetrics
import os
import SimpleITK as sitk
import pandas as pd


def get_structures(input_folder: str):
    contents = os.path.join(input_folder, "standard_contents.csv")
    df = pd.read_csv(contents)
    masks = list(df[df["Type"] == "OAR"]["Structure"])
    masks += list(df[df["Type"] == "Target"]["Structure"])
    mask_files = [os.path.join(input_folder, f + ".nii.gz") for f in masks]
    return mask_files


def read_mask_files(mask_files):
    structure_masks = {}
    for mask_file in mask_files:
        mask_volume = dosemetrics.read_from_nifti(mask_file)
        struct_name = mask_file.split("/")[-1].split(".")[0]
        structure_masks[struct_name] = mask_volume
    return structure_masks


def compute_geometric_metrics(structure_pair: tuple, output_folder: str):

    first_structures, last_structures = structure_pair
    first_masks = read_mask_files(first_structures)
    last_masks = read_mask_files(last_structures)

    stats = {}
    for struct_name in first_masks.keys():
        first_mask = first_masks[struct_name]
        last_mask = last_masks[struct_name]

        intersection = np.logical_and(first_mask, last_mask)

        dice = 2 * intersection.sum() / (first_mask.sum() + last_mask.sum())
        if last_mask.sum() > first_mask.sum():
            print(f"For {struct_name}, DSC: {dice}; Last larger than first")
            stats[struct_name] = {dice, "Last larger than first"}
        elif last_mask.sum() < first_mask.sum():
            print(f"For {struct_name}, DSC: {dice}; Last smaller than first")
            stats[struct_name] = {dice, "Last smaller than first"}
        else:
            print(f"For {struct_name}, DSC: {dice}; Unchanged!")
            stats[struct_name] = {dice, "unchanged"}
    df = pd.DataFrame.from_dict(stats, orient="index", columns=["DSC", "Change"])
    df.to_csv(os.path.join(output_folder, "geometric_difference.csv"))


if __name__ == "__main__":
    repo_root = os.path.dirname(os.path.abspath(__file__))
    """
    first_folder = os.path.join(repo_root, "..", "data", "compare_plans", "first")
    last_folder = os.path.join(repo_root, "..", "data", "compare_plans", "last")
    structures_first = get_structures(first_folder)
    structures_last = get_structures(last_folder)

    results_folder = os.path.join(repo_root, "..", "results")
    compute_geometric_metrics((structures_first, structures_last), results_folder)
    """

    dataset_folder = (
        "/Users/amithkamath/data/EORTC-ICR/ICR-unacceptable-variation-output"
    )
    subject_folders = [f.path for f in os.scandir(dataset_folder) if f.is_dir()]
    for subject_folder in subject_folders:
        print(f"Processing {subject_folder} ...\n")
        try:
            first_folder = os.path.join(subject_folder, "first")
            last_folder = os.path.join(subject_folder, "last")
            structures_first = get_structures(first_folder)
            structures_last = get_structures(last_folder)

            results_folder = os.path.join(subject_folder)
            compute_geometric_metrics(
                (structures_first, structures_last), results_folder
            )
        except:
            print(f"Failed to process {subject_folder}.\n")
        print(f"Completed {subject_folder}.\n")
