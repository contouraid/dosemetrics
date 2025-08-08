"""
GENERATE_DVH_VARIATION: Generate dose-volume histogram variation analysis.
Updated for the new dosemetrics package structure.
"""

import dosemetrics

import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import matplotlib as mpl

# Use the new package structure
read_from_nifti = dosemetrics.read_from_nifti
dose_summary = dosemetrics.dose_summary
check_compliance = dosemetrics.check_compliance
get_default_constraints = dosemetrics.get_default_constraints
compute_dvh = dosemetrics.metrics.dvh.compute_dvh

# Rest of the original functionality would be updated similarly...
# This is a template showing how to update examples for the new structure.


def generate_dvh_with_variations(
    dose_file, structure_files, structure_of_interest, constraints
):
    """
    Example function showing how to use the new dosemetrics package structure.
    """
    # Read dose volume
    dose_volume = read_from_nifti(dose_file)

    # Read structure masks
    structure_masks = {}
    for mask_file in structure_files:
        structure_name = os.path.basename(mask_file).split(".")[0]
        structure_masks[structure_name] = read_from_nifti(mask_file)

    # Compute dose summary
    dose_df = dose_summary(
        dose_volume, {structure_of_interest: structure_masks[structure_of_interest]}
    )

    # Check compliance
    dose_compliance = check_compliance(dose_df, constraints)

    print(f"Dose compliance for {structure_of_interest}: {dose_compliance}")

    return dose_df


if __name__ == "__main__":
    print("Updated example for dosemetrics v0.2.0")
    print("Please update paths and run generate_dvh_with_variations() with your data")
