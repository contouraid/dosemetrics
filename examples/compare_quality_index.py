import os
import dosemetrics


def compare_quality_index(input_folder: str, output_folder: str):
    """
    Compare quality indices between actual and predicted doses.

    This example demonstrates the new compare_quality_indices function.
    """
    # Read dose and structure data
    dose_array, structure_masks = dosemetrics.read_dose_and_mask_files_from_folder(
        input_folder, dose_filename="Dose.nii.gz"
    )

    # Read predicted dose
    predicted_dose_array = dosemetrics.read_from_nifti(
        os.path.join(input_folder, "Predicted_Dose.nii.gz")
    )

    # Compare quality indices using new function
    comparison_df = dosemetrics.compare_quality_indices(
        dose_arrays=[dose_array, predicted_dose_array],
        structure_masks=structure_masks,
        labels=["Actual", "Predicted"],
    )

    # Save results
    subject_name = os.path.basename(input_folder)
    output_file = os.path.join(
        output_folder, f"compare_quality_index_{subject_name}.csv"
    )
    comparison_df.to_csv(output_file, index=False)

    print(f"Quality index comparison saved to: {output_file}")
    print("\nQuality Index Comparison Summary:")
    print(comparison_df.to_string(index=False))


if __name__ == "__main__":
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_folder = os.path.join(repo_root, "data", "test_subject")
    result_folder = os.path.join(repo_root, "results")
    os.makedirs(result_folder, exist_ok=True)
    compare_quality_index(data_folder, result_folder)
