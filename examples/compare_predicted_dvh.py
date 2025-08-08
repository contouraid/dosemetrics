import os
import dosemetrics


def compare_predicted_dvh(input_folder: str, output_folder: str):
    """
    Compare predicted vs actual DVH using the new dosemetrics functionality.

    This example demonstrates how to use the new compare_predicted_doses function
    which replaces the previous manual implementation.
    """
    # Read dose and structure data using new utility functions
    dose_array, structure_masks = dosemetrics.read_dose_and_mask_files_from_folder(
        input_folder, dose_filename="Dose.nii.gz"
    )

    # Read predicted dose
    predicted_dose_array = dosemetrics.read_from_nifti(
        os.path.join(input_folder, "Predicted_Dose.nii.gz")
    )

    # Use the new comparison function
    output_file = os.path.join(output_folder, "compare_prediction.pdf")
    dosemetrics.compare_predicted_doses(
        dose_array=dose_array,
        predicted_array=predicted_dose_array,
        structure_masks=structure_masks,
        output_file=output_file,
    )

    print(f"DVH comparison saved to: {output_file}")


if __name__ == "__main__":
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_folder = os.path.join(repo_root, "data", "test_subject")
    result_folder = os.path.join(repo_root, "results")
    os.makedirs(result_folder, exist_ok=True)
    compare_predicted_dvh(data_folder, result_folder)
