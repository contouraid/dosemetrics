import os
import dosemetrics


def visualize_dose_differences(input_folder: str, output_file: str, n_slices: int = 30):
    """
    Visualize dose differences using the new plot_dose_differences function.

    This example demonstrates the new advanced visualization functionality.
    """
    # Read dose and predicted dose
    dose_array = dosemetrics.read_from_nifti(os.path.join(input_folder, "Dose.nii.gz"))
    predicted_array = dosemetrics.read_from_nifti(
        os.path.join(input_folder, "Predicted_Dose.nii.gz")
    )

    # Use the new dose difference visualization function
    dosemetrics.plot_dose_differences(
        dose_array=dose_array,
        predicted_array=predicted_array,
        output_file=output_file,
        n_slices=n_slices,
        figsize=(30, 15),
    )

    print(f"Dose difference visualization saved to: {output_file}")


if __name__ == "__main__":
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_folder = os.path.join(repo_root, "data", "test_subject")
    results_folder = os.path.join(repo_root, "results")
    os.makedirs(results_folder, exist_ok=True)

    output_file = os.path.join(results_folder, "visualize_dose_differences.pdf")
    visualize_dose_differences(data_folder, output_file)
