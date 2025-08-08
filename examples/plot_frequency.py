import os
import dosemetrics


def plot_frequency(input_folder: str, output_folder: str):
    """
    Plot frequency domain analysis using the new plot_frequency_analysis function.

    This example demonstrates the new frequency analysis visualization.
    """
    # Read dose and predicted dose
    dose_array = dosemetrics.read_from_nifti(os.path.join(input_folder, "Dose.nii.gz"))
    predicted_array = dosemetrics.read_from_nifti(
        os.path.join(input_folder, "Predicted_Dose.nii.gz")
    )

    # Use the new frequency analysis function
    output_file = os.path.join(output_folder, "fft.pdf")
    dosemetrics.plot_frequency_analysis(
        dose_arrays=[dose_array, predicted_array],
        output_file=output_file,
        labels=["Ground Truth", "Predicted"],
    )

    print(f"Frequency analysis saved to: {output_file}")


if __name__ == "__main__":
    repo_root = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(repo_root, "..", "data", "test_subject")
    results_folder = os.path.join(repo_root, "..", "results")
    os.makedirs(results_folder, exist_ok=True)
    plot_frequency(data_folder, results_folder)
