import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf

plt.rcParams["figure.figsize"] = [30, 15]


def visualize_doses(subject_path: str = None, output_file: str = None):
    pp = pdf.PdfPages(output_file)

    target_volume_path = os.path.join(subject_path, "Target.nii.gz")
    target_image = sitk.ReadImage(target_volume_path)
    target_array = sitk.GetArrayFromImage(target_image)

    onr_volume_path = os.path.join(subject_path, "OpticNerve_R.nii.gz")
    onr_image = sitk.ReadImage(onr_volume_path)
    onr_array = sitk.GetArrayFromImage(onr_image)

    # Read Dose
    dose_path = os.path.join(subject_path, "Dose.nii.gz")
    dose_image = sitk.ReadImage(dose_path)
    dose_array = sitk.GetArrayFromImage(dose_image)

    # Get the samples from the image
    selected_idx = np.linspace(0.0, 1.0, num=30)

    for index in selected_idx:
        slice_num = int((dose_array.shape[0] - 1) * index)
        fig, axes = plt.subplots(1, 1)
        axes.imshow(dose_array[slice_num, :, :], cmap="grey", vmin=0, vmax=70)
        axes.imshow(target_array[slice_num, :, :], cmap='viridis', alpha=0.75)  # interpolation='none'
        axes.imshow(onr_array[slice_num, :, :], cmap='viridis', alpha=0.75)  # interpolation='none'
        fig.suptitle(f"Slice: {slice_num}")
        pp.savefig(fig)
        plt.close()
    pp.close()


if __name__ == "__main__":
    repo_root = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(repo_root, "..", "data", "test_subject")
    output_file = os.path.join(repo_root, "..", "results", "visualize_doses.pdf")
    visualize_doses(data_folder, output_file)
