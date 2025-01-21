import os

import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("dark_background")
from matplotlib.backends.backend_pdf import PdfPages

import dosemetrics.data_utils as data_utils


def compute_surface(volume):
    """
    Compute the surface of a binary volume
    :param volume: 3D numpy array
    :return: 3D numpy array
    """
    surface = np.zeros_like(volume)
    for i in range(1, volume.shape[0] - 1):
        for j in range(1, volume.shape[1] - 1):
            for k in range(1, volume.shape[2] - 1):
                if volume[i, j, k] > 0:
                    if np.sum(volume[i - 1:i + 2, j - 1:j + 2, k - 1:k + 2]) < 27:
                        surface[i, j, k] = 1
    return surface


def compute_dose_gradient_on_surface(data_folder, output_folder_):

    contents_files = [f for f in os.listdir(data_folder) if f.endswith(".csv")]
    if len(contents_files) > 0:
        dose_file = os.path.join(data_folder, "Dose.nii.gz")
        dose_volume = data_utils.read_from_nifti(dose_file)
        dose_image = sitk.ReadImage(dose_file)

        dose_gradients = np.gradient(dose_volume)
        gradient_magnitude = np.sqrt(np.sum(np.square(dose_gradients), axis=0))

        for target in ["GTV", "CTV", "PTV"]:
            target_file = os.path.join(data_folder, target + ".nii.gz")
            target_volume = data_utils.read_from_nifti(target_file)

            target_surface = compute_surface(target_volume)
            target_gradient = gradient_magnitude * target_surface
            target_gradient_image = sitk.GetImageFromArray(target_gradient)
            target_gradient_image.CopyInformation(dose_image)
            sitk.WriteImage(target_gradient_image, os.path.join(output_folder_, target + "_gradient.nii.gz"))


if __name__ == "__main__":

    dataset_root = "/mnt/5b9b7229-4179-4263-babd-004c30510079/data/ICR-test/ICR-output"
    output_folder = os.path.join(dataset_root, "..", "gradient-data")
    os.makedirs(output_folder, exist_ok=True)

    subfolders = [f.path for f in os.scandir(dataset_root) if f.is_dir()]
    subfolders = sorted(subfolders)

    for subject_folder in subfolders[:10]:
        subject_name = subject_folder.split("/")[-1]
        gradient_folder = os.path.join(output_folder, subject_name)
        os.makedirs(gradient_folder, exist_ok=True)
        compute_dose_gradient_on_surface(subject_folder, gradient_folder)
