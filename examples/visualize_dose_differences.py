import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf

plt.rcParams["figure.figsize"] = [30, 15]


def visualize_dose_differences(input_folder: str, output_file: str):
    for subject in [input_folder]:

        pp = pdf.PdfPages(output_file)

        # Read GT
        dose_path = os.path.join(subject, "Dose.nii.gz")
        dose_image = sitk.ReadImage(dose_path)
        dose_array = sitk.GetArrayFromImage(dose_image)

        # Read Pred
        pred_path = os.path.join(subject, "Predicted_Dose.nii.gz")
        pred_image = sitk.ReadImage(pred_path)
        pred_array = sitk.GetArrayFromImage(pred_image)

        diff_array = pred_array - dose_array

        # Get the samples from the image
        selected_idx = np.linspace(0.0, 1.0, num=30)

        for index in selected_idx:
            slice_num = int((dose_array.shape[0] - 1) * index)
            fig, axes = plt.subplots(1, 3)
            im_gt = axes[0].imshow(
                dose_array[slice_num, :, :], cmap="coolwarm", vmin=0, vmax=70
            )
            # axes[0].title.set_text("GT")
            im_pred = axes[1].imshow(
                pred_array[slice_num, :, :], cmap="coolwarm", vmin=0, vmax=70
            )
            # axes[1].title.set_text("Pred")
            diff = axes[2].imshow(
                diff_array[slice_num, :, :], cmap="bwr", vmin=-15, vmax=15
            )

            p0 = axes[0].get_position().get_points().flatten()
            p1 = axes[1].get_position().get_points().flatten()
            p2 = axes[2].get_position().get_points().flatten()

            ax_cbar = fig.add_axes([p0[0], 0.2, p1[2] - p0[0], 0.05])
            plt.colorbar(im_gt, cax=ax_cbar, orientation="horizontal")

            ax_cbar1 = fig.add_axes([p2[0], 0.2, p2[2] - p2[0], 0.05])
            plt.colorbar(diff, cax=ax_cbar1, orientation="horizontal")

            fig.suptitle(f"Slice: {slice_num}")
            pp.savefig(fig)
            plt.close()
        pp.close()


if __name__ == "__main__":
    subject_path = os.path.abspath("..")
    base_gt_path = os.path.join(subject_path, "data", "test_subject")
    results_path = os.path.join(subject_path, "results")
    results_file = os.path.join(results_path, "visualize_dose_differences.pdf")

    visualize_dose_differences(base_gt_path, results_file)
