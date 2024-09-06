import os
import glob
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf


def main():
    repo_root = os.path.abspath("/Users/amithkamath/repo/dosemetrics")
    data_folder = os.path.join(repo_root, "data/test_subject")
    
    dose_image = sitk.ReadImage(data_folder + "/Dose.nii.gz")
    dose_array = sitk.GetArrayFromImage(dose_image)

    prediction_image = sitk.ReadImage(data_folder + "/Predicted_Dose.nii.gz")
    prediction_array = sitk.GetArrayFromImage(prediction_image)

    dose_fs = np.fft.fftn(dose_array)
    prediction_fs = np.fft.fftn(prediction_array)

    pp = pdf.PdfPages(os.path.join(data_folder, "..", "fft_test.pdf"))

    for index in range(128):
        slice_num = index
        fig, axes = plt.subplots(1, 2)
        im_gt = axes[0].imshow(np.abs(np.fft.fftshift(dose_fs[:, :, slice_num]))**2, vmax=1000000, vmin=0)
        im_pred = axes[1].imshow(np.abs(np.fft.fftshift(prediction_fs[:, :, slice_num]))**2, vmax=1000000, vmin=0)
        fig.colorbar(im_gt, ax=axes[0])
        fig.colorbar(im_pred, ax=axes[1])
        fig.suptitle(f"Slice: {slice_num}")
        pp.savefig(fig)
        plt.close()
    pp.close()


if __name__ == "__main__":
    main()
