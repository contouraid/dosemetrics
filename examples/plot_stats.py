import os

import SimpleITK as sitk

import tkinter as tk
from tkinter.filedialog import askopenfilename
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

plt.style.use("dark_background")

from dosemetrics import dvh


def plotGraph(X, Y):
    fig = plt.figure()
    ### Plotting arrangements ###
    return fig


def main():
    root = tk.Tk()
    root.withdraw()

    dose_file_name = askopenfilename(
        title="Choose Dose file", filetypes=[("Image files", ".gz .nii")]
    )
    print(f"Dose file name entered is: {dose_file_name}")
    file_path = os.path.dirname(dose_file_name)
    dose_image = sitk.ReadImage(dose_file_name)
    dose_array = sitk.GetArrayFromImage(dose_image)

    print(f"File path entered is: {file_path}")
    pp = PdfPages(os.path.join(file_path, "report.pdf"))

    oar_file_names = askopenfilename(
        initialdir=file_path,
        title="Choose OAR files",
        filetypes=[("Image files", ".gz .nii")],
        multiple=True,
    )
    oar_files = list(oar_file_names)

    for file in oar_files:
        struct_name = file.split("/")[-1].split(".")[0]
        struct_image = sitk.ReadImage(file)
        struct_array = sitk.GetArrayFromImage(struct_image)
        bins, values = dvh.compute_dvh(dose_array, struct_array)
        max = dvh.max_dose(dose_array, struct_array)
        mean = dvh.mean_dose(dose_array, struct_array)
        volume = dvh.volume(struct_array, struct_image.GetSpacing())

        fig = plt.figure()
        plt.plot(bins, values, color="b", label=struct_name)
        plt.legend(loc="best")
        plt.xlabel("Gray")
        plt.ylabel("Percentage of Volume")
        plt.title(
            f"Volume: {volume:4.3f} (cc); Max Dose: {max:2.3f}; Mean Dose: {mean:2.3f}"
        )
        plt.axvline(x=mean, color="y")
        plt.axvline(x=max, color="r")
        plt.grid()
        pp.savefig(fig)

    target_file_name = askopenfilename(
        initialdir=file_path,
        title="Choose Target file",
        filetypes=[("Image files", ".gz .nii")],
    )
    target_name = target_file_name.split("/")[-1].split(".")[0]
    target_image = sitk.ReadImage(target_file_name)
    target_array = sitk.GetArrayFromImage(target_image)
    bins, values = dvh.compute_dvh(dose_array, target_array)
    max = dvh.max_dose(dose_array, target_array)
    mean = dvh.mean_dose(dose_array, target_array)
    volume = dvh.volume(target_array, target_image.GetSpacing())

    fig = plt.figure()
    plt.plot(bins, values, color="r", label=target_name)
    plt.legend(loc="best")
    plt.xlabel("Gray")
    plt.ylabel("Percentage of Volume")
    plt.title(
        f"Volume: {volume:4.3f} (cc); Max Dose: {max:2.3f}; Mean Dose: {mean:2.3f}"
    )
    plt.axvline(x=mean, color="y")
    plt.axvline(x=max, color="r")
    plt.grid()
    pp.savefig(fig)
    pp.close()


if __name__ == "__main__":
    main()
