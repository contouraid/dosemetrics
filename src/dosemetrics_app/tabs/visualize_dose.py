import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from dosemetrics.data import read_byte_data


def request_dose_and_masks(instruction_text):
    """Helper function to request dose and mask file uploads"""
    st.markdown(instruction_text)
    st.markdown(f"Check instructions on the sidebar for more information.")

    dose_file = st.file_uploader(
        "Upload a dose distribution volume (in .nii.gz)", type=["gz"]
    )
    mask_files = st.file_uploader(
        "Upload mask volumes (in .nii.gz)", accept_multiple_files=True, type=["gz"]
    )

    return dose_file, mask_files


def panel():
    """Main panel function for Visualize Dose tab"""
    st.sidebar.success("Select an option above.")

    instruction_text = f"## Step 1: Upload dose distribution volume and mask files"
    dose_file, mask_files = request_dose_and_masks(instruction_text)
    files_uploaded = (dose_file is not None) and (mask_files is not None and len(mask_files) > 0)

    if files_uploaded:
        st.divider()
        st.markdown(f"## Step 2: Visualize Dose")
        dose_volume, structure_masks = read_byte_data(dose_file, mask_files)
        plt.figure(figsize=(6, 6), dpi=80)
        fig, ax = plt.subplots()
        slice_num = st.slider("Choose an axial slice number:", 1, 128, 64)
        plt.imshow(np.rot90(dose_volume[:, :, slice_num], 3), cmap="hot")
        plt.tick_params(
            axis="x",  # changes apply to the x-axis
            which="both",  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False,
        )
        plt.tick_params(
            axis="y",  # changes apply to the x-axis
            which="both",  # both major and minor ticks are affected
            left=False,  # ticks along the bottom edge are off
            right=False,  # ticks
            labelleft=False,
        )
        plt.title("Dose Volume")
        st.pyplot(fig)
