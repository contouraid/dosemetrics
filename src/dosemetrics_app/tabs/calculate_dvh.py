import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from dosemetrics.data import read_byte_data
from dosemetrics.metrics import dvh_by_structure


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
    """Main panel function for Calculate DVH tab"""
    st.sidebar.success("Select an option above.")

    instruction_text = f"## Step 1: Upload dose distribution volume and mask files"
    dose_file, mask_files = request_dose_and_masks(instruction_text)
    files_uploaded = (dose_file is not None) and (mask_files is not None and len(mask_files) > 0)

    if files_uploaded:
        st.divider()
        st.markdown(f"## Step 2: Visualize DVH")
        dose_volume, structure_masks = read_byte_data(dose_file, mask_files)
        df = dvh_by_structure(dose_volume, structure_masks)
        fig = px.line(df, x="Dose", y="Volume", color="Structure")
        fig.update_xaxes(showgrid=True)
        fig.update_yaxes(showgrid=True)
        st.plotly_chart(fig, use_container_width=True)
