import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO

from dosemetrics_app.utils import (
    read_byte_data,
    get_example_datasets,
    load_example_files,
    dvh_by_structure,
)


def request_dose_and_masks(instruction_text):
    """Helper function to request dose and mask file uploads or example selection"""
    st.markdown(instruction_text)
    st.markdown(f"Check instructions on the sidebar for more information.")

    # Add option to use example data
    data_source = st.radio(
        "Data source:", ["Upload your own files", "Use example data"], horizontal=True
    )

    dose_file = None
    mask_files = None

    if data_source == "Upload your own files":
        dose_file = st.file_uploader(
            "Upload a dose distribution volume (in .nii.gz)", type=["gz"]
        )
        mask_files = st.file_uploader(
            "Upload mask volumes (in .nii.gz)", accept_multiple_files=True, type=["gz"]
        )
    else:
        # Load example data
        example_datasets = get_example_datasets()
        if example_datasets:
            # Get list of dataset names with test_subject first
            dataset_names = list(example_datasets.keys())
            default_index = (
                dataset_names.index("test_subject")
                if "test_subject" in dataset_names
                else 0
            )

            selected_dataset = st.selectbox(
                "Select example dataset:", options=dataset_names, index=default_index
            )

            if selected_dataset:
                dataset_path = example_datasets[selected_dataset]
                with st.spinner("Loading example data..."):
                    dose_path, mask_paths = load_example_files(dataset_path)

                    if dose_path:
                        # Read files and create BytesIO objects for compatibility
                        with open(dose_path, "rb") as f:
                            dose_bytes = BytesIO(f.read())
                            dose_bytes.name = dose_path.name
                            dose_file = dose_bytes

                        mask_files = []
                        for mask_path in mask_paths:
                            with open(mask_path, "rb") as f:
                                mask_bytes = BytesIO(f.read())
                                mask_bytes.name = mask_path.name
                                mask_files.append(mask_bytes)

                        st.success(
                            f"Loaded {len(mask_files)} structures from {selected_dataset}"
                        )
        else:
            st.warning("Example data not available. Please upload your own files.")
            data_source = "Upload your own files"

    return dose_file, mask_files


def panel():
    """Main panel function for Calculate DVH tab"""
    st.sidebar.success("Select an option above.")

    instruction_text = f"## Step 1: Upload dose distribution volume and mask files"
    dose_file, mask_files = request_dose_and_masks(instruction_text)
    files_uploaded = (dose_file is not None) and (
        mask_files is not None and len(mask_files) > 0
    )

    if files_uploaded:
        st.divider()
        st.markdown(f"## Step 2: Visualize DVH")
        dose_volume, structure_masks = read_byte_data(dose_file, mask_files)
        df = dvh_by_structure(dose_volume, structure_masks)
        fig = px.line(df, x="Dose", y="Volume", color="Structure")
        fig.update_xaxes(showgrid=True)
        fig.update_yaxes(showgrid=True)
        st.plotly_chart(fig, use_container_width=True)
