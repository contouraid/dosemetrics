"""
Dose statistics analysis tab for the Streamlit app.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from io import BytesIO

from dosemetrics_app.utils import read_byte_data
from dosemetrics import Dose, StructureSet
from dosemetrics.metrics import dvh
from dosemetrics_app.utils import get_example_datasets, load_example_files


def request_dose_and_masks(instruction_text):
    """Helper function to request dose and mask file uploads or example selection"""
    st.markdown(instruction_text)
    st.markdown("Check instructions on the sidebar for more information.")

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
    """Main panel function for Dose Statistics tab"""
    st.sidebar.success("Select an option above.")

    instruction_text = "## Step 1: Upload dose distribution volume and mask files"
    dose_file, mask_files = request_dose_and_masks(instruction_text)
    files_uploaded = (dose_file is not None) and (
        mask_files is not None and len(mask_files) > 0
    )

    if files_uploaded:
        st.divider()
        st.markdown("## Step 2: Compute dose statistics")

        with st.spinner("Loading data and computing statistics..."):
            dose_volume, structure_masks = read_byte_data(dose_file, mask_files)

            # Create Dose and StructureSet objects
            dose = Dose(dose_volume)
            structure_set = StructureSet()
            for name, mask in structure_masks.items():
                structure_set.add_structure(name, mask)

            # Compute statistics for all structures
            results = []
            for struct in structure_set.structures.values():
                stats = {
                    "Structure": struct.name,
                    "Volume (cc)": struct.volume_cc,
                    "Mean Dose (Gy)": dvh.compute_mean_dose(dose, struct),
                    "Max Dose (Gy)": dvh.compute_max_dose(dose, struct),
                    "Min Dose (Gy)": dvh.compute_min_dose(dose, struct),
                    "Std Dose (Gy)": dvh.compute_dose_statistics(dose, struct).get("std_dose", 0),
                }

                # Add dose at volume metrics
                for volume_pct in [2, 5, 50, 95, 98]:
                    dose_at_vol = dvh.compute_dose_at_volume(dose, struct, volume_pct)
                    stats[f"D{volume_pct}% (Gy)"] = dose_at_vol

                # Add volume at dose metrics (if applicable)
                for dose_val in [10, 20, 30, 40, 50, 60]:
                    if dose_val <= dose.max_dose:
                        vol_at_dose = dvh.compute_volume_at_dose(dose, struct, dose_val)
                        stats[f"V{dose_val}Gy (%)"] = vol_at_dose

                results.append(stats)

            stats_df = pd.DataFrame(results)

        st.success("Statistics computed successfully")

        # Display statistics table
        st.markdown("### Dose Statistics")
        st.dataframe(stats_df, use_container_width=True)

        # Download button
        csv = stats_df.to_csv(index=False)
        st.download_button(
            label="Download statistics as CSV",
            data=csv,
            file_name="dose_statistics.csv",
            mime="text/csv",
        )

        # Visualizations
        st.divider()
        st.markdown("### Visualizations")

        # Mean dose bar chart
        fig_mean = px.bar(
            stats_df,
            x="Structure",
            y="Mean Dose (Gy)",
            title="Mean Dose by Structure",
            labels={"Mean Dose (Gy)": "Mean Dose (Gy)"},
        )
        st.plotly_chart(fig_mean, use_container_width=True)

        # Max dose bar chart
        fig_max = px.bar(
            stats_df,
            x="Structure",
            y="Max Dose (Gy)",
            title="Maximum Dose by Structure",
            labels={"Max Dose (Gy)": "Maximum Dose (Gy)"},
        )
        st.plotly_chart(fig_max, use_container_width=True)

        # Volume bar chart
        fig_vol = px.bar(
            stats_df,
            x="Structure",
            y="Volume (cc)",
            title="Structure Volumes",
            labels={"Volume (cc)": "Volume (cc)"},
        )
        st.plotly_chart(fig_vol, use_container_width=True)
