"""
Homogeneity analysis tab for the Streamlit app.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from io import BytesIO

from dosemetrics_app.utils import read_byte_data
from dosemetrics import Dose, StructureSet
from dosemetrics.metrics import homogeneity, dvh
from dosemetrics_app.utils import get_example_datasets, load_example_files


def request_dose_and_target(instruction_text):
    """Helper function to request dose and target file uploads or example selection"""
    st.markdown(instruction_text)
    st.markdown("Check instructions on the sidebar for more information.")

    # Add option to use example data
    data_source = st.radio(
        "Data source:", ["Upload your own files", "Use example data"], horizontal=True
    )

    dose_file = None
    target_file = None

    if data_source == "Upload your own files":
        dose_file = st.file_uploader(
            "Upload a dose distribution volume (in .nii.gz)", type=["gz"]
        )
        target_file = st.file_uploader(
            "Upload target mask volume (in .nii.gz)", type=["gz"]
        )
    else:
        # Load example data
        example_datasets = get_example_datasets()
        if example_datasets:
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
                        # Read dose file
                        with open(dose_path, "rb") as f:
                            dose_bytes = BytesIO(f.read())
                            dose_bytes.name = dose_path.name
                            dose_file = dose_bytes

                        # Find target file
                        target_path = None
                        for mask_path in mask_paths:
                            if any(
                                t in mask_path.name.upper()
                                for t in ["PTV", "GTV", "CTV", "TARGET"]
                            ):
                                target_path = mask_path
                                break

                        if target_path:
                            with open(target_path, "rb") as f:
                                target_bytes = BytesIO(f.read())
                                target_bytes.name = target_path.name
                                target_file = target_bytes
                            st.success(
                                f"Loaded dose and target ({target_path.name}) from {selected_dataset}"
                            )
                        else:
                            st.warning(
                                "No target structure found in example data. Please upload your own target file."
                            )
        else:
            st.warning("Example data not available. Please upload your own files.")
            data_source = "Upload your own files"

    return dose_file, target_file


def panel():
    """Main panel function for Homogeneity Analysis tab"""
    st.sidebar.success("Select an option above.")

    instruction_text = "## Step 1: Upload dose distribution volume and target mask"
    dose_file, target_file = request_dose_and_target(instruction_text)
    files_uploaded = (dose_file is not None) and (target_file is not None)

    if files_uploaded:
        st.divider()
        st.markdown("## Step 2: Specify prescription dose")

        prescription_dose = st.number_input(
            "Prescription dose (Gy):",
            min_value=0.1,
            max_value=200.0,
            value=60.0,
            step=0.1,
            help="The prescribed dose to the target volume in Gray (Gy)",
        )

        st.divider()
        st.markdown("## Step 3: Compute homogeneity index")

        if st.button("Compute Homogeneity Index"):
            with st.spinner("Loading data and computing homogeneity index..."):
                # Load data
                dose_volume, structure_masks = read_byte_data(dose_file, [target_file])

                # Create Dose object
                dose = Dose(dose_volume)

                # Get target structure
                target_name = list(structure_masks.keys())[0]
                target_mask = structure_masks[target_name]

                structure_set = StructureSet()
                structure_set.add_structure(
                    target_name, target_mask, structure_type="target"
                )
                target = structure_set.structures[target_name]

                # Compute homogeneity index
                hi = homogeneity.compute_homogeneity_index(
                    dose, target, prescription_dose
                )

                # Also compute dose statistics for context
                max_dose = dvh.compute_max_dose(dose, target)
                min_dose = dvh.compute_min_dose(dose, target)
                mean_dose = dvh.compute_mean_dose(dose, target)

            st.success("Homogeneity index computed successfully")

            # Display results
            st.markdown("### Results")

            results_df = pd.DataFrame(
                {
                    "Metric": [
                        "Homogeneity Index (HI)",
                        "Maximum Dose (Gy)",
                        "Minimum Dose (Gy)",
                        "Mean Dose (Gy)",
                        "Prescription Dose (Gy)",
                    ],
                    "Value": [hi, max_dose, min_dose, mean_dose, prescription_dose],
                    "Interpretation": [
                        "Measure of dose uniformity within target (lower is better, < 0.15 is good)",
                        "Highest dose delivered to target",
                        "Lowest dose delivered to target",
                        "Average dose delivered to target",
                        "Prescribed dose level",
                    ],
                }
            )

            st.dataframe(results_df, use_container_width=True)

            # Visualize results
            st.markdown("### Visualization")

            # Homogeneity index gauge
            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=hi,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "Homogeneity Index"},
                    gauge={
                        "axis": {"range": [None, 0.5]},
                        "bar": {"color": "darkblue"},
                        "steps": [
                            {"range": [0, 0.15], "color": "lightgreen"},
                            {"range": [0.15, 0.30], "color": "yellow"},
                            {"range": [0.30, 0.50], "color": "red"},
                        ],
                        "threshold": {
                            "line": {"color": "red", "width": 4},
                            "thickness": 0.75,
                            "value": 0.15,
                        },
                    },
                )
            )

            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

            # Dose comparison bar chart
            fig_dose = go.Figure()
            fig_dose.add_trace(
                go.Bar(
                    x=["Min Dose", "Mean Dose", "Prescription Dose", "Max Dose"],
                    y=[min_dose, mean_dose, prescription_dose, max_dose],
                    text=[
                        f"{min_dose:.2f}",
                        f"{mean_dose:.2f}",
                        f"{prescription_dose:.2f}",
                        f"{max_dose:.2f}",
                    ],
                    textposition="auto",
                    marker_color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
                )
            )

            fig_dose.update_layout(
                title="Dose Statistics in Target Volume",
                yaxis_title="Dose (Gy)",
                showlegend=False,
            )

            st.plotly_chart(fig_dose, use_container_width=True)

            # Download results
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download results as CSV",
                data=csv,
                file_name="homogeneity_analysis.csv",
                mime="text/csv",
            )

            # Explanation
            st.markdown(
                """
            ### Metric Definition
            
            The **Homogeneity Index (HI)** quantifies the uniformity of dose distribution within the target volume.
            It is calculated as:
            
            HI = (D_max - D_min) / D_prescription
            
            Where:
            - D_max is the maximum dose in the target
            - D_min is the minimum dose in the target
            - D_prescription is the prescribed dose
            
            **Interpretation:**
            - HI < 0.15: Excellent homogeneity
            - 0.15 ≤ HI < 0.30: Acceptable homogeneity
            - HI ≥ 0.30: Poor homogeneity
            
            Lower HI values indicate more uniform dose distribution, which is generally desirable for target volumes.
            """
            )
