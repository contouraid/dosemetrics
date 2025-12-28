"""
Conformity analysis tab for the Streamlit app.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from io import BytesIO

from dosemetrics_app.utils import read_byte_data
from dosemetrics import Dose, StructureSet
from dosemetrics.metrics import conformity
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

                        # Find target file (look for PTV, GTV, CTV, or Target)
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
    """Main panel function for Conformity Analysis tab"""
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
        st.markdown("## Step 3: Compute conformity indices")

        if st.button("Compute Conformity Indices"):
            with st.spinner("Loading data and computing conformity indices..."):
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

                # Compute conformity indices
                ci = conformity.compute_conformity_index(
                    dose, target, prescription_dose
                )
                cn = conformity.compute_conformation_number(
                    dose, target, prescription_dose
                )
                gi = conformity.compute_gradient_index(dose, target, prescription_dose)

            st.success("Conformity indices computed successfully")

            # Display results
            st.markdown("### Results")

            results_df = pd.DataFrame(
                {
                    "Metric": [
                        "Conformity Index (CI)",
                        "Conformation Number (CN)",
                        "Gradient Index (GI)",
                    ],
                    "Value": [ci, cn, gi],
                    "Interpretation": [
                        "Ratio of prescription isodose volume to target volume (optimal: 1.0)",
                        "Product of target coverage and dose selectivity (optimal: 1.0)",
                        "Measure of dose fall-off outside target (lower is better)",
                    ],
                }
            )

            st.dataframe(results_df, use_container_width=True)

            # Visualize results
            st.markdown("### Visualization")

            fig = go.Figure()

            fig.add_trace(
                go.Bar(
                    x=["Conformity Index", "Conformation Number"],
                    y=[ci, cn],
                    text=[f"{ci:.3f}", f"{cn:.3f}"],
                    textposition="auto",
                    marker_color=["#1f77b4", "#ff7f0e"],
                )
            )

            fig.update_layout(
                title="Conformity Metrics",
                yaxis_title="Value",
                yaxis_range=[0, max(1.5, ci * 1.2, cn * 1.2)],
                showlegend=False,
            )

            # Add reference line at 1.0
            fig.add_hline(
                y=1.0,
                line_dash="dash",
                line_color="green",
                annotation_text="Optimal value = 1.0",
            )

            st.plotly_chart(fig, use_container_width=True)

            # Gradient index separately
            fig_gi = go.Figure()
            fig_gi.add_trace(
                go.Bar(
                    x=["Gradient Index"],
                    y=[gi],
                    text=[f"{gi:.3f}"],
                    textposition="auto",
                    marker_color="#d62728",
                )
            )

            fig_gi.update_layout(
                title="Gradient Index (lower is better)",
                yaxis_title="Value",
                showlegend=False,
            )

            st.plotly_chart(fig_gi, use_container_width=True)

            # Download results
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download results as CSV",
                data=csv,
                file_name="conformity_analysis.csv",
                mime="text/csv",
            )

            # Explanation
            st.markdown(
                """
            ### Metric Definitions
            
            - **Conformity Index (CI)**: Ratio of the prescription isodose volume to the target volume. 
              An ideal CI is 1.0, indicating the prescription isodose perfectly conforms to the target.
              
            - **Conformation Number (CN)**: Product of target coverage fraction and dose selectivity. 
              Accounts for both target underdosage and normal tissue overdosage. Optimal value is 1.0.
              
            - **Gradient Index (GI)**: Ratio of the 50% isodose volume to the prescription isodose volume. 
              Measures dose fall-off outside the target. Lower values indicate steeper dose gradients.
            """
            )
