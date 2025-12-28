"""
Gamma analysis tab for the Streamlit app.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from io import BytesIO

from dosemetrics_app.utils import read_byte_data
from dosemetrics import Dose
from dosemetrics.metrics import gamma as gamma_module
from dosemetrics_app.utils import get_example_datasets, load_example_files


def request_two_dose_files(instruction_text):
    """Helper function to request two dose files for gamma analysis"""
    st.markdown(instruction_text)
    st.markdown("Check instructions on the sidebar for more information.")

    # Add option to use example data
    data_source = st.radio(
        "Data source:", ["Upload your own files", "Use example data"], horizontal=True
    )

    dose_file1 = None
    dose_file2 = None

    if data_source == "Upload your own files":
        dose_file1 = st.file_uploader(
            "Upload reference dose distribution (in .nii.gz)", type=["gz"], key="dose1"
        )
        dose_file2 = st.file_uploader(
            "Upload evaluated dose distribution (in .nii.gz)", type=["gz"], key="dose2"
        )
    else:
        # Load example data
        example_datasets = get_example_datasets()
        if example_datasets and len(example_datasets) >= 2:
            dataset_names = list(example_datasets.keys())

            col1, col2 = st.columns(2)

            with col1:
                selected_dataset1 = st.selectbox(
                    "Select reference dataset:",
                    options=dataset_names,
                    index=0,
                    key="ref_dataset",
                )

            with col2:
                selected_dataset2 = st.selectbox(
                    "Select evaluated dataset:",
                    options=dataset_names,
                    index=min(1, len(dataset_names) - 1),
                    key="eval_dataset",
                )

            if selected_dataset1 and selected_dataset2:
                with st.spinner("Loading example data..."):
                    # Load first dose
                    dataset_path1 = example_datasets[selected_dataset1]
                    dose_path1, _ = load_example_files(dataset_path1)

                    if dose_path1:
                        with open(dose_path1, "rb") as f:
                            dose_bytes1 = BytesIO(f.read())
                            dose_bytes1.name = dose_path1.name
                            dose_file1 = dose_bytes1

                    # Load second dose
                    dataset_path2 = example_datasets[selected_dataset2]
                    dose_path2, _ = load_example_files(dataset_path2)

                    if dose_path2:
                        with open(dose_path2, "rb") as f:
                            dose_bytes2 = BytesIO(f.read())
                            dose_bytes2.name = dose_path2.name
                            dose_file2 = dose_bytes2

                    if dose_file1 and dose_file2:
                        st.success(
                            f"Loaded dose from {selected_dataset1} (reference) and {selected_dataset2} (evaluated)"
                        )
        else:
            st.warning(
                "Not enough example datasets available. Please upload your own files."
            )
            data_source = "Upload your own files"

    return dose_file1, dose_file2


def panel():
    """Main panel function for Gamma Analysis tab"""
    st.sidebar.success("Select an option above.")

    instruction_text = "## Step 1: Upload reference and evaluated dose distributions"
    dose_file1, dose_file2 = request_two_dose_files(instruction_text)
    files_uploaded = (dose_file1 is not None) and (dose_file2 is not None)

    if files_uploaded:
        st.divider()
        st.markdown("## Step 2: Configure gamma criteria")

        col1, col2, col3 = st.columns(3)

        with col1:
            dose_criteria = st.number_input(
                "Dose difference criterion (%):",
                min_value=0.1,
                max_value=10.0,
                value=3.0,
                step=0.1,
                help="Dose difference tolerance in percent",
            )

        with col2:
            distance_criteria = st.number_input(
                "Distance-to-agreement criterion (mm):",
                min_value=0.1,
                max_value=10.0,
                value=3.0,
                step=0.1,
                help="Distance-to-agreement tolerance in millimeters",
            )

        with col3:
            threshold = st.number_input(
                "Low dose threshold (%):",
                min_value=0.0,
                max_value=50.0,
                value=10.0,
                step=1.0,
                help="Doses below this percentage of maximum are excluded",
            )

        st.divider()
        st.markdown("## Step 3: Compute gamma analysis")

        if st.button("Compute Gamma Analysis"):
            with st.spinner(
                "Loading data and computing gamma analysis (this may take a while)..."
            ):
                # Load dose distributions (read_byte_data returns Dose objects)
                reference, _ = read_byte_data(dose_file1, [])
                evaluated, _ = read_byte_data(dose_file2, [])

                # Compute gamma analysis
                gamma_map = gamma_module.compute_gamma_index(
                    reference,
                    evaluated,
                    dose_criterion_percent=dose_criteria,
                    distance_criterion_mm=distance_criteria,
                    dose_threshold_percent=threshold,
                )

                # Compute statistics
                gamma_stats = gamma_module.compute_gamma_statistics(gamma_map)

            st.success("Gamma analysis computed successfully")

            # Display results
            st.markdown("### Results")

            results_df = pd.DataFrame(
                {
                    "Metric": [
                        "Gamma Criteria",
                        "Low Dose Threshold",
                        "Passing Rate (%)",
                        "Mean Gamma",
                        "Maximum Gamma",
                        "Points Evaluated",
                    ],
                    "Value": [
                        f"{dose_criteria}%/{distance_criteria}mm",
                        f"{threshold}%",
                        f"{gamma_stats['passing_rate']:.2f}",
                        f"{gamma_stats['mean_gamma']:.3f}",
                        f"{gamma_stats['max_gamma']:.3f}",
                        f"{gamma_stats.get('n_points', 'N/A')}",
                    ],
                }
            )

            st.dataframe(results_df, use_container_width=True)

            # Visualizations
            st.divider()
            st.markdown("### Visualizations")

            # Passing rate gauge
            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=gamma_stats["passing_rate"],
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "Gamma Passing Rate (%)"},
                    delta={"reference": 95.0},
                    gauge={
                        "axis": {"range": [None, 100]},
                        "bar": {"color": "darkblue"},
                        "steps": [
                            {"range": [0, 80], "color": "red"},
                            {"range": [80, 90], "color": "yellow"},
                            {"range": [90, 100], "color": "lightgreen"},
                        ],
                        "threshold": {
                            "line": {"color": "red", "width": 4},
                            "thickness": 0.75,
                            "value": 95.0,
                        },
                    },
                )
            )

            fig_gauge.update_layout(height=400)
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Gamma histogram
            gamma_flat = gamma_map[~np.isnan(gamma_map)].flatten()
            if len(gamma_flat) > 0:
                fig_hist = go.Figure()
                fig_hist.add_trace(
                    go.Histogram(
                        x=gamma_flat,
                        nbinsx=50,
                        name="Gamma values",
                        marker_color="#1f77b4",
                    )
                )

                fig_hist.add_vline(
                    x=1.0,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Gamma = 1.0",
                )

                fig_hist.update_layout(
                    title="Gamma Index Distribution",
                    xaxis_title="Gamma Index",
                    yaxis_title="Frequency",
                    showlegend=True,
                )

                st.plotly_chart(fig_hist, use_container_width=True)

            # Gamma map slice visualization
            st.markdown("### Gamma Map Visualization")

            slice_idx = st.slider(
                "Select slice:", 0, gamma_map.shape[2] - 1, gamma_map.shape[2] // 2
            )

            fig_slice = go.Figure()
            fig_slice.add_trace(
                go.Heatmap(
                    z=np.rot90(gamma_map[:, :, slice_idx], k=3),
                    colorscale="RdYlGn_r",
                    zmin=0,
                    zmax=2,
                    colorbar=dict(title="Gamma"),
                )
            )

            fig_slice.update_layout(
                title=f"Gamma Map - Slice {slice_idx}",
                xaxis_title="",
                yaxis_title="",
                xaxis=dict(showticklabels=False),
                yaxis=dict(showticklabels=False),
            )

            st.plotly_chart(fig_slice, use_container_width=True)

            # Download results
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download results as CSV",
                data=csv,
                file_name="gamma_analysis.csv",
                mime="text/csv",
            )

            # Explanation
            st.markdown(
                f"""
            ### Gamma Analysis Summary
            
            Gamma analysis with **{dose_criteria}%/{distance_criteria}mm** criteria:
            
            - **Passing Rate**: {gamma_stats['passing_rate']:.2f}% of points have gamma ≤ 1.0
            - **Mean Gamma**: {gamma_stats['mean_gamma']:.3f}
            - **Maximum Gamma**: {gamma_stats['max_gamma']:.3f}
            
            ### Interpretation
            
            The gamma index combines dose difference and distance-to-agreement into a single metric:
            - Gamma ≤ 1.0: Point passes (dose agreement within criteria)
            - Gamma > 1.0: Point fails (dose disagreement exceeds criteria)
            
            A passing rate of ≥ 95% is typically considered acceptable for clinical treatment verification.
            
            ### Criteria Used
            
            - **Dose Difference**: {dose_criteria}% (percentage of reference dose)
            - **Distance-to-Agreement**: {distance_criteria}mm (spatial tolerance)
            - **Low Dose Threshold**: {threshold}% (doses below this are excluded)
            """
            )
