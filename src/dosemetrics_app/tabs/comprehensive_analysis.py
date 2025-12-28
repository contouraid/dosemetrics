"""
Comprehensive dosimetric analysis tab with DVH, statistics, conformity, and homogeneity metrics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import BytesIO

from dosemetrics_app.utils import (
    read_byte_data,
    get_example_datasets,
    load_example_files,
    dvh_by_structure,
)
from dosemetrics import Dose, StructureSet
from dosemetrics.metrics import dvh, conformity, homogeneity


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
                        dose_file = dose_path
                        mask_files = mask_paths

                        st.success(
                            f"Loaded example data: {len(mask_paths)} structures found"
                        )

    return dose_file, mask_files


def plot_structure_slice(
    main_struct, other_structures, slice_idx=None, axis="axial", dose=None
):
    """Plot a slice of a structure with other structure overlays and optional dose"""
    if slice_idx is None:
        slice_idx = (
            main_struct.shape[0] // 2 if axis == "axial" else main_struct.shape[1] // 2
        )

    # Get structure slice
    if axis == "axial":
        main_slice = main_struct.mask[slice_idx, :, :]
    elif axis == "coronal":
        main_slice = main_struct.mask[:, slice_idx, :]
    else:  # sagittal
        main_slice = main_struct.mask[:, :, slice_idx]

    # Create figure
    fig = go.Figure()

    # Add main structure as heatmap
    fig.add_trace(
        go.Heatmap(
            z=main_slice.astype(float),
            colorscale="Viridis",
            name=main_struct.name,
            colorbar=dict(title=main_struct.name),
        )
    )

    # Optionally add dose as contours
    if dose is not None:
        if axis == "axial":
            dose_slice = dose.dose_array[slice_idx, :, :]
        elif axis == "coronal":
            dose_slice = dose.dose_array[:, slice_idx, :]
        else:
            dose_slice = dose.dose_array[:, :, slice_idx]

        fig.add_trace(
            go.Contour(
                z=dose_slice,
                showscale=True,
                contours=dict(coloring="lines"),
                line=dict(width=1),
                name="Dose",
                colorbar=dict(title="Dose (Gy)", x=1.1),
            )
        )

    # Add other structure contours
    colors = ["cyan", "yellow", "magenta", "red", "blue", "orange", "white"]
    for idx, (name, struct) in enumerate(other_structures.items()):
        if axis == "axial":
            mask_slice = struct.mask[slice_idx, :, :]
        elif axis == "coronal":
            mask_slice = struct.mask[:, slice_idx, :]
        else:
            mask_slice = struct.mask[:, :, slice_idx]

        if mask_slice.sum() > 0:
            fig.add_trace(
                go.Contour(
                    z=mask_slice.astype(float),
                    showscale=False,
                    contours=dict(start=0.5, end=0.5, size=1),
                    line=dict(color=colors[idx % len(colors)], width=2),
                    name=name,
                    hoverinfo="name",
                )
            )

    fig.update_layout(
        title=f"{main_struct.name} - {axis.capitalize()} view (slice {slice_idx})",
        height=600,
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1),
    )

    return fig


def plot_dose_only(dose, slice_idx=None, axis="axial"):
    """Plot a slice of the dose distribution without structure overlays"""
    if slice_idx is None:
        if axis == "axial":
            slice_idx = dose.shape[0] // 2
        elif axis == "coronal":
            slice_idx = dose.shape[1] // 2
        else:
            slice_idx = dose.shape[2] // 2

    # Get dose slice
    if axis == "axial":
        dose_slice = dose.dose_array[slice_idx, :, :]
    elif axis == "coronal":
        dose_slice = dose.dose_array[:, slice_idx, :]
    else:  # sagittal
        dose_slice = dose.dose_array[:, :, slice_idx]

    # Create figure
    fig = go.Figure()

    # Add dose as heatmap
    fig.add_trace(
        go.Heatmap(
            z=dose_slice,
            colorscale="Hot",
            name="Dose",
            colorbar=dict(title="Dose (Gy)"),
        )
    )

    fig.update_layout(
        title=f"Dose Distribution - {axis.capitalize()} view (slice {slice_idx})",
        height=600,
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1),
    )

    return fig


def plot_dose_slice(dose, structures, slice_idx=None, axis="axial"):
    """Plot a slice of the dose distribution with structure overlays"""
    if slice_idx is None:
        if axis == "axial":
            slice_idx = dose.shape[0] // 2
        elif axis == "coronal":
            slice_idx = dose.shape[1] // 2
        else:
            slice_idx = dose.shape[2] // 2

    # Get dose slice
    if axis == "axial":
        dose_slice = dose.dose_array[slice_idx, :, :]
    elif axis == "coronal":
        dose_slice = dose.dose_array[:, slice_idx, :]
    else:  # sagittal
        dose_slice = dose.dose_array[:, :, slice_idx]

    # Create figure
    fig = go.Figure()

    # Add dose as heatmap
    fig.add_trace(
        go.Heatmap(
            z=dose_slice,
            colorscale="Hot",
            name="Dose",
            colorbar=dict(title="Dose (Gy)"),
        )
    )

    # Add structure contours
    colors = ["cyan", "green", "yellow", "magenta", "red", "blue", "orange"]
    for idx, (name, struct) in enumerate(structures.items()):
        if axis == "axial":
            mask_slice = struct.mask[slice_idx, :, :]
        elif axis == "coronal":
            mask_slice = struct.mask[:, slice_idx, :]
        else:
            mask_slice = struct.mask[:, :, slice_idx]

        # Find contours
        if mask_slice.sum() > 0:
            fig.add_trace(
                go.Contour(
                    z=mask_slice.astype(float),
                    showscale=False,
                    contours=dict(start=0.5, end=0.5, size=1),
                    line=dict(color=colors[idx % len(colors)], width=2),
                    name=name,
                    hoverinfo="name",
                )
            )

    fig.update_layout(
        title=f"Dose Distribution ({axis.capitalize()} view, slice {slice_idx})",
        xaxis_title="X",
        yaxis_title="Y",
        height=500,
    )

    return fig


def panel():
    """Main panel function for Comprehensive Analysis tab"""
    st.sidebar.success("Select an option above.")

    instruction_text = "## Step 1: Upload dose distribution volume and structure masks"
    dose_file, mask_files = request_dose_and_masks(instruction_text)
    files_uploaded = (dose_file is not None) and (
        mask_files is not None and len(mask_files) > 0
    )

    if files_uploaded:
        with st.spinner("Loading and analyzing data..."):
            try:
                # Load data
                dose, structure_masks = read_byte_data(dose_file, mask_files)

                # Validate compatibility between dose and structures
                incompatible_structures = []
                compatible_structures = {}
                for name, struct in structure_masks.items():
                    if dose.is_compatible_with_structure(struct):
                        compatible_structures[name] = struct
                    else:
                        incompatible_structures.append(
                            f"{name}: shape={struct.mask.shape}, spacing={struct.spacing}"
                        )

                structure_masks = compatible_structures

                if incompatible_structures:
                    st.warning(
                        f"The following structures are incompatible with the dose distribution "
                        f"(dose shape={dose.shape}, spacing={dose.spacing}) and will be skipped:\n"
                        + "\n".join(f"- {s}" for s in incompatible_structures)
                    )

                if len(structure_masks) == 0:
                    st.error(
                        "No compatible structures found. Please check that your dose and structure files have matching dimensions and spacing."
                    )
                    return

                # Create structure set and add structures directly
                structure_set = StructureSet()
                structure_set.spacing = dose.spacing
                structure_set.origin = dose.origin
                for name, struct in structure_masks.items():
                    structure_set.structures[name] = struct

            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
                import traceback

                with st.expander("Show error details"):
                    st.code(traceback.format_exc())
                return

        st.success(f"Loaded {len(structure_masks)} compatible structures")

        # Create tabs for different visualizations and analyses
        viz_tab, dvh_tab, stats_tab, quality_tab = st.tabs(
            ["Dose Visualization", "DVH Analysis", "Dose Statistics", "Quality Metrics"]
        )

        with viz_tab:
            st.markdown("### Dose Distribution Visualization")

            # Volume selector
            volume_to_viz = st.radio(
                "Select volume to visualize:",
                ["Dose Distribution"] + list(structure_masks.keys()),
                horizontal=True,
            )

            # Slice selector
            col1, col2 = st.columns(2)
            with col1:
                axis = st.selectbox("View axis:", ["axial", "coronal", "sagittal"])
            with col2:
                # Dynamically calculate max slice based on selected axis
                if axis == "axial":
                    max_slice = dose.shape[0]
                elif axis == "coronal":
                    max_slice = dose.shape[1]
                else:  # sagittal
                    max_slice = dose.shape[2]
                slice_idx = st.slider("Slice:", 0, max_slice - 1, max_slice // 2)

            # Plot selected volume with structure overlays
            if volume_to_viz == "Dose Distribution":
                # Plot dose only, no structure overlays
                fig = plot_dose_only(dose, slice_idx, axis)
            else:
                # Show selected structure as main volume with other structures as overlays
                selected_struct = structure_masks[volume_to_viz]
                other_structures = {
                    k: v for k, v in structure_masks.items() if k != volume_to_viz
                }
                fig = plot_structure_slice(
                    selected_struct, other_structures, slice_idx, axis, dose
                )
            st.plotly_chart(fig, use_container_width=True)

        with dvh_tab:
            st.markdown("### Dose-Volume Histogram")

            # Compute DVH
            dvh_df = dvh_by_structure(dose, structure_masks)

            # Plot DVH
            fig = px.line(
                dvh_df,
                x="Dose",
                y="Volume",
                color="Structure",
                labels={"Dose": "Dose (Gy)", "Volume": "Volume (%)"},
            )
            fig.update_xaxes(showgrid=True)
            fig.update_yaxes(showgrid=True)
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

            # Download DVH data
            csv = dvh_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download DVH data as CSV",
                data=csv,
                file_name="dvh_data.csv",
                mime="text/csv",
            )

        with stats_tab:
            st.markdown("### Dose Statistics")

            # Compute statistics for all structures
            results = []
            for struct in structure_set.structures.values():
                stats = {
                    "Structure": struct.name,
                    "Volume (cc)": struct.volume_cc(),
                    "Mean Dose (Gy)": dvh.compute_mean_dose(dose, struct),
                    "Max Dose (Gy)": dvh.compute_max_dose(dose, struct),
                    "Min Dose (Gy)": dvh.compute_min_dose(dose, struct),
                    "Std Dose (Gy)": dvh.compute_dose_statistics(dose, struct).get(
                        "std_dose", 0
                    ),
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

            # Display statistics table
            st.dataframe(stats_df, use_container_width=True)

            # Download statistics
            csv = stats_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download statistics as CSV",
                data=csv,
                file_name="dose_statistics.csv",
                mime="text/csv",
            )

        with quality_tab:
            st.markdown("### Plan Quality Metrics")

            # Find target structures (PTVs, CTVs, GTVs)
            target_structures = {
                name: struct
                for name, struct in structure_masks.items()
                if any(
                    keyword in name.upper()
                    for keyword in ["PTV", "CTV", "GTV", "TARGET"]
                )
            }

            if target_structures:
                selected_target = st.selectbox(
                    "Select target structure:", options=list(target_structures.keys())
                )

                prescription_dose = st.number_input(
                    "Prescription dose (Gy):",
                    min_value=0.0,
                    max_value=100.0,
                    value=60.0,
                    step=1.0,
                )

                if selected_target:
                    target = target_structures[selected_target]

                    # Compute metrics
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("#### Conformity Metrics")

                        ci = conformity.compute_conformity_index(
                            dose, target, prescription_dose
                        )
                        cn = conformity.compute_conformity_number(
                            dose, target, prescription_dose
                        )
                        paddick_ci = conformity.compute_paddick_conformity_index(
                            dose, target, prescription_dose
                        )
                        coverage = conformity.compute_coverage(
                            dose, target, prescription_dose
                        )
                        spillage = conformity.compute_spillage(
                            dose, target, prescription_dose
                        )

                        conformity_df = pd.DataFrame(
                            {
                                "Metric": [
                                    "Conformity Index (CI)",
                                    "Conformity Number (CN)",
                                    "Paddick CI",
                                    "Coverage",
                                    "Spillage",
                                ],
                                "Value": [ci, cn, paddick_ci, coverage, spillage],
                                "Ideal": [1.0, 1.0, 1.0, 1.0, 0.0],
                            }
                        )
                        st.dataframe(conformity_df, use_container_width=True)

                        # Gauge chart for CI
                        fig = go.Figure(
                            go.Indicator(
                                mode="gauge+number",
                                value=ci,
                                domain={"x": [0, 1], "y": [0, 1]},
                                title={"text": "Conformity Index"},
                                gauge={
                                    "axis": {"range": [0, 1]},
                                    "bar": {"color": "darkblue"},
                                    "steps": [
                                        {"range": [0, 0.6], "color": "lightgray"},
                                        {"range": [0.6, 0.8], "color": "gray"},
                                        {"range": [0.8, 1.0], "color": "lightgreen"},
                                    ],
                                    "threshold": {
                                        "line": {"color": "red", "width": 4},
                                        "thickness": 0.75,
                                        "value": 1.0,
                                    },
                                },
                            )
                        )
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)

                    with col2:
                        st.markdown("#### Homogeneity Metric")

                        hi = homogeneity.compute_homogeneity_index(
                            dose, target, prescription_dose
                        )

                        # Also compute dose statistics for context
                        max_dose = dvh.compute_max_dose(dose, target)
                        min_dose = dvh.compute_min_dose(dose, target)
                        mean_dose = dvh.compute_mean_dose(dose, target)

                        homogeneity_df = pd.DataFrame(
                            {
                                "Parameter": [
                                    "Homogeneity Index",
                                    "Max Dose",
                                    "Min Dose",
                                    "Mean Dose",
                                ],
                                "Value": [hi, max_dose, min_dose, mean_dose],
                                "Unit": ["", "Gy", "Gy", "Gy"],
                            }
                        )
                        st.dataframe(homogeneity_df, use_container_width=True)

                        # Gauge chart for HI
                        fig = go.Figure(
                            go.Indicator(
                                mode="gauge+number",
                                value=hi,
                                domain={"x": [0, 1], "y": [0, 1]},
                                title={"text": "Homogeneity Index"},
                                gauge={
                                    "axis": {"range": [0, 0.5]},
                                    "bar": {"color": "darkblue"},
                                    "steps": [
                                        {"range": [0, 0.15], "color": "lightgreen"},
                                        {"range": [0.15, 0.25], "color": "gray"},
                                        {"range": [0.25, 0.5], "color": "lightgray"},
                                    ],
                                    "threshold": {
                                        "line": {"color": "green", "width": 4},
                                        "thickness": 0.75,
                                        "value": 0.15,
                                    },
                                },
                            )
                        )
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)

                        st.markdown(
                            """
                        **Interpretation:**
                        - HI < 0.15: Excellent homogeneity
                        - HI 0.15-0.25: Acceptable homogeneity
                        - HI > 0.25: Poor homogeneity
                        """
                        )
            else:
                st.warning(
                    "No target structures (PTV, CTV, GTV) found. Please upload target structures to compute quality metrics."
                )
