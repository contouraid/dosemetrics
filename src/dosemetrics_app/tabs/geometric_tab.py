"""
Geometric comparison tab for the Streamlit app.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from io import BytesIO

from dosemetrics_app.utils import read_byte_data
from dosemetrics import StructureSet
from dosemetrics.metrics import geometric
from dosemetrics_app.utils import get_example_datasets, load_example_files
import dosemetrics


def request_two_structure_sets(instruction_text):
    """Helper function to request two structure sets for comparison"""
    st.markdown(instruction_text)
    st.markdown("Check instructions on the sidebar for more information.")

    # Add option to use example data
    data_source = st.radio(
        "Data source:", ["Upload your own files", "Use example data"], horizontal=True
    )

    mask_files1 = None
    mask_files2 = None

    if data_source == "Upload your own files":
        st.markdown("### First Structure Set")
        mask_files1 = st.file_uploader(
            "Upload mask volumes for first set (in .nii.gz)",
            accept_multiple_files=True,
            type=["gz"],
            key="masks1",
        )

        st.markdown("### Second Structure Set")
        mask_files2 = st.file_uploader(
            "Upload mask volumes for second set (in .nii.gz)",
            accept_multiple_files=True,
            type=["gz"],
            key="masks2",
        )
    else:
        # Load example data
        example_datasets = get_example_datasets()
        if example_datasets and len(example_datasets) >= 2:
            dataset_names = list(example_datasets.keys())

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### First Structure Set")
                selected_dataset1 = st.selectbox(
                    "Select first dataset:",
                    options=dataset_names,
                    index=0,
                    key="dataset1",
                )

            with col2:
                st.markdown("### Second Structure Set")
                selected_dataset2 = st.selectbox(
                    "Select second dataset:",
                    options=dataset_names,
                    index=min(1, len(dataset_names) - 1),
                    key="dataset2",
                )

            if selected_dataset1 and selected_dataset2:
                with st.spinner("Loading example data..."):
                    # Load first set
                    dataset_path1 = example_datasets[selected_dataset1]
                    _, mask_paths1 = load_example_files(dataset_path1)

                    mask_files1 = []
                    for mask_path in mask_paths1:
                        with open(mask_path, "rb") as f:
                            mask_bytes = BytesIO(f.read())
                            mask_bytes.name = mask_path.name
                            mask_files1.append(mask_bytes)

                    # Load second set
                    dataset_path2 = example_datasets[selected_dataset2]
                    _, mask_paths2 = load_example_files(dataset_path2)

                    mask_files2 = []
                    for mask_path in mask_paths2:
                        with open(mask_path, "rb") as f:
                            mask_bytes = BytesIO(f.read())
                            mask_bytes.name = mask_path.name
                            mask_files2.append(mask_bytes)

                    st.success(
                        f"Loaded {len(mask_files1)} structures from {selected_dataset1} and {len(mask_files2)} from {selected_dataset2}"
                    )
        else:
            st.warning(
                "Not enough example datasets available. Please upload your own files."
            )
            data_source = "Upload your own files"

    return mask_files1, mask_files2


def panel():
    """Main panel function for Geometric Comparison tab"""
    st.sidebar.success("Select an option above.")

    instruction_text = "## Step 1: Upload two structure sets to compare"
    mask_files1, mask_files2 = request_two_structure_sets(instruction_text)
    files_uploaded = (mask_files1 is not None and len(mask_files1) > 0) and (
        mask_files2 is not None and len(mask_files2) > 0
    )

    if files_uploaded:
        st.divider()
        st.markdown("## Step 2: Compute geometric comparisons")

        if st.button("Compute Geometric Metrics"):
            with st.spinner("Loading data and computing geometric comparisons..."):
                # Load structure sets
                # For structure comparison, we only need structures, not dose
                # Create a dummy dose file to satisfy read_byte_data
                import numpy as np
                import tempfile
                from pathlib import Path

                with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as f:
                    dummy_dose_path = Path(f.name)
                    # Create minimal dose volume
                    dummy_array = np.zeros((10, 10, 10))
                    dosemetrics.nifti_io.write_nifti_volume(
                        dummy_array, str(dummy_dose_path), spacing=(1.0, 1.0, 1.0)
                    )

                try:
                    _, structure_masks1 = read_byte_data(dummy_dose_path, mask_files1)
                    _, structure_masks2 = read_byte_data(dummy_dose_path, mask_files2)
                finally:
                    # Clean up dummy file
                    if dummy_dose_path.exists():
                        dummy_dose_path.unlink()

                # Create StructureSets
                structure_set1 = StructureSet()
                structure_set1.spacing = structure_masks1[
                    list(structure_masks1.keys())[0]
                ].spacing
                structure_set1.origin = structure_masks1[
                    list(structure_masks1.keys())[0]
                ].origin
                for name, struct in structure_masks1.items():
                    structure_set1.structures[name] = struct

                structure_set2 = StructureSet()
                structure_set2.spacing = structure_masks2[
                    list(structure_masks2.keys())[0]
                ].spacing
                structure_set2.origin = structure_masks2[
                    list(structure_masks2.keys())[0]
                ].origin
                for name, struct in structure_masks2.items():
                    structure_set2.structures[name] = struct

                # Find common structures
                common_names = set(structure_set1.structures.keys()) & set(
                    structure_set2.structures.keys()
                )

                if not common_names:
                    st.error(
                        "No common structures found between the two sets. Ensure structure names match."
                    )
                    return

                st.info(
                    f"Found {len(common_names)} common structure(s): {', '.join(sorted(common_names))}"
                )

                # Compute geometric comparisons
                results = []
                for name in sorted(common_names):
                    struct1 = structure_set1.structures[name]
                    struct2 = structure_set2.structures[name]

                    result = {
                        "Structure": name,
                        "Dice": geometric.compute_dice_coefficient(struct1, struct2),
                        "Jaccard": geometric.compute_jaccard_index(struct1, struct2),
                        "Volume Difference (cc)": geometric.compute_volume_difference(
                            struct1, struct2
                        ),
                        "Volume Ratio": geometric.compute_volume_ratio(
                            struct1, struct2
                        ),
                        "Sensitivity": geometric.compute_sensitivity(struct1, struct2),
                        "Specificity": geometric.compute_specificity(struct1, struct2),
                    }

                    # Hausdorff distance (may be slow for large structures)
                    try:
                        result["Hausdorff Distance (mm)"] = (
                            geometric.compute_hausdorff_distance(struct1, struct2)
                        )
                        result["Mean Surface Distance (mm)"] = (
                            geometric.compute_mean_surface_distance(struct1, struct2)
                        )
                    except Exception as e:
                        st.warning(
                            f"Could not compute surface distances for {name}: {str(e)}"
                        )
                        result["Hausdorff Distance (mm)"] = None
                        result["Mean Surface Distance (mm)"] = None

                    results.append(result)

                results_df = pd.DataFrame(results)

            st.success("Geometric comparisons computed successfully")

            # Display results table
            st.markdown("### Results")
            st.dataframe(results_df, use_container_width=True)

            # Download button
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download results as CSV",
                data=csv,
                file_name="geometric_comparison.csv",
                mime="text/csv",
            )

            # Visualizations
            st.divider()
            st.markdown("### Visualizations")

            # Dice coefficient bar chart
            fig_dice = px.bar(
                results_df,
                x="Structure",
                y="Dice",
                title="Dice Coefficient by Structure",
                labels={"Dice": "Dice Coefficient"},
                color="Dice",
                color_continuous_scale="RdYlGn",
                range_color=[0, 1],
            )
            fig_dice.add_hline(
                y=0.7,
                line_dash="dash",
                line_color="red",
                annotation_text="Good agreement threshold (0.7)",
            )
            st.plotly_chart(fig_dice, use_container_width=True)

            # Jaccard index bar chart
            fig_jaccard = px.bar(
                results_df,
                x="Structure",
                y="Jaccard",
                title="Jaccard Index by Structure",
                labels={"Jaccard": "Jaccard Index"},
                color="Jaccard",
                color_continuous_scale="RdYlGn",
                range_color=[0, 1],
            )
            st.plotly_chart(fig_jaccard, use_container_width=True)

            # Volume comparison
            if "Volume Difference (cc)" in results_df.columns:
                fig_vol = px.bar(
                    results_df,
                    x="Structure",
                    y="Volume Difference (cc)",
                    title="Volume Difference by Structure",
                    labels={"Volume Difference (cc)": "Volume Difference (cc)"},
                )
                st.plotly_chart(fig_vol, use_container_width=True)

            # Surface distances (if available)
            if (
                "Hausdorff Distance (mm)" in results_df.columns
                and results_df["Hausdorff Distance (mm)"].notna().any()
            ):
                fig_hd = px.bar(
                    results_df[results_df["Hausdorff Distance (mm)"].notna()],
                    x="Structure",
                    y="Hausdorff Distance (mm)",
                    title="Hausdorff Distance by Structure",
                    labels={"Hausdorff Distance (mm)": "Hausdorff Distance (mm)"},
                )
                st.plotly_chart(fig_hd, use_container_width=True)

            # Explanation
            st.markdown(
                """
            ### Metric Definitions
            
            - **Dice Coefficient**: Measure of overlap between two segmentations (0 = no overlap, 1 = perfect overlap). 
              Values > 0.7 generally indicate good agreement.
              
            - **Jaccard Index**: Alternative overlap metric, more sensitive to size differences than Dice.
              
            - **Volume Difference**: Absolute difference in volume (cc) between the two structures.
              
            - **Volume Ratio**: Ratio of volumes (Set1 / Set2). Values close to 1.0 indicate similar volumes.
              
            - **Sensitivity**: Fraction of Set1 that overlaps with Set2 (measures false negatives).
              
            - **Specificity**: Fraction of Set2 that overlaps with Set1 (measures false positives).
              
            - **Hausdorff Distance**: Maximum distance from a point in one set to the nearest point in the other. 
              Sensitive to outliers.
              
            - **Mean Surface Distance**: Average distance between the surfaces of the two structures.
            """
            )
