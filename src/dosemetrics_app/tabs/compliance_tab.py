"""
Compliance checking tab for the Streamlit app.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from io import BytesIO

from dosemetrics_app.utils import read_byte_data
from dosemetrics import Dose, StructureSet, get_default_constraints, check_compliance
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
    """Main panel function for Compliance Checking tab"""
    st.sidebar.success("Select an option above.")

    instruction_text = "## Step 1: Upload dose distribution volume and mask files"
    dose_file, mask_files = request_dose_and_masks(instruction_text)
    files_uploaded = (dose_file is not None) and (
        mask_files is not None and len(mask_files) > 0
    )

    if files_uploaded:
        st.divider()
        st.markdown("## Step 2: Select constraint set")

        constraint_option = st.radio(
            "Constraint set:",
            ["Use default constraints", "Upload custom constraints"],
            horizontal=True,
        )

        constraints = None
        if constraint_option == "Use default constraints":
            constraints = get_default_constraints()
            st.info(f"Using default constraints for {len(constraints)} structures")
        else:
            constraints_file = st.file_uploader(
                "Upload custom constraints CSV file",
                type=["csv"],
                help="CSV file with columns: Structure (index), Constraint Type, Level",
            )
            if constraints_file:
                constraints = pd.read_csv(constraints_file, index_col=0)
                st.info(f"Loaded custom constraints for {len(constraints)} structures")

        if constraints is not None:
            # Show constraint preview
            with st.expander("View constraints"):
                st.dataframe(constraints, use_container_width=True)

        st.divider()
        st.markdown("## Step 3: Check compliance")

        if st.button("Check Compliance") and constraints is not None:
            with st.spinner("Loading data and checking compliance..."):
                # Load data (read_byte_data returns Dose object and structures)
                dose, structure_masks = read_byte_data(dose_file, mask_files)

                # Create StructureSet and add structures
                structure_set = StructureSet()
                structure_set.spacing = dose.spacing
                structure_set.origin = dose.origin
                for name, struct in structure_masks.items():
                    structure_set.structures[name] = struct

                # Compute statistics for all structures
                stats_data = []
                for struct in structure_set.structures.values():
                    stats_data.append(
                        {
                            "Structure": struct.name,
                            "Mean Dose": dvh.compute_mean_dose(dose, struct),
                            "Max Dose": dvh.compute_max_dose(dose, struct),
                            "Min Dose": dvh.compute_min_dose(dose, struct),
                        }
                    )

                stats_df = pd.DataFrame(stats_data).set_index("Structure")

                # Check compliance
                compliance_df = check_compliance(stats_df, constraints)

            st.success("Compliance checking completed")

            # Display results
            st.markdown("### Compliance Results")

            # Add color coding to compliance column
            def highlight_compliance(row):
                if "No" in str(row["Compliance"]) or "❌" in str(row["Compliance"]):
                    return ["background-color: #ffcccc"] * len(row)
                elif "Yes" in str(row["Compliance"]) or "✅" in str(row["Compliance"]):
                    return ["background-color: #ccffcc"] * len(row)
                else:
                    return [""] * len(row)

            styled_df = compliance_df.style.apply(highlight_compliance, axis=1)
            st.dataframe(styled_df, use_container_width=True)

            # Summary statistics
            st.divider()
            st.markdown("### Summary")

            # Count compliant vs non-compliant
            compliant_count = sum(
                "Yes" in str(c) or "✅" in str(c) for c in compliance_df["Compliance"]
            )
            non_compliant_count = sum(
                "No" in str(c) or "❌" in str(c) for c in compliance_df["Compliance"]
            )
            total_count = len(compliance_df)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Structures", total_count)

            with col2:
                st.metric(
                    "Compliant",
                    compliant_count,
                    delta=f"{100*compliant_count/total_count:.1f}%",
                )

            with col3:
                st.metric(
                    "Non-Compliant",
                    non_compliant_count,
                    delta=f"{100*non_compliant_count/total_count:.1f}%",
                    delta_color="inverse",
                )

            # Visualization
            st.divider()
            st.markdown("### Visualization")

            # Compliance pie chart
            compliance_summary = pd.DataFrame(
                {
                    "Status": ["Compliant", "Non-Compliant"],
                    "Count": [compliant_count, non_compliant_count],
                }
            )

            fig_pie = px.pie(
                compliance_summary,
                values="Count",
                names="Status",
                title="Compliance Status Distribution",
                color="Status",
                color_discrete_map={"Compliant": "green", "Non-Compliant": "red"},
            )
            st.plotly_chart(fig_pie, use_container_width=True)

            # Dose statistics with constraint overlay
            if len(stats_df) > 0:
                st.markdown("### Dose Statistics vs Constraints")

                # Merge stats with constraints for structures that have constraints
                merged_data = []
                for struct_name in stats_df.index:
                    if struct_name in constraints.index:
                        constraint_type = constraints.loc[
                            struct_name, "Constraint Type"
                        ]
                        constraint_level = constraints.loc[struct_name, "Level"]

                        if constraint_type == "mean":
                            actual_dose = stats_df.loc[struct_name, "Mean Dose"]
                        elif constraint_type == "max":
                            actual_dose = stats_df.loc[struct_name, "Max Dose"]
                        elif constraint_type == "min":
                            actual_dose = stats_df.loc[struct_name, "Min Dose"]
                        else:
                            continue

                        merged_data.append(
                            {
                                "Structure": struct_name,
                                "Constraint Type": constraint_type,
                                "Actual Dose": actual_dose,
                                "Constraint Level": constraint_level,
                                "Difference": actual_dose - constraint_level,
                            }
                        )

                if merged_data:
                    merged_df = pd.DataFrame(merged_data)

                    fig_comparison = px.bar(
                        merged_df,
                        x="Structure",
                        y=["Actual Dose", "Constraint Level"],
                        title="Actual Dose vs Constraint Levels",
                        labels={"value": "Dose (Gy)", "variable": "Metric"},
                        barmode="group",
                    )
                    st.plotly_chart(fig_comparison, use_container_width=True)

            # Download results
            csv = compliance_df.to_csv()
            st.download_button(
                label="Download compliance results as CSV",
                data=csv,
                file_name="compliance_results.csv",
                mime="text/csv",
            )

            # Explanation
            st.markdown(
                """
            ### Interpretation
            
            This analysis checks whether dose statistics for each structure meet the specified constraints:
            
            - **Max constraint**: Maximum dose in structure must not exceed the constraint level
            - **Mean constraint**: Mean dose in structure must not exceed the constraint level
            - **Min constraint**: Minimum dose in structure must meet or exceed the constraint level (typically for targets)
            
            Structures marked as compliant meet their respective constraints, while non-compliant structures 
            exceed the constraint thresholds. The reason column provides details on the specific constraint 
            violation or compliance margin.
            
            ### Clinical Relevance
            
            Compliance checking helps ensure treatment plans meet clinical protocol requirements and 
            safety constraints. Non-compliant structures may require plan optimization or protocol 
            deviation documentation.
            """
            )
