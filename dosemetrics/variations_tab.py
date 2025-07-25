import streamlit as st
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from dosemetrics import data_utils
from dosemetrics import dvh
from dosemetrics import scores
from dosemetrics import compliance
from dosemetrics import plot


def display_summary(doses, structure_mask):
    df = dvh.dvh_by_structure(doses, structure_mask)
    fig = px.line(df, x="Dose", y="Volume", color="Structure")
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    st.plotly_chart(fig, use_container_width=True)

    summary_df = scores.dose_summary(doses, structure_mask)
    st.table(summary_df)
    return summary_df


def compare_differences(summary_df, selected_structures, ref_id):
    diff_table = pd.DataFrame()
    st.markdown(f"#### Dose differences between Dose: {id} vs Reference: {ref_id}")
    for structure in selected_structures:
        diff_table.loc[:, structure] = (
            summary_df[id].loc[structure, :] - summary_df[ref_id].loc[structure, :]
        )
    st.table(diff_table)


def display_difference_dvh(doses, structure_mask, selected_structures, ref_id):
    for structure in selected_structures:
        st.markdown(f"#### DVH comparisons for {structure}")
        df = dvh.dvh_by_dose(doses, structure_mask[structure], structure)
        fig = px.line(df, x="Dose", y="Volume", color="Structure")
        fig.update_xaxes(showgrid=True)
        fig.update_yaxes(showgrid=True)
        st.plotly_chart(fig, use_container_width=True)


def generate_dvh_family(
    dose_volume, structure_masks, constraints: pd.DataFrame, structure_of_interest: str
):

    structure_mask = structure_masks[structure_of_interest]
    constraint_limit = constraints.loc[structure_of_interest, "Level"]
    fig, dsc_range = plot.dvh_family(
        dose_volume, structure_mask, constraint_limit, structure_of_interest
    )
    st.pyplot(fig, clear_figure=True)
    st.markdown(f"Maximum Dice: {dsc_range[0]}, Minimum Dice: {dsc_range[1]:.2f}")


def panel():
    step_1_complete = False
    step_2_complete = False
    step_3_complete = False

    structure_mask = {}

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "🗃️Upload Data",
            "📊 View Dose Metrics",
            "🔍 Compute Compliance",
            "✅Evaluate Contour",
        ]
    )

    with tab1:
        st.markdown(f"## Step 1: Upload dose distribution volume and mask files")

        st.markdown("Upload the dose volume:")
        dose_file = st.file_uploader(
            f"Upload dose volume: (in .nii.gz)", type=["nii", "gz"], key=0
        )

        st.markdown("Upload the contour masks:")
        mask_files = st.file_uploader(
            "Upload mask volumes (in .nii.gz)",
            accept_multiple_files=True,
            type=["nii", "gz"],
            key=1,
        )

        files_uploaded = (dose_file is not None) and (len(mask_files) > 0)

        if files_uploaded:
            st.markdown(
                f"Both dose and mask files are uploaded. Click the toggle button below to proceed."
            )
            step_1_complete = st.toggle("Compute")

            dose, _ = data_utils.read_dose(dose_file)
            structure_mask = data_utils.read_masks(mask_files)

        st.divider()

    with tab2:
        st.markdown(f"## Step 2: Dose Metrics")
        st.markdown(f"Complete Step 1 to view metrics.")
        if step_1_complete:
            st.markdown(f"Dose Metrics: contours, dose distribution.")
            dose_summary_df = display_summary(dose, structure_mask)
            csv = dose_summary_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"dose_summary_df.csv",
                mime="text/csv",
                key=999,
            )

            st.divider()
            step_2_complete = True

    with tab3:
        st.markdown(f"## Step 3: Display Compliance")
        st.markdown(f"Complete Step 2 to proceed.")
        if step_2_complete:
            st.markdown(f"Clinical Compliance: contours, dose distribution.")
            compliance_results = compliance.compute_mirage_compliance(
                dose, structure_mask
            )
            st.table(compliance_results)
            compliance_csv = compliance_results.to_csv(index=True)
            st.download_button(
                label="Download compliance CSV",
                data=compliance_csv,
                file_name="compliance.csv",
                mime="text/csv",
                key=500,
            )

            st.divider()
            step_3_complete = True

    with tab4:
        st.markdown(f"## Step 4: Check Contour Error Impact")
        st.markdown(f"Complete Step 3 to proceed.")
        if step_3_complete:
            st.markdown(f"Contour quality check: dice versus dose distribution.")
            constraints = compliance.get_custom_constraints()

            option = st.pills(
                "Choose structure:",
                tuple(structure_mask.keys()),
                selection_mode="single",
            )
            st.divider()

            if option is not None:
                generate_dvh_family(dose, structure_mask, constraints, option)
