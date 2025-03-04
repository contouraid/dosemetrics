import streamlit as st
import plotly.express as px
import pandas as pd

from dosemetrics import data_utils
from dosemetrics import dvh
from dosemetrics import scores
from dosemetrics import compliance


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
        diff_table.loc[:, structure] = summary_df[id].loc[structure, :] - summary_df[ref_id].loc[structure, :]
    st.table(diff_table)


def display_difference_dvh(doses, structure_mask, selected_structures, ref_id):
    for structure in selected_structures:
        st.markdown(f"#### DVH comparisons for {structure}")
        df = dvh.dvh_by_dose(doses, structure_mask[structure], structure)
        fig = px.line(df, x="Dose", y="Volume", color="Structure")
        fig.update_xaxes(showgrid=True)
        fig.update_yaxes(showgrid=True)
        st.plotly_chart(fig, use_container_width=True)


def panel():

    step_1_complete = False
    step_2_complete = False

    tab1, tab2, tab3 = st.tabs(["🗃️Upload Data", "📊 View Dose Metrics", "🔍 Compute Compliance"])

    with (tab1):
        st.markdown(f"## Step 1: Upload dose distribution volume and mask files")

        st.markdown("Upload the 'first' dose volume:")
        first_dose_file = st.file_uploader(f"Upload dose volume: (in .nii.gz)", type=['nii', 'gz'], key=0)

        st.markdown("Upload the 'first' contour masks:")
        first_mask_files = st.file_uploader("Upload mask volumes (in .nii.gz)", accept_multiple_files=True,
                                          type=['nii', 'gz'], key=1)

        st.markdown("Upload the 'last' dose volume:")
        last_dose_file = st.file_uploader(f"Upload dose volume: (in .nii.gz)", type=['nii', 'gz'], key=2)

        st.markdown("Upload the 'last' contour masks:")
        last_mask_files = st.file_uploader("Upload mask volumes (in .nii.gz)", accept_multiple_files=True,
                                          type=['nii', 'gz'], key=3)

        files_uploaded = (first_dose_file is not None) and (len(first_mask_files) > 0) and \
                         (last_dose_file is not None) and (len(last_mask_files) > 0)

        if files_uploaded:
            st.markdown(f"Both dose and mask files are uploaded. Click the toggle button below to proceed.")
            step_1_complete = st.toggle("Compute")

            first_dose, _ = data_utils.read_dose(first_dose_file)
            last_dose, _ = data_utils.read_dose(last_dose_file)

            first_structure_mask = data_utils.read_masks(first_mask_files)
            last_structure_mask = data_utils.read_masks(last_mask_files)

        st.divider()

    with tab2:
        st.markdown(f"## Step 2: Dose Metrics")
        st.markdown(f"Complete Step 1 to view metrics.")
        if step_1_complete:
            st.markdown(f"Dose Metrics: first contours, first dose distribution.")
            a_dose_summary_df = display_summary(first_dose, first_structure_mask)
            a_csv = a_dose_summary_df.to_csv(index=False)
            st.download_button(label="Download CSV", data=a_csv, file_name=f"a_dose_summary_df.csv", mime="text/csv", key=999)

            st.markdown(f"Dose Metrics: first contours, last dose distribution.")
            b_dose_summary_df = display_summary(last_dose, first_structure_mask)
            b_csv = b_dose_summary_df.to_csv(index=False)
            st.download_button(label="Download CSV", data=b_csv, file_name=f"b_dose_summary_df.csv", mime="text/csv", key=998)

            st.markdown(f"Dose Metrics: last contours, last dose distribution.")
            c_dose_summary_df = display_summary(last_dose, last_structure_mask)
            c_csv = c_dose_summary_df.to_csv(index=False)
            st.download_button(label="Download CSV", data=c_csv, file_name=f"c_dose_summary_df.csv", mime="text/csv", key=997)

            st.divider()
            step_2_complete = True

    with tab3:
        st.markdown(f"## Step 3: Display Compliance")
        st.markdown(f"Complete Step 2 to proceed.")
        if step_2_complete:
            st.markdown(f"Clinical Compliance: first contours, first dose distribution.")
            a_compliance = compliance.compute_mirage_compliance(first_dose, first_structure_mask)
            st.table(a_compliance)
            a_compliance_csv = a_compliance.to_csv(index=True)
            st.download_button(label="Download compliance CSV", data=a_compliance_csv, file_name="a_compliance.csv", mime="text/csv", key=500)

            st.markdown(f"Clinical Compliance: first contours, last dose distribution.")
            b_compliance = compliance.compute_mirage_compliance(last_dose, first_structure_mask)
            st.table(b_compliance)
            b_compliance_csv = b_compliance.to_csv(index=True)
            st.download_button(label="Download compliance CSV", data=b_compliance_csv, file_name="b_compliance.csv", mime="text/csv", key=499)

            st.markdown(f"Clinical Compliance: last contours, last dose distribution.")
            c_compliance = compliance.compute_mirage_compliance(last_dose, last_structure_mask)
            st.table(c_compliance)
            c_compliance_csv = c_compliance.to_csv(index=True)
            st.download_button(label="Download compliance CSV", data=c_compliance_csv, file_name="c_compliance.csv", mime="text/csv", key=498)

            st.divider()

    """
    with tab4: 
        ref_id = st.number_input("Choose reference dose: ", min_value=1, max_value=n_compares, value="min", step=1)
        
        all_structures = list(structure_mask.keys())
        selected_structures = st.multiselect(
            "Choose structures to compare:",
            list(all_structures),
            [],
        )
        
        compare_differences(summary_df, selected_structures, ref_id)
        display_difference_dvh(doses, structure_mask, selected_structures, ref_id)
    """