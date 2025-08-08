import hmac
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
import plotly.express as px

from dosemetrics.data_utils import read_byte_data
from dosemetrics.dvh import dvh_by_structure
import dosemetrics.variations_tab as variations_tab

# Run this from >> streamlit run app.py


def request_dose_and_masks(instruction_text):
    st.markdown(instruction_text)
    st.markdown(f"Check instructions on the sidebar for more information.")

    dose_file = st.file_uploader(
        "Upload a dose distribution volume (in .nii.gz)", type=["gz"]
    )
    mask_files = st.file_uploader(
        "Upload mask volumes (in .nii.gz)", accept_multiple_files=True, type=["gz"]
    )

    return dose_file, mask_files


def check_password():
    """Returns `True` if the user had a correct password."""

    def login_form():
        """Form with widgets to collect user information"""
        with st.form("Credentials"):
            st.text_input("Username", key="username")
            st.text_input("Password", type="password", key="password")
            st.form_submit_button("Log in", on_click=password_entered)

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["username"] in st.secrets[
            "passwords"
        ] and hmac.compare_digest(
            st.session_state["password"],
            st.secrets.passwords[st.session_state["username"]],
        ):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the username or password.
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    # Return True if the username + password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show inputs for username + password.
    login_form()
    if "password_correct" in st.session_state:
        st.error("😕 User not known or password incorrect")
    return False


def display_dvh(df, structures):
    if not structures:
        st.error("Please select at least one structure.")
    else:
        data = df.loc[structures]
        data = data.T.reset_index()
        data = pd.melt(data, id_vars=["Dose"]).rename(
            columns={"index": "Dose", "value": "Dose Volume Histogram"}
        )
        chart = (
            alt.Chart(data)
            .mark_area(opacity=0.3)
            .encode(
                x="Dose:T",
                y=alt.Y("Dose Volume Histogram:Q", stack=None),
                color="variable:N",
            )
        )
        st.altair_chart(chart, use_container_width=True)


def main_loop():
    if not check_password():
        st.stop()

    def calculate_dvh():
        st.markdown(f"# {list(page_names_to_funcs.keys())[0]}")
        st.sidebar.success("Select an option above.")

        instruction_text = f"## Step 1: Upload dose distribution volume and mask files"
        dose_file, mask_files = request_dose_and_masks(instruction_text)
        files_uploaded = (dose_file is not None) and (len(mask_files) > 0)

        if files_uploaded:
            st.divider()
            st.markdown(f"## Step 2: Visualize DVH")
            dose_volume, structure_masks = read_byte_data(dose_file, mask_files)
            df = dvh_by_structure(dose_volume, structure_masks)
            fig = px.line(df, x="Dose", y="Volume", color="Structure")
            fig.update_xaxes(showgrid=True)
            fig.update_yaxes(showgrid=True)
            st.plotly_chart(fig, use_container_width=True)

    def visualize_dose():
        st.markdown(f"# {list(page_names_to_funcs.keys())[1]}")
        st.sidebar.success("Select an option above.")

        instruction_text = f"## Step 1: Upload dose distribution volume and mask files"
        dose_file, mask_files = request_dose_and_masks(instruction_text)
        files_uploaded = (dose_file is not None) and (len(mask_files) > 0)

        if files_uploaded:
            st.divider()
            st.markdown(f"## Step 2: Visualize Dose")
            dose_volume, structure_masks = read_byte_data(dose_file, mask_files)
            plt.figure(figsize=(6, 6), dpi=80)
            fig, ax = plt.subplots()
            slice_num = st.slider("Choose an axial slice number:", 1, 128, 64)
            plt.imshow(np.rot90(dose_volume[:, :, slice_num], 3), cmap="hot")
            plt.tick_params(
                axis="x",  # changes apply to the x-axis
                which="both",  # both major and minor ticks are affected
                bottom=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False,
            )
            plt.tick_params(
                axis="y",  # changes apply to the x-axis
                which="both",  # both major and minor ticks are affected
                left=False,  # ticks along the bottom edge are off
                right=False,  # ticks
                labelleft=False,
            )
            plt.title("Dose Volume")
            st.pyplot(fig)

    def dice_dvh_analysis():
        st.markdown(f"# {list(page_names_to_funcs.keys())[2]}")
        variations_tab.panel()

    def instructions():
        st.markdown(f"# {list(page_names_to_funcs.keys())[3]}")
        st.sidebar.success("Select an option above.")

        st.markdown(
            """
            This web-app calculates statistics from a Dose distribution + segmentation masks.
            This lives here: [dosemetrics.streamlit.app](https://dosemetrics.streamlit.app).
            
            ### Want to learn more?
            - Check out [www.contouraid.com](https://www.contouraid.com) for more information.
        """
        )

    page_names_to_funcs = {
        "Calculate DVH": calculate_dvh,
        "Visualize Dose": visualize_dose,
        "Contour Variation Robustness": dice_dvh_analysis,
        "Instructions": instructions,
    }

    task_selection = st.sidebar.selectbox("Choose a task:", page_names_to_funcs.keys())
    page_names_to_funcs[task_selection]()


if __name__ == "__main__":
    main_loop()
