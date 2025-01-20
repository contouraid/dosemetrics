import hmac
import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
from dosemetrics.dvh import dvh_from_files
# Initial code from here: https://docs.streamlit.io/get-started/tutorials/create-a-multipage-app
# Run this from >> streamlit run app.py


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

    def instructions():
        st.write("# DVH Calculator:")
        st.sidebar.success("Select an option above.")

        st.markdown(
            """
            This web-app calculates DVH curves and allied statistics from a Dose distribution + segmentation masks.
    
            This lives here: [contouraid-dvh-calculator.streamlit.app](https://contouraid-dvh-calculator.streamlit.app).
    
            ### Want to learn more?
    
            - Check out [www.contouraid.com](https://www.contouraid.com) for more info
            - Reach out back to us at [www.contouraid.com/contact](https://www.contouraid.com/contact)
        """
        )


    def calculate_dvh():
        st.markdown(f"# {list(page_names_to_funcs.keys())[1]}")

        st.markdown(f"## Step 1: Upload dose distribution volume and mask files")
        st.markdown(f"Look [here](https://pyradise.readthedocs.io) to format your files into NIfTI format using Pyradise.")
        dose_file = st.file_uploader("Upload a dose distribution volume (in .nii.gz)", type=['nii', 'gz'])

        mask_files = st.file_uploader("Upload mask volumes (in .nii.gz)", accept_multiple_files=True, type=['nii', 'gz'])

        files_uploaded = (dose_file is not None) and (len(mask_files) > 0)

        if files_uploaded:
            st.markdown(f"Both dose and mask files are uploaded. Click the button below to proceed.")
            x = st.button("Compute")

            st.divider()

            if x:
                st.markdown(f"## Step 2: Visualize DVH")
                df = dvh_from_files(dose_file, mask_files)
                fig = px.line(df, x="Dose", y="Volume", color="Structure")
                fig.update_xaxes(showgrid=True)
                fig.update_yaxes(showgrid=True)
                st.plotly_chart(fig, use_container_width=True)


    page_names_to_funcs = {
        "Instructions": instructions,
        "Calculate DVH": calculate_dvh
    }

    task_selection = st.sidebar.selectbox("Choose a task:", page_names_to_funcs.keys())
    page_names_to_funcs[task_selection]()


if __name__ == "__main__":
    main_loop()