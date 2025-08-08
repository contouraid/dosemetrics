import hmac
import streamlit as st

from dosemetrics_app.tabs import calculate_dvh, visualize_dose, instructions, variations


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


def main_loop():
    # Authentication disabled for development/testing
    # if not check_password():
    #     st.stop()

    def calculate_dvh_page():
        st.markdown("# Calculate DVH")
        calculate_dvh.panel()

    def visualize_dose_page():
        st.markdown("# Visualize Dose")
        visualize_dose.panel()

    def dice_dvh_analysis():
        st.markdown("# Contour Variation Robustness")
        variations.panel()

    def instructions_page():
        st.markdown("# Instructions")
        instructions.panel()

    page_names_to_funcs = {
        "Calculate DVH": calculate_dvh_page,
        "Visualize Dose": visualize_dose_page,
        "Contour Variation Robustness": dice_dvh_analysis,
        "Instructions": instructions_page,
    }

    task_selection = st.sidebar.selectbox("Choose a task:", page_names_to_funcs.keys())
    page_names_to_funcs[task_selection]()


if __name__ == "__main__":
    main_loop()
