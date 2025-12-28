import hmac
import streamlit as st

from dosemetrics_app.tabs import (
    comprehensive_analysis,
    geometric_tab,
    gamma_tab,
    compliance_tab,
    instructions,
)


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
        st.error("User not known or password incorrect")
    return False


def main_loop():
    # Authentication disabled for development/testing
    # if not check_password():
    #     st.stop()

    def comprehensive_analysis_page():
        st.markdown("# Dosimetric Analysis")
        comprehensive_analysis.panel()

    def geometric_page():
        st.markdown("# Geometric Comparison")
        geometric_tab.panel()

    def gamma_page():
        st.markdown("# Gamma Analysis")
        gamma_tab.panel()

    def compliance_page():
        st.markdown("# Compliance Checking")
        compliance_tab.panel()

    def instructions_page():
        st.markdown("# Instructions")
        instructions.panel()

    page_names_to_funcs = {
        "Instructions": instructions_page,
        "Dosimetric Analysis": comprehensive_analysis_page,
        "Geometric Comparison": geometric_page,
        "Gamma Analysis": gamma_page,
        "Compliance Checking": compliance_page,
    }

    task_selection = st.sidebar.selectbox("Choose a task:", page_names_to_funcs.keys())
    page_names_to_funcs[task_selection]()


if __name__ == "__main__":
    main_loop()
