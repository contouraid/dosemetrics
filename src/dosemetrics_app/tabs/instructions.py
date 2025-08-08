import streamlit as st


def panel():
    """Main panel function for Instructions tab"""
    st.sidebar.success("Select an option above.")

    st.markdown(
        """
        This web-app calculates statistics from a Dose distribution + segmentation masks.
        This lives here: [dosemetrics.streamlit.app](https://dosemetrics.streamlit.app).
        
        ### Want to learn more?
        - Check out [www.contouraid.com](https://www.contouraid.com) for more information.
    """
    )
