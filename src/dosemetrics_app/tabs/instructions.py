import streamlit as st


def panel():
    """Main panel function for Instructions tab"""
    st.sidebar.success("Select an option above.")

    st.markdown(
        """
        # DoseMetrics - Radiotherapy Dose Analysis Tool
        
        This web application provides comprehensive tools for analyzing radiotherapy dose distributions 
        and structure segmentations. Calculate dose-volume histograms (DVH), evaluate clinical constraints, 
        and assess treatment plan quality.
        
        ## 🚀 Getting Started
        
        ### Option 1: Use Example Data
        Select any analysis tab and choose **"Use example data"** to try the app with pre-loaded datasets:
        - **Local data**: Test subject and longitudinal timepoints from your local installation
        - **HuggingFace data**: Example datasets from [contouraid/dosemetrics-data](https://huggingface.co/datasets/contouraid/dosemetrics-data)
        
        ### Option 2: Upload Your Own Data
        Upload your dose distribution and structure masks in NIfTI format (`.nii.gz`):
        - **Dose file**: 3D dose distribution volume
        - **Structure masks**: One file per anatomical structure (organs at risk, targets)
        
        ## 📊 Available Analyses
        
        1. **Calculate DVH**: Compute dose-volume histograms for all structures
        2. **Visualize Dose**: View dose distributions slice-by-slice
        3. **Contour Variation Robustness**: Assess plan sensitivity to structure delineation variations
        
        ## 📚 Resources
        
        - **Live App**: [dosemetrics.streamlit.app](https://dosemetrics.streamlit.app)
        - **Example Dataset**: [HuggingFace Dataset](https://huggingface.co/datasets/contouraid/dosemetrics-data)
        - **Documentation**: [GitHub Repository](https://github.com/contouraid/dosemetrics)
        - **More Info**: [www.contouraid.com](https://www.contouraid.com)
        
        ## 💡 Tips
        
        - Start with example data to familiarize yourself with the interface
        - Ensure your structure files follow consistent naming conventions
        - Download results as CSV for further analysis
        - All processing happens locally in your browser
        
        ---
        
        **Questions or feedback?** Visit [ContourAId](https://www.contouraid.com) for support.
    """
    )
