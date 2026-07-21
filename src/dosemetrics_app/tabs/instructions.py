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
        
        ## Getting Started
        
        ### Option 1: Use Hosted Example Data
        Open **Dosimetric Analysis**, choose **Hosted example**, and select a study from
        [contouraid/dosemetrics-data](https://huggingface.co/datasets/contouraid/dosemetrics-data):

        - NIfTI: test subject or either longitudinal time point
        - DICOM: select any of the included RTDOSE objects; the RTSTRUCT is rasterized on that dose grid
        
        ### Option 2: Upload Your Own Data
        **Dosimetric Analysis** accepts either:

        - NIfTI: one `.nii`/`.nii.gz` dose volume and one or more binary structure masks
        - DICOM-RT: at least one RTDOSE and one RTSTRUCT; CT and RTPLAN files are optional

        Uploaded DICOM objects are identified by their Modality tag, so their local filenames do not
        need to follow a special convention.
        
        ## Available Analyses
        
        ### Basic Analysis
        1. **Calculate DVH**: Compute dose-volume histograms for all structures
        2. **Visualize Dose**: View dose distributions slice-by-slice
        3. **Dose Statistics**: Calculate comprehensive dose statistics (mean, max, min, DVH metrics)
        
        ### Quality Metrics
        4. **Conformity Analysis**: Evaluate conformity indices for target volumes (CI, CN, GI)
        5. **Homogeneity Analysis**: Assess dose homogeneity within target volumes (HI)
        6. **Compliance Checking**: Verify compliance with clinical dose constraints
        
        ### Comparison Tools
        7. **Geometric Comparison**: Compare structure sets using geometric metrics (Dice, Jaccard, Hausdorff distance)
        8. **Gamma Analysis**: Perform gamma analysis between dose distributions
        
        ## Resources
        
        - Live App: dosemetrics.streamlit.app
        - Example Dataset: HuggingFace Dataset (https://huggingface.co/datasets/contouraid/dosemetrics-data)
        - Documentation: GitHub Repository (https://github.com/contouraid/dosemetrics)
        - More Info: www.contouraid.com
        
        ## Usage Tips
        
        - Start with example data to familiarize yourself with the interface
        - Ensure your structure files follow consistent naming conventions
        - Download results as CSV for further analysis
        - Uploaded files are processed in a temporary server-side directory that is deleted after loading
        
        ---
        
        Questions or feedback? Visit ContourAId (https://www.contouraid.com) for support.
    """
    )
