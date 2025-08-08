import os
import dosemetrics


def batch_analysis_example(subjects_root: str, output_root: str):
    """
    Example demonstrating batch analysis of multiple subjects.

    This example shows how to use the new batch processing utilities
    to analyze multiple subjects at once.
    """
    print("=== Batch Analysis Example ===\n")

    # 1. Find all subject folders automatically
    print("1. Finding subject folders...")
    subject_folders = dosemetrics.find_subject_folders(
        root_path=subjects_root,
        pattern="*",  # Match all folders
        must_contain_dose=True,
    )
    print(f"Found {len(subject_folders)} subject folders:")
    for folder in subject_folders:
        print(f"  - {os.path.basename(folder)}")

    # 2. Validate folder structure
    print("\n2. Validating folder structures...")
    validation_report = dosemetrics.batch_folder_validation(
        input_folders=subject_folders,
        output_file=os.path.join(output_root, "validation_report.csv"),
    )
    print("Validation Summary:")
    print(validation_report.to_string(index=False))

    # 3. Set up standardized output structure
    print("\n3. Setting up output structure...")
    subject_names = [os.path.basename(folder) for folder in subject_folders]
    output_structure = dosemetrics.setup_output_structure(
        output_root=output_root,
        subject_names=subject_names,
        analysis_types=["dvh", "quality_index", "compliance", "plots"],
    )
    print(f"Output structure created at: {output_structure['root']}")

    # 4. Perform batch DVH analysis
    print("\n4. Performing batch DVH analysis...")
    batch_results = dosemetrics.batch_dvh_analysis(
        input_folders=subject_folders,
        output_folder=output_structure["summary"],
    )

    # 5. Process each subject individually with comprehensive analysis
    print("\n5. Processing individual subjects...")
    for folder in subject_folders:
        subject_name = os.path.basename(folder)
        subject_output = os.path.join(output_structure["individual"], subject_name)

        print(f"Processing {subject_name}...")
        try:
            results = dosemetrics.process_subject_folder(
                input_folder=folder,
                output_folder=subject_output,
                analysis_types=["dvh", "quality_index", "compliance"],
            )
            print(f"  ✓ Completed {subject_name}")
        except Exception as e:
            print(f"  ✗ Error processing {subject_name}: {e}")

    print(f"\nBatch analysis complete! Results saved to: {output_root}")


def single_subject_comprehensive_example(input_folder: str, output_folder: str):
    """
    Example demonstrating comprehensive analysis of a single subject.
    """
    print("=== Single Subject Comprehensive Analysis ===\n")

    # 1. Validate the folder
    print("1. Validating folder structure...")
    validation = dosemetrics.validate_folder_structure(input_folder)
    print(f"Validation results: {validation}")

    if not validation.get("Dose.nii.gz", False):
        print("Error: No dose file found!")
        return

    # 2. Auto-detect structures
    print("\n2. Auto-detecting structures...")
    structures = dosemetrics.get_structures_from_folder(input_folder)
    print(f"Found {len(structures)} structures: {structures}")

    # 3. Create standard contents file
    print("\n3. Creating standard contents file...")
    contents_df = dosemetrics.create_standard_contents_csv(
        input_folder=input_folder,
        output_file=os.path.join(output_folder, "standard_contents.csv"),
    )
    print("Contents file created:")
    print(contents_df.to_string(index=False))

    # 4. Comprehensive analysis
    print("\n4. Performing comprehensive analysis...")
    results = dosemetrics.process_subject_folder(
        input_folder=input_folder,
        output_folder=output_folder,
        analysis_types=["dvh", "quality_index", "compliance"],
    )

    # 5. Additional analysis if predicted dose exists
    predicted_dose_file = os.path.join(input_folder, "Predicted_Dose.nii.gz")
    if os.path.exists(predicted_dose_file):
        print("\n5. Comparing with predicted dose...")

        # Read data
        dose_array, structure_masks = dosemetrics.read_dose_and_mask_files_from_folder(
            input_folder
        )
        predicted_array = dosemetrics.read_from_nifti(predicted_dose_file)

        # DVH comparison
        dosemetrics.compare_predicted_doses(
            dose_array=dose_array,
            predicted_array=predicted_array,
            structure_masks=structure_masks,
            output_file=os.path.join(output_folder, "dvh_comparison.pdf"),
        )

        # Quality index comparison
        qi_comparison = dosemetrics.compare_quality_indices(
            dose_arrays=[dose_array, predicted_array],
            structure_masks=structure_masks,
            labels=["Actual", "Predicted"],
        )
        qi_comparison.to_csv(
            os.path.join(output_folder, "qi_comparison.csv"), index=False
        )

        # Dose difference visualization
        dosemetrics.plot_dose_differences(
            dose_array=dose_array,
            predicted_array=predicted_array,
            output_file=os.path.join(output_folder, "dose_differences.pdf"),
        )

        print("Prediction comparison completed!")

    print(f"\nComprehensive analysis complete! Results saved to: {output_folder}")


if __name__ == "__main__":
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Example 1: Single subject analysis
    print("Running single subject example...")
    single_subject_folder = os.path.join(repo_root, "data", "test_subject")
    single_output = os.path.join(repo_root, "results", "comprehensive_analysis")
    os.makedirs(single_output, exist_ok=True)

    if os.path.exists(single_subject_folder):
        single_subject_comprehensive_example(single_subject_folder, single_output)
    else:
        print(f"Test subject folder not found: {single_subject_folder}")

    # Example 2: Batch analysis (if multiple subjects exist)
    print("\n" + "=" * 60 + "\n")

    data_root = os.path.join(repo_root, "data")
    batch_output = os.path.join(repo_root, "results", "batch_analysis")

    if os.path.exists(data_root):
        print("Running batch analysis example...")
        os.makedirs(batch_output, exist_ok=True)
        batch_analysis_example(data_root, batch_output)
    else:
        print(f"Data root folder not found: {data_root}")
