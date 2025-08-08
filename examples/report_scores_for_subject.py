# -*- encoding: utf-8 -*-
import os
import dosemetrics


def compute_dosemetrics(data_dir):
    """
    Compute comprehensive dosimetrics using the new process_subject_folder function.

    This example demonstrates the new high-level workflow function.
    """
    # Set up output directory
    output_dir = os.path.join(
        os.path.dirname(data_dir), "results", os.path.basename(data_dir)
    )

    # Use the new comprehensive analysis function
    results = dosemetrics.process_subject_folder(
        input_folder=data_dir,
        output_folder=output_dir,
        analysis_types=["dvh", "quality_index", "compliance"],
    )

    print(f"Analysis complete for {os.path.basename(data_dir)}")
    print(f"Results saved to: {output_dir}")

    # Print summary statistics
    if "dvh" in results:
        print(f"\nDVH analysis completed for {len(results['dvh'])} structures")

    if "quality_index" in results:
        print(f"Quality index computed for {len(results['quality_index'])} structures")
        print("\nQuality Index Summary:")
        print(results["quality_index"].to_string(index=False))

    if "compliance" in results:
        print(f"\nCompliance checking completed")
        print("Compliance Summary:")
        print(results["compliance"].to_string(index=False))

    return results


if __name__ == "__main__":
    repo_root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(repo_root, "..", "data", "test_subject")
    compute_dosemetrics(data_dir)
