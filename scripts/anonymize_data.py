#!/usr/bin/env python3
"""
Anonymization script for DoseMetrics example data.

This script creates anonymized versions of the example data by:
1. Replacing CT/MR imaging data with synthetic dummy data
2. Preserving all structure masks, dose distributions, and relationships
3. Maintaining spatial alignment and metadata

The output can be safely shared publicly without privacy concerns.
"""

import nibabel as nib
import numpy as np
from pathlib import Path
import shutil
import argparse


def create_synthetic_ct(shape, spacing, affine, output_path):
    """
    Create synthetic CT-like data that looks realistic but contains no patient info.
    
    Uses smooth random noise with anatomically plausible HU values.
    """
    print(f"  Creating synthetic CT with shape {shape}")
    
    # Create base with typical brain HU values (20-40 HU)
    synthetic_data = np.random.normal(30, 5, shape).astype(np.float32)
    
    # Add some smooth anatomical-like variation
    # Create gradient from superior to inferior
    for z in range(shape[2]):
        z_factor = 1.0 - (z / shape[2]) * 0.3
        synthetic_data[:, :, z] *= z_factor
    
    # Add some smooth random variations to simulate tissue differences
    from scipy.ndimage import gaussian_filter
    variations = gaussian_filter(np.random.normal(0, 5, shape), sigma=10)
    synthetic_data += variations
    
    # Clip to reasonable CT range
    synthetic_data = np.clip(synthetic_data, -100, 100)
    
    # Save as NIfTI
    nii = nib.Nifti1Image(synthetic_data, affine)
    nii.header.set_zooms(spacing)
    nib.save(nii, output_path)
    print(f"  Saved synthetic CT to {output_path}")


def create_synthetic_mri(shape, spacing, affine, output_path):
    """
    Create synthetic MRI-like data (T1-weighted appearance).
    """
    print(f"  Creating synthetic MRI with shape {shape}")
    
    # T1-weighted has different contrast (higher signal in GM/WM)
    synthetic_data = np.random.normal(500, 100, shape).astype(np.float32)
    
    # Add smooth variations
    from scipy.ndimage import gaussian_filter
    variations = gaussian_filter(np.random.normal(0, 50, shape), sigma=15)
    synthetic_data += variations
    
    # Clip to reasonable MRI range
    synthetic_data = np.clip(synthetic_data, 0, 2000)
    
    # Save as NIfTI
    nii = nib.Nifti1Image(synthetic_data, affine)
    nii.header.set_zooms(spacing)
    nib.save(nii, output_path)
    print(f"  Saved synthetic MRI to {output_path}")


def anonymize_dataset(input_dir, output_dir, dataset_name="test_subject"):
    """
    Anonymize a dataset by replacing imaging data with synthetic data.
    
    Preserves:
    - All structure masks (Brain, BrainStem, etc.)
    - All dose distributions
    - Spatial alignment and metadata
    
    Replaces:
    - CT/MRI imaging data with synthetic data
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Anonymizing dataset: {dataset_name}")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"{'='*60}\n")
    
    # Files to replace with synthetic data (using specific patterns)
    # Using word boundaries to avoid matching CTV, GTV, PTV as CT
    def is_imaging_file(filename):
        """Check if file is imaging data that should be replaced."""
        # Check for exact CT filename patterns (not CTV, GTV, PTV)
        if filename.startswith('CT.') or filename.startswith('CT_'):
            return True
        # Check for MRI patterns
        if any(pattern in filename for pattern in ['T1c', 'T2', 'FLAIR', 'MRI']):
            return True
        return False
    
    # Files to copy as-is (structures and doses)
    preserve_patterns = [
        'Brain', 'BrainStem', 'Chiasm',
        'Cochlea', 'Eye', 'Hippocampus',
        'LacrimalGland', 'OpticNerve',
        'Pituitary', 'Target', 'Dose',
        'Predicted_Dose', 'Dose_Mask',
        'OARs', 'CTV', 'GTV', 'PTV',  # Planning target volumes
        'Brainstem'  # Alternative spelling
    ]
    
    # Get reference shape/spacing from any available file
    reference_file = None
    
    # Try standard naming first
    for ref_name in ['Brain.nii.gz', 'Dose.nii.gz', 'Target.nii.gz']:
        ref_path = input_path / ref_name
        if ref_path.exists():
            reference_file = ref_path
            break
    
    # Try with patterns (e.g., Dose_086.nii.gz)
    if not reference_file:
        for pattern in preserve_patterns:
            matches = list(input_path.glob(f"{pattern}*.nii.gz"))
            if matches:
                reference_file = matches[0]
                break
    
    # Try .seg.nrrd files
    if not reference_file:
        seg_files = list(input_path.glob("*.seg.nrrd"))
        if seg_files:
            reference_file = seg_files[0]
    
    if reference_file:
        ref_nii = nib.load(reference_file)
        shape = ref_nii.shape
        spacing = ref_nii.header.get_zooms()
        affine = ref_nii.affine
        print(f"Reference image: {reference_file.name}")
        print(f"Shape: {shape}, Spacing: {spacing}\n")
    else:
        print(f"ERROR: Could not find reference file in {input_path}")
        return
    
    # Process all files (.nii.gz and .seg.nrrd)
    all_files = list(input_path.glob('*.nii.gz')) + list(input_path.glob('*.seg.nrrd')) + list(input_path.glob('*.txt'))
    
    for file in all_files:
        filename = file.name
        
        # Always preserve .seg.nrrd and .txt files (they're segmentations/metadata)
        if filename.endswith('.seg.nrrd') or filename.endswith('.txt'):
            print(f"Processing {filename} (preserving)")
            shutil.copy2(file, output_path / filename)
            continue
        
        # Determine if this is an imaging file (to replace) or preserve file
        is_imaging_file_flag = is_imaging_file(filename)
        is_preserve = any(pattern in filename for pattern in preserve_patterns)
        
        if is_imaging_file_flag:
            # Replace with synthetic data
            if 'CT' in filename or 'ct' in filename:
                print(f"Processing {filename} (CT imaging - creating synthetic)")
                create_synthetic_ct(shape, spacing, affine, output_path / filename)
            else:
                print(f"Processing {filename} (MRI imaging - creating synthetic)")
                create_synthetic_mri(shape, spacing, affine, output_path / filename)
        
        elif is_preserve:
            # Copy as-is (structures, doses)
            print(f"Processing {filename} (preserving)")
            shutil.copy2(file, output_path / filename)
        
        else:
            # Check if it's a mask/structure/dose or imaging based on values
            try:
                nii = nib.load(file)
                data = nii.get_fdata()
                unique_values = len(np.unique(data))
                
                if unique_values <= 10:  # Likely a binary mask or label
                    print(f"Processing {filename} (preserving - appears to be mask)")
                    shutil.copy2(file, output_path / filename)
                else:
                    print(f"Processing {filename} (creating synthetic - appears to be imaging)")
                    create_synthetic_mri(shape, spacing, affine, output_path / filename)
            except Exception as e:
                print(f"Processing {filename} (error: {e}, skipping)")
                continue
    
    print(f"\n{'='*60}")
    print(f"✅ Anonymization complete!")
    print(f"   Anonymized data saved to: {output_path}")
    print(f"{'='*60}\n")


def process_compare_plans_dir(input_base, output_base):
    """Process the compare_plans directory structure."""
    compare_input = Path(input_base) / "compare_plans"
    compare_output = Path(output_base) / "compare_plans"
    
    if not compare_input.exists():
        print(f"No compare_plans directory found at {compare_input}")
        return
    
    # Process first and last subdirectories
    for subdir in ['first', 'last']:
        subdir_input = compare_input / subdir
        if subdir_input.exists():
            subdir_output = compare_output / subdir
            anonymize_dataset(subdir_input, subdir_output, f"compare_plans/{subdir}")


def process_visualization_dir(input_base, output_base):
    """Skip the visualization directory - not needed for HuggingFace dataset."""
    print("\nSkipping visualization directory (not needed for HuggingFace dataset)")


def main():
    parser = argparse.ArgumentParser(
        description='Anonymize DoseMetrics example data for public sharing'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='data',
        help='Input data directory (default: data)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data_anonymized',
        help='Output directory for anonymized data (default: data_anonymized)'
    )
    
    args = parser.parse_args()
    
    input_base = Path(args.input_dir)
    output_base = Path(args.output_dir)
    
    if not input_base.exists():
        print(f"ERROR: Input directory does not exist: {input_base}")
        return
    
    print("\n" + "="*80)
    print("DoseMetrics Data Anonymization Script")
    print("="*80)
    print("\nThis script will:")
    print("  ✓ Replace CT/MRI imaging data with synthetic data")
    print("  ✓ Preserve all structure masks")
    print("  ✓ Preserve all dose distributions")
    print("  ✓ Maintain spatial alignment")
    print("\nThe anonymized data can be safely shared publicly.\n")
    
    # Process main test_subject directory
    test_subject_input = input_base / "test_subject"
    if test_subject_input.exists():
        test_subject_output = output_base / "test_subject"
        anonymize_dataset(test_subject_input, test_subject_output, "test_subject")
    
    # Process compare_plans
    process_compare_plans_dir(input_base, output_base)
    
    # Process visualization
    process_visualization_dir(input_base, output_base)
    
    # Copy text files
    for txt_file in input_base.glob('*.txt'):
        output_txt = output_base / txt_file.name
        print(f"\nCopying {txt_file.name}")
        shutil.copy2(txt_file, output_txt)
    
    print("\n" + "="*80)
    print("✅ ALL ANONYMIZATION COMPLETE!")
    print(f"   Anonymized data ready for upload: {output_base}")
    print("="*80 + "\n")
    print("Next steps:")
    print("  1. Review the anonymized data")
    print("  2. Upload to HuggingFace datasets: huggingface.co/new-dataset")
    print("  3. Update examples to use the dataset\n")


if __name__ == "__main__":
    main()
