"""
Command-line interface for dosemetrics.
"""

import argparse
import sys
from pathlib import Path

import dosemetrics


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Dosemetrics: Tools for radiotherapy dose analysis"
    )
    parser.add_argument(
        "--version", action="version", version=f"dosemetrics {dosemetrics.__version__}"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # DVH command
    dvh_parser = subparsers.add_parser("dvh", help="Generate dose-volume histogram")
    dvh_parser.add_argument("dose_file", help="Path to dose file (.nii.gz)")
    dvh_parser.add_argument(
        "mask_files", nargs="+", help="Paths to mask files (.nii.gz)"
    )
    dvh_parser.add_argument("-o", "--output", help="Output file path")

    # Quality command
    quality_parser = subparsers.add_parser("quality", help="Compute quality metrics")
    quality_parser.add_argument("dose_file", help="Path to dose file (.nii.gz)")
    quality_parser.add_argument(
        "mask_files", nargs="+", help="Paths to mask files (.nii.gz)"
    )
    quality_parser.add_argument("-o", "--output", help="Output file path")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    if args.command == "dvh":
        return run_dvh_command(args)
    elif args.command == "quality":
        return run_quality_command(args)

    return 0


def run_dvh_command(args):
    """Run DVH generation command."""
    try:
        # Load dose and masks
        dose_volume = dosemetrics.read_from_nifti(args.dose_file)
        structure_masks = {}

        for mask_file in args.mask_files:
            structure_name = Path(mask_file).stem.replace(".nii", "")
            structure_masks[structure_name] = dosemetrics.read_from_nifti(mask_file)

        # Generate DVH
        dvh_df = dosemetrics.dvh_by_structure(dose_volume, structure_masks)

        # Save or display results
        if args.output:
            dvh_df.to_csv(args.output, index=False)
            print(f"DVH saved to {args.output}")
        else:
            print(dvh_df)

    except Exception as e:
        print(f"Error generating DVH: {e}", file=sys.stderr)
        return 1

    return 0


def run_quality_command(args):
    """Run quality metrics command."""
    try:
        # Load dose and masks
        dose_volume = dosemetrics.read_from_nifti(args.dose_file)
        structure_masks = {}

        for mask_file in args.mask_files:
            structure_name = Path(mask_file).stem.replace(".nii", "")
            structure_masks[structure_name] = dosemetrics.read_from_nifti(mask_file)

        # Compute quality metrics
        summary_df = dosemetrics.dose_summary(dose_volume, structure_masks)

        # Save or display results
        if args.output:
            summary_df.to_csv(args.output)
            print(f"Quality metrics saved to {args.output}")
        else:
            print(summary_df)

    except Exception as e:
        print(f"Error computing quality metrics: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
