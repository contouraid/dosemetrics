"""
Data management utilities for radiotherapy dose and structure data.

This module provides I/O utilities for reading and writing dose and structure data
in DICOM and NIfTI formats.
"""

# High-level unified I/O
from .data_io import (
    load_from_folder,
    load_structure_set,
    load_volume,
    load_structure,
    detect_folder_format,
)

# Format-specific I/O modules
from . import dicom_io
from . import nifti_io

__all__ = [
    # High-level I/O
    "load_from_folder",
    "load_structure_set",
    "load_volume",
    "load_structure",
    "detect_folder_format",
    # Format-specific modules
    "dicom_io",
    "nifti_io",
]
