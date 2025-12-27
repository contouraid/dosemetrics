"""
Unified I/O module for radiotherapy data.

This module provides high-level functions for loading radiotherapy data from
various sources (DICOM, NIfTI) with automatic format detection and intelligent
structure organization.

The API is organized in layers:
1. Low-level: Format-specific readers (dicom_io, nifti_io)
2. Mid-level: Type-specific loaders (load_volume, load_structure, etc.)
3. High-level: Auto-detecting folder loaders (load_from_folder, load_structure_set)
"""

import os
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    from ..structures import Structure, StructureType
    from ..structure_set import StructureSet

from . import dicom_io
from . import nifti_io


def detect_folder_format(folder_path: Union[str, Path]) -> str:
    """
    Detect the data format in a folder (DICOM or NIfTI).

    Args:
        folder_path: Path to folder to inspect

    Returns:
        'dicom' if DICOM files found, 'nifti' if NIfTI files found, 'unknown' otherwise
    """
    folder_path = Path(folder_path)
    
    if not folder_path.exists():
        return 'unknown'
    
    # Check for DICOM files directly
    if list(folder_path.rglob('*.dcm')):
        return 'dicom'
    
    # Check for NIfTI files
    if list(folder_path.rglob('*.nii.gz')) or list(folder_path.rglob('*.nii')):
        return 'nifti'
    
    return 'unknown'


def load_from_folder(
    folder_path: Union[str, Path],
    format: Optional[str] = None,
    **kwargs
) -> Dict[str, Union[np.ndarray, Dict, Tuple]]:
    """
    Load radiotherapy data from a folder, auto-detecting format.

    This is the highest-level function for loading data. It automatically detects
    whether the folder contains DICOM or NIfTI data and loads accordingly.

    Args:
        folder_path: Path to folder containing data
        format: Force specific format ('dicom' or 'nifti'). If None, auto-detects.
        **kwargs: Additional arguments passed to format-specific loaders

    Returns:
        Dictionary with loaded data. Keys depend on format:
            For DICOM: 'ct_volume', 'dose_volumes', 'structures', 'spacing', 'origin'
            For NIfTI: 'dose_volume', 'structure_masks', 'image_volumes', 'spacing', 'origin'

    Raises:
        FileNotFoundError: If folder doesn't exist
        ValueError: If format cannot be determined
    """
    folder_path = Path(folder_path)
    
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    # Auto-detect format if not specified
    if format is None:
        format = detect_folder_format(folder_path)
    
    if format == 'dicom':
        return dicom_io.load_dicom_folder(folder_path, **kwargs)
    elif format == 'nifti':
        return nifti_io.load_nifti_folder(folder_path, **kwargs)
    else:
        raise ValueError(
            f"Unknown format in {folder_path}. "
            f"Folder should contain either DICOM files or NIfTI files."
        )


def load_structure_set(
    folder_path: Union[str, Path],
    format: Optional[str] = None,
    name: Optional[str] = None,
    structure_type_mapping: Optional[Dict[str, "StructureType"]] = None,
    **kwargs
) -> "StructureSet":
    """
    Load a complete StructureSet from a folder, auto-detecting format.

    This is the primary high-level function for loading radiotherapy data as a
    unified StructureSet object. It handles both DICOM and NIfTI formats.

    Args:
        folder_path: Path to folder containing data
        format: Force specific format ('dicom' or 'nifti'). If None, auto-detects.
        name: Name for the structure set. If None, uses folder name.
        structure_type_mapping: Optional dict mapping structure names to StructureType
        **kwargs: Additional arguments passed to format-specific loaders
                 For NIfTI: dose_filename (default: "Dose.nii.gz")
                 For DICOM: dose_file_name (specific dose file to use)

    Returns:
        StructureSet object with loaded structures and dose

    Raises:
        FileNotFoundError: If folder doesn't exist
        ValueError: If format cannot be determined or no structures found

    Examples:
        >>> # Load from DICOM folder
        >>> structure_set = load_structure_set('path/to/dicom_folder')
        
        >>> # Load from NIfTI folder with custom dose filename
        >>> structure_set = load_structure_set('path/to/nifti_folder', 
        ...                                     dose_filename='dose_distribution.nii.gz')
        
        >>> # Force format and provide structure types
        >>> type_mapping = {'Liver': StructureType.OAR, 'PTV': StructureType.TARGET}
        >>> structure_set = load_structure_set('path/to/folder',
        ...                                     format='nifti',
        ...                                     structure_type_mapping=type_mapping)
    """
    from ..structure_set import StructureSet  # Import here to avoid circular dependency
    
    folder_path = Path(folder_path)
    
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    # Auto-detect format if not specified
    if format is None:
        format = detect_folder_format(folder_path)
    
    # Use folder name as default name
    if name is None:
        name = folder_path.name
    
    # Load based on format
    if format == 'dicom':
        return dicom_io.create_structure_set_from_dicom(
            folder_path=folder_path,
            name=name,
            structure_type_mapping=structure_type_mapping,
            **kwargs
        )
    elif format == 'nifti':
        return nifti_io.create_structure_set_from_nifti_folder(
            folder_path=folder_path,
            name=name,
            structure_type_mapping=structure_type_mapping,
            **kwargs
        )
    else:
        raise ValueError(
            f"Unknown format in {folder_path}. "
            f"Folder should contain either DICOM files or NIfTI files."
        )


def load_volume(
    file_path: Union[str, Path],
    format: Optional[str] = None,
) -> Tuple[np.ndarray, Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Load a single volume file (DICOM RTDOSE or NIfTI).

    Args:
        file_path: Path to file or folder (for DICOM CT series)
        format: Force specific format ('dicom' or 'nifti'). If None, auto-detects.

    Returns:
        Tuple of (volume, spacing, origin)

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If format cannot be determined
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Auto-detect format if not specified
    if format is None:
        if file_path.is_dir():
            format = 'dicom'  # Assume directory is DICOM CT series
        elif file_path.suffix in ['.nii', '.gz']:
            format = 'nifti'
        elif file_path.suffix == '.dcm':
            format = 'dicom'
        else:
            raise ValueError(f"Cannot determine format for: {file_path}")
    
    # Load based on format
    if format == 'dicom':
        if file_path.is_dir():
            # CT series
            volume, spacing, origin = dicom_io.read_dicom_ct_volume(file_path)
            return volume, spacing, origin
        else:
            # Single RTDOSE file
            volume, spacing, origin, _ = dicom_io.read_dicom_rtdose(file_path)
            return volume, spacing, origin
    elif format == 'nifti':
        return nifti_io.read_nifti_volume(file_path)
    else:
        raise ValueError(f"Unknown format: {format}")


def load_structure(
    file_path: Union[str, Path],
    name: Optional[str] = None,
    structure_type: "StructureType" = None,
    format: Optional[str] = None,
    **kwargs
) -> "Structure":
    """
    Load a single structure from a file.

    Args:
        file_path: Path to NIfTI file containing structure mask
        name: Name for the structure. If None, uses filename.
        structure_type: Type of structure (OAR, TARGET, etc.)
        format: Force specific format ('nifti'). If None, auto-detects.
        **kwargs: Additional arguments (e.g., threshold for binarization)

    Returns:
        Structure object

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If format is not supported (currently only NIfTI single files)

    Note:
        For DICOM RTSTRUCT files, use load_structure_set() instead as they
        typically contain multiple structures.
    """
    from ..structures import StructureType  # Import here to avoid circular dependency
    
    if structure_type is None:
        structure_type = StructureType.OAR
    
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Auto-detect format if not specified
    if format is None:
        if file_path.suffix in ['.nii', '.gz']:
            format = 'nifti'
        else:
            raise ValueError(
                f"Cannot determine format for: {file_path}. "
                f"For DICOM RTSTRUCT files, use load_structure_set() instead."
            )
    
    # Currently only NIfTI single-file structures supported
    if format == 'nifti':
        return nifti_io.read_nifti_structure(
            file_path,
            name=name,
            structure_type=structure_type,
            **kwargs
        )
    else:
        raise ValueError(
            f"Format '{format}' not supported for single structure loading. "
            f"Use load_structure_set() for DICOM RTSTRUCT files."
        )