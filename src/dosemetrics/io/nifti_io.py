"""
NIfTI I/O utilities for radiotherapy data.

This module provides functions to read NIfTI files for:
- CT/MR image volumes (real-valued)
- Dose distributions (real-valued)
- Structure masks (binary)

Uses SimpleITK for robust NIfTI reading with proper handling of spacing and origin.
"""

import os
import numpy as np
import SimpleITK as sitk
from typing import Dict, List, Optional, Tuple, Union, TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    from ..structures import Structure, OAR, Target, StructureType
    from ..structure_set import StructureSet


def read_nifti_volume(
    nifti_file: Union[str, Path],
) -> Tuple[np.ndarray, Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Read a NIfTI file and return volume with geometric information.

    Args:
        nifti_file: Path to NIfTI file (.nii or .nii.gz)

    Returns:
        Tuple of (volume, spacing, origin) where:
            - volume: 3D numpy array
            - spacing: (x, y, z) voxel spacing in mm
            - origin: (x, y, z) origin coordinates in mm

    Raises:
        FileNotFoundError: If file doesn't exist
        RuntimeError: If file cannot be read
    """
    nifti_file = Path(nifti_file)
    
    if not nifti_file.exists():
        raise FileNotFoundError(f"NIfTI file not found: {nifti_file}")
    
    # Read with SimpleITK
    image = sitk.ReadImage(str(nifti_file))
    
    # Get volume as numpy array
    volume = sitk.GetArrayFromImage(image)
    
    # Get spacing and origin
    spacing = image.GetSpacing()  # (x, y, z)
    origin = image.GetOrigin()    # (x, y, z)
    
    return volume, spacing, origin


def read_nifti_mask(
    nifti_file: Union[str, Path],
    threshold: float = 0.5,
) -> Tuple[np.ndarray, Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Read a NIfTI file as a binary mask.

    Args:
        nifti_file: Path to NIfTI file containing binary or probability mask
        threshold: Threshold for binarization (values > threshold become True)

    Returns:
        Tuple of (mask, spacing, origin) where:
            - mask: 3D boolean numpy array
            - spacing: (x, y, z) voxel spacing in mm
            - origin: (x, y, z) origin coordinates in mm

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    volume, spacing, origin = read_nifti_volume(nifti_file)
    
    # Binarize
    mask = volume > threshold
    
    return mask, spacing, origin


def read_nifti_dose(
    nifti_file: Union[str, Path],
) -> Tuple[np.ndarray, Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Read a NIfTI file containing dose distribution.

    This is an alias for read_nifti_volume with a more semantic name.

    Args:
        nifti_file: Path to NIfTI file containing dose data

    Returns:
        Tuple of (dose_array, spacing, origin) where:
            - dose_array: 3D numpy array with dose values (typically in Gy)
            - spacing: (x, y, z) voxel spacing in mm
            - origin: (x, y, z) origin coordinates in mm

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    return read_nifti_volume(nifti_file)


def read_from_nifti(nifti_file: Union[str, Path]) -> np.ndarray:
    """
    Read a NIfTI file and return only the volume array (backward compatibility).

    This function provides backward compatibility with older code that expects
    only the numpy array. For new code, use read_nifti_volume() to also get
    spacing and origin information.

    Args:
        nifti_file: Path to NIfTI file

    Returns:
        3D numpy array

    Raises:
        FileNotFoundError: If file doesn't exist
    
    Note:
        Deprecated. Use read_nifti_volume() for new code to get spacing and origin.
    """
    volume, _, _ = read_nifti_volume(nifti_file)
    return volume


def is_binary_volume(volume: np.ndarray, tolerance: float = 1e-6) -> bool:
    """
    Check if a volume contains only binary values (0 and 1).

    Args:
        volume: Numpy array to check
        tolerance: Tolerance for checking if values are 0 or 1

    Returns:
        True if volume is binary, False otherwise
    """
    unique_values = np.unique(volume)
    
    # Check if all unique values are close to 0 or 1
    for val in unique_values:
        if not (np.abs(val) < tolerance or np.abs(val - 1) < tolerance):
            return False
    
    return True


def load_nifti_folder(
    folder_path: Union[str, Path],
    dose_filename: str = "Dose.nii.gz",
    auto_detect_masks: bool = True,
    return_as_structureset: bool = True,
    structure_type_mapping: Optional[Dict[str, "StructureType"]] = None,
) -> Union["StructureSet", Dict[str, Union[np.ndarray, Dict, Tuple]]]:
    """
    Load all NIfTI files from a folder.

    This function automatically:
    1. Detects and loads the dose file
    2. Auto-detects whether each file is a binary mask (structure) or real-valued volume (image)
    3. Returns a StructureSet (default) or dictionary with organized data

    Args:
        folder_path: Path to folder containing NIfTI files
        dose_filename: Name of the dose file (default: "Dose.nii.gz")
        auto_detect_masks: Whether to auto-detect binary masks vs real-valued volumes
        return_as_structureset: If True (default), returns a StructureSet object.
                               If False, returns raw dictionary.
        structure_type_mapping: Optional dict mapping structure names to StructureType
                               (only used if return_as_structureset=True)

    Returns:
        If return_as_structureset=True: StructureSet object with loaded data
        If return_as_structureset=False: Dictionary with keys:
            - 'dose_volume': Dose distribution array (if found)
            - 'dose_spacing': Dose spacing tuple (if found)
            - 'dose_origin': Dose origin tuple (if found)
            - 'image_volumes': Dict of real-valued volumes {name: {'volume', 'spacing', 'origin'}}
            - 'structure_masks': Dict of binary masks {name: {'mask', 'spacing', 'origin'}}
            - 'spacing': Common spacing for all data
            - 'origin': Common origin for all data

    Raises:
        FileNotFoundError: If folder doesn't exist
    """
    folder_path = Path(folder_path)
    
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    result = {
        'image_volumes': {},
        'structure_masks': {},
    }
    
    # Get all NIfTI files
    nifti_files = list(folder_path.glob("*.nii.gz")) + list(folder_path.glob("*.nii"))
    
    if not nifti_files:
        return result
    
    # Try to load dose file first
    dose_file = folder_path / dose_filename
    if dose_file.exists():
        try:
            dose_volume, dose_spacing, dose_origin = read_nifti_dose(dose_file)
            result['dose_volume'] = dose_volume
            result['dose_spacing'] = dose_spacing
            result['dose_origin'] = dose_origin
            result['spacing'] = dose_spacing
            result['origin'] = dose_origin
        except Exception as e:
            print(f"Warning: Could not load dose file {dose_filename}: {e}")
    
    # Process other NIfTI files
    for nifti_file in nifti_files:
        # Skip dose file
        if nifti_file.name == dose_filename:
            continue
        
        try:
            volume, spacing, origin = read_nifti_volume(nifti_file)
            
            # Set common spacing/origin from first file if not set
            if 'spacing' not in result:
                result['spacing'] = spacing
                result['origin'] = origin
            
            # Extract name from filename (remove .nii.gz or .nii)
            if nifti_file.name.endswith('.nii.gz'):
                name = nifti_file.name[:-7]
            else:
                name = nifti_file.stem
            
            # Auto-detect if this is a binary mask
            if auto_detect_masks and is_binary_volume(volume):
                # It's a binary mask (structure)
                mask = volume.astype(bool)
                result['structure_masks'][name] = {
                    'mask': mask,
                    'spacing': spacing,
                    'origin': origin,
                }
            else:
                # It's a real-valued volume (image)
                result['image_volumes'][name] = {
                    'volume': volume,
                    'spacing': spacing,
                    'origin': origin,
                }
        
        except Exception as e:
            print(f"Warning: Could not load {nifti_file.name}: {e}")
    
    # Return as StructureSet if requested
    if return_as_structureset:
        if not result['structure_masks']:
            raise ValueError(f"No structure masks found in: {folder_path}")
        return create_structure_set_from_nifti_folder(
            folder_path,
            dose_filename=dose_filename,
            structure_type_mapping=structure_type_mapping,
            name=folder_path.name if isinstance(folder_path, Path) else Path(folder_path).name,
        )
    
    return result


def create_structure_set_from_nifti_folder(
    folder_path: Union[str, Path],
    dose_filename: str = "Dose.nii.gz",
    structure_type_mapping: Optional[Dict[str, "StructureType"]] = None,
    name: Optional[str] = None,
) -> "StructureSet":
    """
    Create a StructureSet from NIfTI files in a folder.

    This high-level function automatically:
    1. Loads dose distribution
    2. Auto-detects binary mask files as structures
    3. Creates Structure objects with appropriate types
    4. Returns a complete StructureSet

    Args:
        folder_path: Path to folder containing NIfTI files
        dose_filename: Name of the dose file (default: "Dose.nii.gz")
        structure_type_mapping: Optional dict mapping structure names to StructureType.
                               If not provided, guesses based on naming conventions.
        name: Name for the structure set. If None, uses folder name.

    Returns:
        StructureSet object with loaded structures and dose

    Raises:
        FileNotFoundError: If folder doesn't exist
        ValueError: If no structures found
    """
    from ..structures import StructureType
    from ..structure_set import StructureSet
    
    folder_path = Path(folder_path)
    
    # Load all data from folder (get raw dict to avoid recursion)
    data = load_nifti_folder(folder_path, dose_filename=dose_filename, return_as_structureset=False)
    
    if not data['structure_masks']:
        raise ValueError(f"No structure masks found in: {folder_path}")
    
    # Get spacing and origin
    spacing = data.get('spacing', (1.0, 1.0, 1.0))
    origin = data.get('origin', (0.0, 0.0, 0.0))
    
    # Create structure set name
    if name is None:
        name = folder_path.name
    
    # Create structure set
    structure_set = StructureSet(spacing=spacing, origin=origin, name=name)
    
    # Add structures
    for struct_name, struct_data in data['structure_masks'].items():
        # Determine structure type
        struct_type = StructureType.OAR  # Default
        
        if structure_type_mapping and struct_name in structure_type_mapping:
            struct_type = structure_type_mapping[struct_name]
        else:
            # Guess based on name
            name_upper = struct_name.upper()
            if any(keyword in name_upper for keyword in ['PTV', 'GTV', 'CTV', 'TARGET']):
                struct_type = StructureType.TARGET
            elif any(keyword in name_upper for keyword in ['AVOID', 'PRV']):
                struct_type = StructureType.AVOIDANCE
        
        structure_set.add_structure(
            name=struct_name,
            mask=struct_data['mask'],
            structure_type=struct_type,
        )
    
    # Note: In the new architecture, dose is loaded separately using Dose.from_nifti()
    # and not attached to the StructureSet
    
    return structure_set


def read_nifti_structure(
    nifti_file: Union[str, Path],
    name: Optional[str] = None,
    structure_type: "StructureType" = None,
    threshold: float = 0.5,
) -> "Structure":
    """
    Read a single NIfTI file as a Structure object.

    Args:
        nifti_file: Path to NIfTI file containing binary mask
        name: Name for the structure. If None, uses filename.
        structure_type: Type of structure (OAR, TARGET, etc.)
        threshold: Threshold for binarization

    Returns:
        Structure object (OAR, Target, or AvoidanceStructure based on type)

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    from ..structures import Structure, OAR, Target, StructureType
    
    if structure_type is None:
        structure_type = StructureType.OAR
    
    nifti_file = Path(nifti_file)
    
    # Extract name from filename if not provided
    if name is None:
        if nifti_file.name.endswith('.nii.gz'):
            name = nifti_file.name[:-7]
        else:
            name = nifti_file.stem
    
    # Read mask
    mask, spacing, origin = read_nifti_mask(nifti_file, threshold=threshold)
    
    # Create appropriate Structure subclass
    if structure_type == StructureType.OAR:
        structure_class = OAR
    elif structure_type == StructureType.TARGET:
        structure_class = Target
    else:
        # For other types, use base class with type override
        structure_class = type(
            f"{structure_type.value.title()}Structure",
            (Structure,),
            {"structure_type": property(lambda self: structure_type)},
        )
    
    structure = structure_class(
        name=name,
        mask=mask,
        spacing=spacing,
        origin=origin,
    )
    
    return structure


def write_nifti_volume(
    volume: np.ndarray,
    output_file: Union[str, Path],
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    """
    Write a numpy array to a NIfTI file.

    Args:
        volume: 3D numpy array to write
        output_file: Path for output NIfTI file
        spacing: Voxel spacing in (x, y, z) mm
        origin: Origin coordinates in (x, y, z) mm

    Raises:
        ValueError: If volume is not 3D
    """
    if volume.ndim != 3:
        raise ValueError(f"Volume must be 3D, got {volume.ndim}D")
    
    output_file = Path(output_file)
    
    # Create SimpleITK image
    image = sitk.GetImageFromArray(volume)
    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    
    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Write to file
    sitk.WriteImage(image, str(output_file))


def write_structure_as_nifti(
    structure: "Structure",
    output_file: Union[str, Path],
) -> None:
    """
    Write a Structure's mask to a NIfTI file.

    Args:
        structure: Structure object to write
        output_file: Path for output NIfTI file

    Raises:
        ValueError: If structure has no mask
    """
    if structure.mask is None:
        raise ValueError(f"Structure '{structure.name}' has no mask")
    
    # Convert boolean mask to uint8 for better compatibility
    mask_uint = structure.mask.astype(np.uint8)
    
    write_nifti_volume(
        volume=mask_uint,
        output_file=output_file,
        spacing=structure.spacing,
        origin=structure.origin,
    )


def write_structure_set_as_nifti(
    structure_set: "StructureSet",
    output_folder: Union[str, Path],
    write_dose: bool = True,
    dose_filename: str = "Dose.nii.gz",
) -> None:
    """
    Write a StructureSet to NIfTI files in a folder.

    Args:
        structure_set: StructureSet to write
        output_folder: Path to output folder
        write_dose: Whether to write dose file
        dose_filename: Name for dose file

    Raises:
        ValueError: If structure_set has no structures
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Write each structure
    for struct_name, structure in structure_set.structures.items():
        if structure.mask is not None:
            output_file = output_folder / f"{struct_name}.nii.gz"
            write_structure_as_nifti(structure, output_file)
    
    # Note: Dose writing removed - use Dose objects separately
