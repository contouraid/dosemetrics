"""
DICOM I/O utilities for radiotherapy data.

This module provides functions to read DICOM radiotherapy data including:
- CT image volumes
- RTDOSE (dose distributions)
- RTSTRUCT (structure sets/contours)
- RTPLAN (treatment plans - metadata only)

Uses pydicom for DICOM parsing and SimpleITK for volume reconstruction.
"""

import os
import numpy as np
import pydicom
import SimpleITK as sitk
from typing import Dict, List, Optional, Tuple, Union, TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    from ..structures import Structure, OAR, Target, StructureType
    from ..structure_set import StructureSet


def read_dicom_ct_volume(
    ct_directory: Union[str, Path],
) -> Tuple[np.ndarray, Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Read a CT volume from a directory of DICOM slices.

    Args:
        ct_directory: Path to directory containing CT DICOM files

    Returns:
        Tuple of (volume, spacing, origin) where:
            - volume: 3D numpy array with shape (slices, rows, cols)
            - spacing: (x, y, z) voxel spacing in mm
            - origin: (x, y, z) origin coordinates in mm

    Raises:
        FileNotFoundError: If directory doesn't exist or contains no DICOM files
        ValueError: If DICOM files don't form a valid CT series
    """
    ct_directory = Path(ct_directory)
    
    if not ct_directory.exists():
        raise FileNotFoundError(f"CT directory not found: {ct_directory}")
    
    # Get all DICOM files in directory
    dicom_files = sorted(ct_directory.glob("*.dcm"))
    
    if not dicom_files:
        raise FileNotFoundError(f"No DICOM files found in: {ct_directory}")
    
    # Use SimpleITK for robust volume reconstruction
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(str(ct_directory))
    
    if not dicom_names:
        raise ValueError(f"No valid DICOM series found in: {ct_directory}")
    
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    
    # Get volume as numpy array (SimpleITK uses (z, y, x) ordering)
    volume = sitk.GetArrayFromImage(image)
    
    # Get spacing and origin
    spacing = image.GetSpacing()  # (x, y, z)
    origin = image.GetOrigin()    # (x, y, z)
    
    return volume, spacing, origin


def read_dicom_rtdose(
    rtdose_file: Union[str, Path],
) -> Tuple[np.ndarray, Tuple[float, float, float], Tuple[float, float, float], float]:
    """
    Read an RTDOSE DICOM file.

    Args:
        rtdose_file: Path to RTDOSE DICOM file

    Returns:
        Tuple of (dose_array, spacing, origin, dose_scaling) where:
            - dose_array: 3D numpy array with dose values in Gy
            - spacing: (x, y, z) voxel spacing in mm
            - origin: (x, y, z) origin coordinates in mm
            - dose_scaling: Dose grid scaling factor

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is not a valid RTDOSE DICOM
    """
    rtdose_file = Path(rtdose_file)
    
    if not rtdose_file.exists():
        raise FileNotFoundError(f"RTDOSE file not found: {rtdose_file}")
    
    # Read with pydicom
    ds = pydicom.dcmread(rtdose_file)
    
    # Verify it's an RTDOSE file
    if ds.Modality != 'RTDOSE':
        raise ValueError(f"File is not RTDOSE, got modality: {ds.Modality}")
    
    # Get dose array
    dose_array = ds.pixel_array.astype(np.float32)
    
    # Apply dose grid scaling to get doses in Gy
    dose_scaling = float(ds.DoseGridScaling)
    dose_array = dose_array * dose_scaling
    
    # Get geometric information
    # DICOM uses (row, col) for pixel spacing, need to add slice thickness
    pixel_spacing = ds.PixelSpacing  # [row_spacing, col_spacing]
    
    # Get slice thickness or use grid frame offset vector
    if hasattr(ds, 'SliceThickness') and ds.SliceThickness is not None:
        slice_spacing = float(ds.SliceThickness)
    elif hasattr(ds, 'GridFrameOffsetVector') and len(ds.GridFrameOffsetVector) > 1:
        # Calculate from frame offset vector
        slice_spacing = abs(float(ds.GridFrameOffsetVector[1]) - float(ds.GridFrameOffsetVector[0]))
    else:
        slice_spacing = 1.0  # Default fallback
    
    # Spacing in (x, y, z) format
    spacing = (float(pixel_spacing[1]), float(pixel_spacing[0]), slice_spacing)
    
    # Get origin (ImagePositionPatient)
    if hasattr(ds, 'ImagePositionPatient'):
        origin = tuple(float(x) for x in ds.ImagePositionPatient)
    else:
        origin = (0.0, 0.0, 0.0)
    
    return dose_array, spacing, origin, dose_scaling


def read_dicom_rtstruct(
    rtstruct_file: Union[str, Path],
    reference_image: Optional[Union[sitk.Image, Tuple[Tuple[int, ...], Tuple[float, ...], Tuple[float, ...]]]] = None,
) -> Dict[str, Dict[str, Union[np.ndarray, List]]]:
    """
    Read an RTSTRUCT DICOM file and extract structure information.

    Args:
        rtstruct_file: Path to RTSTRUCT DICOM file
        reference_image: Optional reference image or (shape, spacing, origin) tuple for mask generation.
                        If None, only contour points are returned without generating binary masks.

    Returns:
        Dictionary mapping structure names to dictionaries containing:
            - 'contours': List of contour point arrays (each is Nx3 array of (x, y, z) points)
            - 'mask': Binary mask array (only if reference_image provided)
            - 'roi_number': ROI number from DICOM
            - 'color': RGB color tuple

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is not a valid RTSTRUCT DICOM
    """
    rtstruct_file = Path(rtstruct_file)
    
    if not rtstruct_file.exists():
        raise FileNotFoundError(f"RTSTRUCT file not found: {rtstruct_file}")
    
    # Read with pydicom
    ds = pydicom.dcmread(rtstruct_file)
    
    # Verify it's an RTSTRUCT file
    if ds.Modality != 'RTSTRUCT':
        raise ValueError(f"File is not RTSTRUCT, got modality: {ds.Modality}")
    
    structures = {}
    
    # Build ROI number to name mapping
    roi_dict = {}
    if hasattr(ds, 'StructureSetROISequence'):
        for roi in ds.StructureSetROISequence:
            roi_number = roi.ROINumber
            roi_name = roi.ROIName
            roi_dict[roi_number] = roi_name
    
    # Extract contours for each ROI
    if hasattr(ds, 'ROIContourSequence'):
        for roi_contour in ds.ROIContourSequence:
            roi_number = roi_contour.ReferencedROINumber
            
            if roi_number not in roi_dict:
                continue
            
            roi_name = roi_dict[roi_number]
            
            # Get color if available
            if hasattr(roi_contour, 'ROIDisplayColor'):
                color = tuple(int(c) for c in roi_contour.ROIDisplayColor)
            else:
                color = (255, 0, 0)  # Default red
            
            # Extract contour points
            contours = []
            if hasattr(roi_contour, 'ContourSequence'):
                for contour in roi_contour.ContourSequence:
                    if hasattr(contour, 'ContourData'):
                        # ContourData is a flat list of [x1, y1, z1, x2, y2, z2, ...]
                        points = np.array(contour.ContourData).reshape(-1, 3)
                        contours.append(points)
            
            structures[roi_name] = {
                'contours': contours,
                'roi_number': roi_number,
                'color': color,
            }
    
    # Generate binary masks if reference image provided
    if reference_image is not None:
        if isinstance(reference_image, sitk.Image):
            shape = reference_image.GetSize()[::-1]  # SimpleITK uses (x,y,z), numpy uses (z,y,x)
            spacing = reference_image.GetSpacing()
            origin = reference_image.GetOrigin()
        else:
            shape, spacing, origin = reference_image
        
        # Generate masks for each structure
        for roi_name, roi_data in structures.items():
            mask = _generate_mask_from_contours(
                roi_data['contours'],
                shape,
                spacing,
                origin
            )
            roi_data['mask'] = mask
    
    return structures


def _generate_mask_from_contours(
    contours: List[np.ndarray],
    shape: Tuple[int, ...],
    spacing: Tuple[float, float, float],
    origin: Tuple[float, float, float],
) -> np.ndarray:
    """
    Generate a binary mask from contour points.

    Args:
        contours: List of contour arrays (each Nx3 with (x, y, z) points)
        shape: (depth, height, width) of output mask
        spacing: (x, y, z) voxel spacing in mm
        origin: (x, y, z) origin in mm

    Returns:
        Binary mask as boolean numpy array with given shape
    """
    from skimage.draw import polygon
    
    mask = np.zeros(shape, dtype=bool)
    
    # Group contours by z-coordinate (slice)
    slice_contours = {}
    for contour in contours:
        if len(contour) < 3:  # Need at least 3 points for a polygon
            continue
        
        # Get z-coordinate (should be constant for a single contour)
        z_coord = contour[0, 2]
        
        # Convert z coordinate to slice index
        slice_idx = int(round((z_coord - origin[2]) / spacing[2]))
        
        if 0 <= slice_idx < shape[0]:
            if slice_idx not in slice_contours:
                slice_contours[slice_idx] = []
            slice_contours[slice_idx].append(contour)
    
    # Fill each slice
    for slice_idx, contour_list in slice_contours.items():
        for contour in contour_list:
            # Convert physical coordinates to pixel coordinates
            cols = (contour[:, 0] - origin[0]) / spacing[0]
            rows = (contour[:, 1] - origin[1]) / spacing[1]
            
            # Fill polygon
            try:
                rr, cc = polygon(rows, cols, shape[1:])
                # Ensure indices are within bounds
                valid_idx = (rr >= 0) & (rr < shape[1]) & (cc >= 0) & (cc < shape[2])
                mask[slice_idx, rr[valid_idx], cc[valid_idx]] = True
            except Exception:
                # Skip invalid contours
                continue
    
    return mask


def load_dicom_folder(
    folder_path: Union[str, Path],
    load_ct: bool = True,
    load_rtdose: bool = True,
    load_rtstruct: bool = True,
    return_as_structureset: bool = True,
    dose_file_name: Optional[str] = None,
    structure_type_mapping: Optional[Dict[str, "StructureType"]] = None,
) -> Union["StructureSet", Dict[str, Union[np.ndarray, Dict, Tuple]]]:
    """
    Load all DICOM data from a folder containing CT, RTDOSE, and RTSTRUCT files.

    This is a high-level function that automatically detects and loads all DICOM
    modalities present in the folder.

    Args:
        folder_path: Path to folder containing DICOM files organized in subfolders
                    (e.g., CT/, RTDOSE/, RTSTRUCT/)
        load_ct: Whether to load CT volume
        load_rtdose: Whether to load dose distributions
        load_rtstruct: Whether to load structure sets
        return_as_structureset: If True (default), returns a StructureSet object.
                               If False, returns raw dictionary.
        dose_file_name: Specific dose file to use (only if return_as_structureset=True)
        structure_type_mapping: Optional dict mapping structure names to StructureType
                               (only used if return_as_structureset=True)

    Returns:
        If return_as_structureset=True: StructureSet object with loaded data
        If return_as_structureset=False: Dictionary with keys:
            - 'ct_volume': CT volume array (if loaded)
            - 'ct_spacing': CT spacing tuple (if loaded)
            - 'ct_origin': CT origin tuple (if loaded)
            - 'dose_volumes': Dict of dose volumes {filename: (array, spacing, origin, scaling)}
            - 'structures': Dict of structures from RTSTRUCT
            - 'spacing': Common spacing for all data
            - 'origin': Common origin for all data

    Raises:
        FileNotFoundError: If folder doesn't exist
    """
    folder_path = Path(folder_path)
    
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    result = {}
    
    # Load CT volume
    if load_ct:
        ct_dir = folder_path / 'CT'
        if ct_dir.exists():
            try:
                ct_volume, ct_spacing, ct_origin = read_dicom_ct_volume(ct_dir)
                result['ct_volume'] = ct_volume
                result['ct_spacing'] = ct_spacing
                result['ct_origin'] = ct_origin
                result['spacing'] = ct_spacing
                result['origin'] = ct_origin
            except Exception as e:
                print(f"Warning: Could not load CT volume: {e}")
    
    # Load RTDOSE files
    if load_rtdose:
        rtdose_dir = folder_path / 'RTDOSE'
        if rtdose_dir.exists():
            dose_volumes = {}
            for dose_file in rtdose_dir.glob('*.dcm'):
                try:
                    dose_array, spacing, origin, scaling = read_dicom_rtdose(dose_file)
                    dose_volumes[dose_file.stem] = {
                        'array': dose_array,
                        'spacing': spacing,
                        'origin': origin,
                        'scaling': scaling,
                    }
                    # Use first dose for common spacing/origin if CT not available
                    if 'spacing' not in result:
                        result['spacing'] = spacing
                        result['origin'] = origin
                except Exception as e:
                    print(f"Warning: Could not load RTDOSE {dose_file.name}: {e}")
            
            if dose_volumes:
                result['dose_volumes'] = dose_volumes
    
    # Load RTSTRUCT files
    if load_rtstruct:
        rtstruct_dir = folder_path / 'RTSTRUCT'
        if rtstruct_dir.exists():
            # Use CT or dose as reference for mask generation
            reference = None
            if 'ct_volume' in result:
                reference = (
                    result['ct_volume'].shape,
                    result['ct_spacing'],
                    result['ct_origin']
                )
            elif 'dose_volumes' in result:
                first_dose = next(iter(result['dose_volumes'].values()))
                reference = (
                    first_dose['array'].shape,
                    first_dose['spacing'],
                    first_dose['origin']
                )
            
            for rtstruct_file in rtstruct_dir.glob('*.dcm'):
                try:
                    structures = read_dicom_rtstruct(rtstruct_file, reference)
                    result['structures'] = structures
                    break  # Usually only one RTSTRUCT file
                except Exception as e:
                    print(f"Warning: Could not load RTSTRUCT {rtstruct_file.name}: {e}")
    
    # Return as StructureSet if requested
    if return_as_structureset:
        if 'structures' not in result or not result['structures']:
            raise ValueError(f"No structures found in: {folder_path}")
        return create_structure_set_from_dicom(
            folder_path,
            dose_file_name=dose_file_name,
            structure_type_mapping=structure_type_mapping,
            name=f"DICOM - {folder_path.name if isinstance(folder_path, Path) else Path(folder_path).name}",
        )
    
    return result


def create_structure_set_from_dicom(
    folder_path: Union[str, Path],
    dose_file_name: Optional[str] = None,
    structure_type_mapping: Optional[Dict[str, "StructureType"]] = None,
    name: str = "DICOM StructureSet",
) -> "StructureSet":
    """
    Create a StructureSet object from DICOM data in a folder.

    This is a high-level convenience function that loads DICOM data and creates
    a complete StructureSet object.

    Args:
        folder_path: Path to folder containing DICOM subfolders
        dose_file_name: Specific dose file to use (e.g., 'RD_1'). If None, uses first found.
        structure_type_mapping: Optional dict mapping structure names to StructureType
        name: Name for the structure set

    Returns:
        StructureSet object with loaded structures and dose data

    Raises:
        FileNotFoundError: If folder doesn't exist
        ValueError: If no structures found
    """
    from ..structures import StructureType
    from ..structure_set import StructureSet
    
    # Load all DICOM data (get raw dict to avoid recursion)
    data = load_dicom_folder(folder_path, return_as_structureset=False)
    
    if 'structures' not in data or not data['structures']:
        raise ValueError(f"No structures found in: {folder_path}")
    
    # Get spacing and origin
    spacing = data.get('spacing', (1.0, 1.0, 1.0))
    origin = data.get('origin', (0.0, 0.0, 0.0))
    
    # Create structure set
    structure_set = StructureSet(spacing=spacing, origin=origin, name=name)
    
    # Add structures
    for struct_name, struct_data in data['structures'].items():
        if 'mask' not in struct_data:
            continue
        
        # Determine structure type
        struct_type = StructureType.OAR  # Default
        if structure_type_mapping and struct_name in structure_type_mapping:
            struct_type = structure_type_mapping[struct_name]
        elif 'PTV' in struct_name.upper() or 'GTV' in struct_name.upper() or 'CTV' in struct_name.upper():
            struct_type = StructureType.TARGET
        
        structure_set.add_structure(
            name=struct_name,
            mask=struct_data['mask'],
            structure_type=struct_type,
        )
    
    # Note: In the new architecture, dose is loaded separately using Dose.from_dicom()
    # and not attached to the StructureSet. Multiple dose files can be loaded independently.
    
    return structure_set
