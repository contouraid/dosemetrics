"""
Data management utilities for radiotherapy dose and structure data.

This module provides:
- I/O utilities for reading and writing dose and mask data
- Structure classes for managing radiotherapy structures
- StructureSet for unified structure management
"""

# Core structure classes
from .structures import (
    Structure,
    OAR,
    Target,
    StructureType,
    AvoidanceStructure,
)

# Structure set management
from .structure_set import (
    StructureSet,
    create_structure_set_from_folder,
    create_structure_set_from_masks,
)

# I/O utilities
from .data_io import (
    read_file,
    read_byte_data,
    read_dose,
    read_masks,
    read_from_eclipse,
    read_dose_and_mask_files,
    find_all_files,
    read_from_nifti,
    get_dose,
    get_structures,
    read_dose_and_mask_files_as_structure_set,
    get_dose_and_structures_as_structure_set,
    create_structure_set_from_existing_data,
)

__all__ = [
    # Structure classes
    "Structure",
    "OAR",
    "Target",
    "StructureType",
    "AvoidanceStructure",
    # Structure set management
    "StructureSet",
    "create_structure_set_from_folder",
    "create_structure_set_from_masks",
    # I/O functions
    "read_file",
    "read_byte_data",
    "read_dose",
    "read_masks",
    "read_from_eclipse",
    "read_dose_and_mask_files",
    "find_all_files",
    "read_from_nifti",
    "get_dose",
    "get_structures",
    "read_dose_and_mask_files_as_structure_set",
    "get_dose_and_structures_as_structure_set",
    "create_structure_set_from_existing_data",
]
