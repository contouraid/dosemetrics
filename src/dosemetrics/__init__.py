"""
Dosemetrics: A library for radiotherapy dose analysis.

This library provides tools for:
- Dose distribution analysis
- Structure set management
- DVH computation and analysis
- Quality metrics (conformity, homogeneity)
- Geometric comparison
- Compliance checking
- Visualization utilities

Architecture:
- dosemetrics.dose: Dose data container
- dosemetrics.structures: Structure/OAR/Target classes
- dosemetrics.structure_set: StructureSet management
- dosemetrics.metrics: All computational metrics
  - dvh: DVH computation and queries
  - statistics: Dose statistics
  - conformity: Conformity indices
  - homogeneity: Homogeneity indices
  - geometric: Geometric comparisons
- dosemetrics.io: Data loading/saving
- dosemetrics.utils: Utilities (plotting, compliance, batch processing)
"""

# Core data classes
from .dose import Dose
from .structures import (
    Structure,
    OAR,
    Target,
    StructureType,
    AvoidanceStructure,
)
from .structure_set import StructureSet
from .utils.compliance import quality_index, get_default_constraints, check_compliance
import numpy as np

# Metrics subpackage (use: from dosemetrics.metrics import dvh, conformity, etc.)
from . import metrics

# I/O utilities
from .io import (
    load_from_folder,
    load_structure_set,
    load_volume,
    load_structure,
    detect_folder_format,
    # Format-specific modules
    dicom_io,
    nifti_io,
)

# Utilities (plotting, compliance, batch)
from . import utils


def create_structure_set_from_masks(
    structure_masks: dict,
    spacing: tuple = (1.0, 1.0, 1.0),
    origin: tuple = (0.0, 0.0, 0.0),
    structure_types: dict = None,
    dose_volume: np.ndarray = None,
    name: str = "StructureSet",
):
    """Create a StructureSet from mask dictionaries.

    Args:
        structure_masks: Dict mapping structure names to binary masks
        spacing: Voxel spacing in mm (x, y, z)
        origin: Origin coordinates in mm
        structure_types: Optional dict mapping structure names to StructureType
        dose_volume: Optional dose array to attach
        name: Name for the structure set

    Returns:
        StructureSet with the structures and optionally dose attached
    """
    ss = StructureSet(spacing=spacing, origin=origin, name=name)
    for struct_name, mask in structure_masks.items():
        stype = None
        if structure_types:
            stype = structure_types.get(struct_name)
            if isinstance(stype, str):
                stype = (
                    StructureType(stype.lower())
                    if stype.lower() in StructureType._value2member_map_
                    else StructureType.OAR
                )
        if stype is None:
            stype = StructureType.OAR
        ss.add_structure(struct_name, mask, stype)
    # Note: dose_volume parameter is ignored - use Dose objects for dose analysis
    return ss


# Version information
__version__ = "0.3.0"

# Public API
__all__ = [
    # Version
    "__version__",
    # Core data classes
    "Dose",
    "Structure",
    "OAR",
    "Target",
    "StructureType",
    "AvoidanceStructure",
    "StructureSet",
    # Helper functions
    "create_structure_set_from_masks",
    # Metrics subpackage (access via metrics.dvh, metrics.conformity, etc.)
    "metrics",
    # I/O subpackage
    "load_from_folder",
    "load_structure_set",
    "load_volume",
    "load_structure",
    "detect_folder_format",
    "dicom_io",
    "nifti_io",
    # Utils subpackage (access via utils.compliance, utils.plot, etc.)
    "utils",
    # Convenience exports from utils
    "check_compliance",
    "from_dataframe",
    "quality_index",
    "get_default_constraints",
    "compare_dvh",
    "generate_dvh_variations",
    "plot_dvh",
    "plot_dvh_variations",
    "variability",
]
