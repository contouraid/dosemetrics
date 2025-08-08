"""
Structure Set classes for managing collections of radiotherapy structures.

This module provides classes to manage collections of radiotherapy structures,
representing complete structure sets similar to DICOM RTSS files. These classes
facilitate bulk operations, analysis, and organization of multiple structures.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Iterator
from glob import glob

from .structures import Structure, OAR, Target, AvoidanceStructure, StructureType
from .data_io import read_from_nifti


class StructureSet:
    """
    Collection of radiotherapy structures representing a complete structure set.

    Similar to a DICOM RTSS file, this class manages multiple structures
    (OARs, targets, avoidance regions) with common geometric properties
    and provides bulk operations for analysis and processing.

    Attributes:
        structures (Dict[str, Structure]): Dictionary mapping structure names to Structure objects
        spacing (Tuple[float, float, float]): Common voxel spacing for all structures
        origin (Tuple[float, float, float]): Common origin for all structures
        dose_data (np.ndarray, optional): Common dose distribution for all structures
    """

    def __init__(
        self,
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        name: str = "StructureSet",
    ):
        """
        Initialize an empty StructureSet.

        Args:
            spacing: Common voxel spacing in (x, y, z) mm
            origin: Common origin coordinates in mm
            name: Name identifier for this structure set
        """
        self.structures: Dict[str, Structure] = {}
        self.spacing = spacing
        self.origin = origin
        self.name = name
        self.dose_data: Optional[np.ndarray] = None

    def add_structure(
        self,
        name: str,
        mask: np.ndarray,
        structure_type: StructureType,
        structure_class: Optional[type] = None,
    ) -> Structure:
        """
        Add a structure to the set.

        Args:
            name: Name of the structure
            mask: 3D binary mask array
            structure_type: Type of structure (OAR, TARGET, etc.)
            structure_class: Specific structure class to use (defaults based on type)

        Returns:
            The created Structure object

        Raises:
            ValueError: If structure name already exists or mask dimensions are invalid
        """
        if name in self.structures:
            raise ValueError(f"Structure '{name}' already exists in the set")

        # Determine structure class if not specified
        if structure_class is None:
            if structure_type == StructureType.OAR:
                structure_class = OAR
            elif structure_type == StructureType.TARGET:
                structure_class = Target
            elif structure_type == StructureType.AVOIDANCE:
                structure_class = AvoidanceStructure
            else:
                # For SUPPORT, EXTERNAL, or custom types, use the base class with type override
                structure_class = type(
                    f"{structure_type.value.title()}Structure",
                    (Structure,),
                    {"structure_type": property(lambda self: structure_type)},
                )

        # Create structure instance
        structure = structure_class(
            name=name, mask=mask, spacing=self.spacing, origin=self.origin
        )

        # Set dose data if available
        if self.dose_data is not None:
            structure.set_dose_data(self.dose_data)

        self.structures[name] = structure
        return structure

    def remove_structure(self, name: str) -> None:
        """Remove a structure from the set."""
        if name not in self.structures:
            raise ValueError(f"Structure '{name}' not found in the set")
        del self.structures[name]

    def get_structure(self, name: str) -> Structure:
        """Get a structure by name."""
        if name not in self.structures:
            raise ValueError(f"Structure '{name}' not found in the set")
        return self.structures[name]

    def set_dose_data(self, dose_data: np.ndarray) -> None:
        """
        Set dose data for all structures in the set.

        Args:
            dose_data: 3D dose array
        """
        self.dose_data = np.asarray(dose_data)

        # Propagate to all existing structures
        for structure in self.structures.values():
            structure.set_dose_data(self.dose_data)

    def get_structures_by_type(
        self, structure_type: StructureType
    ) -> Dict[str, Structure]:
        """Get all structures of a specific type."""
        return {
            name: struct
            for name, struct in self.structures.items()
            if struct.structure_type == structure_type
        }

    def get_oars(self) -> Dict[str, OAR]:
        """Get all OAR structures."""
        return self.get_structures_by_type(StructureType.OAR)

    def get_targets(self) -> Dict[str, Target]:
        """Get all target structures."""
        return self.get_structures_by_type(StructureType.TARGET)

    def get_avoidance_structures(self) -> Dict[str, AvoidanceStructure]:
        """Get all avoidance structures."""
        return self.get_structures_by_type(StructureType.AVOIDANCE)

    @property
    def structure_names(self) -> List[str]:
        """Get list of all structure names."""
        return list(self.structures.keys())

    @property
    def oar_names(self) -> List[str]:
        """Get list of OAR structure names."""
        return list(self.get_oars().keys())

    @property
    def target_names(self) -> List[str]:
        """Get list of target structure names."""
        return list(self.get_targets().keys())

    @property
    def structure_count(self) -> int:
        """Get total number of structures."""
        return len(self.structures)

    @property
    def has_dose(self) -> bool:
        """Check if dose data is available."""
        return self.dose_data is not None

    def total_volume_cc(self) -> float:
        """Calculate total volume of all structures in cc."""
        return sum(struct.volume_cc() for struct in self.structures.values())

    def compute_bulk_dvh(
        self, max_dose: Optional[float] = None, step_size: float = 0.1
    ) -> pd.DataFrame:
        """
        Compute DVH for all structures and return as DataFrame.

        Args:
            max_dose: Maximum dose for DVH bins
            step_size: Bin width in Gy

        Returns:
            DataFrame with dose bins and volume percentages for each structure
        """
        if not self.has_dose:
            raise ValueError("No dose data available for DVH computation")

        # Determine max_dose if not provided
        if max_dose is None:
            max_dose = float(np.max(self.dose_data))

        bins = np.arange(0, max_dose + step_size, step_size)
        dvh_data = {"Dose": bins}

        # Compute DVH for each structure
        for name, structure in self.structures.items():
            _, volumes = structure.dvh(max_dose, step_size)
            dvh_data[name] = volumes

        return pd.DataFrame(dvh_data)

    def dose_statistics_summary(self) -> pd.DataFrame:
        """
        Generate summary statistics for all structures.

        Returns:
            DataFrame with dose statistics for each structure
        """
        if not self.has_dose:
            raise ValueError("No dose data available for statistics")

        stats_data = []
        for name, structure in self.structures.items():
            stats = {
                "Structure": name,
                "Type": structure.structure_type.value.upper(),
                "Volume_cc": structure.volume_cc(),
                "Mean_Dose_Gy": structure.mean_dose(),
                "Max_Dose_Gy": structure.max_dose(),
                "Min_Dose_Gy": structure.min_dose(),
                "D95_Gy": structure.percentile_dose(95),
                "D05_Gy": structure.percentile_dose(5),
            }
            stats_data.append(stats)

        return pd.DataFrame(stats_data)

    def compliance_check(self, constraints: Dict[str, Dict]) -> pd.DataFrame:
        """
        Check dose constraints for all applicable structures.

        Args:
            constraints: Dictionary mapping structure names to constraint dictionaries
                        e.g., {"Brainstem": {"max_dose": 54, "unit": "Gy"}}

        Returns:
            DataFrame with compliance results
        """
        if not self.has_dose:
            raise ValueError("No dose data available for compliance checking")

        results = []
        for struct_name, constraint in constraints.items():
            if struct_name not in self.structures:
                continue

            structure = self.structures[struct_name]
            result = {
                "Structure": struct_name,
                "Constraint_Type": None,
                "Constraint_Value": None,
                "Actual_Value": None,
                "Compliant": None,
                "Difference": None,
            }

            if "max_dose" in constraint:
                max_dose = structure.max_dose()
                limit = constraint["max_dose"]
                result.update(
                    {
                        "Constraint_Type": "Max Dose",
                        "Constraint_Value": limit,
                        "Actual_Value": max_dose,
                        "Compliant": (
                            max_dose <= limit if max_dose is not None else None
                        ),
                        "Difference": (
                            max_dose - limit if max_dose is not None else None
                        ),
                    }
                )
            elif "mean_dose" in constraint:
                mean_dose = structure.mean_dose()
                limit = constraint["mean_dose"]
                result.update(
                    {
                        "Constraint_Type": "Mean Dose",
                        "Constraint_Value": limit,
                        "Actual_Value": mean_dose,
                        "Compliant": (
                            mean_dose <= limit if mean_dose is not None else None
                        ),
                        "Difference": (
                            mean_dose - limit if mean_dose is not None else None
                        ),
                    }
                )

            results.append(result)

        return pd.DataFrame(results)

    def geometric_summary(self) -> pd.DataFrame:
        """
        Generate geometric summary for all structures.

        Returns:
            DataFrame with geometric properties of each structure
        """
        geom_data = []
        for name, structure in self.structures.items():
            centroid = structure.centroid()
            bbox = structure.bounding_box()

            geom = {
                "Structure": name,
                "Type": structure.structure_type.value.upper(),
                "Volume_cc": structure.volume_cc(),
                "Volume_voxels": structure.volume_voxels(),
                "Centroid_X": centroid[0] if centroid else None,
                "Centroid_Y": centroid[1] if centroid else None,
                "Centroid_Z": centroid[2] if centroid else None,
                "BBox_X_Range": f"{bbox[0][0]}-{bbox[0][1]}" if bbox else None,
                "BBox_Y_Range": f"{bbox[1][0]}-{bbox[1][1]}" if bbox else None,
                "BBox_Z_Range": f"{bbox[2][0]}-{bbox[2][1]}" if bbox else None,
            }
            geom_data.append(geom)

        return pd.DataFrame(geom_data)

    def __len__(self) -> int:
        """Return number of structures in the set."""
        return len(self.structures)

    def __iter__(self) -> Iterator[Tuple[str, Structure]]:
        """Iterate over structure name-object pairs."""
        return iter(self.structures.items())

    def __getitem__(self, name: str) -> Structure:
        """Access structure by name using bracket notation."""
        return self.get_structure(name)

    def __contains__(self, name: str) -> bool:
        """Check if structure name exists in the set."""
        return name in self.structures

    def __str__(self) -> str:
        """String representation of the structure set."""
        oar_count = len(self.get_oars())
        target_count = len(self.get_targets())
        total_volume = self.total_volume_cc()
        dose_status = "with dose" if self.has_dose else "without dose"

        return (
            f"StructureSet '{self.name}': {self.structure_count} structures "
            f"({target_count} targets, {oar_count} OARs) "
            f"- Total volume: {total_volume:.1f} cc ({dose_status})"
        )

    def __repr__(self) -> str:
        """Detailed representation of the structure set."""
        return (
            f"StructureSet(name='{self.name}', "
            f"structures={self.structure_count}, "
            f"has_dose={self.has_dose}, "
            f"spacing={self.spacing}, "
            f"origin={self.origin})"
        )


def create_structure_set_from_folder(
    data_path: str,
    dose_file: str = "Dose.nii.gz",
    contents_file: Optional[str] = None,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    name: Optional[str] = None,
) -> StructureSet:
    """
    Create a StructureSet from a folder containing NIfTI files.

    Args:
        data_path: Path to folder containing structure files
        dose_file: Name of dose file (default: "Dose.nii.gz")
        contents_file: Optional CSV file specifying structure types
        spacing: Voxel spacing in mm
        origin: Origin coordinates in mm
        name: Name for the structure set (defaults to folder name)

    Returns:
        StructureSet with loaded structures and dose data
    """
    if name is None:
        name = os.path.basename(os.path.normpath(data_path))

    structure_set = StructureSet(spacing=spacing, origin=origin, name=name)

    # Load dose data if available
    dose_path = os.path.join(data_path, dose_file)
    if os.path.exists(dose_path):
        dose_volume = read_from_nifti(dose_path)
        structure_set.set_dose_data(dose_volume)

    # Look for contents CSV file
    structure_info = {}
    if contents_file is None:
        contents_files = glob(os.path.join(data_path, "*.csv"))
        if contents_files:
            contents_file = contents_files[0]  # Use first CSV found

    if contents_file and os.path.exists(contents_file):
        df = pd.read_csv(contents_file)
        if "Structure" in df.columns and "Type" in df.columns:
            for _, row in df.iterrows():
                struct_name = row["Structure"]
                struct_type = row["Type"]
                if struct_type in ["Target", "OAR", "Avoidance"]:
                    # Map string types to StructureType enum
                    if struct_type == "Target":
                        structure_info[struct_name] = StructureType.TARGET
                    elif struct_type == "OAR":
                        structure_info[struct_name] = StructureType.OAR
                    elif struct_type == "Avoidance":
                        structure_info[struct_name] = StructureType.AVOIDANCE

    # Find all NIfTI structure files
    nifti_files = glob(os.path.join(data_path, "*.nii.gz"))
    structure_files = [f for f in nifti_files if not f.endswith(dose_file)]

    # Load structures
    for struct_file in structure_files:
        struct_name = os.path.basename(struct_file).replace(".nii.gz", "")

        # Skip dose file
        if struct_name == "Dose":
            continue

        # Load mask
        mask_volume = read_from_nifti(struct_file)

        # Determine structure type
        if struct_name in structure_info:
            struct_type = structure_info[struct_name]
        else:
            # Default assignment based on naming conventions
            struct_name_lower = struct_name.lower()
            if any(
                term in struct_name_lower for term in ["ptv", "ctv", "gtv", "target"]
            ):
                struct_type = StructureType.TARGET
            else:
                struct_type = StructureType.OAR  # Default to OAR

        # Add to structure set
        structure_set.add_structure(struct_name, mask_volume, struct_type)

    return structure_set


def create_structure_set_from_masks(
    structure_masks: Dict[str, np.ndarray],
    dose_volume: Optional[np.ndarray] = None,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    structure_types: Optional[Dict[str, StructureType]] = None,
    name: str = "StructureSet",
) -> StructureSet:
    """
    Create a StructureSet from dictionaries of masks and optional dose data.

    Args:
        structure_masks: Dictionary mapping structure names to 3D mask arrays
        dose_volume: Optional 3D dose array
        spacing: Voxel spacing in mm
        origin: Origin coordinates in mm
        structure_types: Optional mapping of structure names to types
        name: Name for the structure set

    Returns:
        StructureSet with loaded structures and dose data
    """
    structure_set = StructureSet(spacing=spacing, origin=origin, name=name)

    # Set dose data if provided
    if dose_volume is not None:
        structure_set.set_dose_data(dose_volume)

    # Add structures
    for struct_name, mask in structure_masks.items():
        # Determine structure type
        if structure_types and struct_name in structure_types:
            struct_type = structure_types[struct_name]
        else:
            # Default assignment based on naming conventions
            struct_name_lower = struct_name.lower()
            if any(
                term in struct_name_lower for term in ["ptv", "ctv", "gtv", "target"]
            ):
                struct_type = StructureType.TARGET
            else:
                struct_type = StructureType.OAR  # Default to OAR

        structure_set.add_structure(struct_name, mask, struct_type)

    return structure_set
