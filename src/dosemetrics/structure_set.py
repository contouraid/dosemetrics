"""
Structure Set classes for managing collections of radiotherapy structures.

This module provides classes to manage collections of radiotherapy structures,
representing complete structure sets similar to DICOM RTSS files. These classes
facilitate bulk operations, organization of multiple structures, and geometric analysis.

StructureSets contain only geometric information. For dose analysis, combine a
StructureSet with a Dose object using the dosemetrics.dose module.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Iterator

from .structures import Structure, OAR, Target, AvoidanceStructure, StructureType
from .dose import Dose
from .metrics import dvh as dvh_metrics


class StructureSet:
    """
    Collection of radiotherapy structures representing a complete structure set.

    Similar to a DICOM RTSS file, this class manages multiple structures
    (OARs, targets, avoidance regions) with common geometric properties.
    Contains only geometric information - dose analysis is performed separately
    by combining with Dose objects.

    Attributes:
        structures (Dict[str, Structure]): Dictionary mapping structure names to Structure objects
        spacing (Tuple[float, float, float]): Common voxel spacing for all structures
        origin (Tuple[float, float, float]): Common origin for all structures
        name (str): Identifier for this structure set
    
    Examples:
        >>> # Create a structure set
        >>> structure_set = StructureSet(spacing=(1.0, 1.0, 3.0), name="Patient001")
        >>> 
        >>> # Add structures
        >>> structure_set.add_structure("PTV", ptv_mask, StructureType.TARGET)
        >>> structure_set.add_structure("Brainstem", brain_mask, StructureType.OAR)
        >>> 
        >>> # Access structures
        >>> ptv = structure_set.get_structure("PTV")
        >>> print(f"PTV volume: {ptv.volume_cc():.2f} cc")
        >>> 
        >>> # For dose analysis, combine with Dose object
        >>> from dosemetrics.dose import Dose
        >>> dose = Dose.from_dicom("rtdose.dcm")
        >>> stats = dose.compute_statistics(ptv)
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
        self.spacing = tuple(spacing)
        self.origin = tuple(origin)
        self.name = name

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
                # For SUPPORT, EXTERNAL, or custom types, create dynamic class
                structure_class = type(
                    f"{structure_type.value.title()}Structure",
                    (Structure,),
                    {"structure_type": property(lambda self: structure_type)},
                )

        # Create structure instance
        structure = structure_class(
            name=name, mask=mask, spacing=self.spacing, origin=self.origin
        )

        self.structures[name] = structure
        return structure

    def remove_structure(self, name: str) -> None:
        """
        Remove a structure from the set.
        
        Args:
            name: Name of the structure to remove
            
        Raises:
            ValueError: If structure name not found
        """
        if name not in self.structures:
            raise ValueError(f"Structure '{name}' not found in the set")
        del self.structures[name]

    def add_structure_object(self, structure: Structure) -> Structure:
        """
        Add an existing `Structure` instance to the set.

        Convenience wrapper to support tests and workflows that create
        `Structure` objects independently and then attach them to a `StructureSet`.

        Args:
            structure: A `Structure` instance to add.

        Returns:
            The same `Structure` instance after being added to the set.

        Raises:
            ValueError: If a structure with the same name already exists,
                       or if spacing/origin are incompatible with the set.
        """
        name = structure.name
        if name in self.structures:
            raise ValueError(f"Structure '{name}' already exists in the set")

        if tuple(structure.spacing) != tuple(self.spacing):
            raise ValueError("Structure spacing incompatible with StructureSet")
        if tuple(structure.origin) != tuple(self.origin):
            raise ValueError("Structure origin incompatible with StructureSet")

        self.structures[name] = structure
        return structure

    def get_structure(self, name: str) -> Structure:
        """
        Get a structure by name.
        
        Args:
            name: Name of the structure
            
        Returns:
            Structure object
            
        Raises:
            ValueError: If structure name not found
        """
        if name not in self.structures:
            raise ValueError(f"Structure '{name}' not found in the set")
        return self.structures[name]

    def get_structures_by_type(
        self, structure_type: StructureType
    ) -> Dict[str, Structure]:
        """
        Get all structures of a specific type.
        
        Args:
            structure_type: Type to filter by
            
        Returns:
            Dictionary of structures matching the type
        """
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

    def total_volume_cc(self) -> float:
        """
        Calculate total volume of all structures in cc.
        
        Returns:
            Sum of all structure volumes in cubic centimeters
        """
        return sum(struct.volume_cc() for struct in self.structures.values())

    def geometric_summary(self) -> pd.DataFrame:
        """
        Generate geometric summary for all structures.

        Returns:
            DataFrame with geometric properties of each structure including:
            - Structure name and type
            - Volume in cc and voxels
            - Centroid coordinates
            - Bounding box ranges
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
                "Centroid_X": centroid[0] if centroid is not None else None,
                "Centroid_Y": centroid[1] if centroid is not None else None,
                "Centroid_Z": centroid[2] if centroid is not None else None,
                "BBox_X_Range": f"{bbox[0][0]}-{bbox[0][1]}" if bbox is not None else None,
                "BBox_Y_Range": f"{bbox[1][0]}-{bbox[1][1]}" if bbox is not None else None,
                "BBox_Z_Range": f"{bbox[2][0]}-{bbox[2][1]}" if bbox is not None else None,
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

        return (
            f"StructureSet '{self.name}': {self.structure_count} structures "
            f"({target_count} targets, {oar_count} OARs) "
            f"- Total volume: {total_volume:.1f} cc"
        )

    def __repr__(self) -> str:
        """Detailed representation of the structure set."""
        return (
            f"StructureSet(name='{self.name}', "
            f"structures={self.structure_count}, "
            f"spacing={self.spacing}, "
            f"origin={self.origin})"
        )
