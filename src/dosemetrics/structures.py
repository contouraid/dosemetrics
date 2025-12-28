"""
Radiotherapy structure classes for anatomical regions of interest.

This module provides core data structures to represent radiotherapy structures from
RTSS DICOM files or NIfTI files. These structures are 3D volumes with binary masks
representing anatomical regions of interest such as organs at risk (OARs), target
volumes, and avoidance regions.

Structures represent pure geometry. Dose analysis is performed by combining Structure
objects with Dose objects using the dosemetrics.dose module.
"""

import numpy as np
from typing import Optional, Tuple
from abc import ABC, abstractmethod
from enum import Enum


class StructureType(Enum):
    """Enumeration for different types of radiotherapy structures."""

    OAR = "oar"  # Organ at Risk
    TARGET = "target"  # Target volume (PTV, CTV, etc.)
    AVOIDANCE = "avoidance"  # Avoidance structure
    SUPPORT = "support"  # Support structure
    EXTERNAL = "external"  # External contour


class Structure(ABC):
    """
    Base class for radiotherapy structures.

    Represents a 3D anatomical structure derived from RTSS DICOM files or
    equivalent NIfTI masks. Contains geometric information only - dose analysis
    is performed by combining with Dose objects.

    Attributes:
        name (str): Name/identifier of the structure
        mask (np.ndarray): 3D binary mask array (boolean)
        spacing (Tuple[float, float, float]): Voxel spacing in (x, y, z) mm
        origin (Tuple[float, float, float]): Origin coordinates in mm
    
    Examples:
        >>> # Create a structure
        >>> ptv = Target(name="PTV", mask=mask_array, spacing=(1.0, 1.0, 3.0))
        >>> 
        >>> # Get volume
        >>> volume_cc = ptv.volume_cc()
        >>> 
        >>> # Compute dose statistics (using Dose object)
        >>> from dosemetrics.dose import Dose
        >>> dose = Dose.from_dicom("rtdose.dcm")
        >>> stats = dose.compute_statistics(ptv)
    """

    def __init__(
        self,
        name: str,
        mask: Optional[np.ndarray] = None,
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        """
        Initialize a Structure.

        Args:
            name: Name/identifier of the structure
            mask: 3D binary mask array (will be converted to bool)
            spacing: Voxel spacing in (x, y, z) mm
            origin: Origin coordinates in mm
        """
        self.name = name
        self.spacing = tuple(spacing)
        self.origin = tuple(origin)

        # Set and validate mask
        if mask is not None:
            self.set_mask(mask)
        else:
            self._mask = None

    def set_mask(self, mask: np.ndarray) -> None:
        """
        Set the binary mask for this structure.

        Args:
            mask: 3D array that will be converted to binary mask

        Raises:
            ValueError: If mask is not 3D
        """
        mask_array = np.asarray(mask)
        if mask_array.ndim != 3:
            raise ValueError(f"Mask must be 3D, got {mask_array.ndim}D")

        # Convert to boolean mask
        self._mask = mask_array.astype(bool)

    @property
    def mask(self) -> Optional[np.ndarray]:
        """Get the binary mask array."""
        return self._mask

    @property
    @abstractmethod
    def structure_type(self) -> StructureType:
        """Return the type of this structure."""
        pass

    @property
    def has_mask(self) -> bool:
        """Check if structure has a valid mask."""
        return self._mask is not None

    def volume_voxels(self) -> int:
        """
        Get structure volume in voxels.

        Returns:
            Number of voxels in the structure (sum of mask)
        """
        if not self.has_mask or self._mask is None:
            return 0
        return int(np.sum(self._mask))

    def volume_cc(self) -> float:
        """
        Get structure volume in cubic centimeters.

        Returns:
            Volume in cc (considering voxel spacing)
        """
        voxel_volume_mm3 = np.prod(self.spacing)  # mm³
        voxel_volume_cc = float(voxel_volume_mm3 / 1000.0)  # Convert mm³ to cc
        return float(self.volume_voxels() * voxel_volume_cc)

    def centroid(self) -> Optional[Tuple[float, float, float]]:
        """
        Calculate the centroid of the structure in world coordinates.

        Returns:
            Tuple of (x, y, z) coordinates in mm, or None if no mask
        """
        if not self.has_mask or self._mask is None:
            return None

        # Get indices of mask voxels
        mask_indices = np.where(self._mask)

        if len(mask_indices[0]) == 0:
            return None

        # Calculate centroid in voxel coordinates as mean of all voxel indices
        centroid_voxel = [
            float(np.mean(indices))
            for indices in mask_indices
        ]

        # Convert to world coordinates
        centroid_world = [
            float(self.origin[i] + centroid_voxel[i] * self.spacing[i])
            for i in range(3)
        ]

        return (centroid_world[0], centroid_world[1], centroid_world[2])

    def bounding_box(
        self,
    ) -> Optional[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]]:
        """
        Get bounding box of the structure in voxel coordinates.

        Returns:
            Tuple of ((min_x, max_x), (min_y, max_y), (min_z, max_z)), or None if no mask
        """
        if not self.has_mask or self._mask is None:
            return None

        mask_indices = np.where(self._mask)

        if len(mask_indices[0]) == 0:
            return None

        bounds = []
        for i in range(3):
            min_idx = int(np.min(mask_indices[i]))
            max_idx = int(np.max(mask_indices[i]))
            bounds.append((min_idx, max_idx))

        return (bounds[0], bounds[1], bounds[2])

    def __str__(self) -> str:
        """String representation of the structure."""
        volume_cc = self.volume_cc() if self.has_mask else 0
        return (
            f"{self.structure_type.value.upper()}: {self.name} "
            f"(Volume: {volume_cc:.2f} cc)"
        )

    def __repr__(self) -> str:
        """Detailed representation of the structure."""
        return (
            f"{self.__class__.__name__}(name='{self.name}', "
            f"type={self.structure_type.value}, "
            f"has_mask={self.has_mask}, "
            f"volume_cc={self.volume_cc():.2f})"
        )


class OAR(Structure):
    """
    Organ at Risk (OAR) structure.

    Represents critical normal organs that should receive limited radiation dose
    to avoid complications (e.g., spinal cord, eyes, heart, brainstem).
    """

    @property
    def structure_type(self) -> StructureType:
        """Return OAR structure type."""
        return StructureType.OAR


class Target(Structure):
    """
    Target volume structure.

    Represents volumes that should receive the prescribed radiation dose
    (e.g., PTV, CTV, GTV).
    """

    @property
    def structure_type(self) -> StructureType:
        """Return Target structure type."""
        return StructureType.TARGET


class AvoidanceStructure(Structure):
    """
    Avoidance structure.

    Represents regions where dose should be minimized during planning
    (e.g., critical OAR expansions, sensitive areas).
    """

    @property
    def structure_type(self) -> StructureType:
        """Return Avoidance structure type."""
        return StructureType.AVOIDANCE
