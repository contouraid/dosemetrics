"""
Radiotherapy structure classes for anatomical regions of interest.

This module provides core data structures to represent radiotherapy structures from
RTSS DICOM files or NIfTI files derived from them. These structures are 3D volumes
with binary masks representing anatomical regions of interest such as organs at risk
(OARs), target volumes, and avoidance regions.

The classes in this module serve as fundamental building blocks for dose analysis,
DVH computation, geometric calculations, and treatment plan evaluation.
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
    equivalent NIfTI masks. Contains both geometric (mask) and dosimetric
    information when dose data is available.

    Attributes:
        name (str): Name/identifier of the structure
        mask (np.ndarray): 3D binary mask array (0 or 1 values)
        spacing (Tuple[float, float, float]): Voxel spacing in (x, y, z) mm
        origin (Tuple[float, float, float]): Origin coordinates in mm
        dose_data (np.ndarray, optional): 3D dose array corresponding to mask
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
        self.spacing = spacing
        self.origin = origin
        self.dose_data: Optional[np.ndarray] = None

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

    @property
    def has_dose(self) -> bool:
        """Check if structure has dose data."""
        return self.dose_data is not None

    def set_dose_data(self, dose: np.ndarray) -> None:
        """
        Set dose data for this structure.

        Args:
            dose: 3D dose array matching mask dimensions

        Raises:
            ValueError: If dose dimensions don't match mask
        """
        if self._mask is not None and dose.shape != self._mask.shape:
            raise ValueError(
                f"Dose shape {dose.shape} doesn't match mask shape {self._mask.shape}"
            )
        self.dose_data = np.asarray(dose)

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

    def get_dose_in_structure(self) -> Optional[np.ndarray]:
        """
        Get dose values within the structure mask.

        Returns:
            1D array of dose values inside the structure, or None if no dose/mask
        """
        if (
            not self.has_mask
            or not self.has_dose
            or self._mask is None
            or self.dose_data is None
        ):
            return None
        return self.dose_data[self._mask]

    def mean_dose(self) -> Optional[float]:
        """
        Calculate mean dose in the structure.

        Returns:
            Mean dose in Gy, or None if no dose data available
        """
        dose_values = self.get_dose_in_structure()
        if dose_values is None or len(dose_values) == 0:
            return None
        return float(np.mean(dose_values))

    def max_dose(self) -> Optional[float]:
        """
        Calculate maximum dose in the structure.

        Returns:
            Maximum dose in Gy, or None if no dose data available
        """
        dose_values = self.get_dose_in_structure()
        if dose_values is None or len(dose_values) == 0:
            return None
        return float(np.max(dose_values))

    def min_dose(self) -> Optional[float]:
        """
        Calculate minimum dose in the structure.

        Returns:
            Minimum dose in Gy, or None if no dose data available
        """
        dose_values = self.get_dose_in_structure()
        if dose_values is None or len(dose_values) == 0:
            return None
        return float(np.min(dose_values))

    def percentile_dose(self, percentile: float) -> Optional[float]:
        """
        Calculate dose at given percentile.

        Args:
            percentile: Percentile value (0-100)

        Returns:
            Dose at percentile in Gy, or None if no dose data available
        """
        dose_values = self.get_dose_in_structure()
        if dose_values is None or len(dose_values) == 0:
            return None
        return float(np.percentile(dose_values, percentile))

    def dvh(
        self, max_dose: Optional[float] = None, step_size: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute dose-volume histogram for the structure.

        Args:
            max_dose: Maximum dose for DVH bins (uses structure max if None)
            step_size: Bin width in Gy

        Returns:
            Tuple of (dose_bins, volume_percentages)
        """
        dose_values = self.get_dose_in_structure()

        if dose_values is None or len(dose_values) == 0:
            # Return empty DVH if no dose data
            bins = np.arange(0, max_dose or 70, step_size)
            return bins, np.zeros(len(bins))

        if max_dose is None:
            max_dose = float(np.max(dose_values))

        bins = np.arange(0, max_dose + step_size, step_size)
        total_voxels = len(dose_values)

        # Calculate cumulative volume percentages
        volume_percentages = []
        for dose_threshold in bins:
            voxels_above_threshold = np.sum(dose_values >= dose_threshold)
            percentage = (voxels_above_threshold / total_voxels) * 100
            volume_percentages.append(percentage)

        return bins, np.array(volume_percentages)

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

        # Calculate centroid in voxel coordinates
        centroid_voxel = [float(np.mean(indices)) for indices in mask_indices]

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
        dose_info = ""
        if self.has_dose:
            mean_d = self.mean_dose()
            max_d = self.max_dose()
            dose_info = f", Mean dose: {mean_d:.2f} Gy, Max dose: {max_d:.2f} Gy"

        return f"{self.structure_type.value.upper()}: {self.name} (Volume: {volume_cc:.2f} cc{dose_info})"

    def __repr__(self) -> str:
        """Detailed representation of the structure."""
        return (
            f"{self.__class__.__name__}(name='{self.name}', "
            f"type={self.structure_type.value}, "
            f"has_mask={self.has_mask}, "
            f"has_dose={self.has_dose}, "
            f"volume_cc={self.volume_cc():.2f})"
        )


class OAR(Structure):
    """
    Organ at Risk (OAR) structure.

    Represents critical normal organs that should receive limited radiation dose
    to avoid complications (e.g., spinal cord, eyes, heart).
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

    def coverage_volume_percentage(self, dose_threshold: float) -> Optional[float]:
        """
        Calculate percentage of target volume receiving at least the threshold dose.

        Args:
            dose_threshold: Dose threshold in Gy

        Returns:
            Percentage of volume above threshold, or None if no dose data
        """
        dose_values = self.get_dose_in_structure()
        if dose_values is None or len(dose_values) == 0:
            return None

        voxels_above_threshold = np.sum(dose_values >= dose_threshold)
        return float((voxels_above_threshold / len(dose_values)) * 100)

    def conformity_index(
        self, prescription_dose: float, tolerance: float = 0.05
    ) -> Optional[float]:
        """
        Calculate conformity index for the target.

        Args:
            prescription_dose: Prescribed dose in Gy
            tolerance: Dose tolerance (e.g., 0.05 for 95% of prescription)

        Returns:
            Conformity index, or None if no dose data available
        """
        dose_values = self.get_dose_in_structure()
        if dose_values is None or len(dose_values) == 0:
            return None

        threshold_dose = prescription_dose * (1 - tolerance)
        coverage = self.coverage_volume_percentage(threshold_dose)

        if coverage is None:
            return None

        return coverage / 100.0  # Return as fraction


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
