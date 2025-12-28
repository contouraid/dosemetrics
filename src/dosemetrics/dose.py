"""
Dose distribution class for radiotherapy dose data.

This module provides the Dose class for representing 3D dose distributions
from RT-DOSE DICOM files or NIfTI files. The Dose class is a pure data
container - dose metrics are computed using functions in the metrics subpackage.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from .structures import Structure


class Dose:
    """
    Represents a 3D dose distribution from RT-DOSE or NIfTI.
    
    A Dose object is a pure data container representing dose distributions.
    Dose analysis is performed by combining Dose with Structure objects using
    functions from the metrics subpackage.
    
    Attributes:
        dose_array (np.ndarray): 3D array of dose values in Gy
        spacing (Tuple[float, float, float]): Voxel spacing in (x, y, z) mm
        origin (Tuple[float, float, float]): Origin coordinates in mm
        name (str): Identifier for this dose distribution
        metadata (Dict): Additional metadata (DICOM tags, beam info, etc.)
    
    Examples:
        >>> from dosemetrics.dose import Dose
        >>> from dosemetrics.metrics import dvh
        >>> 
        >>> # Load dose from DICOM
        >>> dose = Dose.from_dicom("path/to/rtdose.dcm", name="Plan_v1")
        >>> 
        >>> # Load dose from NIfTI
        >>> dose = Dose.from_nifti("path/to/dose.nii.gz", name="Predicted")
        >>> 
        >>> # Compute dose statistics (use metrics module)
        >>> ptv = structure_set.get_structure("PTV")
        >>> stats = statistics.compute_dose_statistics(dose, ptv)
        >>> print(f"Mean dose: {stats['mean_dose']:.2f} Gy")
        >>> 
        >>> # Compute DVH (use metrics module)
        >>> dose_bins, volumes = dvh.compute_dvh(dose, ptv)
    """
    
    def __init__(
        self,
        dose_array: np.ndarray,
        spacing: Tuple[float, float, float],
        origin: Tuple[float, float, float],
        name: str = "Dose",
        metadata: Optional[Dict] = None,
    ):
        """
        Initialize a Dose distribution.
        
        Args:
            dose_array: 3D array of dose values (Gy)
            spacing: Voxel spacing in (x, y, z) mm
            origin: Origin coordinates in mm
            name: Identifier for this dose (e.g., "Plan_v1", "Sum", "Predicted")
            metadata: Additional metadata (DICOM tags, beam info, etc.)
        
        Raises:
            ValueError: If dose_array is not 3D
        """
        self.dose_array = np.asarray(dose_array)
        self.spacing = tuple(spacing)
        self.origin = tuple(origin)
        self.name = name
        self.metadata = metadata or {}
        
        # Validate 3D
        if self.dose_array.ndim != 3:
            raise ValueError(
                f"Dose array must be 3D, got {self.dose_array.ndim}D array"
            )
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        """Shape of the dose array."""
        return self.dose_array.shape
    
    @property
    def max_dose(self) -> float:
        """Maximum dose in the entire distribution (Gy)."""
        return float(np.max(self.dose_array))
    
    @property
    def mean_dose(self) -> float:
        """Mean dose across the entire volume (Gy)."""
        return float(np.mean(self.dose_array))
    
    @property
    def min_dose(self) -> float:
        """Minimum dose in the distribution (Gy)."""
        return float(np.min(self.dose_array))
    
    def is_compatible_with_structure(self, structure: Structure) -> bool:
        """
        Check if this dose is spatially compatible with a structure.
        
        Args:
            structure: Structure to check compatibility with
        
        Returns:
            True if shapes, spacing, and origin match
        """
        if structure.mask is None:
            return False
        
        return (
            self.shape == structure.mask.shape
            and np.allclose(self.spacing, structure.spacing, rtol=1e-5)
            and np.allclose(self.origin, structure.origin, rtol=1e-5)
        )
    
    def get_dose_in_structure(self, structure: Structure) -> np.ndarray:
        """
        Extract dose values within a structure mask.
        
        Args:
            structure: Structure to extract dose from
        
        Returns:
            1D array of dose values inside the structure
        
        Raises:
            ValueError: If dose and structure are not spatially compatible
        """
        if not self.is_compatible_with_structure(structure):
            raise ValueError(
                f"Dose '{self.name}' (shape={self.shape}) is not compatible "
                f"with structure '{structure.name}' (shape={structure.mask.shape}). "
                f"Dose spacing: {self.spacing}, Structure spacing: {structure.spacing}"
            )
        
        return self.dose_array[structure.mask]
    
    @classmethod
    def from_nifti(
        cls, 
        file_path: Union[str, Path], 
        name: Optional[str] = None
    ) -> Dose:
        """
        Load dose distribution from a NIfTI file.
        
        Args:
            file_path: Path to NIfTI file (.nii or .nii.gz)
            name: Name for this dose (uses filename stem if None)
        
        Returns:
            Dose object
        
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file cannot be loaded
        """
        from .io.data_io import load_volume
        
        file_path = Path(file_path)
        volume, spacing, origin = load_volume(file_path)
        
        if name is None:
            name = file_path.stem.replace('.nii', '')
        
        return cls(volume, spacing, origin, name=name)
    
    @classmethod
    def from_dicom(
        cls, 
        file_path: Union[str, Path], 
        name: Optional[str] = None
    ) -> Dose:
        """
        Load dose distribution from a DICOM RT-DOSE file.
        
        Args:
            file_path: Path to RT-DOSE DICOM file
            name: Name for this dose (uses filename stem if None)
        
        Returns:
            Dose object
        
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is not a valid RT-DOSE
        """
        from .io.dicom_io import read_dicom_rtdose
        
        file_path = Path(file_path)
        dose_array, spacing, origin, scaling = read_dicom_rtdose(file_path)
        
        if name is None:
            name = file_path.stem
        
        metadata = {'dose_scaling': scaling}
        
        return cls(dose_array, spacing, origin, name=name, metadata=metadata)
    
    def __repr__(self) -> str:
        """String representation of the Dose object."""
        return (
            f"Dose(name='{self.name}', shape={self.shape}, "
            f"max={self.max_dose:.2f} Gy, mean={self.mean_dose:.2f} Gy)"
        )
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"Dose Distribution '{self.name}':\n"
            f"  Shape: {self.shape}\n"
            f"  Spacing: {self.spacing} mm\n"
            f"  Max dose: {self.max_dose:.2f} Gy\n"
            f"  Mean dose: {self.mean_dose:.2f} Gy\n"
            f"  Min dose: {self.min_dose:.2f} Gy"
        )
