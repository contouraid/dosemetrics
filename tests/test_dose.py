"""
Comprehensive tests for the Dose class.

Tests cover:
- Loading from NIfTI and DICOM
- Dose properties and methods
- Dose-structure compatibility
- DVH computation
- Statistics computation
- Edge cases and error handling
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from huggingface_hub import snapshot_download

from dosemetrics.dose import Dose
from dosemetrics.structures import OAR, Target, StructureType
from dosemetrics.structure_set import StructureSet


@pytest.fixture(scope="module")
def hf_data_path():
    """Download test data from HuggingFace once per module."""
    data_path = snapshot_download(
        repo_id="contouraid/dosemetrics-data",
        repo_type="dataset"
    )
    return Path(data_path)


@pytest.fixture
def sample_dose_array():
    """Create a sample 3D dose array."""
    # Create a dose distribution with a gradient
    x = np.linspace(0, 60, 50)
    y = np.linspace(0, 60, 50)
    z = np.linspace(0, 60, 30)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Create a peaked dose distribution
    dose = 60 * np.exp(-((X-30)**2 + (Y-30)**2 + (Z-15)**2) / 200)
    return dose


@pytest.fixture
def sample_structure():
    """Create a sample structure for testing."""
    mask = np.zeros((50, 50, 30), dtype=bool)
    mask[20:30, 20:30, 10:20] = True  # Small cube in center
    
    return Target(
        name="TestPTV",
        mask=mask,
        spacing=(2.0, 2.0, 3.0),
        origin=(0.0, 0.0, 0.0)
    )


class TestDoseConstruction:
    """Test Dose object construction and validation."""
    
    def test_dose_init_basic(self, sample_dose_array):
        """Test basic Dose initialization."""
        dose = Dose(
            dose_array=sample_dose_array,
            spacing=(2.0, 2.0, 3.0),
            origin=(0.0, 0.0, 0.0),
            name="TestDose"
        )
        
        assert dose.name == "TestDose"
        assert dose.shape == sample_dose_array.shape
        assert dose.spacing == (2.0, 2.0, 3.0)
        assert dose.origin == (0.0, 0.0, 0.0)
        assert isinstance(dose.dose_array, np.ndarray)
    
    def test_dose_init_validates_3d(self):
        """Test that Dose requires 3D array."""
        with pytest.raises(ValueError, match="must be 3D"):
            Dose(
                dose_array=np.zeros((10, 10)),  # 2D
                spacing=(1.0, 1.0, 1.0),
                origin=(0.0, 0.0, 0.0)
            )
        
        with pytest.raises(ValueError, match="must be 3D"):
            Dose(
                dose_array=np.zeros((10,)),  # 1D
                spacing=(1.0, 1.0, 1.0),
                origin=(0.0, 0.0, 0.0)
            )
    
    def test_dose_properties(self, sample_dose_array):
        """Test Dose properties."""
        dose = Dose(sample_dose_array, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
        
        assert dose.max_dose > 0
        assert dose.mean_dose > 0
        assert dose.min_dose >= 0
        assert dose.min_dose <= dose.mean_dose <= dose.max_dose
        assert isinstance(dose.max_dose, float)
        assert isinstance(dose.mean_dose, float)
        assert isinstance(dose.min_dose, float)
    
    def test_dose_repr(self, sample_dose_array):
        """Test string representations."""
        dose = Dose(sample_dose_array, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), name="TestDose")
        
        repr_str = repr(dose)
        assert "TestDose" in repr_str
        assert "shape=" in repr_str
        assert "max=" in repr_str
        
        str_str = str(dose)
        assert "TestDose" in str_str
        assert "Shape:" in str_str
        assert "Max dose:" in str_str


class TestDoseFromNIfTI:
    """Test loading Dose from NIfTI files."""
    
    def test_load_nifti_dose(self, hf_data_path):
        """Test loading dose from NIfTI file."""
        dose_file = hf_data_path / "test_subject" / "Dose.nii.gz"
        
        if not dose_file.exists():
            pytest.skip("Test data not available")
        
        dose = Dose.from_nifti(dose_file, name="TestDose")
        
        assert isinstance(dose, Dose)
        assert dose.name == "TestDose"
        assert len(dose.shape) == 3  # 3D array
        assert dose.max_dose > 0
        assert dose.mean_dose > 0
    
    def test_load_nifti_dose_auto_name(self, hf_data_path):
        """Test automatic name extraction from filename."""
        dose_file = hf_data_path / "test_subject" / "Dose.nii.gz"
        
        if not dose_file.exists():
            pytest.skip("Test data not available")
        
        dose = Dose.from_nifti(dose_file)
        
        assert dose.name == "Dose"
    
    def test_load_nifti_nonexistent(self):
        """Test error handling for non-existent file."""
        with pytest.raises(FileNotFoundError):
            Dose.from_nifti("nonexistent.nii.gz")


class TestDoseFromDICOM:
    """Test loading Dose from DICOM RT-DOSE files."""
    
    def test_load_dicom_dose(self, hf_data_path):
        """Test loading dose from DICOM RT-DOSE file."""
        dicom_path = hf_data_path / "dicom"
        
        if not dicom_path.exists():
            pytest.skip("DICOM test data not available")
        
        rtdose_files = list((dicom_path / "RTDOSE").glob("*.dcm"))
        if not rtdose_files:
            pytest.skip("No RT-DOSE files found")
        
        dose = Dose.from_dicom(rtdose_files[0], name="Plan1")
        
        assert isinstance(dose, Dose)
        assert dose.name == "Plan1"
        assert len(dose.shape) == 3
        assert dose.max_dose > 0
        assert 'dose_scaling' in dose.metadata
    
    def test_load_multiple_dicom_doses(self, hf_data_path):
        """Test loading multiple RT-DOSE files."""
        dicom_path = hf_data_path / "dicom"
        
        if not dicom_path.exists():
            pytest.skip("DICOM test data not available")
        
        rtdose_files = list((dicom_path / "RTDOSE").glob("*.dcm"))
        if len(rtdose_files) < 2:
            pytest.skip("Need at least 2 RT-DOSE files")
        
        doses = []
        for i, dose_file in enumerate(rtdose_files[:3]):
            dose = Dose.from_dicom(dose_file, name=f"Plan_{i+1}")
            doses.append(dose)
        
        # Verify each dose is independent
        assert len(doses) >= 2
        for i, dose in enumerate(doses):
            assert dose.name == f"Plan_{i+1}"
            assert isinstance(dose, Dose)


class TestDoseStructureCompatibility:
    """Test spatial compatibility checking between Dose and Structure."""
    
    def test_compatible_dose_structure(self, sample_dose_array, sample_structure):
        """Test compatible dose and structure."""
        dose = Dose(
            sample_dose_array,
            spacing=sample_structure.spacing,
            origin=sample_structure.origin
        )
        
        assert dose.is_compatible_with_structure(sample_structure)
    
    def test_incompatible_shape(self, sample_dose_array, sample_structure):
        """Test incompatible shapes."""
        dose = Dose(
            np.zeros((40, 40, 20)),  # Different shape
            spacing=sample_structure.spacing,
            origin=sample_structure.origin
        )
        
        assert not dose.is_compatible_with_structure(sample_structure)
    
    def test_incompatible_spacing(self, sample_dose_array, sample_structure):
        """Test incompatible spacing."""
        dose = Dose(
            sample_dose_array,
            spacing=(1.0, 1.0, 1.0),  # Different spacing
            origin=sample_structure.origin
        )
        
        assert not dose.is_compatible_with_structure(sample_structure)
    
    def test_incompatible_origin(self, sample_dose_array, sample_structure):
        """Test incompatible origin."""
        dose = Dose(
            sample_dose_array,
            spacing=sample_structure.spacing,
            origin=(10.0, 10.0, 10.0)  # Different origin
        )
        
        assert not dose.is_compatible_with_structure(sample_structure)


class TestDoseInStructure:
    """Test extracting dose values within structures."""
    
    def test_get_dose_in_structure(self, sample_dose_array, sample_structure):
        """Test extracting dose values in structure."""
        dose = Dose(
            sample_dose_array,
            spacing=sample_structure.spacing,
            origin=sample_structure.origin
        )
        
        dose_values = dose.get_dose_in_structure(sample_structure)
        
        assert isinstance(dose_values, np.ndarray)
        assert dose_values.ndim == 1
        assert len(dose_values) == sample_structure.volume_voxels()
        assert np.all(dose_values >= 0)
    
    def test_get_dose_incompatible_raises(self, sample_dose_array):
        """Test error when dose and structure are incompatible."""
        dose = Dose(sample_dose_array, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
        
        # Create incompatible structure
        incompatible_structure = Target(
            name="Incompatible",
            mask=np.zeros((30, 30, 30), dtype=bool),  # Different shape
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0)
        )
        
        with pytest.raises(ValueError, match="not compatible"):
            dose.get_dose_in_structure(incompatible_structure)


# NOTE: Statistics, DVH, and volume queries have been moved to metrics modules
# See test_statistics.py, test_dvh.py for new tests using metrics functions


class TestDoseWithRealData:
    """Test Dose with real HuggingFace data."""
    
    def test_nifti_dose_loading(self, hf_data_path):
        """Test dose loading from real NIfTI data."""
        from dosemetrics.io import load_structure_set
        
        subject_path = hf_data_path / "test_subject"
        if not subject_path.exists():
            pytest.skip("Test data not available")
        
        # Load dose
        dose_file = subject_path / "Dose.nii.gz"
        dose = Dose.from_nifti(dose_file, name="Clinical")
        
        # Load structures
        structures = load_structure_set(subject_path)
        
        # Verify compatibility with all structures
        for name in structures.structure_names:
            structure = structures.get_structure(name)
            assert dose.is_compatible_with_structure(structure)
            
            # Verify can extract dose values
            dose_values = dose.get_dose_in_structure(structure)
            assert len(dose_values) > 0
            assert np.all(dose_values >= 0)
    
    def test_dicom_dose_loading(self, hf_data_path):
        """Test dose loading from real DICOM data."""
        from dosemetrics.io import load_structure_set
        
        dicom_path = hf_data_path / "dicom"
        if not dicom_path.exists():
            pytest.skip("DICOM test data not available")
        
        # Load first dose file
        rtdose_files = list((dicom_path / "RTDOSE").glob("*.dcm"))
        if not rtdose_files:
            pytest.skip("No RT-DOSE files found")
        
        dose = Dose.from_dicom(rtdose_files[0], name="Plan1")
        
        # Verify basic properties
        assert dose.max_dose > 0
        assert dose.mean_dose > 0
        assert 'dose_scaling' in dose.metadata

