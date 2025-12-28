"""
Comprehensive tests for the new I/O functionality.

Tests cover:
- NIfTI reading (volumes, masks, dose)
- DICOM reading (CT, RTDOSE, RTSTRUCT)
- High-level API (auto-detection, folder loading)
- StructureSet creation from different sources
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path

from dosemetrics.io import (
    # High-level API
    load_from_folder,
    load_structure_set,
    load_volume,
    load_structure,
    detect_folder_format,
    # Format-specific modules
    nifti_io,
    dicom_io,
)
from dosemetrics import (
    # Structure classes
    Structure,
    StructureType,
    StructureSet,
)


class TestNIfTIIO:
    """Tests for NIfTI I/O functionality."""
    
    def setup_method(self):
        """Create temporary directory for test files."""
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        """Clean up temporary directory."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def create_test_volume(self, filename, shape=(10, 20, 20), is_binary=False):
        """Create a test NIfTI volume."""
        if is_binary:
            data = np.random.randint(0, 2, shape).astype(np.uint8)
        else:
            data = np.random.rand(*shape).astype(np.float32) * 10.0
        
        file_path = self.temp_dir / filename
        spacing = (1.5, 1.5, 3.0)
        origin = (10.0, 20.0, 30.0)
        
        nifti_io.write_nifti_volume(data, file_path, spacing, origin)
        return file_path, data, spacing, origin
    
    def test_read_nifti_volume(self):
        """Test reading a NIfTI volume."""
        file_path, expected_data, expected_spacing, expected_origin = self.create_test_volume(
            "test_volume.nii.gz"
        )
        
        volume, spacing, origin = nifti_io.read_nifti_volume(file_path)
        
        assert volume.shape == expected_data.shape
        assert np.allclose(volume, expected_data, atol=1e-5)
        assert spacing == expected_spacing
        assert origin == expected_origin
    
    def test_read_nifti_mask(self):
        """Test reading a NIfTI mask."""
        file_path, expected_data, _, _ = self.create_test_volume(
            "test_mask.nii.gz", is_binary=True
        )
        
        mask, spacing, origin = nifti_io.read_nifti_mask(file_path)
        
        assert mask.dtype == bool
        assert mask.shape == expected_data.shape
        assert np.all(mask == expected_data.astype(bool))
    
    def test_is_binary_volume(self):
        """Test binary volume detection."""
        binary_vol = np.array([[[0, 1, 0], [1, 0, 1]], [[1, 1, 0], [0, 0, 1]]])
        real_vol = np.array([[[0.5, 1.2, 0.8], [1.5, 0.3, 1.1]], [[1.8, 1.9, 0.1], [0.2, 0.7, 1.4]]])
        
        assert nifti_io.is_binary_volume(binary_vol)
        assert not nifti_io.is_binary_volume(real_vol)
    
    def test_load_nifti_folder(self):
        """Test loading all NIfTI files from a folder."""
        # Create test files
        dose_path, dose_data, _, _ = self.create_test_volume("Dose.nii.gz", is_binary=False)
        liver_path, _, _, _ = self.create_test_volume("Liver.nii.gz", is_binary=True)
        ptv_path, _, _, _ = self.create_test_volume("PTV.nii.gz", is_binary=True)
        
        # Load folder (use return_as_structureset=False to get dict)
        data = nifti_io.load_nifti_folder(self.temp_dir, return_as_structureset=False)
        
        # Check dose volume
        assert 'dose_volume' in data
        assert np.allclose(data['dose_volume'], dose_data, atol=1e-5)
        
        # Check structure masks
        assert 'structure_masks' in data
        assert 'Liver' in data['structure_masks']
        assert 'PTV' in data['structure_masks']
        
        # Check spacing and origin
        assert 'spacing' in data
        assert 'origin' in data
    
    def test_create_structure_set_from_nifti_folder(self):
        """Test creating StructureSet from NIfTI folder."""
        # Create test files
        self.create_test_volume("Dose.nii.gz", is_binary=False)
        self.create_test_volume("Liver.nii.gz", is_binary=True)
        self.create_test_volume("PTV_High.nii.gz", is_binary=True)
        
        # Create structure set
        structure_set = nifti_io.create_structure_set_from_nifti_folder(
            self.temp_dir,
            name="Test Structure Set"
        )
        
        assert structure_set.name == "Test Structure Set"
        assert len(structure_set.structures) == 2
        assert 'Liver' in structure_set.structures
        assert 'PTV_High' in structure_set.structures
        # Note: In new architecture, dose is loaded separately via Dose.from_nifti()
        # StructureSet no longer holds dose data by default
        
        # Check structure types (should auto-detect)
        assert structure_set.structures['PTV_High'].structure_type == StructureType.TARGET
    
    def test_read_nifti_structure(self):
        """Test reading single structure from NIfTI file."""
        file_path, expected_data, expected_spacing, expected_origin = self.create_test_volume(
            "Spinal_Cord.nii.gz", is_binary=True
        )
        
        structure = nifti_io.read_nifti_structure(
            file_path,
            structure_type=StructureType.OAR
        )
        
        assert structure.name == "Spinal_Cord"
        assert structure.structure_type == StructureType.OAR
        assert structure.mask is not None
        assert structure.spacing == expected_spacing
        assert structure.origin == expected_origin
    
    def test_write_structure_as_nifti(self):
        """Test writing structure to NIfTI file."""
        from dosemetrics import OAR
        
        # Create a structure
        mask = np.random.randint(0, 2, (10, 20, 20)).astype(bool)
        spacing = (1.0, 1.0, 2.0)
        origin = (0.0, 0.0, 0.0)
        
        structure = OAR("TestOAR", mask=mask, spacing=spacing, origin=origin)
        
        # Write to file
        output_file = self.temp_dir / "TestOAR.nii.gz"
        nifti_io.write_structure_as_nifti(structure, output_file)
        
        # Read back and verify
        loaded_mask, loaded_spacing, loaded_origin = nifti_io.read_nifti_mask(output_file)
        
        assert np.all(loaded_mask == mask)
        assert loaded_spacing == spacing
        assert loaded_origin == origin
    
    def test_write_structure_set_as_nifti(self):
        """Test writing entire structure set to NIfTI files."""
        # Create structure set
        structure_set = StructureSet(
            spacing=(1.0, 1.0, 2.0),
            origin=(0.0, 0.0, 0.0),
            name="Test Set"
        )
        
        # Add structures
        structure_set.add_structure(
            "Liver",
            np.random.randint(0, 2, (10, 20, 20)).astype(bool),
            StructureType.OAR
        )
        structure_set.add_structure(
            "PTV",
            np.random.randint(0, 2, (10, 20, 20)).astype(bool),
            StructureType.TARGET
        )
        
        # Write to folder
        output_folder = self.temp_dir / "output"
        nifti_io.write_structure_set_as_nifti(structure_set, output_folder)
        
        # Verify files exist
        assert (output_folder / "Liver.nii.gz").exists()
        assert (output_folder / "PTV.nii.gz").exists()
        # Note: Dose file is not written anymore - use Dose objects separately


class TestHighLevelAPI:
    """Tests for high-level unified API."""
    
    def setup_method(self):
        """Create temporary directory for test files."""
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        """Clean up temporary directory."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def create_nifti_test_data(self):
        """Create test NIfTI data."""
        dose_data = np.random.rand(10, 20, 20) * 50.0
        liver_mask = np.random.randint(0, 2, (10, 20, 20)).astype(np.uint8)
        ptv_mask = np.random.randint(0, 2, (10, 20, 20)).astype(np.uint8)
        
        spacing = (1.0, 1.0, 2.0)
        origin = (0.0, 0.0, 0.0)
        
        nifti_io.write_nifti_volume(dose_data, self.temp_dir / "Dose.nii.gz", spacing, origin)
        nifti_io.write_nifti_volume(liver_mask, self.temp_dir / "Liver.nii.gz", spacing, origin)
        nifti_io.write_nifti_volume(ptv_mask, self.temp_dir / "PTV.nii.gz", spacing, origin)
    
    def test_detect_folder_format_nifti(self):
        """Test format detection for NIfTI folder."""
        self.create_nifti_test_data()
        
        format = detect_folder_format(self.temp_dir)
        assert format == 'nifti'
    
    def test_detect_folder_format_unknown(self):
        """Test format detection for empty folder."""
        format = detect_folder_format(self.temp_dir)
        assert format == 'unknown'
    
    def test_load_from_folder_nifti(self):
        """Test loading from NIfTI folder with auto-detection."""
        self.create_nifti_test_data()
        
        # load_from_folder returns a StructureSet by default for NIfTI folders
        structure_set = load_from_folder(self.temp_dir)
        
        assert isinstance(structure_set, StructureSet)
        assert 'Liver' in structure_set.structures
        assert 'PTV' in structure_set.structures
    
    def test_load_structure_set_nifti(self):
        """Test loading StructureSet from NIfTI folder."""
        self.create_nifti_test_data()
        
        structure_set = load_structure_set(self.temp_dir)
        
        assert isinstance(structure_set, StructureSet)
        assert len(structure_set.structures) >= 2
        # Note: In new architecture, dose is loaded separately via Dose.from_nifti()
    
    def test_load_structure_set_with_type_mapping(self):
        """Test loading StructureSet with custom structure type mapping."""
        self.create_nifti_test_data()
        
        type_mapping = {
            'Liver': StructureType.OAR,
            'PTV': StructureType.TARGET,
        }
        
        structure_set = load_structure_set(
            self.temp_dir,
            structure_type_mapping=type_mapping
        )
        
        assert structure_set.structures['Liver'].structure_type == StructureType.OAR
        assert structure_set.structures['PTV'].structure_type == StructureType.TARGET
    
    def test_load_volume(self):
        """Test loading a single volume file."""
        dose_data = np.random.rand(10, 20, 20) * 50.0
        spacing = (1.0, 1.0, 2.0)
        origin = (0.0, 0.0, 0.0)
        
        dose_file = self.temp_dir / "Dose.nii.gz"
        nifti_io.write_nifti_volume(dose_data, dose_file, spacing, origin)
        
        volume, loaded_spacing, loaded_origin = load_volume(dose_file)
        
        assert np.allclose(volume, dose_data, atol=1e-5)
        assert loaded_spacing == spacing
        assert loaded_origin == origin
    
    def test_load_structure(self):
        """Test loading a single structure file."""
        mask = np.random.randint(0, 2, (10, 20, 20)).astype(np.uint8)
        spacing = (1.0, 1.0, 2.0)
        origin = (0.0, 0.0, 0.0)
        
        mask_file = self.temp_dir / "Heart.nii.gz"
        nifti_io.write_nifti_volume(mask, mask_file, spacing, origin)
        
        structure = load_structure(mask_file, structure_type=StructureType.OAR)
        
        assert isinstance(structure, Structure)
        assert structure.name == "Heart"
        assert structure.structure_type == StructureType.OAR
        assert structure.mask is not None


class TestErrorHandling:
    """Tests for error handling and edge cases."""
    
    def test_load_nonexistent_folder(self):
        """Test loading from non-existent folder."""
        with pytest.raises(FileNotFoundError):
            load_from_folder("/path/that/does/not/exist")
    
    def test_load_nonexistent_file(self):
        """Test loading non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_volume("/path/that/does/not/exist.nii.gz")
    
    def test_load_empty_folder(self):
        """Test loading from empty folder."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(ValueError):
                load_structure_set(temp_dir)
    
    def test_invalid_mask_dimension(self):
        """Test that invalid mask dimensions are caught."""
        from dosemetrics import OAR
        
        invalid_mask = np.random.rand(10, 20)  # 2D instead of 3D
        
        with pytest.raises(ValueError):
            OAR("test", mask=invalid_mask)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
