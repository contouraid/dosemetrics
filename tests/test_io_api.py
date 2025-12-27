"""
Tests for the I/O API.
"""

import pytest
import tempfile
import numpy as np
from pathlib import Path

from dosemetrics.io import nifti_io, dicom_io
from dosemetrics import StructureSet, StructureType


class TestNIfTIFolderLoading:
    """Test load_nifti_folder with StructureSet defaults."""
    
    def test_load_nifti_folder_returns_structureset_by_default(self):
        """Test that load_nifti_folder returns StructureSet by default."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create test data
            dose = np.random.rand(10, 10, 10) * 50
            mask1 = np.random.rand(10, 10, 10) > 0.5
            mask2 = np.random.rand(10, 10, 10) > 0.7
            
            # Write files
            nifti_io.write_nifti_volume(dose, tmpdir / "Dose.nii.gz")
            nifti_io.write_nifti_volume(mask1.astype(float), tmpdir / "PTV.nii.gz")
            nifti_io.write_nifti_volume(mask2.astype(float), tmpdir / "BrainStem.nii.gz")
            
            # Load with default (should return StructureSet)
            result = nifti_io.load_nifti_folder(tmpdir)
            
            assert isinstance(result, StructureSet)
            # Note: In new architecture, dose is loaded separately
            assert len(result) == 2  # PTV and BrainStem
            assert 'PTV' in result
            assert 'BrainStem' in result
    
    def test_load_nifti_folder_returns_dict_when_requested(self):
        """Test that load_nifti_folder can return raw dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create test data
            dose = np.random.rand(10, 10, 10) * 50
            mask = np.random.rand(10, 10, 10) > 0.5
            
            # Write files
            nifti_io.write_nifti_volume(dose, tmpdir / "Dose.nii.gz")
            nifti_io.write_nifti_volume(mask.astype(float), tmpdir / "Structure.nii.gz")
            
            # Load with return_as_structureset=False
            result = nifti_io.load_nifti_folder(tmpdir, return_as_structureset=False)
            
            assert isinstance(result, dict)
            assert 'dose_volume' in result
            assert 'structure_masks' in result
            assert 'Structure' in result['structure_masks']
    
    def test_load_nifti_folder_with_custom_type_mapping(self):
        """Test custom structure type mapping."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create test data
            mask1 = np.random.rand(10, 10, 10) > 0.5
            mask2 = np.random.rand(10, 10, 10) > 0.7
            
            # Write files
            nifti_io.write_nifti_volume(mask1.astype(float), tmpdir / "MyTarget.nii.gz")
            nifti_io.write_nifti_volume(mask2.astype(float), tmpdir / "MyOAR.nii.gz")
            
            # Load with custom mapping
            custom_mapping = {
                'MyTarget': StructureType.TARGET,
                'MyOAR': StructureType.OAR,
            }
            result = nifti_io.load_nifti_folder(
                tmpdir,
                structure_type_mapping=custom_mapping
            )
            
            assert isinstance(result, StructureSet)
            target = result.get_structure('MyTarget')
            oar = result.get_structure('MyOAR')
            assert target.structure_type == StructureType.TARGET
            assert oar.structure_type == StructureType.OAR
    
    def test_load_nifti_folder_raises_on_no_structures(self):
        """Test that loading folder with no structures raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create only dose file
            dose = np.random.rand(10, 10, 10) * 50
            nifti_io.write_nifti_volume(dose, tmpdir / "Dose.nii.gz")
            
            # Should raise ValueError when no structures found
            with pytest.raises(ValueError, match="No structure masks found"):
                nifti_io.load_nifti_folder(tmpdir)


class TestDICOMFolderLoading:
    """Test load_dicom_folder with StructureSet defaults."""
    
    def test_load_dicom_folder_signature_updated(self):
        """Test that load_dicom_folder has new parameters."""
        import inspect
        sig = inspect.signature(dicom_io.load_dicom_folder)
        params = sig.parameters
        
        assert 'return_as_structureset' in params
        assert params['return_as_structureset'].default is True
        assert 'dose_file_name' in params
        assert 'structure_type_mapping' in params


class TestBackwardCompatibility:
    """Test that existing code still works."""
    
    def test_nifti_folder_dict_access_still_works(self):
        """Test that explicitly requesting dict still works."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create test data
            dose = np.random.rand(10, 10, 10) * 50
            mask = np.random.rand(10, 10, 10) > 0.5
            
            # Write files
            nifti_io.write_nifti_volume(dose, tmpdir / "Dose.nii.gz")
            nifti_io.write_nifti_volume(mask.astype(float), tmpdir / "OAR.nii.gz")
            
            # Old way: get dict
            data = nifti_io.load_nifti_folder(tmpdir, return_as_structureset=False)
            
            # Verify old dict structure
            assert 'dose_volume' in data
            assert 'structure_masks' in data
            assert 'spacing' in data
            assert 'origin' in data
            
            # Can still access arrays directly
            dose_array = data['dose_volume']
            assert isinstance(dose_array, np.ndarray)
            assert dose_array.shape == (10, 10, 10)


class TestIntegration:
    """Integration tests for the full workflow."""
    
    def test_roundtrip_structureset_to_nifti(self):
        """Test saving and loading a StructureSet."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create a StructureSet
            original_set = StructureSet(
                spacing=(1.0, 1.0, 1.0),
                origin=(0.0, 0.0, 0.0),
                name="Test Set"
            )
            
            # Add structures
            mask1 = np.random.rand(10, 10, 10) > 0.5
            mask2 = np.random.rand(10, 10, 10) > 0.7
            dose_array = np.random.rand(10, 10, 10) * 50
            
            original_set.add_structure("PTV", mask1, StructureType.TARGET)
            original_set.add_structure("OAR", mask2, StructureType.OAR)
            
            # In new architecture, save dose separately
            nifti_io.write_nifti_volume(dose_array, tmpdir / "Dose.nii.gz")
            
            # Save structures to NIfTI
            nifti_io.write_structure_set_as_nifti(
                original_set,
                tmpdir,
                write_dose=False  # Dose saved separately
            )
            
            # Load back
            loaded_set = nifti_io.load_nifti_folder(tmpdir)
            
            # Verify
            assert isinstance(loaded_set, StructureSet)
            assert 'PTV' in loaded_set
            assert 'OAR' in loaded_set
            assert len(loaded_set) == 2
    
    def test_mixed_usage_patterns(self):
        """Test that both StructureSet and dict usage work together."""
        from dosemetrics import Dose
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create test data
            dose_array = np.random.rand(10, 10, 10) * 50
            mask = np.random.rand(10, 10, 10) > 0.5
            
            nifti_io.write_nifti_volume(dose_array, tmpdir / "Dose.nii.gz")
            nifti_io.write_nifti_volume(mask.astype(float), tmpdir / "PTV.nii.gz")
            
            # Load as StructureSet
            struct_set = nifti_io.load_nifti_folder(tmpdir)
            assert isinstance(struct_set, StructureSet)
            
            # Load dose separately in new architecture
            dose = Dose.from_nifti(tmpdir / "Dose.nii.gz")
            
            # Load as dict
            data_dict = nifti_io.load_nifti_folder(tmpdir, return_as_structureset=False)
            assert isinstance(data_dict, dict)
            
            # Verify dose matches
            np.testing.assert_array_equal(dose.dose_array, data_dict['dose_volume'])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
