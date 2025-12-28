"""
Comprehensive tests for the simplified Structure classes.

Tests cover:
- Structure creation and validation
- Geometry calculations (volume, centroid, bounding box)
- Subclass behavior (OAR, Target, AvoidanceStructure)
- Loading from NIfTI and DICOM
- Edge cases and error handling
"""

import pytest
import numpy as np
from pathlib import Path
from huggingface_hub import snapshot_download

from dosemetrics.structures import (
    Structure, OAR, Target, AvoidanceStructure, StructureType
)


@pytest.fixture(scope="module")
def hf_data_path():
    """Download test data from HuggingFace once per module."""
    data_path = snapshot_download(
        repo_id="contouraid/dosemetrics-data",
        repo_type="dataset"
    )
    return Path(data_path)


@pytest.fixture
def sample_mask():
    """Create a sample 3D binary mask."""
    mask = np.zeros((50, 50, 30), dtype=bool)
    mask[20:30, 20:30, 10:20] = True  # 10x10x10 cube
    return mask


@pytest.fixture
def sample_spacing():
    """Sample voxel spacing."""
    return (2.0, 2.0, 3.0)


@pytest.fixture
def sample_origin():
    """Sample origin."""
    return (0.0, 0.0, 0.0)


class TestStructureCreation:
    """Test Structure object creation and validation."""
    
    def test_create_oar(self, sample_mask, sample_spacing, sample_origin):
        """Test OAR creation."""
        oar = OAR(
            name="Brainstem",
            mask=sample_mask,
            spacing=sample_spacing,
            origin=sample_origin
        )
        
        assert oar.name == "Brainstem"
        assert oar.structure_type == StructureType.OAR
        assert oar.mask.shape == sample_mask.shape
        assert oar.spacing == sample_spacing
        assert oar.origin == sample_origin
    
    def test_create_target(self, sample_mask, sample_spacing, sample_origin):
        """Test Target creation."""
        target = Target(
            name="PTV",
            mask=sample_mask,
            spacing=sample_spacing,
            origin=sample_origin
        )
        
        assert target.name == "PTV"
        assert target.structure_type == StructureType.TARGET
    
    def test_create_avoidance(self, sample_mask, sample_spacing, sample_origin):
        """Test AvoidanceStructure creation."""
        avoidance = AvoidanceStructure(
            name="PRV_SpinalCord",
            mask=sample_mask,
            spacing=sample_spacing,
            origin=sample_origin
        )
        
        assert avoidance.name == "PRV_SpinalCord"
        assert avoidance.structure_type == StructureType.AVOIDANCE
    
    def test_mask_validation(self, sample_spacing, sample_origin):
        """Test mask validation."""
        # Non-boolean mask should be converted
        float_mask = np.random.rand(10, 10, 10)
        oar = OAR(
            name="Test",
            mask=float_mask > 0.5,  # Convert to boolean
            spacing=sample_spacing,
            origin=sample_origin
        )
        assert oar.mask.dtype == bool
        
        # 2D mask should raise error
        with pytest.raises(ValueError, match="must be 3D"):
            OAR(
                name="Test",
                mask=np.zeros((10, 10), dtype=bool),
                spacing=sample_spacing,
                origin=sample_origin
            )
    
    def test_default_origin(self, sample_mask, sample_spacing):
        """Test default origin."""
        oar = OAR(
            name="Test",
            mask=sample_mask,
            spacing=sample_spacing
        )
        
        assert oar.origin == (0.0, 0.0, 0.0)
    
    def test_repr(self, sample_mask, sample_spacing, sample_origin):
        """Test string representations."""
        oar = OAR("Brainstem", sample_mask, sample_spacing, sample_origin)
        
        repr_str = repr(oar)
        assert "Brainstem" in repr_str
        assert "OAR" in repr_str
        assert "volume_cc" in repr_str
        
        str_str = str(oar)
        assert "Brainstem" in str_str
        assert "OAR" in str_str


class TestGeometryCalculations:
    """Test geometric calculations on structures."""
    
    def test_volume_voxels(self, sample_mask, sample_spacing, sample_origin):
        """Test volume calculation in voxels."""
        oar = OAR("Test", sample_mask, sample_spacing, sample_origin)
        
        expected_voxels = np.sum(sample_mask)
        assert oar.volume_voxels() == expected_voxels
        assert oar.volume_voxels() == 1000  # 10x10x10 cube
    
    def test_volume_cc(self, sample_mask, sample_spacing, sample_origin):
        """Test volume calculation in cubic centimeters."""
        oar = OAR("Test", sample_mask, sample_spacing, sample_origin)
        
        voxel_volume_mm3 = 2.0 * 2.0 * 3.0  # 12 mm³
        total_volume_mm3 = 1000 * voxel_volume_mm3
        expected_cc = total_volume_mm3 / 1000.0
        
        assert oar.volume_cc() == pytest.approx(expected_cc)
        assert oar.volume_cc() == pytest.approx(12.0)
    
    def test_empty_structure_volume(self, sample_spacing, sample_origin):
        """Test volume of empty structure."""
        empty_mask = np.zeros((10, 10, 10), dtype=bool)
        oar = OAR("Empty", empty_mask, sample_spacing, sample_origin)
        
        assert oar.volume_voxels() == 0
        assert oar.volume_cc() == 0.0
    
    def test_centroid(self, sample_spacing, sample_origin):
        """Test centroid calculation."""
        # Create a centered cube
        mask = np.zeros((50, 50, 30), dtype=bool)
        mask[20:30, 20:30, 10:20] = True
        
        oar = OAR("Test", mask, sample_spacing, sample_origin)
        centroid = oar.centroid()
        
        assert len(centroid) == 3
        # Expected centroid at center of cube (24.5, 24.5, 14.5) in voxel coords
        # In physical coords: (24.5*2, 24.5*2, 14.5*3)
        expected = (24.5 * 2.0, 24.5 * 2.0, 14.5 * 3.0)
        
        assert centroid[0] == pytest.approx(expected[0])
        assert centroid[1] == pytest.approx(expected[1])
        assert centroid[2] == pytest.approx(expected[2])
    
    def test_centroid_with_origin(self):
        """Test centroid calculation with non-zero origin."""
        mask = np.zeros((10, 10, 10), dtype=bool)
        mask[4:6, 4:6, 4:6] = True  # Small centered cube
        
        oar = OAR(
            "Test",
            mask,
            spacing=(1.0, 1.0, 1.0),
            origin=(10.0, 20.0, 30.0)
        )
        
        centroid = oar.centroid()
        
        # Centroid in voxel coords is (4.5, 4.5, 4.5)
        # In physical coords: (4.5 + 10, 4.5 + 20, 4.5 + 30)
        assert centroid[0] == pytest.approx(14.5)
        assert centroid[1] == pytest.approx(24.5)
        assert centroid[2] == pytest.approx(34.5)
    
    def test_bounding_box(self, sample_mask, sample_spacing, sample_origin):
        """Test bounding box calculation."""
        oar = OAR("Test", sample_mask, sample_spacing, sample_origin)
        bbox = oar.bounding_box()
        
        # Expected bbox in voxel coords: ((20,29), (20,29), (10,19))
        assert bbox == ((20, 29), (20, 29), (10, 19))
    
    def test_empty_structure_centroid(self, sample_spacing, sample_origin):
        """Test centroid of empty structure."""
        empty_mask = np.zeros((10, 10, 10), dtype=bool)
        oar = OAR("Empty", empty_mask, sample_spacing, sample_origin)
        
        assert oar.centroid() is None
    
    def test_empty_structure_bbox(self, sample_spacing, sample_origin):
        """Test bounding box of empty structure."""
        empty_mask = np.zeros((10, 10, 10), dtype=bool)
        oar = OAR("Empty", empty_mask, sample_spacing, sample_origin)
        
        assert oar.bounding_box() is None


class TestStructureComparison:
    """Test structure comparison and equality."""
    
    def test_equality_same(self, sample_mask, sample_spacing, sample_origin):
        """Test equality of identical structures."""
        oar1 = OAR("Brainstem", sample_mask, sample_spacing, sample_origin)
        oar2 = OAR("Brainstem", sample_mask, sample_spacing, sample_origin)
        
        # Note: Structures are different objects, so they're not equal
        assert oar1 is not oar2
        
        # But their properties should match
        assert oar1.name == oar2.name
        assert oar1.structure_type == oar2.structure_type
        assert oar1.volume_voxels() == oar2.volume_voxels()
    
    def test_hash(self, sample_mask, sample_spacing, sample_origin):
        """Test that structures are hashable."""
        oar1 = OAR("Brainstem", sample_mask, sample_spacing, sample_origin)
        oar2 = OAR("SpinalCord", sample_mask, sample_spacing, sample_origin)
        
        # Should be able to use in sets/dicts
        structure_set = {oar1, oar2}
        assert len(structure_set) == 2


class TestStructureSubclasses:
    """Test behavior specific to structure subclasses."""
    
    def test_oar_type(self, sample_mask, sample_spacing, sample_origin):
        """Test OAR has correct type."""
        oar = OAR("Brainstem", sample_mask, sample_spacing, sample_origin)
        
        assert isinstance(oar, OAR)
        assert isinstance(oar, Structure)
        assert oar.structure_type == StructureType.OAR
    
    def test_target_type(self, sample_mask, sample_spacing, sample_origin):
        """Test Target has correct type."""
        target = Target("PTV", sample_mask, sample_spacing, sample_origin)
        
        assert isinstance(target, Target)
        assert isinstance(target, Structure)
        assert target.structure_type == StructureType.TARGET
    
    def test_avoidance_type(self, sample_mask, sample_spacing, sample_origin):
        """Test AvoidanceStructure has correct type."""
        avoidance = AvoidanceStructure("PRV", sample_mask, sample_spacing, sample_origin)
        
        assert isinstance(avoidance, AvoidanceStructure)
        assert isinstance(avoidance, Structure)
        assert avoidance.structure_type == StructureType.AVOIDANCE


class TestStructureDoseMethods:
    """Verify that Structure has dose-related methods."""
    
    def test_has_dose_data_attribute(self, sample_mask, sample_spacing, sample_origin):
        """Verify dose_data attribute exists and works."""
        oar = OAR("Test", sample_mask, sample_spacing, sample_origin)
        
        # Structure should not have dose-related attributes
        assert not hasattr(oar, '_dose_data')
        assert not hasattr(oar, 'has_dose')
    
    def test_no_legacy_dose_methods(self, sample_mask, sample_spacing, sample_origin):
        """Verify legacy dose-related methods don't exist."""
        oar = OAR("Test", sample_mask, sample_spacing, sample_origin)
        
        assert not hasattr(oar, 'set_dose_data')
        assert not hasattr(oar, 'get_dose_data')
        assert not hasattr(oar, 'mean_dose')
        assert not hasattr(oar, 'max_dose')
        assert not hasattr(oar, 'min_dose')
        assert not hasattr(oar, 'dvh')


class TestWithRealNIfTIData:
    """Test Structure with real NIfTI data from HuggingFace."""
    
    def test_load_nifti_structures(self, hf_data_path):
        """Test loading structures from NIfTI files."""
        from dosemetrics.io import load_structure_set
        
        subject_path = hf_data_path / "test_subject"
        if not subject_path.exists():
            pytest.skip("Test data not available")
        
        structures = load_structure_set(subject_path)
        
        assert len(structures) > 0
        
        for name in structures.structure_names:
            structure = structures.get_structure(name)
            
            # Verify it's a Structure instance
            assert isinstance(structure, Structure)
            
            # Verify geometry methods work
            assert structure.volume_voxels() > 0
            assert structure.volume_cc() > 0
            
            centroid = structure.centroid()
            assert len(centroid) == 3
            
            bbox = structure.bounding_box()
            assert bbox is None or (isinstance(bbox, tuple) and len(bbox) == 3)
            
            # Verify no dose attributes/methods
            assert not hasattr(structure, 'dose_data')
            assert not hasattr(structure, 'compute_statistics')


class TestWithRealDICOMData:
    """Test Structure with real DICOM data from HuggingFace."""
    
    def test_load_dicom_structures(self, hf_data_path):
        """Test loading structures from DICOM RT-STRUCT."""
        from dosemetrics.io import load_structure_set
        
        dicom_path = hf_data_path / "dicom"
        if not dicom_path.exists():
            pytest.skip("DICOM test data not available")
        
        structures = load_structure_set(dicom_path)
        
        assert len(structures) > 0
        
        for name in structures.structure_names:
            structure = structures.get_structure(name)
            
            # Verify it's a Structure instance
            assert isinstance(structure, Structure)
            
            # Verify correct subclass based on type
            if structure.structure_type == StructureType.OAR:
                assert isinstance(structure, OAR)
            elif structure.structure_type == StructureType.TARGET:
                assert isinstance(structure, Target)
            elif structure.structure_type == StructureType.AVOIDANCE:
                assert isinstance(structure, AvoidanceStructure)
            
            # Verify geometry
            if structure.volume_voxels() > 0:  # Skip empty structures
                assert structure.volume_cc() > 0
                centroid = structure.centroid()
                assert len(centroid) == 3


class TestStructureEdgeCases:
    """Test edge cases and error handling."""
    
    def test_single_voxel_structure(self):
        """Test structure with single voxel."""
        mask = np.zeros((10, 10, 10), dtype=bool)
        mask[5, 5, 5] = True
        
        oar = OAR("SingleVoxel", mask, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
        
        assert oar.volume_voxels() == 1
        assert oar.volume_cc() == pytest.approx(0.001)  # 1 mm³ = 0.001 cc
        
        centroid = oar.centroid()
        assert len(centroid) == 3
    
    def test_large_spacing(self):
        """Test with very large voxel spacing."""
        mask = np.zeros((10, 10, 10), dtype=bool)
        mask[5:7, 5:7, 5:7] = True  # 2x2x2 = 8 voxels
        
        oar = OAR("LargeSpacing", mask, (10.0, 10.0, 10.0), (0.0, 0.0, 0.0))
        
        # 8 voxels * (10mm)^3 = 8000 mm³ = 8 cc
        assert oar.volume_cc() == pytest.approx(8.0)
    
    def test_anisotropic_spacing(self):
        """Test with highly anisotropic spacing."""
        mask = np.zeros((10, 10, 10), dtype=bool)
        mask[5, 5, 5:8] = True  # 3 voxels in z direction
        
        oar = OAR("Anisotropic", mask, (0.5, 0.5, 5.0), (0.0, 0.0, 0.0))
        
        # 3 voxels * 0.5 * 0.5 * 5.0 = 3.75 mm³
        assert oar.volume_cc() == pytest.approx(0.00375)
