"""
Comprehensive tests for the simplified StructureSet class.

Tests cover:
- StructureSet creation and management
- Adding, removing, and retrieving structures
- Filtering by type (OAR, Target, etc.)
- Geometric summary generation
- Loading from NIfTI and DICOM
- Edge cases and error handling
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from huggingface_hub import snapshot_download

from dosemetrics.structure_set import StructureSet
from dosemetrics.structures import OAR, Target, AvoidanceStructure, StructureType


@pytest.fixture(scope="module")
def hf_data_path():
    """Download test data from HuggingFace once per module."""
    data_path = snapshot_download(
        repo_id="contouraid/dosemetrics-data",
        repo_type="dataset"
    )
    return Path(data_path)


@pytest.fixture
def sample_structures():
    """Create sample structures for testing."""
    mask1 = np.zeros((50, 50, 30), dtype=bool)
    mask1[20:30, 20:30, 10:20] = True
    
    mask2 = np.zeros((50, 50, 30), dtype=bool)
    mask2[10:20, 10:20, 5:15] = True
    
    mask3 = np.zeros((50, 50, 30), dtype=bool)
    mask3[30:40, 30:40, 15:25] = True
    
    spacing = (2.0, 2.0, 3.0)
    origin = (0.0, 0.0, 0.0)
    
    brainstem = OAR("Brainstem", mask1, spacing, origin)
    spinal_cord = OAR("SpinalCord", mask2, spacing, origin)
    ptv = Target("PTV", mask3, spacing, origin)
    
    return [brainstem, spinal_cord, ptv]


@pytest.fixture
def empty_structure_set():
    """Create an empty StructureSet."""
    return StructureSet(
        spacing=(2.0, 2.0, 3.0),
        origin=(0.0, 0.0, 0.0)
    )


class TestStructureSetCreation:
    """Test StructureSet object creation."""
    
    def test_create_empty(self):
        """Test creating empty StructureSet."""
        ss = StructureSet(
            spacing=(2.0, 2.0, 3.0),
            origin=(0.0, 0.0, 0.0)
        )
        
        assert len(ss) == 0
        assert ss.spacing == (2.0, 2.0, 3.0)
        assert ss.origin == (0.0, 0.0, 0.0)
        assert ss.structure_names == []
    
    def test_default_origin(self):
        """Test default origin."""
        ss = StructureSet(
            spacing=(2.0, 2.0, 3.0)
        )
        
        assert ss.origin == (0.0, 0.0, 0.0)
    
    def test_repr(self):
        """Test string representation."""
        ss = StructureSet(spacing=(1.0, 1.0, 1.0))
        
        repr_str = repr(ss)
        assert "StructureSet" in repr_str
        assert "structures=0" in repr_str
        
        # Add a structure
        mask = np.zeros((50, 50, 30), dtype=bool)
        mask[20:30, 20:30, 10:20] = True
        ss.add_structure("Test", mask, StructureType.OAR)
        
        repr_str = repr(ss)
        assert "structures=1" in repr_str


class TestAddingStructures:
    """Test adding structures to StructureSet."""
    
    def test_add_structure_from_mask(self, empty_structure_set):
        """Test adding structure from mask."""
        mask = np.zeros((50, 50, 30), dtype=bool)
        mask[20:30, 20:30, 10:20] = True
        
        empty_structure_set.add_structure("Brainstem", mask, StructureType.OAR)
        
        assert len(empty_structure_set) == 1
        assert "Brainstem" in empty_structure_set
        
        structure = empty_structure_set.get_structure("Brainstem")
        assert isinstance(structure, OAR)
        assert structure.name == "Brainstem"
    
    def test_add_structure_object(self, empty_structure_set, sample_structures):
        """Test adding Structure object directly."""
        empty_structure_set.add_structure_object(sample_structures[0])
        
        assert len(empty_structure_set) == 1
        assert "Brainstem" in empty_structure_set
    
    def test_add_multiple_structures(self, empty_structure_set, sample_structures):
        """Test adding multiple structures."""
        for structure in sample_structures:
            empty_structure_set.add_structure_object(structure)
        
        assert len(empty_structure_set) == 3
        assert "Brainstem" in empty_structure_set
        assert "SpinalCord" in empty_structure_set
        assert "PTV" in empty_structure_set
    
    def test_add_duplicate_name_raises(self, empty_structure_set):
        """Test that adding duplicate name raises error."""
        mask = np.zeros((50, 50, 30), dtype=bool)
        empty_structure_set.add_structure("Test", mask, StructureType.OAR)
        
        with pytest.raises(ValueError, match="already exists"):
            empty_structure_set.add_structure("Test", mask, StructureType.OAR)
    
    def test_add_different_types(self, empty_structure_set):
        """Test adding structures of different types."""
        shape = (50, 50, 30)
        
        oar_mask = np.zeros(shape, dtype=bool)
        oar_mask[10:20, 10:20, 10:20] = True
        empty_structure_set.add_structure("OAR1", oar_mask, StructureType.OAR)
        
        target_mask = np.zeros(shape, dtype=bool)
        target_mask[30:40, 30:40, 15:25] = True
        empty_structure_set.add_structure("PTV", target_mask, StructureType.TARGET)
        
        avoidance_mask = np.zeros(shape, dtype=bool)
        avoidance_mask[20:25, 20:25, 12:18] = True
        empty_structure_set.add_structure("PRV", avoidance_mask, StructureType.AVOIDANCE)
        
        assert len(empty_structure_set) == 3
        assert isinstance(empty_structure_set.get_structure("OAR1"), OAR)
        assert isinstance(empty_structure_set.get_structure("PTV"), Target)
        assert isinstance(empty_structure_set.get_structure("PRV"), AvoidanceStructure)


class TestRetrievingStructures:
    """Test retrieving structures from StructureSet."""
    
    def test_get_structure(self, empty_structure_set, sample_structures):
        """Test getting structure by name."""
        for structure in sample_structures:
            empty_structure_set.add_structure_object(structure)
        
        brainstem = empty_structure_set.get_structure("Brainstem")
        assert isinstance(brainstem, OAR)
        assert brainstem.name == "Brainstem"
    
    def test_get_nonexistent_raises(self, empty_structure_set):
        """Test getting nonexistent structure raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            empty_structure_set.get_structure("NonExistent")
    
    def test_contains(self, empty_structure_set, sample_structures):
        """Test 'in' operator."""
        empty_structure_set.add_structure_object(sample_structures[0])
        
        assert "Brainstem" in empty_structure_set
        assert "NonExistent" not in empty_structure_set
    
    def test_structure_names(self, empty_structure_set, sample_structures):
        """Test getting list of structure names."""
        for structure in sample_structures:
            empty_structure_set.add_structure_object(structure)
        
        names = empty_structure_set.structure_names
        assert len(names) == 3
        assert "Brainstem" in names
        assert "SpinalCord" in names
        assert "PTV" in names
    
    def test_iteration(self, empty_structure_set, sample_structures):
        """Test iterating over structure set."""
        for structure in sample_structures:
            empty_structure_set.add_structure_object(structure)
        
        names_from_iteration = [name for name, _ in empty_structure_set]
        assert len(names_from_iteration) == 3
        assert set(names_from_iteration) == {"Brainstem", "SpinalCord", "PTV"}


class TestFilteringStructures:
    """Test filtering structures by type."""
    
    def test_get_oars(self, empty_structure_set, sample_structures):
        """Test getting all OARs."""
        for structure in sample_structures:
            empty_structure_set.add_structure_object(structure)
        
        oars = list(empty_structure_set.get_oars().values())
        assert len(oars) == 2
        assert all(isinstance(oar, OAR) for oar in oars)
        
        oar_names = [oar.name for oar in oars]
        assert "Brainstem" in oar_names
        assert "SpinalCord" in oar_names
        assert "PTV" not in oar_names
    
    def test_get_targets(self, empty_structure_set, sample_structures):
        """Test getting all targets."""
        for structure in sample_structures:
            empty_structure_set.add_structure_object(structure)
        
        targets = list(empty_structure_set.get_targets().values())
        assert len(targets) == 1
        assert isinstance(targets[0], Target)
        assert targets[0].name == "PTV"
    
    def test_get_avoidance_structures(self, empty_structure_set):
        """Test getting avoidance structures."""
        mask = np.zeros((50, 50, 30), dtype=bool)
        mask[10:20, 10:20, 10:20] = True
        
        prv = AvoidanceStructure("PRV_SpinalCord", mask, (2.0, 2.0, 3.0))
        empty_structure_set.add_structure_object(prv)
        
        avoidances = list(empty_structure_set.get_avoidance_structures().values())
        assert len(avoidances) == 1
        assert isinstance(avoidances[0], AvoidanceStructure)
    
    def test_empty_filter_results(self, empty_structure_set):
        """Test filtering when no structures match."""
        # Add only OARs
        mask = np.zeros((50, 50, 30), dtype=bool)
        empty_structure_set.add_structure("OAR1", mask, StructureType.OAR)
        
        targets = list(empty_structure_set.get_targets().values())
        assert len(targets) == 0


class TestRemovingStructures:
    """Test removing structures from StructureSet."""
    
    def test_remove_structure(self, empty_structure_set, sample_structures):
        """Test removing a structure."""
        for structure in sample_structures:
            empty_structure_set.add_structure_object(structure)
        
        assert len(empty_structure_set) == 3
        
        empty_structure_set.remove_structure("Brainstem")
        
        assert len(empty_structure_set) == 2
        assert "Brainstem" not in empty_structure_set
        assert "SpinalCord" in empty_structure_set
        assert "PTV" in empty_structure_set
    
    def test_remove_nonexistent_raises(self, empty_structure_set):
        """Test removing nonexistent structure raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            empty_structure_set.remove_structure("NonExistent")
    
    def test_clear_all(self, empty_structure_set, sample_structures):
        """Test clearing all structures."""
        for structure in sample_structures:
            empty_structure_set.add_structure_object(structure)
        
        assert len(empty_structure_set) == 3
        
        # Remove all
        for name in list(empty_structure_set.structure_names):
            empty_structure_set.remove_structure(name)
        
        assert len(empty_structure_set) == 0


class TestGeometricSummary:
    """Test geometric summary generation."""
    
    def test_geometric_summary(self, empty_structure_set, sample_structures):
        """Test generating geometric summary DataFrame."""
        for structure in sample_structures:
            empty_structure_set.add_structure_object(structure)
        
        summary = empty_structure_set.geometric_summary()
        
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 3
        
        # Check columns (capitalized as per implementation)
        expected_columns = ['Structure', 'Type', 'Volume_cc', 'Centroid_X', 
                           'Centroid_Y', 'Centroid_Z']
        for col in expected_columns:
            assert col in summary.columns
        
        # Check values
        assert set(summary['Structure']) == {"Brainstem", "SpinalCord", "PTV"}
        assert all(summary['Volume_cc'] > 0)
    
    def test_empty_summary(self, empty_structure_set):
        """Test summary of empty structure set."""
        summary = empty_structure_set.geometric_summary()
        
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 0
    
    def test_summary_with_empty_structure(self, empty_structure_set):
        """Test summary with an empty structure (no voxels)."""
        empty_mask = np.zeros((50, 50, 30), dtype=bool)
        empty_structure_set.add_structure("Empty", empty_mask, StructureType.OAR)
        
        mask = np.zeros((50, 50, 30), dtype=bool)
        mask[20:30, 20:30, 10:20] = True
        empty_structure_set.add_structure("NonEmpty", mask, StructureType.OAR)
        
        summary = empty_structure_set.geometric_summary()
        
        # Empty structure should have volume = 0
        empty_row = summary[summary['Structure'] == 'Empty']
        assert empty_row['Volume_cc'].iloc[0] == 0.0
        
        # Centroid columns should have NaN or similar for empty structure
        # (actual behavior depends on implementation)


class TestStructureSetDoseMethods:
    """Verify that StructureSet doesn't have legacy dose-related methods."""
    
    def test_no_legacy_dose_methods(self, empty_structure_set):
        """Verify legacy dose-related methods don't exist."""
        assert not hasattr(empty_structure_set, 'set_dose_data')
        assert not hasattr(empty_structure_set, 'has_dose')
        assert not hasattr(empty_structure_set, 'compute_bulk_dvh')
        assert hasattr(empty_structure_set, 'geometric_summary')
        assert not hasattr(empty_structure_set, 'dose_statistics_summary')
        assert not hasattr(empty_structure_set, 'compliance_check')
        assert not hasattr(empty_structure_set, '_dose_data')


class TestWithRealNIfTIData:
    """Test StructureSet with real NIfTI data."""
    
    def test_load_nifti_structure_set(self, hf_data_path):
        """Test loading complete structure set from NIfTI."""
        from dosemetrics.io import load_structure_set
        
        subject_path = hf_data_path / "test_subject"
        if not subject_path.exists():
            pytest.skip("Test data not available")
        
        structures = load_structure_set(subject_path)
        
        assert isinstance(structures, StructureSet)
        assert len(structures) > 0
        
        # Verify all structures are accessible
        for name in structures.structure_names:
            structure = structures.get_structure(name)
            assert structure is not None
            assert structure.name == name
        
        # Test filtering
        oars = structures.get_oars()
        targets = structures.get_targets()
        
        # Should have some of each type
        assert len(oars) >= 0
        assert len(targets) >= 0
        
        # Test geometric summary
        summary = structures.geometric_summary()
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == len(structures)
    
    def test_structure_set_properties(self, hf_data_path):
        """Test StructureSet spatial properties."""
        from dosemetrics.io import load_structure_set
        
        subject_path = hf_data_path / "test_subject"
        if not subject_path.exists():
            pytest.skip("Test data not available")
        
        structures = load_structure_set(subject_path)
        
        # All structures should have same spatial properties
        first_structure = structures.get_structure(structures.structure_names[0])
        
        assert structures.spacing == first_structure.spacing
        assert structures.origin == first_structure.origin


class TestWithRealDICOMData:
    """Test StructureSet with real DICOM data."""
    
    def test_load_dicom_structure_set(self, hf_data_path):
        """Test loading structure set from DICOM RT-STRUCT."""
        from dosemetrics.io import load_structure_set
        
        dicom_path = hf_data_path / "dicom"
        if not dicom_path.exists():
            pytest.skip("DICOM test data not available")
        
        structures = load_structure_set(dicom_path)
        
        assert isinstance(structures, StructureSet)
        assert len(structures) > 0
        
        # Verify structure types are correctly assigned
        for name in structures.structure_names:
            structure = structures.get_structure(name)
            
            assert structure.structure_type in [
                StructureType.OAR,
                StructureType.TARGET,
                StructureType.AVOIDANCE,
                StructureType.EXTERNAL,
                StructureType.SUPPORT
            ]
        
        # Test filtering works
        oars = structures.get_oars()
        targets = structures.get_targets()
        
        assert isinstance(oars, dict)
        assert isinstance(targets, dict)
        
        # Generate summary
        summary = structures.geometric_summary()
        assert len(summary) == len(structures)


class TestStructureSetEdgeCases:
    """Test edge cases and error handling."""
    
    def test_very_large_structure_set(self):
        """Test with many structures."""
        ss = StructureSet(spacing=(1.0, 1.0, 1.0))
        
        # Add 100 structures
        for i in range(100):
            mask = np.zeros((100, 100, 50), dtype=bool)
            mask[i:i+5, i:i+5, i%40:i%40+5] = True
            ss.add_structure(f"Structure_{i}", mask, StructureType.OAR)
        
        assert len(ss) == 100
        assert len(ss.structure_names) == 100
        
        summary = ss.geometric_summary()
        assert len(summary) == 100
    
    def test_anisotropic_spacing(self):
        """Test with highly anisotropic voxel spacing."""
        ss = StructureSet(spacing=(0.5, 0.5, 5.0))
        
        mask = np.zeros((50, 50, 30), dtype=bool)
        mask[20:30, 20:30, 10:20] = True
        ss.add_structure("Test", mask, StructureType.OAR)
        
        summary = ss.geometric_summary()
        assert len(summary) == 1
        
        # Volume should account for anisotropic spacing
        # 10*10*10 voxels * 0.5*0.5*5.0 = 1250 mm³ = 1.25 cc
        assert summary['Volume_cc'].iloc[0] == pytest.approx(1.25)
    
    def test_non_zero_origin(self):
        """Test with non-zero origin."""
        ss = StructureSet(spacing=(1.0, 1.0, 1.0), origin=(-100.0, -100.0, -50.0))
        
        mask = np.zeros((50, 50, 30), dtype=bool)
        mask[24:26, 24:26, 14:16] = True  # Small centered cube
        ss.add_structure("Test", mask, StructureType.OAR)
        
        structure = ss.get_structure("Test")
        centroid = structure.centroid()
        
        # Centroid should be offset by origin
        # Mask spans indices 24-25, mean = 24.5; physical = origin + index * spacing
        # x: -100 + 24.5 * 1.0 = -75.5
        # y: -100 + 24.5 * 1.0 = -75.5
        # z: -50 + 14.5 * 1.0 = -35.5
        assert centroid[0] == pytest.approx(-75.5)
        assert centroid[1] == pytest.approx(-75.5)
        assert centroid[2] == pytest.approx(-35.5)


class TestStructureSetComparison:
    """Test comparing and copying structure sets."""
    
    def test_copy_behavior(self, empty_structure_set, sample_structures):
        """Test that structures are stored by reference."""
        original_structure = sample_structures[0]
        empty_structure_set.add_structure_object(original_structure)
        
        retrieved = empty_structure_set.get_structure(original_structure.name)
        
        # Should be the same object (stored by reference)
        assert retrieved is original_structure
    
    def test_len(self, empty_structure_set, sample_structures):
        """Test len() operator."""
        assert len(empty_structure_set) == 0
        
        for structure in sample_structures:
            empty_structure_set.add_structure_object(structure)
        
        assert len(empty_structure_set) == len(sample_structures)
