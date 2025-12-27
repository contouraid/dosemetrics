"""
Comprehensive tests for dosemetrics.metrics.geometric module.

Tests geometric similarity metrics for structure comparison.
"""

import pytest
import numpy as np
from pathlib import Path
from huggingface_hub import snapshot_download

from dosemetrics.structures import Target, OAR
from dosemetrics.metrics import geometric


@pytest.fixture(scope="module")
def hf_data_path():
    """Download test data from HuggingFace once per module."""
    data_path = snapshot_download(
        repo_id="contouraid/dosemetrics-data",
        repo_type="dataset"
    )
    return Path(data_path)


@pytest.fixture
def identical_structures():
    """Create two identical structures."""
    mask = np.zeros((50, 50, 30), dtype=bool)
    mask[20:30, 20:30, 10:20] = True
    
    struct1 = Target("PTV1", mask.copy(), (2.0, 2.0, 3.0), (0.0, 0.0, 0.0))
    struct2 = Target("PTV2", mask.copy(), (2.0, 2.0, 3.0), (0.0, 0.0, 0.0))
    
    return struct1, struct2


@pytest.fixture
def overlapping_structures():
    """Create two partially overlapping structures."""
    mask1 = np.zeros((50, 50, 30), dtype=bool)
    mask1[20:30, 20:30, 10:20] = True
    
    mask2 = np.zeros((50, 50, 30), dtype=bool)
    mask2[25:35, 25:35, 15:25] = True  # Partial overlap
    
    struct1 = Target("PTV1", mask1, (2.0, 2.0, 3.0), (0.0, 0.0, 0.0))
    struct2 = Target("PTV2", mask2, (2.0, 2.0, 3.0), (0.0, 0.0, 0.0))
    
    return struct1, struct2


@pytest.fixture
def no_overlap_structures():
    """Create two non-overlapping structures."""
    mask1 = np.zeros((50, 50, 30), dtype=bool)
    mask1[10:20, 10:20, 5:15] = True
    
    mask2 = np.zeros((50, 50, 30), dtype=bool)
    mask2[30:40, 30:40, 20:25] = True  # No overlap
    
    struct1 = Target("PTV1", mask1, (2.0, 2.0, 3.0), (0.0, 0.0, 0.0))
    struct2 = Target("PTV2", mask2, (2.0, 2.0, 3.0), (0.0, 0.0, 0.0))
    
    return struct1, struct2


class TestDiceCoefficient:
    """Test Dice coefficient computation."""
    
    def test_dice_identical(self, identical_structures):
        """Test Dice=1 for identical structures."""
        struct1, struct2 = identical_structures
        dice = geometric.compute_dice_coefficient(struct1, struct2)
        
        assert isinstance(dice, (float, np.floating))
        assert dice == 1.0
    
    def test_dice_no_overlap(self, no_overlap_structures):
        """Test Dice=0 for non-overlapping structures."""
        struct1, struct2 = no_overlap_structures
        dice = geometric.compute_dice_coefficient(struct1, struct2)
        
        assert dice == 0.0
    
    def test_dice_partial_overlap(self, overlapping_structures):
        """Test 0<Dice<1 for partial overlap."""
        struct1, struct2 = overlapping_structures
        dice = geometric.compute_dice_coefficient(struct1, struct2)
        
        assert 0 < dice < 1.0
    
    def test_dice_symmetry(self, overlapping_structures):
        """Test Dice is symmetric."""
        struct1, struct2 = overlapping_structures
        
        dice12 = geometric.compute_dice_coefficient(struct1, struct2)
        dice21 = geometric.compute_dice_coefficient(struct2, struct1)
        
        assert abs(dice12 - dice21) < 1e-10


class TestJaccardIndex:
    """Test Jaccard index (IoU) computation."""
    
    def test_jaccard_identical(self, identical_structures):
        """Test Jaccard=1 for identical structures."""
        struct1, struct2 = identical_structures
        jaccard = geometric.compute_jaccard_index(struct1, struct2)
        
        assert jaccard == 1.0
    
    def test_jaccard_no_overlap(self, no_overlap_structures):
        """Test Jaccard=0 for non-overlapping structures."""
        struct1, struct2 = no_overlap_structures
        jaccard = geometric.compute_jaccard_index(struct1, struct2)
        
        assert jaccard == 0.0
    
    def test_jaccard_partial_overlap(self, overlapping_structures):
        """Test 0<Jaccard<1 for partial overlap."""
        struct1, struct2 = overlapping_structures
        jaccard = geometric.compute_jaccard_index(struct1, struct2)
        
        assert 0 < jaccard < 1.0
    
    def test_jaccard_dice_relationship(self, overlapping_structures):
        """Test Jaccard <= Dice (always true)."""
        struct1, struct2 = overlapping_structures
        
        dice = geometric.compute_dice_coefficient(struct1, struct2)
        jaccard = geometric.compute_jaccard_index(struct1, struct2)
        
        assert jaccard <= dice + 1e-10


class TestVolumeDifference:
    """Test volume difference computation."""
    
    def test_volume_diff_identical(self, identical_structures):
        """Test volume difference = 0 for identical structures."""
        struct1, struct2 = identical_structures
        
        vol_diff = geometric.compute_volume_difference(struct1, struct2)
        
        assert abs(vol_diff) < 0.01
    
    def test_volume_diff_different_sizes(self):
        """Test volume difference for different sized structures."""
        # Small structure
        mask1 = np.zeros((50, 50, 30), dtype=bool)
        mask1[20:25, 20:25, 10:15] = True  # 5x5x5 = 125 voxels
        struct1 = Target("Small", mask1, (2.0, 2.0, 3.0), (0.0, 0.0, 0.0))
        
        # Large structure
        mask2 = np.zeros((50, 50, 30), dtype=bool)
        mask2[20:30, 20:30, 10:20] = True  # 10x10x10 = 1000 voxels
        struct2 = Target("Large", mask2, (2.0, 2.0, 3.0), (0.0, 0.0, 0.0))
        
        vol_diff = geometric.compute_volume_difference(struct1, struct2)
        
        # Volume difference is absolute, always positive
        assert vol_diff > 0
        
        # Absolute difference should be (1000-125) * 2*2*3 cc
        expected_diff = (1000 - 125) * 2.0 * 2.0 * 3.0 / 1000  # to cc
        assert abs(vol_diff - expected_diff) < 1.0


class TestVolumeRatio:
    """Test volume ratio computation."""
    
    def test_volume_ratio_identical(self, identical_structures):
        """Test volume ratio = 1 for identical structures."""
        struct1, struct2 = identical_structures
        
        ratio = geometric.compute_volume_ratio(struct1, struct2)
        
        assert abs(ratio - 1.0) < 0.01
    
    def test_volume_ratio_different_sizes(self):
        """Test volume ratio for different sized structures."""
        # Small structure (125 voxels)
        mask1 = np.zeros((50, 50, 30), dtype=bool)
        mask1[20:25, 20:25, 10:15] = True
        struct1 = Target("Small", mask1, (2.0, 2.0, 3.0), (0.0, 0.0, 0.0))
        
        # Large structure (1000 voxels)
        mask2 = np.zeros((50, 50, 30), dtype=bool)
        mask2[20:30, 20:30, 10:20] = True
        struct2 = Target("Large", mask2, (2.0, 2.0, 3.0), (0.0, 0.0, 0.0))
        
        ratio = geometric.compute_volume_ratio(struct1, struct2)
        
        # Ratio should be 125/1000 = 0.125
        assert abs(ratio - 0.125) < 0.01
    
    def test_volume_ratio_reciprocal(self, overlapping_structures):
        """Test volume ratio reciprocal relationship."""
        struct1, struct2 = overlapping_structures
        
        ratio12 = geometric.compute_volume_ratio(struct1, struct2)
        ratio21 = geometric.compute_volume_ratio(struct2, struct1)
        
        # ratio12 * ratio21 should equal 1
        assert abs(ratio12 * ratio21 - 1.0) < 0.01


class TestSensitivitySpecificity:
    """Test sensitivity and specificity metrics."""
    
    def test_sensitivity_identical(self, identical_structures):
        """Test sensitivity = 1 for identical structures."""
        struct1, struct2 = identical_structures
        
        sens = geometric.compute_sensitivity(struct1, struct2)
        
        assert sens == 1.0
    
    def test_specificity_identical(self, identical_structures):
        """Test specificity = 1 for identical structures."""
        struct1, struct2 = identical_structures
        
        spec = geometric.compute_specificity(struct1, struct2)
        
        assert spec == 1.0
    
    def test_sensitivity_no_overlap(self, no_overlap_structures):
        """Test sensitivity = 0 when no overlap."""
        struct1, struct2 = no_overlap_structures
        
        sens = geometric.compute_sensitivity(struct1, struct2)
        
        assert sens == 0.0
    
    def test_sensitivity_partial(self, overlapping_structures):
        """Test 0 < sensitivity < 1 for partial overlap."""
        struct1, struct2 = overlapping_structures
        
        sens = geometric.compute_sensitivity(struct1, struct2)
        
        assert 0 < sens < 1.0


class TestCompareStructureSets:
    """Test structure set comparison."""
    
    def test_compare_structure_sets_basic(self, hf_data_path):
        """Test comparing two structure sets."""
        from dosemetrics.io import load_structure_set
        
        subject_path = hf_data_path / "test_subject"
        if not subject_path.exists():
            pytest.skip("Test data not available")
        
        # Load same structure set twice (perfect match)
        structures1 = load_structure_set(subject_path)
        structures2 = load_structure_set(subject_path)
        
        # Compare all structures
        df = geometric.compare_structure_sets(
            structures1,
            structures2,
            structure_names=structures1.structure_names[:3]  # First 3
        )
        
        # Check DataFrame structure (capitalized columns)
        assert 'Structure' in df.columns
        assert 'Dice' in df.columns
        assert 'Jaccard' in df.columns
        assert 'Volume_Difference_cc' in df.columns
        
        # All metrics should be perfect (comparing to self)
        assert (df['Dice'] == 1.0).all()
        assert (df['Jaccard'] == 1.0).all()
        assert (df['Volume_Difference_cc'].abs() < 0.1).all()


class TestRealDataGeometric:
    """Test geometric metrics with real HuggingFace data."""
    
    def test_geometric_on_real_structures(self, hf_data_path):
        """Test geometric metrics on real structure pairs."""
        from dosemetrics.io import load_structure_set
        
        subject_path = hf_data_path / "test_subject"
        if not subject_path.exists():
            pytest.skip("Test data not available")
        
        structures = load_structure_set(subject_path)
        
        if len(structures.structure_names) < 2:
            pytest.skip("Need at least 2 structures")
        
        # Get first two structures
        struct1 = structures.get_structure(structures.structure_names[0])
        struct2 = structures.get_structure(structures.structure_names[1])
        
        # Compute all geometric metrics
        dice = geometric.compute_dice_coefficient(struct1, struct2)
        jaccard = geometric.compute_jaccard_index(struct1, struct2)
        vol_diff = geometric.compute_volume_difference(struct1, struct2)
        vol_ratio = geometric.compute_volume_ratio(struct1, struct2)
        sens = geometric.compute_sensitivity(struct1, struct2)
        spec = geometric.compute_specificity(struct1, struct2)
        
        # Check all are in valid ranges
        assert 0 <= dice <= 1.0
        assert 0 <= jaccard <= 1.0
        assert vol_ratio >= 0
        assert 0 <= sens <= 1.0
        assert 0 <= spec <= 1.0


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_structures(self):
        """Test with empty structures."""
        mask = np.zeros((50, 50, 30), dtype=bool)
        
        struct1 = Target("Empty1", mask.copy(), (2.0, 2.0, 3.0), (0.0, 0.0, 0.0))
        struct2 = Target("Empty2", mask.copy(), (2.0, 2.0, 3.0), (0.0, 0.0, 0.0))
        
        dice = geometric.compute_dice_coefficient(struct1, struct2)
        jaccard = geometric.compute_jaccard_index(struct1, struct2)
        
        # Empty structures: Dice and Jaccard undefined, but should return 0
        assert dice == 0.0
        assert jaccard == 0.0
    
    def test_incompatible_shapes(self):
        """Test error with incompatible structure shapes."""
        mask1 = np.zeros((50, 50, 30), dtype=bool)
        mask2 = np.zeros((40, 40, 20), dtype=bool)  # Different shape
        
        struct1 = Target("S1", mask1, (2.0, 2.0, 3.0), (0.0, 0.0, 0.0))
        struct2 = Target("S2", mask2, (2.0, 2.0, 3.0), (0.0, 0.0, 0.0))
        
        # Should raise error due to shape mismatch
        with pytest.raises((ValueError, AssertionError)):
            geometric.compute_dice_coefficient(struct1, struct2)
    
    def test_one_empty_one_full(self):
        """Test with one empty and one filled structure."""
        mask1 = np.zeros((50, 50, 30), dtype=bool)
        mask2 = np.ones((50, 50, 30), dtype=bool)
        
        struct1 = Target("Empty", mask1, (2.0, 2.0, 3.0), (0.0, 0.0, 0.0))
        struct2 = Target("Full", mask2, (2.0, 2.0, 3.0), (0.0, 0.0, 0.0))
        
        dice = geometric.compute_dice_coefficient(struct1, struct2)
        sens = geometric.compute_sensitivity(struct1, struct2)
        
        assert dice == 0.0
        assert sens == 0.0  # No true positives
