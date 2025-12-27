"""
Comprehensive tests for dosemetrics.metrics.statistics module.

Tests all dose statistics functions with synthetic and real data.
"""

import pytest
import numpy as np
from pathlib import Path
from huggingface_hub import snapshot_download

from dosemetrics.dose import Dose
from dosemetrics.structures import OAR, Target
from dosemetrics.metrics import statistics


@pytest.fixture(scope="module")
def hf_data_path():
    """Download test data from HuggingFace once per module."""
    data_path = snapshot_download(
        repo_id="contouraid/dosemetrics-data",
        repo_type="dataset"
    )
    return Path(data_path)


@pytest.fixture
def sample_dose():
    """Create a sample dose distribution with known properties."""
    # Create a dose distribution with predictable values
    dose_array = np.zeros((50, 50, 30))
    
    # Create a gradient: 0-60 Gy
    for z in range(30):
        dose_array[:, :, z] = z * 2.0  # 0, 2, 4, ..., 58 Gy
    
    return Dose(
        dose_array=dose_array,
        spacing=(2.0, 2.0, 3.0),
        origin=(0.0, 0.0, 0.0),
        name="TestDose"
    )


@pytest.fixture
def uniform_dose():
    """Create a uniform dose distribution."""
    dose_array = np.ones((50, 50, 30)) * 50.0  # Uniform 50 Gy
    return Dose(dose_array, (2.0, 2.0, 3.0), (0.0, 0.0, 0.0))


@pytest.fixture
def sample_structure():
    """Create a sample structure."""
    mask = np.zeros((50, 50, 30), dtype=bool)
    mask[20:30, 20:30, 10:20] = True  # Small cube
    
    return Target(
        name="TestPTV",
        mask=mask,
        spacing=(2.0, 2.0, 3.0),
        origin=(0.0, 0.0, 0.0)
    )


class TestComputeDoseStatistics:
    """Test comprehensive dose statistics computation."""
    
    def test_compute_dose_statistics_basic(self, sample_dose, sample_structure):
        """Test basic statistics computation."""
        stats = statistics.compute_dose_statistics(sample_dose, sample_structure)
        
        # Check all expected keys
        expected_keys = ['mean_dose', 'max_dose', 'min_dose', 'median_dose',
                        'std_dose', 'D95', 'D50', 'D05', 'D02', 'D98']
        for key in expected_keys:
            assert key in stats
            assert isinstance(stats[key], (float, np.floating))
            assert stats[key] >= 0
    
    def test_statistics_relationships(self, sample_dose, sample_structure):
        """Test logical relationships between statistics."""
        stats = statistics.compute_dose_statistics(sample_dose, sample_structure)
        
        # Ordering relationships
        assert stats['min_dose'] <= stats['mean_dose'] <= stats['max_dose']
        assert stats['D98'] <= stats['D95'] <= stats['D50']
        assert stats['D50'] <= stats['D05'] <= stats['D02']
        assert stats['D02'] <= stats['max_dose']
        
        # Median relationship
        assert abs(stats['median_dose'] - stats['D50']) < 0.01
    
    def test_uniform_dose_statistics(self, uniform_dose, sample_structure):
        """Test statistics on uniform dose (std should be ~0)."""
        stats = statistics.compute_dose_statistics(uniform_dose, sample_structure)
        
        # All statistics should be 50.0 for uniform dose
        assert abs(stats['mean_dose'] - 50.0) < 0.01
        assert abs(stats['max_dose'] - 50.0) < 0.01
        assert abs(stats['min_dose'] - 50.0) < 0.01
        assert abs(stats['median_dose'] - 50.0) < 0.01
        assert stats['std_dose'] < 0.01  # Near zero
    
    def test_empty_structure(self, sample_dose):
        """Test with empty structure."""
        empty_structure = Target(
            name="Empty",
            mask=np.zeros((50, 50, 30), dtype=bool),
            spacing=(2.0, 2.0, 3.0),
            origin=(0.0, 0.0, 0.0)
        )
        
        stats = statistics.compute_dose_statistics(sample_dose, empty_structure)
        
        # All stats should be 0 for empty structure
        for key, value in stats.items():
            assert value == 0.0


class TestIndividualStatistics:
    """Test individual statistic functions."""
    
    def test_compute_mean_dose(self, sample_dose, sample_structure):
        """Test mean dose computation."""
        mean = statistics.compute_mean_dose(sample_dose, sample_structure)
        
        assert isinstance(mean, (float, np.floating))
        assert mean > 0
        
        # Verify against full stats
        stats = statistics.compute_dose_statistics(sample_dose, sample_structure)
        assert abs(mean - stats['mean_dose']) < 0.01
    
    def test_compute_max_dose(self, sample_dose, sample_structure):
        """Test max dose computation."""
        max_dose = statistics.compute_max_dose(sample_dose, sample_structure)
        
        assert isinstance(max_dose, (float, np.floating))
        assert max_dose > 0
        
        # Should match statistics
        stats = statistics.compute_dose_statistics(sample_dose, sample_structure)
        assert abs(max_dose - stats['max_dose']) < 0.01
    
    def test_compute_min_dose(self, sample_dose, sample_structure):
        """Test min dose computation."""
        min_dose = statistics.compute_min_dose(sample_dose, sample_structure)
        
        assert isinstance(min_dose, (float, np.floating))
        assert min_dose >= 0
        
        # Should match statistics
        stats = statistics.compute_dose_statistics(sample_dose, sample_structure)
        assert abs(min_dose - stats['min_dose']) < 0.01
    
    def test_compute_median_dose(self, sample_dose, sample_structure):
        """Test median dose computation."""
        median = statistics.compute_median_dose(sample_dose, sample_structure)
        
        assert isinstance(median, (float, np.floating))
        assert median > 0
        
        # Should match D50
        stats = statistics.compute_dose_statistics(sample_dose, sample_structure)
        assert abs(median - stats['D50']) < 0.01
    
    def test_compute_dose_percentile(self, sample_dose, sample_structure):
        """Test dose percentile computation."""
        d95 = statistics.compute_dose_percentile(sample_dose, sample_structure, 95)
        d50 = statistics.compute_dose_percentile(sample_dose, sample_structure, 50)
        d05 = statistics.compute_dose_percentile(sample_dose, sample_structure, 5)
        
        # Check ordering
        assert d95 <= d50 <= d05
        
        # Verify against full stats
        stats = statistics.compute_dose_statistics(sample_dose, sample_structure)
        assert abs(d95 - stats['D95']) < 0.01
        assert abs(d50 - stats['D50']) < 0.01
        assert abs(d05 - stats['D05']) < 0.01
    
    def test_percentile_bounds(self, sample_dose, sample_structure):
        """Test percentile boundary values."""
        d0 = statistics.compute_dose_percentile(sample_dose, sample_structure, 0)
        d100 = statistics.compute_dose_percentile(sample_dose, sample_structure, 100)
        
        max_dose = statistics.compute_max_dose(sample_dose, sample_structure)
        min_dose = statistics.compute_min_dose(sample_dose, sample_structure)
        
        # D0 should be close to max, D100 close to min
        assert abs(d0 - max_dose) < 1.0
        assert abs(d100 - min_dose) < 1.0


class TestRealDataStatistics:
    """Test statistics with real HuggingFace data."""
    
    def test_nifti_dose_statistics(self, hf_data_path):
        """Test statistics on real NIfTI dose data."""
        from dosemetrics.io import load_structure_set
        
        subject_path = hf_data_path / "test_subject"
        if not subject_path.exists():
            pytest.skip("Test data not available")
        
        # Load dose and structures
        dose = Dose.from_nifti(subject_path / "Dose.nii.gz")
        structures = load_structure_set(subject_path)
        
        # Test on all structures
        for name in structures.structure_names:
            structure = structures.get_structure(name)
            
            # Compute statistics
            stats = statistics.compute_dose_statistics(dose, structure)
            
            # Verify reasonable values
            assert stats['mean_dose'] > 0
            assert stats['max_dose'] > 0
            assert stats['min_dose'] >= 0
            assert stats['std_dose'] >= 0
            
            # Check relationships
            assert stats['min_dose'] <= stats['mean_dose'] <= stats['max_dose']
            assert stats['D98'] <= stats['D50'] <= stats['D02']
    
    def test_multiple_structures_comparison(self, hf_data_path):
        """Test comparing statistics across structures."""
        from dosemetrics.io import load_structure_set
        
        subject_path = hf_data_path / "test_subject"
        if not subject_path.exists():
            pytest.skip("Test data not available")
        
        dose = Dose.from_nifti(subject_path / "Dose.nii.gz")
        structures = load_structure_set(subject_path)
        
        all_stats = {}
        for name in structures.structure_names:
            structure = structures.get_structure(name)
            all_stats[name] = statistics.compute_dose_statistics(dose, structure)
        
        # Verify we got statistics for all structures
        assert len(all_stats) == len(structures.structure_names)
        
        # Each structure should have all keys
        for name, stats in all_stats.items():
            assert 'mean_dose' in stats
            assert 'max_dose' in stats
            assert 'D95' in stats


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_incompatible_dose_structure(self, sample_dose):
        """Test error when dose and structure are incompatible."""
        incompatible = Target(
            name="Incompatible",
            mask=np.zeros((30, 30, 20), dtype=bool),  # Different shape
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0)
        )
        
        # Should raise ValueError
        with pytest.raises(ValueError):
            statistics.compute_dose_statistics(sample_dose, incompatible)
    
    def test_single_voxel_structure(self, sample_dose):
        """Test statistics on single-voxel structure."""
        single_voxel = Target(
            name="SingleVoxel",
            mask=np.zeros((50, 50, 30), dtype=bool),
            spacing=(2.0, 2.0, 3.0),
            origin=(0.0, 0.0, 0.0)
        )
        single_voxel.mask[25, 25, 15] = True
        
        stats = statistics.compute_dose_statistics(sample_dose, single_voxel)
        
        # For single voxel, all stats should be the same
        assert abs(stats['mean_dose'] - stats['max_dose']) < 0.01
        assert abs(stats['mean_dose'] - stats['min_dose']) < 0.01
        assert stats['std_dose'] < 0.01
    
    def test_zero_dose(self, sample_structure):
        """Test statistics on zero dose."""
        zero_dose = Dose(
            np.zeros((50, 50, 30)),
            spacing=(2.0, 2.0, 3.0),
            origin=(0.0, 0.0, 0.0)
        )
        
        stats = statistics.compute_dose_statistics(zero_dose, sample_structure)
        
        # All stats should be 0
        for value in stats.values():
            assert value == 0.0
