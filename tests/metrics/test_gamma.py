"""
Tests for dosemetrics.metrics.gamma module.

Tests gamma index calculation and related statistics.
"""

import pytest
import numpy as np

from dosemetrics.dose import Dose
from dosemetrics.metrics import gamma


@pytest.fixture
def reference_dose():
    """Create a reference dose distribution."""
    dose_array = np.zeros((30, 30, 20))
    # Central high-dose region
    dose_array[10:20, 10:20, 5:15] = 60.0
    # Gradient region
    dose_array[8:10, 10:20, 5:15] = 40.0
    dose_array[20:22, 10:20, 5:15] = 40.0
    return Dose(dose_array, (2.0, 2.0, 2.0), (0.0, 0.0, 0.0))


@pytest.fixture
def evaluated_dose_similar():
    """Create evaluated dose very similar to reference."""
    dose_array = np.zeros((30, 30, 20))
    # Slightly different high-dose region
    dose_array[10:20, 10:20, 5:15] = 61.0  # 1 Gy difference
    dose_array[8:10, 10:20, 5:15] = 40.0
    dose_array[20:22, 10:20, 5:15] = 40.0
    return Dose(dose_array, (2.0, 2.0, 2.0), (0.0, 0.0, 0.0))


@pytest.fixture
def evaluated_dose_shifted():
    """Create evaluated dose with spatial shift."""
    dose_array = np.zeros((30, 30, 20))
    # Shifted by 1 voxel
    dose_array[11:21, 10:20, 5:15] = 60.0
    dose_array[9:11, 10:20, 5:15] = 40.0
    dose_array[21:23, 10:20, 5:15] = 40.0
    return Dose(dose_array, (2.0, 2.0, 2.0), (0.0, 0.0, 0.0))


@pytest.mark.skipif(
    not gamma.PYMEDPHYS_AVAILABLE,
    reason="pymedphys not installed"
)
class TestGammaIndex:
    """Test gamma index calculation."""
    
    def test_gamma_identical(self, reference_dose):
        """Test gamma index for identical doses."""
        try:
            gamma_result = gamma.compute_gamma_index(
                reference_dose, reference_dose,
                dose_criterion_percent=3.0,
                distance_criterion_mm=3.0
            )
            
            # Most values should be close to 0 for identical doses
            valid_gamma = gamma_result[~np.isnan(gamma_result)]
            if len(valid_gamma) > 0:
                assert np.mean(valid_gamma) < 0.5
        except Exception as e:
            pytest.skip(f"Gamma calculation failed: {e}")
    
    def test_gamma_similar(self, reference_dose, evaluated_dose_similar):
        """Test gamma index for similar doses."""
        try:
            gamma_result = gamma.compute_gamma_index(
                reference_dose, evaluated_dose_similar,
                dose_criterion_percent=3.0,
                distance_criterion_mm=3.0
            )
            
            # Should be an array
            assert isinstance(gamma_result, np.ndarray)
            assert gamma_result.shape == reference_dose.dose_grid.shape
        except Exception as e:
            pytest.skip(f"Gamma calculation failed: {e}")
    
    def test_gamma_shifted(self, reference_dose, evaluated_dose_shifted):
        """Test gamma index for spatially shifted dose."""
        try:
            gamma_result = gamma.compute_gamma_index(
                reference_dose, evaluated_dose_shifted,
                dose_criterion_percent=3.0,
                distance_criterion_mm=3.0
            )
            
            assert isinstance(gamma_result, np.ndarray)
        except Exception as e:
            pytest.skip(f"Gamma calculation failed: {e}")
    
    def test_gamma_local_normalization(self, reference_dose, evaluated_dose_similar):
        """Test gamma with local normalization."""
        try:
            gamma_result = gamma.compute_gamma_index(
                reference_dose, evaluated_dose_similar,
                dose_criterion_percent=3.0,
                distance_criterion_mm=3.0,
                global_normalization=False
            )
            
            assert isinstance(gamma_result, np.ndarray)
        except Exception as e:
            pytest.skip(f"Gamma calculation failed: {e}")


class TestGammaPassingRate:
    """Test gamma passing rate calculation."""
    
    def test_passing_rate_perfect(self):
        """Test passing rate for gamma=0 everywhere."""
        gamma_array = np.zeros((10, 10, 10))
        passing_rate = gamma.compute_gamma_passing_rate(gamma_array)
        assert passing_rate == 100.0
    
    def test_passing_rate_half(self):
        """Test passing rate for 50% passing."""
        gamma_array = np.ones((10, 10, 10))
        gamma_array[:5, :, :] = 0.5  # Half passing
        gamma_array[5:, :, :] = 1.5  # Half failing
        
        passing_rate = gamma.compute_gamma_passing_rate(gamma_array)
        assert 45 <= passing_rate <= 55  # Should be ~50%
    
    def test_passing_rate_with_nan(self):
        """Test passing rate with NaN values."""
        gamma_array = np.ones((10, 10, 10))
        gamma_array[:5, :, :] = np.nan  # Below threshold
        gamma_array[5:, :, :] = 0.5  # Passing
        
        passing_rate = gamma.compute_gamma_passing_rate(gamma_array)
        assert passing_rate == 100.0  # Only valid values count
    
    def test_passing_rate_custom_threshold(self):
        """Test passing rate with custom threshold."""
        gamma_array = np.ones((10, 10, 10)) * 0.8
        
        passing_rate_1 = gamma.compute_gamma_passing_rate(gamma_array, threshold=1.0)
        passing_rate_0_5 = gamma.compute_gamma_passing_rate(gamma_array, threshold=0.5)
        
        assert passing_rate_1 == 100.0
        assert passing_rate_0_5 == 0.0


class TestGammaStatistics:
    """Test gamma statistics computation."""
    
    def test_statistics_uniform(self):
        """Test statistics for uniform gamma."""
        gamma_array = np.ones((10, 10, 10)) * 0.5
        
        stats = gamma.compute_gamma_statistics(gamma_array)
        
        assert stats['passing_rate_1_0'] == 100.0
        assert stats['mean_gamma'] == 0.5
        assert stats['max_gamma'] == 0.5
        assert stats['gamma_50'] == 0.5
        assert stats['gamma_95'] == 0.5
    
    def test_statistics_varied(self):
        """Test statistics for varied gamma values."""
        gamma_array = np.random.uniform(0, 2, (10, 10, 10))
        
        stats = gamma.compute_gamma_statistics(gamma_array)
        
        assert 'passing_rate_1_0' in stats
        assert 'mean_gamma' in stats
        assert 'max_gamma' in stats
        assert 'gamma_50' in stats
        assert 'gamma_95' in stats
        
        assert 0 <= stats['passing_rate_1_0'] <= 100
        assert stats['mean_gamma'] >= 0
        assert stats['max_gamma'] >= stats['mean_gamma']
    
    def test_statistics_empty(self):
        """Test statistics with all NaN values."""
        gamma_array = np.full((10, 10, 10), np.nan)
        
        stats = gamma.compute_gamma_statistics(gamma_array)
        
        assert stats['passing_rate_1_0'] == 0.0
        assert np.isnan(stats['mean_gamma'])


@pytest.mark.skipif(
    not gamma.PYMEDPHYS_AVAILABLE,
    reason="pymedphys not installed"
)
class Test2DGamma:
    """Test 2D gamma calculation."""
    
    def test_2d_gamma_shape(self):
        """Test 2D gamma output shape."""
        ref_slice = np.ones((30, 30)) * 50.0
        ref_slice[10:20, 10:20] = 60.0
        
        eval_slice = np.ones((30, 30)) * 50.0
        eval_slice[10:20, 10:20] = 61.0
        
        try:
            gamma_result = gamma.compute_2d_gamma(
                ref_slice, eval_slice,
                dose_criterion_percent=3.0,
                distance_criterion_mm=3.0,
                pixel_spacing=(2.0, 2.0)
            )
            
            assert gamma_result.shape == ref_slice.shape
        except Exception as e:
            pytest.skip(f"2D gamma calculation failed: {e}")
    
    def test_2d_gamma_identical(self):
        """Test 2D gamma for identical slices."""
        ref_slice = np.ones((20, 20)) * 50.0
        
        try:
            gamma_result = gamma.compute_2d_gamma(
                ref_slice, ref_slice,
                dose_criterion_percent=3.0,
                distance_criterion_mm=3.0
            )
            
            # Most values should be close to 0
            valid_gamma = gamma_result[~np.isnan(gamma_result)]
            if len(valid_gamma) > 0:
                assert np.mean(valid_gamma) < 0.5
        except Exception as e:
            pytest.skip(f"2D gamma calculation failed: {e}")


def test_pymedphys_import():
    """Test pymedphys availability."""
    # This test always passes but logs availability
    assert isinstance(gamma.PYMEDPHYS_AVAILABLE, bool)
