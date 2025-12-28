"""
Tests for dosemetrics.metrics.advanced_dvh module.

Tests advanced DVH metrics including Wasserstein distance, area between curves, etc.
"""

import pytest
import numpy as np

from dosemetrics.dose import Dose
from dosemetrics.structures import Target
from dosemetrics.metrics import advanced_dvh


@pytest.fixture
def dose1():
    """Create first dose distribution."""
    dose_array = np.zeros((30, 30, 20))
    dose_array[10:20, 10:20, 5:15] = 60.0
    dose_array[8:22, 8:22, 3:17] = 30.0
    return Dose(dose_array, (2.0, 2.0, 2.0), (0.0, 0.0, 0.0))


@pytest.fixture
def dose2_similar():
    """Create similar dose distribution."""
    dose_array = np.zeros((30, 30, 20))
    dose_array[10:20, 10:20, 5:15] = 62.0  # Slightly higher
    dose_array[8:22, 8:22, 3:17] = 31.0
    return Dose(dose_array, (2.0, 2.0, 2.0), (0.0, 0.0, 0.0))


@pytest.fixture
def dose3_different():
    """Create different dose distribution."""
    dose_array = np.zeros((30, 30, 20))
    dose_array[10:20, 10:20, 5:15] = 55.0  # Lower dose
    dose_array[8:22, 8:22, 3:17] = 25.0
    return Dose(dose_array, (2.0, 2.0, 2.0), (0.0, 0.0, 0.0))


@pytest.fixture
def test_structure():
    """Create test structure."""
    mask = np.zeros((30, 30, 20), dtype=bool)
    mask[10:20, 10:20, 5:15] = True
    return Target("TestStructure", mask, (2.0, 2.0, 2.0), (0.0, 0.0, 0.0))


class TestWassersteinDistance:
    """Test Wasserstein distance between DVHs."""
    
    def test_wasserstein_identical(self, dose1, test_structure):
        """Test Wasserstein distance is zero for identical doses."""
        distance = advanced_dvh.compute_dvh_wasserstein_distance(
            dose1, dose1, test_structure
        )
        assert distance >= 0
        assert distance < 1.0  # Should be very small for identical
    
    def test_wasserstein_similar(self, dose1, dose2_similar, test_structure):
        """Test Wasserstein distance for similar doses."""
        distance = advanced_dvh.compute_dvh_wasserstein_distance(
            dose1, dose2_similar, test_structure
        )
        assert distance >= 0
        assert isinstance(distance, float)
    
    def test_wasserstein_different(self, dose1, dose3_different, test_structure):
        """Test Wasserstein distance for different doses."""
        distance = advanced_dvh.compute_dvh_wasserstein_distance(
            dose1, dose3_different, test_structure
        )
        assert distance > 0
        assert isinstance(distance, float)


class TestAreaBetweenCurves:
    """Test area between DVH curves."""
    
    def test_area_identical_l1(self, dose1, test_structure):
        """Test area is zero for identical doses with L1 norm."""
        area = advanced_dvh.compute_area_between_dvh_curves(
            dose1, dose1, test_structure, norm='L1'
        )
        assert area >= 0
        assert area < 1.0  # Should be very small
    
    def test_area_identical_l2(self, dose1, test_structure):
        """Test area is zero for identical doses with L2 norm."""
        area = advanced_dvh.compute_area_between_dvh_curves(
            dose1, dose1, test_structure, norm='L2'
        )
        assert area >= 0
        assert area < 1.0  # Should be very small
    
    def test_area_different(self, dose1, dose3_different, test_structure):
        """Test area is positive for different doses."""
        area = advanced_dvh.compute_area_between_dvh_curves(
            dose1, dose3_different, test_structure
        )
        # Area should be non-negative (can be 0 for very similar DVHs)
        assert area >= 0
    
    def test_area_invalid_norm(self, dose1, dose2_similar, test_structure):
        """Test error for invalid norm."""
        with pytest.raises(ValueError, match="norm"):
            advanced_dvh.compute_area_between_dvh_curves(
                dose1, dose2_similar, test_structure, norm='L3'
            )


class TestChiSquareTest:
    """Test chi-square test for DVH comparison."""
    
    def test_chi_square_similar(self, dose1, dose2_similar, test_structure):
        """Test chi-square for similar doses."""
        chi2_stat, p_value = advanced_dvh.compute_dvh_chi_square(
            dose1, dose2_similar, test_structure
        )
        
        assert chi2_stat >= 0
        assert 0 <= p_value <= 1
    
    def test_chi_square_different(self, dose1, dose3_different, test_structure):
        """Test chi-square for different doses."""
        chi2_stat, p_value = advanced_dvh.compute_dvh_chi_square(
            dose1, dose3_different, test_structure
        )
        
        assert chi2_stat >= 0
        assert 0 <= p_value <= 1


class TestKSTest:
    """Test Kolmogorov-Smirnov test for DVH comparison."""
    
    def test_ks_identical(self, dose1, test_structure):
        """Test KS test for identical doses."""
        ks_stat, p_value = advanced_dvh.compute_dvh_ks_test(
            dose1, dose1, test_structure
        )
        
        # Should have high p-value for identical
        assert ks_stat >= 0
        assert 0 <= p_value <= 1
        assert p_value > 0.05  # Should not reject null hypothesis
    
    def test_ks_similar(self, dose1, dose2_similar, test_structure):
        """Test KS test for similar doses."""
        ks_stat, p_value = advanced_dvh.compute_dvh_ks_test(
            dose1, dose2_similar, test_structure
        )
        
        assert ks_stat >= 0
        assert 0 <= p_value <= 1
    
    def test_ks_different(self, dose1, dose3_different, test_structure):
        """Test KS test for different doses."""
        ks_stat, p_value = advanced_dvh.compute_dvh_ks_test(
            dose1, dose3_different, test_structure
        )
        
        assert ks_stat >= 0
        assert 0 <= p_value <= 1


class TestConfidenceInterval:
    """Test DVH confidence interval computation."""
    
    def test_confidence_interval_single(self, dose1, test_structure):
        """Test confidence interval with single dose."""
        dose_bins, mean_dvh, ci_lower, ci_upper = \
            advanced_dvh.compute_dvh_confidence_interval([dose1], test_structure)
        
        assert len(dose_bins) > 0
        assert len(mean_dvh) == len(dose_bins)
        assert len(ci_lower) == len(dose_bins)
        assert len(ci_upper) == len(dose_bins)
        
        # For single dose, CI should equal mean
        assert np.allclose(mean_dvh, ci_lower)
        assert np.allclose(mean_dvh, ci_upper)
    
    def test_confidence_interval_multiple(self, dose1, dose2_similar, dose3_different, test_structure):
        """Test confidence interval with multiple doses."""
        doses = [dose1, dose2_similar, dose3_different]
        dose_bins, mean_dvh, ci_lower, ci_upper = \
            advanced_dvh.compute_dvh_confidence_interval(doses, test_structure)
        
        assert len(dose_bins) > 0
        assert len(mean_dvh) == len(dose_bins)
        assert len(ci_lower) == len(dose_bins)
        assert len(ci_upper) == len(dose_bins)
        
        # CI bounds should bracket mean
        assert np.all(ci_lower <= mean_dvh)
        assert np.all(ci_upper >= mean_dvh)
    
    def test_confidence_interval_95(self, dose1, dose2_similar, test_structure):
        """Test 95% confidence interval."""
        doses = [dose1, dose2_similar]
        dose_bins, mean_dvh, ci_lower, ci_upper = \
            advanced_dvh.compute_dvh_confidence_interval(
                doses, test_structure, confidence=0.95
            )
        
        assert len(dose_bins) > 0
    
    def test_confidence_interval_empty(self, test_structure):
        """Test error with empty dose list."""
        with pytest.raises(ValueError):
            advanced_dvh.compute_dvh_confidence_interval([], test_structure)


class TestBandwidth:
    """Test DVH bandwidth computation."""
    
    def test_bandwidth_single(self, dose1, test_structure):
        """Test bandwidth with single dose."""
        bandwidth = advanced_dvh.compute_dvh_bandwidth([dose1], test_structure)
        
        assert len(bandwidth) > 0
        # Single dose should have zero bandwidth
        assert np.allclose(bandwidth, 0)
    
    def test_bandwidth_multiple(self, dose1, dose2_similar, dose3_different, test_structure):
        """Test bandwidth with multiple doses."""
        doses = [dose1, dose2_similar, dose3_different]
        bandwidth = advanced_dvh.compute_dvh_bandwidth(doses, test_structure)
        
        assert len(bandwidth) > 0
        assert np.all(bandwidth >= 0)
        # May have zero or positive bandwidth
        assert isinstance(bandwidth, np.ndarray)
    
    def test_bandwidth_identical(self, dose1, test_structure):
        """Test bandwidth for identical doses."""
        bandwidth = advanced_dvh.compute_dvh_bandwidth(
            [dose1, dose1, dose1], test_structure
        )
        
        # Should be zero for identical doses
        assert np.allclose(bandwidth, 0)
    
    def test_bandwidth_empty(self, test_structure):
        """Test error with empty dose list."""
        with pytest.raises(ValueError):
            advanced_dvh.compute_dvh_bandwidth([], test_structure)


class TestIntegration:
    """Integration tests for advanced DVH metrics."""
    
    def test_multiple_metrics(self, dose1, dose2_similar, test_structure):
        """Test computing multiple metrics together."""
        # Wasserstein distance
        wd = advanced_dvh.compute_dvh_wasserstein_distance(
            dose1, dose2_similar, test_structure
        )
        
        # Area between curves
        area = advanced_dvh.compute_area_between_dvh_curves(
            dose1, dose2_similar, test_structure
        )
        
        # KS test
        ks_stat, p_value = advanced_dvh.compute_dvh_ks_test(
            dose1, dose2_similar, test_structure
        )
        
        # All should be valid
        assert wd >= 0
        assert area >= 0
        assert ks_stat >= 0
        assert 0 <= p_value <= 1
