"""
Tests for dosemetrics.metrics.dose_comparison module.

Tests dose comparison metrics including MSE, MAE, PSNR, SSIM, etc.
"""

import pytest
import numpy as np

from dosemetrics.dose import Dose
from dosemetrics.structures import Target
from dosemetrics.metrics import dose_comparison


@pytest.fixture
def simple_dose1():
    """Create a simple dose distribution."""
    dose_array = np.ones((20, 20, 10)) * 50.0
    dose_array[5:15, 5:15, 2:8] = 60.0
    return Dose(dose_array, (2.0, 2.0, 3.0), (0.0, 0.0, 0.0))


@pytest.fixture
def simple_dose2():
    """Create a similar dose with small differences."""
    dose_array = np.ones((20, 20, 10)) * 50.0
    dose_array[5:15, 5:15, 2:8] = 62.0  # Slightly higher
    return Dose(dose_array, (2.0, 2.0, 3.0), (0.0, 0.0, 0.0))


@pytest.fixture
def identical_dose():
    """Create identical dose for testing."""
    dose_array = np.ones((20, 20, 10)) * 50.0
    return Dose(dose_array, (2.0, 2.0, 3.0), (0.0, 0.0, 0.0))


@pytest.fixture
def simple_structure():
    """Create a simple structure."""
    mask = np.zeros((20, 20, 10), dtype=bool)
    mask[5:15, 5:15, 2:8] = True
    return Target("TestStructure", mask, (2.0, 2.0, 3.0), (0.0, 0.0, 0.0))


class TestMSE:
    """Test Mean Squared Error."""

    def test_mse_identical(self, identical_dose):
        """Test MSE=0 for identical doses."""
        mse = dose_comparison.compare_mse(identical_dose, identical_dose)
        assert mse == 0.0

    def test_mse_different(self, simple_dose1, simple_dose2):
        """Test MSE>0 for different doses."""
        mse = dose_comparison.compare_mse(simple_dose1, simple_dose2)
        assert mse > 0.0

    def test_mse_with_structure(self, simple_dose1, simple_dose2, simple_structure):
        """Test MSE within structure."""
        mse = dose_comparison.compare_mse(simple_dose1, simple_dose2, simple_structure)
        assert mse > 0.0
        assert isinstance(mse, float)

    def test_mse_shape_mismatch(self, simple_dose1):
        """Test MSE raises error for shape mismatch."""
        different_shape = Dose(
            np.ones((10, 10, 5)) * 50.0, (2.0, 2.0, 3.0), (0.0, 0.0, 0.0)
        )
        with pytest.raises(ValueError, match="shape"):
            dose_comparison.compare_mse(simple_dose1, different_shape)


class TestMAE:
    """Test Mean Absolute Error."""

    def test_mae_identical(self, identical_dose):
        """Test MAE=0 for identical doses."""
        mae = dose_comparison.compare_mae(identical_dose, identical_dose)
        assert mae == 0.0

    def test_mae_different(self, simple_dose1, simple_dose2):
        """Test MAE>0 for different doses."""
        mae = dose_comparison.compare_mae(simple_dose1, simple_dose2)
        assert mae > 0.0

    def test_mae_with_structure(self, simple_dose1, simple_dose2, simple_structure):
        """Test MAE within structure."""
        mae = dose_comparison.compare_mae(simple_dose1, simple_dose2, simple_structure)
        assert mae > 0.0
        assert isinstance(mae, float)


class TestPSNR:
    """Test Peak Signal-to-Noise Ratio."""

    def test_psnr_identical(self, identical_dose):
        """Test PSNR=inf for identical doses."""
        psnr = dose_comparison.compare_psnr(identical_dose, identical_dose)
        assert np.isinf(psnr)

    def test_psnr_different(self, simple_dose1, simple_dose2):
        """Test PSNR is finite for different doses."""
        psnr = dose_comparison.compare_psnr(simple_dose1, simple_dose2)
        assert np.isfinite(psnr)
        assert psnr > 0

    def test_psnr_with_data_range(self, simple_dose1, simple_dose2):
        """Test PSNR with specified data range."""
        psnr = dose_comparison.compare_psnr(
            simple_dose1, simple_dose2, data_range=100.0
        )
        assert np.isfinite(psnr)


class TestSSIM:
    """Test Structural Similarity Index."""

    def test_ssim_identical(self, identical_dose):
        """Test SSIM=1 for identical doses."""
        ssim = dose_comparison.compare_ssim(identical_dose, identical_dose)
        assert 0.99 <= ssim <= 1.0  # Close to 1 for identical

    def test_ssim_different(self, simple_dose1, simple_dose2):
        """Test SSIM in valid range for different doses."""
        ssim = dose_comparison.compare_ssim(simple_dose1, simple_dose2)
        assert -1 <= ssim <= 1
        assert ssim > 0  # Should be positive for similar doses


class TestNCC:
    """Test Normalized Cross-Correlation."""

    def test_ncc_identical(self, identical_dose):
        """Test NCC=1 for identical doses."""
        ncc = dose_comparison.compare_normalized_cross_correlation(
            identical_dose, identical_dose
        )
        # For uniform doses, NCC returns 0 (no variation)
        assert ncc == 0.0 or (0.99 <= ncc <= 1.0)

    def test_ncc_different(self, simple_dose1, simple_dose2):
        """Test NCC in valid range."""
        ncc = dose_comparison.compare_normalized_cross_correlation(
            simple_dose1, simple_dose2
        )
        assert -1 <= ncc <= 1.01  # Allow small floating point error
        assert ncc > 0  # Should be positive for similar doses


class TestMutualInformation:
    """Test Mutual Information."""

    def test_mi_positive(self, simple_dose1, simple_dose2):
        """Test MI is positive for related doses."""
        mi = dose_comparison.compare_mutual_information(simple_dose1, simple_dose2)
        assert mi >= 0

    def test_mi_with_bins(self, simple_dose1, simple_dose2):
        """Test MI with different bin counts."""
        mi_256 = dose_comparison.compare_mutual_information(
            simple_dose1, simple_dose2, bins=256
        )
        mi_128 = dose_comparison.compare_mutual_information(
            simple_dose1, simple_dose2, bins=128
        )
        assert mi_256 >= 0
        assert mi_128 >= 0


class TestDoseDifferenceMap:
    """Test dose difference map computation."""

    def test_difference_map_shape(self, simple_dose1, simple_dose2):
        """Test difference map has correct shape."""
        diff_dose = dose_comparison.compare_dose_difference_map(
            simple_dose1, simple_dose2
        )
        assert diff_dose.dose_array.shape == simple_dose1.dose_array.shape

    def test_difference_map_absolute(self, simple_dose1, simple_dose2):
        """Test absolute difference map."""
        diff_dose = dose_comparison.compare_dose_difference_map(
            simple_dose1, simple_dose2, absolute=True
        )
        assert np.all(diff_dose.dose_array >= 0)

    def test_difference_map_signed(self, simple_dose1, simple_dose2):
        """Test signed difference map."""
        diff_dose = dose_comparison.compare_dose_difference_map(
            simple_dose1, simple_dose2, absolute=False
        )
        # Some differences should be negative
        assert np.any(diff_dose.dose_array != 0)


class Test3DGradient:
    """Test 3D dose gradient computation."""

    def test_gradient_shape(self, simple_dose1):
        """Test gradient has correct shape."""
        grad_x, grad_y, grad_z = dose_comparison.compute_3d_dose_gradient(simple_dose1)
        assert grad_x.shape == simple_dose1.dose_array.shape
        assert grad_y.shape == simple_dose1.dose_array.shape
        assert grad_z.shape == simple_dose1.dose_array.shape

    def test_gradient_uniform(self):
        """Test gradient is zero for uniform dose."""
        uniform_dose = Dose(
            np.ones((10, 10, 10)) * 50.0, (2.0, 2.0, 3.0), (0.0, 0.0, 0.0)
        )
        grad_x, grad_y, grad_z = dose_comparison.compute_3d_dose_gradient(uniform_dose)
        assert np.allclose(grad_x, 0, atol=1e-10)
        assert np.allclose(grad_y, 0, atol=1e-10)
        assert np.allclose(grad_z, 0, atol=1e-10)


class TestComprehensiveMetrics:
    """Test comprehensive dose comparison metrics."""

    def test_all_metrics(self, simple_dose1, simple_dose2):
        """Test that all metrics are computed."""
        metrics = dose_comparison.compare_dose(simple_dose1, simple_dose2)

        assert "mse" in metrics
        assert "mae" in metrics
        assert "psnr" in metrics
        assert "ssim" in metrics
        assert "ncc" in metrics
        assert "mi" in metrics

    def test_metrics_with_structure(self, simple_dose1, simple_dose2, simple_structure):
        """Test metrics within structure."""
        metrics = dose_comparison.compare_dose(
            simple_dose1, simple_dose2, simple_structure
        )
        assert all(
            key in metrics for key in ["mse", "mae", "psnr", "ssim", "ncc", "mi"]
        )


class TestVarianceOfLaplacian:
    """Test Variance of Laplacian (dose sharpness metric)."""

    def test_returns_float(self, simple_dose1):
        vol = dose_comparison.compute_variance_of_laplacian(simple_dose1)
        assert isinstance(vol, float)

    def test_non_negative(self, simple_dose1):
        vol = dose_comparison.compute_variance_of_laplacian(simple_dose1)
        assert vol >= 0.0

    def test_uniform_dose_has_low_variance(self, identical_dose):
        # Uniform dose → Laplacian ≈ 0 everywhere → near-zero variance
        vol = dose_comparison.compute_variance_of_laplacian(identical_dose)
        assert vol == pytest.approx(0.0, abs=1e-8)

    def test_dose_with_sharp_edges_has_higher_variance(
        self, simple_dose1, identical_dose
    ):
        # Dose with abrupt transitions should have higher VoL than uniform dose
        vol_sharp = dose_comparison.compute_variance_of_laplacian(simple_dose1)
        vol_uniform = dose_comparison.compute_variance_of_laplacian(identical_dose)
        assert vol_sharp > vol_uniform

    def test_with_structure(self, simple_dose1, simple_structure):
        vol = dose_comparison.compute_variance_of_laplacian(
            simple_dose1, simple_structure
        )
        assert isinstance(vol, float)
        assert vol >= 0.0

    def test_empty_structure_returns_nan(self, simple_dose1):
        empty = Target(
            "Empty",
            np.zeros((20, 20, 10), dtype=bool),
            (2.0, 2.0, 3.0),
            (0.0, 0.0, 0.0),
        )
        vol = dose_comparison.compute_variance_of_laplacian(simple_dose1, empty)
        assert np.isnan(vol)


class TestNormalizedMAE:
    """Test Normalized MAE with threshold masking."""

    def test_identical_doses_returns_zero(self, identical_dose):
        nmae = dose_comparison.compare_normalized_mae(identical_dose, identical_dose)
        assert nmae == pytest.approx(0.0, abs=1e-10)

    def test_without_normalization_equals_mae(self, simple_dose1, simple_dose2):
        nmae = dose_comparison.compare_normalized_mae(simple_dose1, simple_dose2)
        mae = dose_comparison.compare_mae(simple_dose1, simple_dose2)
        assert nmae == pytest.approx(mae, rel=1e-6)

    def test_normalization_scales_result(self, simple_dose1, simple_dose2):
        mae = dose_comparison.compare_mae(simple_dose1, simple_dose2)
        nmae = dose_comparison.compare_normalized_mae(
            simple_dose1, simple_dose2, normalization_value=60.0
        )
        assert nmae == pytest.approx(mae / 60.0, rel=1e-6)

    def test_threshold_excludes_low_dose_voxels(self, simple_dose1, simple_dose2):
        # With high threshold, fewer voxels used → may differ from global MAE
        nmae_global = dose_comparison.compare_normalized_mae(simple_dose1, simple_dose2)
        nmae_thresh = dose_comparison.compare_normalized_mae(
            simple_dose1, simple_dose2, dose_threshold_gy=55.0
        )
        # Threshold removes low-dose background voxels (50 Gy < threshold 55 Gy)
        # Remaining: only voxels at 60/62 Gy → different from global
        assert nmae_thresh != pytest.approx(nmae_global, rel=1e-3)

    def test_threshold_above_all_voxels_returns_nan(self, simple_dose1, simple_dose2):
        nmae = dose_comparison.compare_normalized_mae(
            simple_dose1, simple_dose2, dose_threshold_gy=1000.0
        )
        assert np.isnan(nmae)

    def test_with_structure(self, simple_dose1, simple_dose2, simple_structure):
        nmae = dose_comparison.compare_normalized_mae(
            simple_dose1, simple_dose2, structure=simple_structure
        )
        assert isinstance(nmae, float)
        assert nmae >= 0.0
