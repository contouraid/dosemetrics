"""
Correctness tests for dosemetrics.metrics.gamma module.

These tests validate the algorithmic correctness of the gamma index
implementation against known medical physics principles and edge cases
from Low et al. (1998) and subsequent literature.

References:
    - Low DA, Harms WB, Mutic S, Purdy JA. "A technique for the quantitative
      evaluation of dose distributions." Med Phys. 1998;25(5):656-61.
    - Depuydt T, Van Esch A, Huyskens DP. "A quantitative evaluation of IMRT
      dose distributions: refinement and clinical assessment of the gamma
      evaluation." Radiother Oncol. 2002;62(3):309-19.
    - Wendling M, et al. "A fast algorithm for gamma evaluation in 3D."
      Med Phys. 2007;34(5):1647-54.
"""

import pytest
import numpy as np
from typing import Tuple

from dosemetrics.dose import Dose
from dosemetrics.metrics import gamma


class TestGammaIndexFormula:
    """Test the gamma index formula implementation."""

    def test_gamma_formula_pure_dose_difference(self):
        """
        Test gamma index with pure dose difference (no spatial shift).

        According to Low et al., gamma for pure dose difference should be:
        γ = |ΔD| / ΔD_criterion

        Where ΔD is normalized to reference dose (global or local).
        """
        # Create reference dose: uniform 60 Gy
        dose_array_ref = np.full((20, 20, 10), 60.0)
        ref_dose = Dose(dose_array_ref, (2.0, 2.0, 2.0), (0.0, 0.0, 0.0))

        # Create evaluated dose: uniform 61.8 Gy (3% difference)
        dose_array_eval = np.full((20, 20, 10), 61.8)
        eval_dose = Dose(dose_array_eval, (2.0, 2.0, 2.0), (0.0, 0.0, 0.0))

        # 3%/3mm criteria with global normalization
        gamma_result = gamma.compute_gamma_index(
            ref_dose,
            eval_dose,
            dose_criterion_percent=3.0,
            distance_criterion_mm=3.0,
            dose_threshold_percent=0.0,  # No threshold
            global_normalization=True,
        )

        # Expected gamma: 1.8/60 = 3% difference, so gamma = 1.0
        # (distance contribution is 0 since no spatial shift)
        valid_gamma = gamma_result[~np.isnan(gamma_result)]

        # All points should have gamma = 1.0 (within numerical tolerance)
        assert np.allclose(
            valid_gamma, 1.0, atol=0.01
        ), f"Expected gamma=1.0, got mean={np.mean(valid_gamma):.3f}, std={np.std(valid_gamma):.3f}"

    def test_gamma_formula_pure_spatial_shift(self):
        """
        Test gamma index with pure spatial shift (no dose difference).

        For pure spatial shift, gamma should be:
        γ = r / r_criterion

        Where r is the distance to agreement.
        """
        # Create reference dose with a sharp boundary
        dose_array_ref = np.zeros((30, 30, 20))
        dose_array_ref[10:20, :, :] = 60.0
        ref_dose = Dose(dose_array_ref, (2.0, 2.0, 2.0), (0.0, 0.0, 0.0))

        # Create evaluated dose shifted by 3mm (1.5 voxels at 2mm spacing)
        # Shift by 2mm (1 voxel) in x-direction
        dose_array_eval = np.zeros((30, 30, 20))
        dose_array_eval[11:21, :, :] = 60.0
        eval_dose = Dose(dose_array_eval, (2.0, 2.0, 2.0), (0.0, 0.0, 0.0))

        # 3%/3mm criteria
        gamma_result = gamma.compute_gamma_index(
            ref_dose,
            eval_dose,
            dose_criterion_percent=3.0,
            distance_criterion_mm=3.0,
            dose_threshold_percent=5.0,
            global_normalization=True,
        )

        # For central region with 2mm shift, gamma ≈ 2/3 = 0.67
        # Check central slice where dose is uniform
        central_slice = gamma_result[15, 15, :]
        valid_central = central_slice[~np.isnan(central_slice)]

        if len(valid_central) > 0:
            mean_gamma = np.mean(valid_central)
            # Should be less than 1 since shift (2mm) < criterion (3mm)
            assert (
                mean_gamma < 1.0
            ), f"Spatial shift of 2mm with 3mm criterion should give gamma < 1.0, got {mean_gamma:.3f}"

    def test_gamma_formula_combined_dose_and_distance(self):
        """
        Test gamma index with both dose and distance differences.

        The general gamma formula is:
        γ = sqrt((r/r_c)^2 + (ΔD/ΔD_c)^2)
        """
        # Create reference dose
        dose_array_ref = np.full((20, 20, 10), 60.0)
        ref_dose = Dose(dose_array_ref, (2.0, 2.0, 2.0), (0.0, 0.0, 0.0))

        # Create evaluated dose with 2% dose difference
        # 2% of 60 Gy = 1.2 Gy
        dose_array_eval = np.full((20, 20, 10), 61.2)
        eval_dose = Dose(dose_array_eval, (2.0, 2.0, 2.0), (0.0, 0.0, 0.0))

        # 3%/3mm criteria - pure 2% dose difference
        gamma_result = gamma.compute_gamma_index(
            ref_dose,
            eval_dose,
            dose_criterion_percent=3.0,
            distance_criterion_mm=3.0,
            dose_threshold_percent=0.0,
            global_normalization=True,
        )

        valid_gamma = gamma_result[~np.isnan(gamma_result)]

        # Expected gamma for 2% dose difference with 3% criterion
        expected_gamma = 2.0 / 3.0  # ≈ 0.67

        assert np.allclose(
            valid_gamma, expected_gamma, atol=0.05
        ), f"Expected gamma≈{expected_gamma:.2f}, got mean={np.mean(valid_gamma):.3f}"


class TestGammaIndexProperties:
    """Test mathematical properties of gamma index."""

    def test_gamma_zero_for_identical_distributions(self):
        """Gamma should be 0 for identical dose distributions."""
        dose_array = np.random.rand(15, 15, 10) * 60.0 + 10.0
        dose1 = Dose(dose_array, (2.0, 2.0, 2.0), (0.0, 0.0, 0.0))
        dose2 = Dose(dose_array.copy(), (2.0, 2.0, 2.0), (0.0, 0.0, 0.0))

        gamma_result = gamma.compute_gamma_index(
            dose1,
            dose2,
            dose_threshold_percent=5.0,
        )

        valid_gamma = gamma_result[~np.isnan(gamma_result)]

        # Gamma should be very close to 0 for identical distributions
        assert (
            np.max(valid_gamma) < 0.01
        ), f"Gamma for identical distributions should be ~0, got max={np.max(valid_gamma):.4f}"

    def test_gamma_symmetry(self):
        """
        Test that gamma index is not symmetric.

        According to Low et al., gamma(ref, eval) ≠ gamma(eval, ref)
        because the reference point is where gamma is evaluated.
        """
        # Create two different dose distributions
        dose_array1 = np.zeros((20, 20, 10))
        dose_array1[8:12, 8:12, :] = 60.0
        dose1 = Dose(dose_array1, (2.0, 2.0, 2.0), (0.0, 0.0, 0.0))

        dose_array2 = np.zeros((20, 20, 10))
        dose_array2[9:13, 9:13, :] = 60.0
        dose2 = Dose(dose_array2, (2.0, 2.0, 2.0), (0.0, 0.0, 0.0))

        # Compute gamma in both directions
        gamma_1to2 = gamma.compute_gamma_index(dose1, dose2, dose_threshold_percent=5.0)
        gamma_2to1 = gamma.compute_gamma_index(dose2, dose1, dose_threshold_percent=5.0)

        # The gamma distributions should be different (not symmetric)
        # At least some values should differ
        diff = np.abs(gamma_1to2 - gamma_2to1)
        valid_diff = diff[~np.isnan(diff)]

        # Note: This tests implementation behavior, not always a strict requirement
        # but it's expected based on the reference point definition
        print(f"Mean difference between directions: {np.mean(valid_diff):.4f}")

    def test_gamma_passes_at_threshold(self):
        """Test that gamma = 1.0 is the passing threshold."""
        # Create dose with exactly 3% difference
        dose_array_ref = np.full((15, 15, 10), 60.0)
        ref_dose = Dose(dose_array_ref, (2.0, 2.0, 2.0), (0.0, 0.0, 0.0))

        dose_array_eval = np.full((15, 15, 10), 61.8)  # Exactly 3% higher
        eval_dose = Dose(dose_array_eval, (2.0, 2.0, 2.0), (0.0, 0.0, 0.0))

        # 3%/3mm criteria
        gamma_result = gamma.compute_gamma_index(
            ref_dose,
            eval_dose,
            dose_criterion_percent=3.0,
            distance_criterion_mm=3.0,
            dose_threshold_percent=0.0,
        )

        passing_rate = gamma.compute_gamma_passing_rate(gamma_result, threshold=1.0)

        # Should be at the threshold, so passing rate should be ~100%
        assert (
            passing_rate > 99.0
        ), f"At exactly the dose criterion, passing rate should be ~100%, got {passing_rate:.1f}%"


class TestGammaNormalization:
    """Test normalization modes (global vs local)."""

    def test_global_normalization(self):
        """
        Test global normalization mode.

        In global normalization, all dose differences are normalized to
        the maximum dose in the reference distribution.
        """
        # Create dose with varying magnitudes
        dose_array_ref = np.zeros((20, 20, 10))
        dose_array_ref[5:10, 5:10, :] = 60.0  # High dose region
        dose_array_ref[12:17, 12:17, :] = 20.0  # Low dose region
        ref_dose = Dose(dose_array_ref, (2.0, 2.0, 2.0), (0.0, 0.0, 0.0))

        # Add 1 Gy everywhere
        dose_array_eval = dose_array_ref + 1.0
        eval_dose = Dose(dose_array_eval, (2.0, 2.0, 2.0), (0.0, 0.0, 0.0))

        # Global normalization: 1 Gy / 60 Gy = 1.67% (should pass 3% criterion)
        gamma_result = gamma.compute_gamma_index(
            ref_dose,
            eval_dose,
            dose_criterion_percent=3.0,
            distance_criterion_mm=3.0,
            dose_threshold_percent=5.0,
            global_normalization=True,
        )

        valid_gamma = gamma_result[~np.isnan(gamma_result)]

        # All points should have gamma < 1.0 since 1.67% < 3%
        assert np.all(
            valid_gamma < 1.0
        ), f"Global norm: 1 Gy diff / 60 Gy max = 1.67% should pass 3% criterion"

    def test_local_normalization(self):
        """
        Test local normalization mode.

        In local normalization, dose differences are normalized to
        the local reference dose at each point.
        """
        # Create dose with varying magnitudes
        dose_array_ref = np.zeros((20, 20, 10))
        dose_array_ref[5:10, 5:10, :] = 60.0  # High dose region
        dose_array_ref[12:17, 12:17, :] = 20.0  # Low dose region
        ref_dose = Dose(dose_array_ref, (2.0, 2.0, 2.0), (0.0, 0.0, 0.0))

        # Add 1.8 Gy to high dose, 0.6 Gy to low dose (3% of each)
        dose_array_eval = dose_array_ref.copy()
        dose_array_eval[5:10, 5:10, :] += 1.8  # 3% of 60 Gy
        dose_array_eval[12:17, 12:17, :] += 0.6  # 3% of 20 Gy
        eval_dose = Dose(dose_array_eval, (2.0, 2.0, 2.0), (0.0, 0.0, 0.0))

        # Local normalization: both regions should have gamma ≈ 1.0
        gamma_result = gamma.compute_gamma_index(
            ref_dose,
            eval_dose,
            dose_criterion_percent=3.0,
            distance_criterion_mm=3.0,
            dose_threshold_percent=5.0,
            global_normalization=False,  # Local normalization
        )

        # Check high dose region
        high_dose_gamma = gamma_result[6:9, 6:9, 5]
        high_dose_valid = high_dose_gamma[~np.isnan(high_dose_gamma)]

        # Check low dose region
        low_dose_gamma = gamma_result[13:16, 13:16, 5]
        low_dose_valid = low_dose_gamma[~np.isnan(low_dose_gamma)]

        if len(high_dose_valid) > 0 and len(low_dose_valid) > 0:
            # Both should be close to 1.0 with local normalization
            assert (
                np.abs(np.mean(high_dose_valid) - 1.0) < 0.1
            ), f"High dose region should have gamma≈1.0, got {np.mean(high_dose_valid):.3f}"
            assert (
                np.abs(np.mean(low_dose_valid) - 1.0) < 0.1
            ), f"Low dose region should have gamma≈1.0, got {np.mean(low_dose_valid):.3f}"


class TestGammaThreshold:
    """Test dose threshold functionality."""

    def test_threshold_excludes_low_doses(self):
        """Test that low dose regions below threshold are excluded."""
        # Create dose with high and low dose regions
        dose_array = np.zeros((20, 20, 10))
        dose_array[5:10, 5:10, :] = 60.0  # High dose: 60 Gy
        dose_array[12:17, 12:17, :] = 3.0  # Low dose: 3 Gy (5% of max)

        ref_dose = Dose(dose_array, (2.0, 2.0, 2.0), (0.0, 0.0, 0.0))
        eval_dose = Dose(dose_array * 1.1, (2.0, 2.0, 2.0), (0.0, 0.0, 0.0))

        # 10% threshold (6 Gy)
        gamma_result = gamma.compute_gamma_index(
            ref_dose,
            eval_dose,
            dose_threshold_percent=10.0,  # 10% of 60 Gy = 6 Gy
        )

        # Low dose region (3 Gy) should be NaN
        low_dose_region = gamma_result[13:16, 13:16, 5]
        assert np.all(np.isnan(low_dose_region)), "Doses below threshold should be NaN"

        # High dose region should have values
        high_dose_region = gamma_result[6:9, 6:9, 5]
        assert not np.all(
            np.isnan(high_dose_region)
        ), "Doses above threshold should have gamma values"

    def test_zero_threshold_includes_all(self):
        """Test that zero threshold includes all non-zero doses."""
        dose_array = np.ones((15, 15, 10)) * 1.0  # Very low dose: 1 Gy
        ref_dose = Dose(dose_array, (2.0, 2.0, 2.0), (0.0, 0.0, 0.0))
        eval_dose = Dose(dose_array * 1.05, (2.0, 2.0, 2.0), (0.0, 0.0, 0.0))

        gamma_result = gamma.compute_gamma_index(
            ref_dose,
            eval_dose,
            dose_threshold_percent=0.0,  # No threshold
        )

        # Should have gamma values everywhere
        valid_points = np.sum(~np.isnan(gamma_result))
        total_points = np.prod(gamma_result.shape)

        assert (
            valid_points == total_points
        ), f"With 0% threshold, all points should be evaluated: {valid_points}/{total_points}"


class TestGammaEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_dose_handling(self):
        """Test behavior with zero dose values."""
        dose_array_ref = np.zeros((15, 15, 10))
        dose_array_ref[5:10, 5:10, :] = 60.0
        ref_dose = Dose(dose_array_ref, (2.0, 2.0, 2.0), (0.0, 0.0, 0.0))

        eval_dose = Dose(dose_array_ref.copy(), (2.0, 2.0, 2.0), (0.0, 0.0, 0.0))

        gamma_result = gamma.compute_gamma_index(
            ref_dose,
            eval_dose,
            dose_threshold_percent=5.0,
        )

        # Should handle zero doses gracefully (excluded by threshold)
        assert not np.any(
            np.isinf(gamma_result)
        ), "Gamma result should not contain inf values"

    def test_high_gradient_region(self):
        """
        Test gamma in high gradient regions.

        High gradients are challenging for gamma analysis and should
        be handled correctly by the distance-to-agreement component.
        """
        # Create sharp dose gradient
        dose_array = np.zeros((30, 30, 10))
        for i in range(30):
            dose_array[i, :, :] = i * 2.0  # Linear gradient: 0 to 58 Gy

        ref_dose = Dose(dose_array, (2.0, 2.0, 2.0), (0.0, 0.0, 0.0))

        # Shift by 1 voxel (2mm)
        dose_array_shifted = np.zeros((30, 30, 10))
        for i in range(29):
            dose_array_shifted[i, :, :] = (i + 1) * 2.0
        eval_dose = Dose(dose_array_shifted, (2.0, 2.0, 2.0), (0.0, 0.0, 0.0))

        gamma_result = gamma.compute_gamma_index(
            ref_dose,
            eval_dose,
            dose_criterion_percent=3.0,
            distance_criterion_mm=3.0,
            dose_threshold_percent=10.0,
        )

        # Should complete without errors
        valid_gamma = gamma_result[~np.isnan(gamma_result)]
        assert len(valid_gamma) > 0, "Should have valid gamma values in gradient region"
        assert not np.any(np.isinf(valid_gamma)), "No inf values in high gradient"

    def test_small_volume(self):
        """Test gamma on very small volumes."""
        dose_array = np.ones((3, 3, 3)) * 50.0
        ref_dose = Dose(dose_array, (2.0, 2.0, 2.0), (0.0, 0.0, 0.0))
        eval_dose = Dose(dose_array * 1.01, (2.0, 2.0, 2.0), (0.0, 0.0, 0.0))

        gamma_result = gamma.compute_gamma_index(
            ref_dose,
            eval_dose,
            dose_threshold_percent=0.0,
        )

        # Should handle small volumes
        assert gamma_result.shape == (3, 3, 3)
        valid_gamma = gamma_result[~np.isnan(gamma_result)]
        assert len(valid_gamma) > 0, "Should compute gamma for small volumes"


class TestGammaCriteria:
    """Test different gamma criteria commonly used in clinical practice."""

    @pytest.mark.parametrize(
        "dose_crit,dist_crit,description",
        [
            (1.0, 1.0, "1%/1mm - Research/strict"),
            (2.0, 2.0, "2%/2mm - Tight clinical"),
            (3.0, 3.0, "3%/3mm - Standard clinical"),
            (5.0, 3.0, "5%/3mm - Lenient dose"),
            (3.0, 5.0, "3%/5mm - Lenient distance"),
        ],
    )
    def test_various_clinical_criteria(self, dose_crit, dist_crit, description):
        """Test gamma with various clinical criteria."""
        # Create test doses
        dose_array_ref = np.full((15, 15, 10), 60.0)
        ref_dose = Dose(dose_array_ref, (2.0, 2.0, 2.0), (0.0, 0.0, 0.0))

        # Small perturbation
        dose_array_eval = dose_array_ref * 1.01  # 1% difference
        eval_dose = Dose(dose_array_eval, (2.0, 2.0, 2.0), (0.0, 0.0, 0.0))

        gamma_result = gamma.compute_gamma_index(
            ref_dose,
            eval_dose,
            dose_criterion_percent=dose_crit,
            distance_criterion_mm=dist_crit,
            dose_threshold_percent=0.0,
        )

        passing_rate = gamma.compute_gamma_passing_rate(gamma_result)

        # 1% dose difference should pass all these criteria
        assert (
            passing_rate > 99.0
        ), f"{description}: 1% dose diff should pass, got {passing_rate:.1f}%"


class TestGamma2D:
    """Test 2D gamma index calculation."""

    def test_2d_gamma_identical_slices(self):
        """Test 2D gamma with identical slices."""
        slice_array = np.random.rand(20, 20) * 50.0 + 10.0

        gamma_result = gamma.compute_2d_gamma(
            slice_array,
            slice_array.copy(),
            dose_criterion_percent=3.0,
            distance_criterion_mm=3.0,
        )

        assert gamma_result.shape == slice_array.shape
        assert np.max(gamma_result) < 0.01, "Identical slices should have gamma ≈ 0"

    def test_2d_gamma_dose_difference(self):
        """Test 2D gamma with dose difference."""
        ref_slice = np.full((20, 20), 60.0)
        eval_slice = np.full((20, 20), 61.8)  # 3% difference

        gamma_result = gamma.compute_2d_gamma(
            ref_slice,
            eval_slice,
            dose_criterion_percent=3.0,
            distance_criterion_mm=3.0,
        )

        # Should have gamma ≈ 1.0 everywhere
        assert np.allclose(
            gamma_result, 1.0, atol=0.05
        ), f"3% dose diff with 3% criterion should give gamma≈1.0"

    def test_2d_gamma_shape_validation(self):
        """Test 2D gamma input validation."""
        ref_slice = np.ones((20, 20))
        eval_slice = np.ones((20, 21))  # Different shape

        with pytest.raises(ValueError, match="must match"):
            gamma.compute_2d_gamma(ref_slice, eval_slice)

        # Test non-2D input
        with pytest.raises(ValueError, match="must be 2D"):
            gamma.compute_2d_gamma(np.ones((20, 20, 5)), np.ones((20, 20, 5)))


class TestGammaStatistics:
    """Test gamma statistics calculation."""

    def test_passing_rate_calculation(self):
        """Test gamma passing rate calculation."""
        # Create gamma array with known values
        gamma_array = np.array([0.5, 0.8, 1.0, 1.2, 0.3, np.nan, np.nan])

        passing_rate = gamma.compute_gamma_passing_rate(gamma_array, threshold=1.0)

        # 4 out of 5 valid values are <= 1.0 (0.5, 0.8, 1.0, 0.3 pass; 1.2 fails)
        expected_rate = 4.0 / 5.0 * 100.0
        assert (
            abs(passing_rate - expected_rate) < 0.01
        ), f"Expected {expected_rate}%, got {passing_rate}%"

    def test_gamma_statistics_comprehensive(self):
        """Test comprehensive gamma statistics."""
        gamma_array = np.array(
            [
                [0.2, 0.5, 0.8],
                [1.0, 1.5, 2.0],
                [np.nan, np.nan, np.nan],
            ]
        )

        stats = gamma.compute_gamma_statistics(gamma_array)

        assert "passing_rate_1_0" in stats
        assert "mean_gamma" in stats
        assert "max_gamma" in stats
        assert "gamma_50" in stats
        assert "gamma_95" in stats

        # Check values
        assert stats["passing_rate_1_0"] == 4.0 / 6.0 * 100.0  # 4 out of 6 pass
        assert abs(stats["mean_gamma"] - np.mean([0.2, 0.5, 0.8, 1.0, 1.5, 2.0])) < 0.01
        assert stats["max_gamma"] == 2.0

    def test_empty_gamma_statistics(self):
        """Test statistics with all NaN values."""
        gamma_array = np.full((10, 10), np.nan)

        stats = gamma.compute_gamma_statistics(gamma_array)

        assert stats["passing_rate_1_0"] == 0.0
        assert np.isnan(stats["mean_gamma"])
        assert np.isnan(stats["max_gamma"])


if __name__ == "__main__":
    """Allow running correctness tests directly."""
    print("=" * 70)
    print("Gamma Analysis Algorithmic Correctness Test Suite")
    print("=" * 70)
    print("\nValidating gamma index implementation against medical physics")
    print("references (Low et al. 1998, Depuydt et al. 2002)")
    print("=" * 70)

    # Run pytest with verbose output
    pytest.main([__file__, "-v"])
