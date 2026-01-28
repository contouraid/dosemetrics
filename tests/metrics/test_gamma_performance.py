"""
Performance tests for dosemetrics.metrics.gamma module.

Tests runtime performance of gamma index calculation with varying volume sizes
to ensure the implementation is efficient enough for clinical use.
"""

import pytest
import numpy as np
import time
from typing import Tuple, Dict

from dosemetrics.dose import Dose
from dosemetrics.metrics import gamma


def create_test_dose_distribution(
    shape: Tuple[int, int, int],
    spacing: Tuple[float, float, float] = (2.0, 2.0, 2.0),
    pattern: str = "gradient",
) -> Dose:
    """
    Create a test dose distribution with specified pattern.

    Parameters
    ----------
    shape : tuple
        Volume dimensions (nx, ny, nz)
    spacing : tuple
        Voxel spacing in mm
    pattern : str
        Pattern type: 'gradient', 'sphere', 'checker', or 'realistic'

    Returns
    -------
    Dose
        Generated dose distribution
    """
    dose_array = np.zeros(shape, dtype=np.float32)
    nx, ny, nz = shape

    if pattern == "gradient":
        # Linear gradient from 0 to 70 Gy
        x = np.linspace(0, 70, nx)
        dose_array = np.repeat(x[:, np.newaxis, np.newaxis], ny, axis=1)
        dose_array = np.repeat(dose_array, nz, axis=2)

    elif pattern == "sphere":
        # Spherical dose distribution centered in volume
        cx, cy, cz = nx // 2, ny // 2, nz // 2
        radius = min(nx, ny, nz) // 3

        x = np.arange(nx) - cx
        y = np.arange(ny) - cy
        z = np.arange(nz) - cz
        xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")

        dist = np.sqrt(xx**2 + yy**2 + zz**2)
        dose_array = np.maximum(0, 70 * (1 - dist / radius))

    elif pattern == "checker":
        # Checkerboard pattern with high/low dose regions
        checker_size = 8
        x_check = (np.arange(nx) // checker_size) % 2
        y_check = (np.arange(ny) // checker_size) % 2
        z_check = (np.arange(nz) // checker_size) % 2

        xx, yy, zz = np.meshgrid(x_check, y_check, z_check, indexing="ij")
        checker = (xx + yy + zz) % 2
        dose_array = np.where(checker == 0, 60.0, 20.0)

    elif pattern == "realistic":
        # More realistic dose distribution with multiple regions
        # Central high-dose region (tumor)
        cx, cy, cz = nx // 2, ny // 2, nz // 2
        tumor_radius = min(nx, ny, nz) // 6

        x = np.arange(nx) - cx
        y = np.arange(ny) - cy
        z = np.arange(nz) - cz
        xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")

        # Tumor region (60 Gy)
        dist_tumor = np.sqrt(xx**2 + yy**2 + zz**2)
        dose_array = np.where(dist_tumor <= tumor_radius, 60.0, 0.0)

        # Surrounding tissue with gradient fall-off
        surrounding = (dist_tumor > tumor_radius) & (dist_tumor <= tumor_radius * 3)
        dose_array = np.where(
            surrounding,
            60.0 * np.exp(-(dist_tumor - tumor_radius) / tumor_radius),
            dose_array,
        )

    return Dose(dose_array, spacing, (0.0, 0.0, 0.0))


def create_perturbed_dose(
    reference_dose: Dose,
    dose_shift_percent: float = 2.0,
    spatial_shift_voxels: Tuple[int, int, int] = (1, 1, 0),
) -> Dose:
    """
    Create a perturbed version of reference dose for testing.

    Parameters
    ----------
    reference_dose : Dose
        Original dose distribution
    dose_shift_percent : float
        Percentage dose shift to apply
    spatial_shift_voxels : tuple
        Voxel shift in each dimension

    Returns
    -------
    Dose
        Perturbed dose distribution
    """
    # Apply spatial shift using np.roll
    shifted_array = reference_dose.dose_array.copy()
    for axis, shift in enumerate(spatial_shift_voxels):
        if shift != 0:
            shifted_array = np.roll(shifted_array, shift, axis=axis)

    # Apply dose shift
    dose_factor = 1.0 + dose_shift_percent / 100.0
    shifted_array = shifted_array * dose_factor

    return Dose(shifted_array, reference_dose.spacing, reference_dose.origin)


def measure_gamma_performance(
    shape: Tuple[int, int, int],
    pattern: str = "gradient",
    dose_criterion: float = 3.0,
    distance_criterion: float = 3.0,
    global_norm: bool = True,
) -> Dict[str, float]:
    """
    Measure gamma index computation performance for given parameters.

    Parameters
    ----------
    shape : tuple
        Volume dimensions
    pattern : str
        Dose distribution pattern
    dose_criterion : float
        Dose criterion (%)
    distance_criterion : float
        Distance criterion (mm)
    global_norm : bool
        Use global normalization

    Returns
    -------
    dict
        Performance metrics including runtime, passing rate, etc.
    """
    # Create test doses
    print(f"\nCreating test volumes ({shape[0]}x{shape[1]}x{shape[2]})...")
    ref_dose = create_test_dose_distribution(shape, pattern=pattern)
    eval_dose = create_perturbed_dose(ref_dose, dose_shift_percent=2.0)

    # Warm-up run (to avoid cold start effects)
    print("Performing warm-up calculation...")
    _ = gamma.compute_gamma_index(
        ref_dose,
        eval_dose,
        dose_criterion_percent=dose_criterion,
        distance_criterion_mm=distance_criterion,
        global_normalization=global_norm,
    )

    # Actual timed run
    print("Performing timed calculation...")
    start_time = time.perf_counter()

    gamma_result = gamma.compute_gamma_index(
        ref_dose,
        eval_dose,
        dose_criterion_percent=dose_criterion,
        distance_criterion_mm=distance_criterion,
        global_normalization=global_norm,
    )

    end_time = time.perf_counter()
    runtime = end_time - start_time

    # Compute statistics
    stats = gamma.compute_gamma_statistics(gamma_result)

    # Calculate voxels per second
    total_voxels = np.prod(shape)
    voxels_per_second = total_voxels / runtime if runtime > 0 else 0

    return {
        "runtime_seconds": runtime,
        "volume_shape": shape,
        "total_voxels": total_voxels,
        "voxels_per_second": voxels_per_second,
        "passing_rate": stats["passing_rate_1_0"],
        "mean_gamma": stats["mean_gamma"],
        "max_gamma": stats["max_gamma"],
    }


class TestGammaPerformance:
    """Performance tests for gamma index calculation."""

    def test_small_volume_64x64x64(self):
        """Test performance on small volume (64x64x64)."""
        print("\n" + "=" * 70)
        print("Testing 64x64x64 volume")
        print("=" * 70)

        results = measure_gamma_performance((64, 64, 64), pattern="gradient")

        print(f"\nResults:")
        print(f"  Runtime: {results['runtime_seconds']:.3f} seconds")
        print(f"  Voxels/second: {results['voxels_per_second']:.0f}")
        print(f"  Passing rate: {results['passing_rate']:.1f}%")

        # Should be fast for small volumes
        assert (
            results["runtime_seconds"] < 5.0
        ), f"Small volume took {results['runtime_seconds']:.2f}s (expected < 5s)"

    def test_medium_volume_128x128x128(self):
        """Test performance on medium volume (128x128x128) - critical size."""
        print("\n" + "=" * 70)
        print("Testing 128x128x128 volume (CRITICAL SIZE)")
        print("=" * 70)

        results = measure_gamma_performance((128, 128, 128), pattern="gradient")

        print(f"\nResults:")
        print(f"  Runtime: {results['runtime_seconds']:.3f} seconds")
        print(f"  Voxels/second: {results['voxels_per_second']:.0f}")
        print(f"  Passing rate: {results['passing_rate']:.1f}%")

        # Critical test: should complete within 10 seconds
        if results["runtime_seconds"] > 10.0:
            pytest.fail(
                f"PERFORMANCE ISSUE: 128^3 volume took {results['runtime_seconds']:.2f}s "
                f"(threshold: 10s). Implementation may need optimization."
            )
        else:
            print(f"  ✓ Performance acceptable (< 10s threshold)")

    @pytest.mark.slow
    def test_large_volume_256x256x256(self):
        """Test performance on large volume (256x256x256) - stress test."""
        print("\n" + "=" * 70)
        print("Testing 256x256x256 volume (STRESS TEST)")
        print("=" * 70)

        results = measure_gamma_performance((256, 256, 256), pattern="gradient")

        print(f"\nResults:")
        print(f"  Runtime: {results['runtime_seconds']:.3f} seconds")
        print(f"  Voxels/second: {results['voxels_per_second']:.0f}")
        print(f"  Passing rate: {results['passing_rate']:.1f}%")

        # This is a stress test - allow more time but flag if excessive
        if results["runtime_seconds"] > 10.0:
            pytest.fail(
                f"PERFORMANCE ISSUE: 256^3 volume took {results['runtime_seconds']:.2f}s "
                f"(threshold: 10s). Implementation definitely needs optimization."
            )
        else:
            print(f"  ✓ Performance acceptable (< 10s threshold)")

    def test_rectangular_volume_256x256x64(self):
        """Test performance on rectangular volume (typical CT scan dimensions)."""
        print("\n" + "=" * 70)
        print("Testing 256x256x64 volume (typical CT scan)")
        print("=" * 70)

        results = measure_gamma_performance((256, 256, 64), pattern="realistic")

        print(f"\nResults:")
        print(f"  Runtime: {results['runtime_seconds']:.3f} seconds")
        print(f"  Voxels/second: {results['voxels_per_second']:.0f}")
        print(f"  Passing rate: {results['passing_rate']:.1f}%")

        # Rectangular volumes are common in clinical practice
        assert (
            results["runtime_seconds"] < 10.0
        ), f"Typical CT volume took {results['runtime_seconds']:.2f}s (expected < 10s)"

    def test_performance_with_different_patterns(self):
        """Test performance with different dose patterns."""
        print("\n" + "=" * 70)
        print("Testing different dose patterns (128x128x64)")
        print("=" * 70)

        patterns = ["gradient", "sphere", "checker", "realistic"]
        results = {}

        for pattern in patterns:
            print(f"\nPattern: {pattern}")
            result = measure_gamma_performance((128, 128, 64), pattern=pattern)
            results[pattern] = result
            print(f"  Runtime: {result['runtime_seconds']:.3f}s")
            print(f"  Passing rate: {result['passing_rate']:.1f}%")

        # Check all patterns complete reasonably
        for pattern, result in results.items():
            assert (
                result["runtime_seconds"] < 10.0
            ), f"Pattern '{pattern}' took {result['runtime_seconds']:.2f}s (expected < 10s)"

    def test_performance_with_different_criteria(self):
        """Test performance with different gamma criteria."""
        print("\n" + "=" * 70)
        print("Testing different gamma criteria (128x128x64)")
        print("=" * 70)

        criteria = [
            (1.0, 1.0, "1%/1mm (strict)"),
            (2.0, 2.0, "2%/2mm (moderate)"),
            (3.0, 3.0, "3%/3mm (standard)"),
            (5.0, 5.0, "5%/5mm (lenient)"),
        ]

        shape = (128, 128, 64)
        ref_dose = create_test_dose_distribution(shape, pattern="realistic")
        eval_dose = create_perturbed_dose(ref_dose)

        for dose_crit, dist_crit, label in criteria:
            print(f"\nCriteria: {label}")

            start_time = time.perf_counter()
            gamma_result = gamma.compute_gamma_index(
                ref_dose,
                eval_dose,
                dose_criterion_percent=dose_crit,
                distance_criterion_mm=dist_crit,
            )
            runtime = time.perf_counter() - start_time

            stats = gamma.compute_gamma_statistics(gamma_result)

            print(f"  Runtime: {runtime:.3f}s")
            print(f"  Passing rate: {stats['passing_rate_1_0']:.1f}%")

            assert (
                runtime < 10.0
            ), f"Criteria {label} took {runtime:.2f}s (expected < 10s)"

    def test_performance_scaling(self):
        """Test how performance scales with volume size."""
        print("\n" + "=" * 70)
        print("Testing performance scaling")
        print("=" * 70)

        sizes = [
            (32, 32, 32),
            (64, 64, 64),
            (96, 96, 96),
            (128, 128, 128),
        ]

        results = []
        for size in sizes:
            print(f"\nVolume: {size[0]}x{size[1]}x{size[2]}")
            result = measure_gamma_performance(size, pattern="gradient")
            results.append(result)
            print(f"  Runtime: {result['runtime_seconds']:.3f}s")
            print(f"  Voxels/second: {result['voxels_per_second']:.0f}")

        # Print scaling analysis
        print("\n" + "-" * 70)
        print("Scaling Analysis:")
        print("-" * 70)
        print(f"{'Size':>15} {'Voxels':>12} {'Runtime (s)':>12} {'Vox/sec':>12}")
        print("-" * 70)

        for result in results:
            shape = result["volume_shape"]
            print(
                f"{shape[0]:>4}x{shape[1]:>4}x{shape[2]:>4} "
                f"{result['total_voxels']:>12,} "
                f"{result['runtime_seconds']:>12.3f} "
                f"{result['voxels_per_second']:>12,.0f}"
            )

        # Check that largest size still meets threshold
        largest_result = results[-1]
        assert (
            largest_result["runtime_seconds"] < 10.0
        ), f"Largest test volume took {largest_result['runtime_seconds']:.2f}s (threshold: 10s)"


class TestGammaPerformanceEdgeCases:
    """Test performance in edge cases and special scenarios."""

    def test_performance_with_local_normalization(self):
        """Test performance with local vs global normalization."""
        print("\n" + "=" * 70)
        print("Testing local vs global normalization (128x128x64)")
        print("=" * 70)

        shape = (128, 128, 64)
        ref_dose = create_test_dose_distribution(shape, pattern="realistic")
        eval_dose = create_perturbed_dose(ref_dose)

        # Global normalization
        print("\nGlobal normalization:")
        start_time = time.perf_counter()
        gamma_global = gamma.compute_gamma_index(
            ref_dose, eval_dose, global_normalization=True
        )
        runtime_global = time.perf_counter() - start_time
        print(f"  Runtime: {runtime_global:.3f}s")

        # Local normalization
        print("\nLocal normalization:")
        start_time = time.perf_counter()
        gamma_local = gamma.compute_gamma_index(
            ref_dose, eval_dose, global_normalization=False
        )
        runtime_local = time.perf_counter() - start_time
        print(f"  Runtime: {runtime_local:.3f}s")

        # Both should be reasonable
        assert (
            runtime_global < 10.0
        ), f"Global normalization took {runtime_global:.2f}s (expected < 10s)"
        assert (
            runtime_local < 10.0
        ), f"Local normalization took {runtime_local:.2f}s (expected < 10s)"

    def test_performance_with_high_threshold(self):
        """Test performance when most voxels are excluded by threshold."""
        print("\n" + "=" * 70)
        print("Testing with high dose threshold (128x128x64)")
        print("=" * 70)

        shape = (128, 128, 64)
        ref_dose = create_test_dose_distribution(shape, pattern="sphere")
        eval_dose = create_perturbed_dose(ref_dose)

        # High threshold - excludes more points
        print("\nHigh threshold (50%):")
        start_time = time.perf_counter()
        gamma_result = gamma.compute_gamma_index(
            ref_dose,
            eval_dose,
            dose_threshold_percent=50.0,  # High threshold
        )
        runtime = time.perf_counter() - start_time

        valid_points = np.sum(~np.isnan(gamma_result))
        total_points = np.prod(shape)

        print(f"  Runtime: {runtime:.3f}s")
        print(
            f"  Valid points: {valid_points}/{total_points} "
            f"({100*valid_points/total_points:.1f}%)"
        )

        # Should be faster with fewer points to evaluate
        assert (
            runtime < 10.0
        ), f"High threshold case took {runtime:.2f}s (expected < 10s)"


if __name__ == "__main__":
    """Allow running performance tests directly."""
    print("=" * 70)
    print("Gamma Analysis Performance Test Suite")
    print("=" * 70)
    print("\nRunning performance benchmarks...")
    print("This will measure gamma index computation speed for various volumes.")
    print(
        "\nNote: Tests will fail if runtime exceeds 10 seconds for 128^3 or 256^3 volumes."
    )
    print("=" * 70)

    # Run pytest with verbose output
    pytest.main([__file__, "-v", "-s"])
