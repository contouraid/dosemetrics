"""
Tests for dosemetrics.structures module.
Tests the radiotherapy structure classes (Structure, OAR, Target, AvoidanceStructure) with functionality including geometric analysis, dose statistics, DVH computation, and target-specific metrics.
"""

import numpy as np
import unittest
import logging
from typing import Tuple

import dosemetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestStructures(unittest.TestCase):
    """Comprehensive test suite for radiotherapy structure classes."""

    def setUp(self):
        """Set up test fixtures for use in test methods."""
        self.shape = (20, 20, 20)
        self.spacing = (2.0, 2.0, 2.0)
        self.origin = (0.0, 0.0, 0.0)

        # Create sample dose distribution (increasing with depth)
        self.dose_volume = np.zeros(self.shape)
        for z in range(self.shape[2]):
            self.dose_volume[:, :, z] = (
                40 + (z / self.shape[2]) * 30
            )  # 40-70 Gy gradient

        # Add some noise to make it realistic
        self.dose_volume += np.random.normal(0, 2, self.shape)
        self.dose_volume = np.clip(
            self.dose_volume, 0, None
        )  # Ensure no negative doses

        # Create target mask: central sphere
        center = np.array(self.shape) // 2
        self.target_mask = np.zeros(self.shape, dtype=bool)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                for k in range(self.shape[2]):
                    if (
                        np.sqrt(
                            (i - center[0]) ** 2
                            + (j - center[1]) ** 2
                            + (k - center[2]) ** 2
                        )
                        <= 5
                    ):
                        self.target_mask[i, j, k] = True

        # Create OAR mask: off-center smaller sphere
        oar_center = center + np.array([6, 0, 0])
        self.oar_mask = np.zeros(self.shape, dtype=bool)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                for k in range(self.shape[2]):
                    if (
                        np.sqrt(
                            (i - oar_center[0]) ** 2
                            + (j - oar_center[1]) ** 2
                            + (k - oar_center[2]) ** 2
                        )
                        <= 3
                    ):
                        self.oar_mask[i, j, k] = True

    def test_basic_structure_creation(self):
        """Test basic structure creation and properties."""
        logger.info("Testing basic structure creation...")

        # Test OAR creation
        oar = dosemetrics.OAR("test_oar")
        self.assertEqual(oar.name, "test_oar")
        self.assertEqual(oar.structure_type, dosemetrics.StructureType.OAR)
        self.assertEqual(oar.volume_voxels(), 0)
        self.assertEqual(oar.volume_cc(), 0.0)
        self.assertFalse(oar.has_mask)
        self.assertFalse(oar.has_dose)

        # Test Target creation
        target = dosemetrics.Target("test_target")
        self.assertEqual(target.name, "test_target")
        self.assertEqual(target.structure_type, dosemetrics.StructureType.TARGET)

        # Test AvoidanceStructure creation
        avoidance = dosemetrics.AvoidanceStructure("test_avoidance")
        self.assertEqual(avoidance.name, "test_avoidance")
        self.assertEqual(avoidance.structure_type, dosemetrics.StructureType.AVOIDANCE)

        logger.info("... completed basic structure creation tests.")

    def test_mask_handling(self):
        """Test mask setting and validation."""
        logger.info("Testing mask handling...")

        structure = dosemetrics.OAR("test_mask")

        # Test setting valid 3D mask
        test_mask = np.ones((5, 5, 5), dtype=bool)
        structure.set_mask(test_mask)
        self.assertTrue(structure.has_mask)
        self.assertEqual(structure.volume_voxels(), 125)

        # Test volume calculation with different spacing
        structure.spacing = (2.0, 2.0, 2.0)  # 2mm voxels
        expected_volume_cc = 125 * (2 * 2 * 2) / 1000.0  # Convert mm³ to cc
        self.assertAlmostEqual(structure.volume_cc(), expected_volume_cc, places=3)

        # Test error handling for invalid mask dimensions
        with self.assertRaises(ValueError) as context:
            invalid_mask = np.random.rand(10, 10)  # 2D instead of 3D
            structure.set_mask(invalid_mask)
        self.assertIn("Mask must be 3D", str(context.exception))

        logger.info("... completed mask handling tests.")

    def test_dose_handling(self):
        """Test dose data setting and validation."""
        logger.info("Testing dose handling...")

        structure = dosemetrics.OAR("test_dose")

        # Set mask first
        test_mask = np.ones((5, 5, 5), dtype=bool)
        structure.set_mask(test_mask)

        # Test setting valid dose data
        test_dose = np.random.rand(5, 5, 5) * 70  # Random dose 0-70 Gy
        structure.set_dose_data(test_dose)
        self.assertTrue(structure.has_dose)

        # Test error handling for mismatched dimensions
        with self.assertRaises(ValueError) as context:
            invalid_dose = np.ones((10, 10, 10))  # Wrong dimensions
            structure.set_dose_data(invalid_dose)
        self.assertIn("Dose shape", str(context.exception))

        logger.info("... completed dose handling tests.")

    def test_dose_statistics(self):
        """Test dose statistical calculations."""
        logger.info("Testing dose statistics...")

        # Create structure with known dose values
        structure = dosemetrics.OAR("test_stats")

        # Create a simple mask
        test_mask = np.zeros((10, 10, 10), dtype=bool)
        test_mask[5:7, 5:7, 5:7] = True  # 2x2x2 = 8 voxels
        structure.set_mask(test_mask)

        # Create dose data with known values
        test_dose = np.zeros((10, 10, 10))
        dose_values = [10, 20, 30, 40, 50, 60, 70, 80]  # 8 different dose values
        flat_indices = np.where(test_mask.flatten())[0]
        test_dose_flat = test_dose.flatten()
        test_dose_flat[flat_indices] = dose_values
        test_dose = test_dose_flat.reshape((10, 10, 10))
        structure.set_dose_data(test_dose)

        # Test statistics
        self.assertEqual(structure.mean_dose(), np.mean(dose_values))
        self.assertEqual(structure.max_dose(), 80.0)
        self.assertEqual(structure.min_dose(), 10.0)
        self.assertEqual(structure.percentile_dose(50), np.percentile(dose_values, 50))

        # Test with no dose data
        no_dose_structure = dosemetrics.OAR("no_dose")
        self.assertIsNone(no_dose_structure.mean_dose())
        self.assertIsNone(no_dose_structure.max_dose())

        logger.info("... completed dose statistics tests.")

    def test_geometric_analysis(self):
        """Test geometric calculations."""
        logger.info("Testing geometric analysis...")

        # Create structure with known geometry
        structure = dosemetrics.OAR("test_geometry")

        # Create a 3x3x3 mask at a known location
        test_mask = np.zeros((10, 10, 10), dtype=bool)
        test_mask[5:8, 5:8, 5:8] = True
        structure.set_mask(test_mask)
        structure.spacing = (1.0, 1.0, 1.0)
        structure.origin = (0.0, 0.0, 0.0)

        # Test centroid calculation
        centroid = structure.centroid()
        self.assertIsNotNone(centroid)
        if centroid is not None:
            # Centroid should be at (6, 6, 6) in voxel coordinates
            self.assertAlmostEqual(centroid[0], 6.0, places=1)
            self.assertAlmostEqual(centroid[1], 6.0, places=1)
            self.assertAlmostEqual(centroid[2], 6.0, places=1)

        # Test bounding box calculation
        bbox = structure.bounding_box()
        self.assertIsNotNone(bbox)
        if bbox is not None:
            self.assertEqual(bbox[0], (5, 7))  # x bounds
            self.assertEqual(bbox[1], (5, 7))  # y bounds
            self.assertEqual(bbox[2], (5, 7))  # z bounds

        logger.info("... completed geometric analysis tests.")

    def test_dvh_computation(self):
        """Test DVH computation with known dose distribution."""
        logger.info("Testing DVH computation...")

        # Create structure with gradient dose
        structure = dosemetrics.OAR("test_dvh")

        # Create a simple mask
        test_mask = np.zeros((10, 10, 10), dtype=bool)
        test_mask[5:7, 5:7, 5:7] = True  # 2x2x2 = 8 voxels
        structure.set_mask(test_mask)

        # Create dose data with known values
        test_dose = np.zeros((10, 10, 10))
        dose_values = [10, 20, 30, 40, 50, 60, 70, 80]  # 8 different dose values
        flat_indices = np.where(test_mask.flatten())[0]
        test_dose_flat = test_dose.flatten()
        test_dose_flat[flat_indices] = dose_values
        test_dose = test_dose_flat.reshape((10, 10, 10))
        structure.set_dose_data(test_dose)

        # Compute DVH
        bins, volumes = structure.dvh(max_dose=100, step_size=10)

        # At 0 Gy, 100% of volume should be included
        self.assertEqual(volumes[0], 100.0)

        # At 50 Gy, 50% of volume should be included (4 out of 8 voxels >= 50)
        bin_50_idx = np.argmin(np.abs(bins - 50))
        self.assertEqual(volumes[bin_50_idx], 50.0)

        logger.info("... completed DVH computation tests.")

    def test_target_specific_features(self):
        """Test Target-specific methods."""
        logger.info("Testing target-specific features...")

        # Create a target structure
        target = dosemetrics.Target("test_target")

        # Create a simple mask
        test_mask = np.ones((5, 5, 5), dtype=bool)  # 125 voxels
        target.set_mask(test_mask)

        # Create dose data with uniform 60 Gy dose
        test_dose = np.full((5, 5, 5), 60.0)
        target.set_dose_data(test_dose)

        # Test coverage
        coverage = target.coverage_volume_percentage(50.0)  # 50 Gy threshold
        self.assertEqual(coverage, 100.0)  # All voxels above 50 Gy

        coverage = target.coverage_volume_percentage(70.0)  # 70 Gy threshold
        self.assertEqual(coverage, 0.0)  # No voxels above 70 Gy

        # Test conformity index
        ci = target.conformity_index(60.0, tolerance=0.05)
        self.assertIsNotNone(ci)
        if ci is not None:
            self.assertAlmostEqual(
                ci, 1.0, places=2
            )  # Perfect conformity for uniform dose

        logger.info("... completed target-specific features tests.")

    def test_structure_type_system(self):
        """Test the structure type enumeration system."""
        logger.info("Testing structure type system...")

        # Test all available structure types
        expected_types = ["oar", "target", "avoidance", "support", "external"]
        available_types = [st.value for st in dosemetrics.StructureType]

        for expected_type in expected_types:
            self.assertIn(expected_type, available_types)

        # Test structure type assignment
        oar = dosemetrics.OAR("test")
        target = dosemetrics.Target("test")
        avoidance = dosemetrics.AvoidanceStructure("test")

        self.assertEqual(oar.structure_type, dosemetrics.StructureType.OAR)
        self.assertEqual(target.structure_type, dosemetrics.StructureType.TARGET)
        self.assertEqual(avoidance.structure_type, dosemetrics.StructureType.AVOIDANCE)

        logger.info("... completed structure type system tests.")

    def test_comprehensive_workflow(self):
        """Test a comprehensive radiotherapy workflow with realistic data."""
        logger.info("Testing comprehensive workflow...")

        # Create Target structure
        target = dosemetrics.Target("PTV")
        target.set_mask(self.target_mask)
        target.spacing = self.spacing
        target.origin = self.origin
        target.set_dose_data(self.dose_volume)

        # Create OAR structure
        oar = dosemetrics.OAR("Spinal_Cord")
        oar.set_mask(self.oar_mask)
        oar.spacing = self.spacing
        oar.origin = self.origin
        oar.set_dose_data(self.dose_volume)

        # Test basic properties
        self.assertGreater(target.volume_cc(), 0)
        self.assertGreater(oar.volume_cc(), 0)
        self.assertTrue(target.has_mask)
        self.assertTrue(target.has_dose)
        self.assertTrue(oar.has_mask)
        self.assertTrue(oar.has_dose)

        # Test dose statistics are reasonable
        target_mean = target.mean_dose()
        oar_mean = oar.mean_dose()
        self.assertIsNotNone(target_mean)
        self.assertIsNotNone(oar_mean)
        if target_mean and oar_mean:
            self.assertGreater(target_mean, 40)  # Should be > 40 Gy
            self.assertLess(target_mean, 80)  # Should be < 80 Gy
            self.assertGreater(oar_mean, 40)  # Should be > 40 Gy
            self.assertLess(oar_mean, 80)  # Should be < 80 Gy

        # Test geometric analysis
        target_centroid = target.centroid()
        oar_centroid = oar.centroid()
        self.assertIsNotNone(target_centroid)
        self.assertIsNotNone(oar_centroid)

        if target_centroid and oar_centroid:
            # Calculate distance between centroids
            distance = np.sqrt(
                sum((a - b) ** 2 for a, b in zip(target_centroid, oar_centroid))
            )
            self.assertGreater(distance, 0)  # Should be separated

        # Test target-specific metrics
        prescription_dose = 60.0  # Gy
        coverage_95 = target.coverage_volume_percentage(prescription_dose * 0.95)
        self.assertIsNotNone(coverage_95)
        if coverage_95 is not None:
            self.assertGreaterEqual(coverage_95, 0)
            self.assertLessEqual(coverage_95, 100)

        # Test DVH computation
        target_bins, target_volumes = target.dvh(max_dose=80, step_size=1.0)
        oar_bins, oar_volumes = oar.dvh(max_dose=80, step_size=1.0)

        # DVH should start at 100% and decrease monotonically
        self.assertEqual(target_volumes[0], 100.0)
        self.assertEqual(oar_volumes[0], 100.0)
        self.assertTrue(
            all(
                target_volumes[i] >= target_volumes[i + 1]
                for i in range(len(target_volumes) - 1)
            )
        )
        self.assertTrue(
            all(
                oar_volumes[i] >= oar_volumes[i + 1]
                for i in range(len(oar_volumes) - 1)
            )
        )

        logger.info("... completed comprehensive workflow tests.")

    def test_string_representations(self):
        """Test string representations of structures."""
        logger.info("Testing string representations...")

        # Create a structure with data
        target = dosemetrics.Target("PTV")
        target.set_mask(self.target_mask)
        target.spacing = self.spacing
        target.set_dose_data(self.dose_volume)

        # Test string representation
        str_repr = str(target)
        self.assertIn("TARGET", str_repr)
        self.assertIn("PTV", str_repr)
        self.assertIn("Volume:", str_repr)
        self.assertIn("Mean dose:", str_repr)

        # Test detailed representation
        repr_str = repr(target)
        self.assertIn("Target", repr_str)
        self.assertIn("name='PTV'", repr_str)
        self.assertIn("type=target", repr_str)
        self.assertIn("has_mask=True", repr_str)
        self.assertIn("has_dose=True", repr_str)

        logger.info("... completed string representation tests.")

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        logger.info("Testing edge cases...")

        # Test empty structure
        empty_structure = dosemetrics.OAR("empty")
        self.assertEqual(empty_structure.volume_voxels(), 0)
        self.assertEqual(empty_structure.volume_cc(), 0.0)
        self.assertIsNone(empty_structure.mean_dose())
        self.assertIsNone(empty_structure.centroid())
        self.assertIsNone(empty_structure.bounding_box())

        # Test structure with single voxel
        single_voxel = dosemetrics.OAR("single")
        single_mask = np.zeros((5, 5, 5), dtype=bool)
        single_mask[2, 2, 2] = True
        single_voxel.set_mask(single_mask)
        self.assertEqual(single_voxel.volume_voxels(), 1)

        # Test DVH with no dose data
        bins, volumes = single_voxel.dvh()
        self.assertTrue(all(v == 0 for v in volumes))

        # Test structure with zero dose
        zero_dose = np.zeros((5, 5, 5))
        single_voxel.set_dose_data(zero_dose)
        self.assertEqual(single_voxel.mean_dose(), 0.0)
        self.assertEqual(single_voxel.max_dose(), 0.0)

        logger.info("... completed edge cases tests.")


if __name__ == "__main__":
    # Set up logging to show test progress
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run the tests
    unittest.main(verbosity=2)
