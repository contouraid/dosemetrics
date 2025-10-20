"""
Tests for dosemetrics.utils.plot module.

Tests plotting and visualization functionality.
"""

import unittest
import logging
import numpy as np
import dosemetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestPlot(unittest.TestCase):
    """Test plotting functionality."""

    def test_import_functions_exist(self):
        """Test that expected plotting functions are available."""
        # Test that key plotting functions are accessible
        self.assertTrue(hasattr(dosemetrics, "plot_dvh"))
        self.assertTrue(hasattr(dosemetrics, "from_dataframe"))
        self.assertTrue(hasattr(dosemetrics, "generate_dvh_variations"))
        self.assertTrue(hasattr(dosemetrics, "plot_dvh_variations"))

    def test_plot_dvh_exists(self):
        """Test that plot_dvh function is callable."""
        # Basic existence test - actual plotting would require test data
        self.assertTrue(callable(dosemetrics.plot_dvh))

    def test_generate_dvh_variations_exists(self):
        """Test that generate_dvh_variations function is callable."""
        self.assertTrue(callable(dosemetrics.generate_dvh_variations))

    def test_plot_dvh_variations_exists(self):
        """Test that plot_dvh_variations function is callable."""
        self.assertTrue(callable(dosemetrics.plot_dvh_variations))

    def test_generate_dvh_variations_basic(self):
        """Test basic functionality of generate_dvh_variations."""
        # Create simple test data
        dose_volume = np.random.rand(10, 10, 10) * 50
        structure_mask = np.zeros((10, 10, 10), dtype=np.uint8)
        structure_mask[3:7, 3:7, 3:7] = 1

        # Generate variations
        dvh_data, dice_coefficients, original_dvh = dosemetrics.generate_dvh_variations(
            dose_volume,
            structure_mask,
            n_variations=5,  # Small number for testing
            dice_range=(0.7, 1.0),
            volume_variation=0.2,
            max_dose=50,
            step_size=1.0,
        )

        # Check outputs
        self.assertEqual(len(dvh_data), 5)
        self.assertEqual(len(dice_coefficients), 5)
        self.assertIsInstance(original_dvh, tuple)
        self.assertEqual(len(original_dvh), 2)

        # Check that all Dice coefficients are in valid range
        for dice in dice_coefficients:
            self.assertGreaterEqual(dice, 0.0)
            self.assertLessEqual(dice, 1.0)

    def test_variability_backward_compatibility(self):
        """Test that legacy variability function still works."""
        # Create simple test data
        dose_volume = np.random.rand(10, 10, 10) * 50
        structure_mask = np.zeros((10, 10, 10), dtype=np.uint8)
        structure_mask[3:7, 3:7, 3:7] = 1

        # Call legacy function
        fig, (max_dice, min_dice) = dosemetrics.variability(
            dose_volume,
            structure_mask,
            constraint_limit=30.0,
            structure_of_interest="Test",
        )

        # Check outputs
        self.assertIsNotNone(fig)
        self.assertGreaterEqual(max_dice, 0.0)
        self.assertLessEqual(max_dice, 1.0)
        self.assertGreaterEqual(min_dice, 0.0)
        self.assertLessEqual(min_dice, 1.0)


if __name__ == "__main__":
    unittest.main()
