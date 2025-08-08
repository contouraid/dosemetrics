"""
Tests for dosemetrics.utils.plot module.

Tests plotting and visualization functionality.
"""

import unittest
import logging
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

    def test_plot_dvh_exists(self):
        """Test that plot_dvh function is callable."""
        # Basic existence test - actual plotting would require test data
        self.assertTrue(callable(dosemetrics.plot_dvh))


if __name__ == "__main__":
    unittest.main()
