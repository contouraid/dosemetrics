"""
Tests for dosemetrics.utils.comparison module.

Tests dose comparison and analysis functionality.
"""

import unittest
import logging
import dosemetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestComparison(unittest.TestCase):
    """Test dose comparison functionality."""

    def test_import_functions_exist(self):
        """Test that expected comparison functions are available."""
        # Test that key comparison functions are accessible
        self.assertTrue(hasattr(dosemetrics, "compare_dvh"))
        self.assertTrue(hasattr(dosemetrics, "quality_index"))

    def test_quality_index_exists(self):
        """Test that quality_index function is callable."""
        # Basic existence test - actual comparison would require test data
        self.assertTrue(callable(dosemetrics.quality_index))


if __name__ == "__main__":
    unittest.main()
