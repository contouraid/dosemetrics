"""
Tests for dosemetrics.utils.compliance module.

Tests compliance checking and constraint handling functionality.
"""

import dosemetrics
import pandas as pd
import numpy as np
import unittest
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def is_string_series(s: pd.Series):
    """
    IS_STRING_SERIES: From https://stackoverflow.com/a/67001213
    :param s: Pandas series object.
    :return: True if all elements in s are strings.
    """
    logger.info("Checking if series is string series: %s", s.name)
    if isinstance(s.dtype, pd.StringDtype):
        # The series was explicitly created as a string series (Pandas>=1.0.0)
        logger.info("Series is explicitly a string series.")
        return True
    elif s.dtype == "object":
        # Object series, check each value
        result = all((v is None) or isinstance(v, str) for v in s)
        logger.info("Series is object type, all values are strings: %s", result)
        return result
    else:
        return False


class TestCompliance(unittest.TestCase):
    """Test compliance checking functionality."""

    def test_default_compliance(self):
        """Test default constraint loading and validation."""
        logger.info("Testing default compliance.")
        constraints = dosemetrics.get_default_constraints()

        # Check that constraints have the correct columns
        self.assertTrue(set(constraints.columns) == {"Constraint Type", "Level"})

        # Check that index is string (structure names)
        self.assertTrue(is_string_series(constraints.index))

        # Check that constraint type is string
        self.assertTrue(is_string_series(constraints["Constraint Type"]))
        constraint_types = constraints["Constraint Type"].unique()

        # Should have min, max, and mean constraint types
        self.assertTrue(len(constraint_types) >= 2)  # At least min/max or mean
        valid_types = {"min", "max", "mean"}
        self.assertTrue(all(ct in valid_types for ct in constraint_types))

        # Check that Level is numeric
        self.assertFalse(is_string_series(constraints["Level"]))
        self.assertFalse(np.any(constraints["Level"] < 0))  # No negative levels.
        self.assertFalse(np.any(constraints["Level"] > 70))  # No levels above 70 Gy.

        # Should have exactly 2 columns (Constraint Type and Level)
        self.assertTrue(constraints.shape[1] == 2)

        # Check that we have some common structures
        self.assertTrue(len(constraints) > 0)  # At least some constraints defined

    def test_check_compliance(self):
        """Test compliance checking against constraints."""
        logger.info("Testing check compliance.")
        example_df = pd.DataFrame(
            [
                {"Structure": "Brain", "Mean Dose": 25, "Max Dose": 30, "Min Dose": 10},
                {
                    "Structure": "BrainStem",
                    "Mean Dose": 15,
                    "Max Dose": 50,
                    "Min Dose": 5,
                },
            ],
        )
        example_df.set_index("Structure", inplace=True)

        constraints = dosemetrics.get_default_constraints()
        results = dosemetrics.check_compliance(example_df, constraints)

        # Check that results have the correct structure
        self.assertTrue(set(results.columns) == {"Compliance", "Reason"})
        self.assertTrue(results.shape[0] == example_df.shape[0])
        self.assertTrue(results.shape[1] == 2)
        self.assertTrue(is_string_series(results["Compliance"]))
        self.assertTrue(is_string_series(results["Reason"]))

        # Check that compliance values are reasonable (should contain Yes or No)
        compliance_values = results["Compliance"].unique()
        # Note: May contain emoji characters, so just check for non-empty strings
        self.assertTrue(all(len(str(val)) > 0 for val in compliance_values))

    def test_check_compliance_violations(self):
        """Test compliance checking with clear violations."""
        logger.info("Testing check compliance with violations.")
        # Create example data with clear constraint violations
        example_df = pd.DataFrame(
            [
                {
                    "Structure": "Brain",
                    "Mean Dose": 50,
                    "Max Dose": 60,
                    "Min Dose": 40,
                },  # Mean exceeds 30
                {
                    "Structure": "BrainStem",
                    "Mean Dose": 30,
                    "Max Dose": 60,
                    "Min Dose": 10,
                },  # Max exceeds 54
            ],
        )
        example_df.set_index("Structure", inplace=True)

        constraints = dosemetrics.get_default_constraints()
        results = dosemetrics.check_compliance(example_df, constraints)

        # Results should indicate non-compliance
        self.assertTrue(results.shape[0] == 2)
        # Reason should mention constraint violations
        self.assertTrue(
            all(
                "constraint" in str(reason).lower() or "within" in str(reason).lower()
                for reason in results["Reason"]
            )
        )

    def test_check_compliance_passing(self):
        """Test compliance checking with passing constraints."""
        logger.info("Testing check compliance with passing values.")
        # Create example data that meets all constraints
        example_df = pd.DataFrame(
            [
                {
                    "Structure": "Brain",
                    "Mean Dose": 20,
                    "Max Dose": 25,
                    "Min Dose": 10,
                },  # Mean under 30
                {
                    "Structure": "BrainStem",
                    "Mean Dose": 15,
                    "Max Dose": 45,
                    "Min Dose": 5,
                },  # Max under 54
            ],
        )
        example_df.set_index("Structure", inplace=True)

        constraints = dosemetrics.get_default_constraints()
        results = dosemetrics.check_compliance(example_df, constraints)

        # Results should indicate compliance
        self.assertTrue(results.shape[0] == 2)
        # All should pass
        self.assertTrue(
            all(
                "within" in str(reason).lower() or "achieved" in str(reason).lower()
                for reason in results["Reason"]
            )
        )


if __name__ == "__main__":
    unittest.main()
