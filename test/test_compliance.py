import dosemetrics.compliance as compliance
import pandas as pd
import numpy as np
import unittest


def is_string_series(s: pd.Series):
    """
    IS_STRING_SERIES: From https://stackoverflow.com/a/67001213
    :param s: Pandas series object.
    :return: True if all elements in s are strings.
    """
    if isinstance(s.dtype, pd.StringDtype):
        # The series was explicitly created as a string series (Pandas>=1.0.0)
        return True
    elif s.dtype == "object":
        # Object series, check each value
        return all((v is None) or isinstance(v, str) for v in s)
    else:
        return False


class TestCompliance(unittest.TestCase):
    def test_default_compliance(self):
        constraints = compliance.get_default_constraints()

        self.assertTrue(
            constraints.columns.isin(["Structure", "Constraint Type", "Level"]).all()
        )

        self.assertTrue(is_string_series(constraints["Structure"]))
        self.assertTrue(is_string_series(constraints["Constraint Type"]))
        constraint_types = constraints["Constraint Type"].unique()
        self.assertTrue(len(constraint_types) == 3)  # min, max and mean.

        self.assertFalse(is_string_series(constraints["Level"]))
        self.assertFalse(np.any(constraints["Level"] < 0))  # No negative levels.
        self.assertFalse(np.any(constraints["Level"] > 70))  # No levels above 70 Gy.

        self.assertTrue(constraints.shape[1] == 3)

    def test_check_compliance(self):
        example_df = pd.DataFrame(
            [
                {"Structure": "Brain", "Mean Dose": 25, "Max Dose": 30},
                {"Structure": "BrainStem", "Mean Dose": 15, "Max Dose": 20},
            ],
        )
        example_df.set_index("Structure", inplace=True)

        results = compliance.check_compliance(
            example_df, compliance.get_default_constraints()
        )

        self.assertTrue(results.columns.isin(["Compliance", "Reason"]).all())
        self.assertTrue(results.shape == (2, 2))
        self.assertTrue(is_string_series(results["Compliance"]))
        self.assertTrue(is_string_series(results["Reason"]))


if __name__ == "__main__":
    unittest.main()
