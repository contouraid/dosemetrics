"""Compliance checking for dose constraints."""

from __future__ import annotations

import pandas as pd


def get_custom_constraints():
    """
    GET_CUSTOM_CONSTRAINTS: Get custom constraints for common structures.
    :return: DataFrame with custom constraints for common structures.
    """
    constraint_df = pd.DataFrame(
        [
            {"Structure": "Brain", "Constraint Type": "mean", "Level": 30},
            {"Structure": "BrainStem", "Constraint Type": "max", "Level": 56},
            {"Structure": "Chiasm", "Constraint Type": "max", "Level": 55},
            {"Structure": "Cochlea_L", "Constraint Type": "max", "Level": 45},
            {"Structure": "Cochlea_R", "Constraint Type": "max", "Level": 45},
            {"Structure": "LacrimalGland_L", "Constraint Type": "max", "Level": 40},
            {"Structure": "LacrimalGland_R", "Constraint Type": "max", "Level": 40},
            {"Structure": "OpticNerve_L", "Constraint Type": "max", "Level": 56},
            {"Structure": "OpticNerve_R", "Constraint Type": "max", "Level": 56},
            {"Structure": "GTV", "Constraint Type": "nmean", "Level": 60},
            {"Structure": "CTV", "Constraint Type": "nmean", "Level": 60},
            {"Structure": "PTV", "Constraint Type": "nmean", "Level": 60},
        ]
    )

    constraint_df.set_index("Structure", inplace=True)
    return constraint_df


def get_default_constraints():
    """
    GET_DEFAULT_CONSTRAINTS: Get default constraints for common structures.
    :return: DataFrame with default constraints for common structures.
    """
    constraint_df = pd.DataFrame(
        [
            {"Structure": "Brain", "Constraint Type": "mean", "Level": 30},
            {"Structure": "BrainStem", "Constraint Type": "max", "Level": 54},
            {"Structure": "Chiasm", "Constraint Type": "max", "Level": 54},
            {"Structure": "Cochlea_L", "Constraint Type": "mean", "Level": 45},
            {"Structure": "Cochlea_R", "Constraint Type": "mean", "Level": 45},
            {"Structure": "Eye_L", "Constraint Type": "max", "Level": 10},
            {"Structure": "Eye_R", "Constraint Type": "max", "Level": 10},
            {"Structure": "Hippocampus_L", "Constraint Type": "mean", "Level": 30},
            {"Structure": "Hippocampus_R", "Constraint Type": "mean", "Level": 30},
            {"Structure": "LacrimalGland_L", "Constraint Type": "mean", "Level": 25},
            {"Structure": "LacrimalGland_R", "Constraint Type": "mean", "Level": 25},
            {"Structure": "OpticNerve_L", "Constraint Type": "max", "Level": 54},
            {"Structure": "OpticNerve_R", "Constraint Type": "max", "Level": 54},
            {"Structure": "Pituitary", "Constraint Type": "mean", "Level": 45},
            {"Structure": "Target", "Constraint Type": "min", "Level": 60},
        ]
    )

    constraint_df.set_index("Structure", inplace=True)
    return constraint_df


def check_compliance(df, constraint):
    """
    CHECK_COMPLIANCE: Check compliance of dose metrics with constraints.
    :param df: DataFrame with dose metrics including columns for max-dose, mean-dose, ...
    :param constraint: DataFrame constructed using get_default_constraints().
    :return: DataFrame with compliance status and failure reason for each structure.
    """
    compliance_df = pd.DataFrame()
    for structure in constraint.index:
        if structure in df.index:
            if constraint.loc[structure, "Constraint Type"] == "max":
                if (
                    pd.to_numeric(df.loc[structure, "Max Dose"])
                    > constraint.loc[structure, "Level"]
                ):
                    compliance_df.loc[structure, "Compliance"] = "❌ No"
                    compliance_df.loc[structure, "Reason"] = (
                        f"Max dose constraint: "
                        f"{float(constraint.loc[structure, 'Level'])}, "
                        f"exceeded: {float(df.loc[structure, 'Max Dose']):.2f}"
                    )
                else:
                    compliance_df.loc[structure, "Compliance"] = "✅ Yes"
                    compliance_df.loc[structure, "Reason"] = (
                        f"Max dose is within constraint! "
                    )
            elif constraint.loc[structure, "Constraint Type"] == "min":
                if (
                    pd.to_numeric(df.loc[structure, "Min Dose"])
                    < constraint.loc[structure, "Level"]
                ):
                    compliance_df.loc[structure, "Compliance"] = "❌ No"
                    compliance_df.loc[structure, "Reason"] = (
                        "Min dose constraint: "
                        + str(constraint.loc[structure, "Level"])
                        + " not met: "
                        + str(df.loc[structure, "Min Dose"])
                    )
                else:
                    compliance_df.loc[structure, "Compliance"] = "✅ Yes"
                    compliance_df.loc[structure, "Reason"] = (
                        f"Min dose is within constraint! "
                    )
            elif constraint.loc[structure, "Constraint Type"] == "mean":
                if (
                    pd.to_numeric(df.loc[structure, "Mean Dose"])
                    > constraint.loc[structure, "Level"]
                ):
                    compliance_df.loc[structure, "Compliance"] = "❌ No"
                    compliance_df.loc[structure, "Reason"] = (
                        f"Mean dose constraint: "
                        f"{constraint.loc[structure, 'Level']},"
                        f" exceeded: {df.loc[structure, 'Mean Dose']:.2f}"
                    )
                else:
                    compliance_df.loc[structure, "Compliance"] = "✅ Yes"
                    compliance_df.loc[structure, "Reason"] = (
                        f"Mean dose is within constraint! "
                    )
            elif constraint.loc[structure, "Constraint Type"] == "nmean":
                # This is negative mean dose, so we want to check if the mean dose
                # is below the constraint. This is used only for targets.
                if df.loc[structure, "Mean Dose"] < constraint.loc[structure, "Level"]:
                    compliance_df.loc[structure, "Compliance"] = "❌ No"
                    compliance_df.loc[structure, "Reason"] = (
                        f"Target mean dose constraint: "
                        f"{constraint.loc[structure, 'Level']},"
                        f" higher than: {df.loc[structure, 'Mean Dose']:.2f}"
                    )
                else:
                    compliance_df.loc[structure, "Compliance"] = "✅ Yes"
                    compliance_df.loc[structure, "Reason"] = (
                        f"Target mean dose is achieved! "
                    )
            elif constraint.loc[structure, "Constraint Type"] == "volume":
                NotImplementedError("Volume constraint not implemented yet!")
                # compliance_df.loc[structure, "Compliance"] = "✅ Yes"
                # compliance_df.loc[
                #    structure, "Reason"
                # ] = f"Volume dose is within constraint! "

    return compliance_df
