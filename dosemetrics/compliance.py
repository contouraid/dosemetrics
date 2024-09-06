import pandas as pd
import numpy as np


def get_default_constraints():

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

    return constraint_df


def check_compliance(df, constraint):
    """
    CHECK_COMPLIANCE: Check compliance of dose metrics with constraints.
    :param df: DataFrame with dose metrics including columns for max-dose, mean-dose, ...
    :param constraint: DataFrame constructed using get_default_constraints().
    :return: DataFrame with compliance status and failure reason for each structure.
    """
    compliance_df = pd.DataFrame()
    for structure in constraint["Structure"]:
        if structure in df.index:
            if (
                constraint.loc[
                    constraint["Structure"] == structure, "Constraint Type"
                ].values[0]
                == "max"
            ):
                if (
                    df.loc[structure, "Max Dose"]
                    > constraint.loc[
                        constraint["Structure"] == structure, "Level"
                    ].values[0]
                ):
                    compliance_df.loc[structure, "Compliance"] = "❌ No"
                    compliance_df.loc[structure, "Reason"] = (
                        f"Max dose constraint: "
                        f"{constraint.loc[constraint['Structure'] == structure, 'Level'].values[0]},"
                        f" exceeded: {df.loc[structure, 'Max Dose']:.2f}"
                    )
                else:
                    compliance_df.loc[structure, "Compliance"] = "✅ Yes"
                    compliance_df.loc[
                        structure, "Reason"
                    ] = f"Max dose is within constraint! "
            elif (
                constraint.loc[
                    constraint["Structure"] == structure, "Constraint Type"
                ].values[0]
                == "min"
            ):
                if (
                    df.loc[structure, "Min Dose"]
                    < constraint.loc[
                        constraint["Structure"] == structure, "Level"
                    ].values[0]
                ):
                    compliance_df.loc[structure, "Compliance"] = "❌ No"
                    compliance_df.loc[structure, "Reason"] = (
                        f"Min dose constraint: "
                        f"{constraint.loc[constraint['Structure'] == structure, 'Level'].values[0]},"
                        f" not met: {df.loc[structure, 'Min Dose']:.2f}"
                    )
                else:
                    compliance_df.loc[structure, "Compliance"] = "✅ Yes"
                    compliance_df.loc[
                        structure, "Reason"
                    ] = f"Min dose is within constraint! "
            elif (
                constraint.loc[
                    constraint["Structure"] == structure, "Constraint Type"
                ].values[0]
                == "mean"
            ):
                if (
                    df.loc[structure, "Mean Dose"]
                    > constraint.loc[
                        constraint["Structure"] == structure, "Level"
                    ].values[0]
                ):
                    compliance_df.loc[structure, "Compliance"] = "❌ No"
                    compliance_df.loc[structure, "Reason"] = (
                        f"Mean dose constraint: "
                        f"{constraint.loc[constraint['Structure'] == structure, 'Level'].values[0]},"
                        f" exceeded: {df.loc[structure, 'Mean Dose']:.2f}"
                    )
                else:
                    compliance_df.loc[structure, "Compliance"] = "✅ Yes"
                    compliance_df.loc[
                        structure, "Reason"
                    ] = f"Mean dose is within constraint! "
            elif (
                constraint.loc[
                    constraint["Structure"] == structure, "Constraint Type"
                ].values[0]
                == "volume"
            ):
                compliance_df.loc[structure, "Compliance"] = "✅ Yes"
                compliance_df.loc[
                    structure, "Reason"
                ] = f"Volume dose is within constraint! "

    return compliance_df
