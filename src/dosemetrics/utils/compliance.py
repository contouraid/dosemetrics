"""
Compliance checking and quality indices for dose constraints.

This module provides functions to check compliance with dose constraints
and compute quality indices for treatment plan evaluation.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..dose import Dose
    from ..structures import Structure


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


def quality_index(
    dose: Dose,
    structure: Structure,
    constraint_type: str,
    constraint_level: float,
) -> float:
    """
    Compute the quality index of a dose distribution relative to a constraint.

    Quality index interpretation:
    - Positive values: Constraint is met (higher is better, 1.0 is ideal)
    - Negative values: Constraint is violated (magnitude indicates severity)

    Args:
        dose: Dose distribution object
        structure: Structure to evaluate
        constraint_type: Type of constraint ('max', 'mean', or 'min')
        constraint_level: Constraint value in Gy

    Returns:
        Quality index (-1 to 1)

    Examples:
        >>> from dosemetrics.dose import Dose
        >>> from dosemetrics.utils.compliance import quality_index
        >>>
        >>> dose = Dose.from_dicom("rtdose.dcm")
        >>> brainstem = structures.get_structure("Brainstem")
        >>>
        >>> # Check max dose constraint
        >>> qi = quality_index(dose, brainstem, "max", 54.0)
        >>> if qi < 0:
        ...     print("Constraint violated!")
    """
    from ..metrics import dvh, statistics

    dose_bins, volumes = dvh.compute_dvh(dose, structure)

    if constraint_type == "mean":
        # Check if mean dose exceeds constraint
        indices = np.where(dose_bins > constraint_level)[0]
        if len(indices) > 0:
            proportion_above = np.max(volumes[indices])
        else:
            proportion_above = 0.0

        if proportion_above > 0:
            # Negative value indicates violation
            # Worst case is -1 (all voxels above constraint)
            return -proportion_above / 100.0
        else:
            # Constraint is met - compute gap
            mean_dose_val = statistics.compute_mean_dose(dose, structure)
            gap_between = (constraint_level - mean_dose_val) / constraint_level
            return float(gap_between)

    elif constraint_type == "max":
        # Check if any dose exceeds constraint
        indices = np.where(dose_bins > constraint_level)[0]
        if len(indices) > 0:
            proportion_above = np.max(volumes[indices])
        else:
            proportion_above = 0.0

        if proportion_above > 0:
            # Negative value indicates violation
            return -proportion_above / 100.0
        else:
            # Constraint is met - compute gap
            max_dose_val = statistics.compute_max_dose(dose, structure)
            gap_between = (constraint_level - max_dose_val) / constraint_level
            return float(gap_between)

    elif constraint_type == "min":
        # For targets - check if dose is below constraint
        indices = np.where(dose_bins < constraint_level)[0]
        if len(indices) > 0:
            proportion_below = np.min(volumes[indices])
        else:
            proportion_below = 0.0

        if proportion_below < 100:
            # Negative value indicates violation
            return -(100 - proportion_below) / 100.0
        else:
            return 1.0

    # Default return
    return 0.0
