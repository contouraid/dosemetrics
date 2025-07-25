import pandas as pd
import numpy as np

from dosemetrics.dvh import compute_dvh, mean_dose, max_dose


def get_custom_constraints():
    """
    GET_CUSTOM_CONSTRAINTS: Get custom constraints for common structures.
    :return: DataFrame with custom constraints for common structures.
    """

    """
    • Optic Chiasm: ≤55 Gy. to 0.03cc
    • Optic Chiasm PRV ≤55Gy to 0.03cc
    • Brainstem: ≤56 Gy. to 0.03cc
    • Optic Nerves ≤ 56 Gy to 0.03cc
    • Optic Nerves PRV: ≤56 Gy. to 0.03cc
    • Eye balls, retina: ≤40 G to Dmax
    • Lens: 10 Gy to 0.03cc
    • Lacrimal glands: <40 Gy to 0.03cc
    • Cochlea: ≤45 Gy if both sides are involved; otherwise ≤60 Gy. (Low priority OaR) to 0.03cc
    • The dose to the normal brain minus the PTV should be kept as low as possible. The Dmean is to be ≤ 30 Gy    
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
                        f"{constraint.loc[structure, 'Level']},"
                        f" exceeded: {df.loc[structure, 'Max Dose']:.2f}"
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
    _dose: np.ndarray,
    _struct_mask: np.ndarray,
    _constraint_type: str,
    _constraint_level: float,
) -> float:
    """
    QUALITY_INDEX: Compute the quality index of a dose distribution.
    :param _dose: Dose distribution.
    :param _struct_mask: Mask of the structure of interest.
    :param _constraint_type: max or mean.
    :param _constraint_level: Constraint value in Gray.
    :return: Quality index.
    """
    bins, values = compute_dvh(_dose, _struct_mask)

    if _constraint_type == "mean":
        proportion_above = np.max(values[np.where(bins > _constraint_level)[0]])
        if proportion_above > 0:
            # negative value here to indicate crossing the constraint,
            # worst case is -1, where all voxels are above constraint.
            return -proportion_above / 100  # percentage to ratio.
        else:
            _mean_dose = mean_dose(_dose, _struct_mask)
            gap_between = (_constraint_level - _mean_dose) / _constraint_level
            # Ideal value here is 1, as max_dose will be 0,
            # and constraint_value will be non-zero positive.
            return float(gap_between)

    elif _constraint_type == "max":
        proportion_above = np.max(values[np.where(bins > _constraint_level)[0]])
        if proportion_above > 0:
            # negative value here to indicate crossing the constraint,
            # worst case is -1, where all voxels are above constraint.
            return -proportion_above / 100  # percentage to ratio.
        else:
            _max_dose = max_dose(_dose, _struct_mask)
            gap_between = (_constraint_level - _max_dose) / _constraint_level
            # Ideal value here is 1, as max_dose will be 0,
            # and constraint_value will be non-zero positive.
            return gap_between

    elif _constraint_type == "min":
        # This is for targets, but could be applied to other structures.
        proportion_below = np.min(values[np.where(bins < _constraint_level)[0]])
        if proportion_below < 100:
            # negative value here to indicate crossing the constraint,
            # worst case is -1, where all voxels are above constraint.
            return -(100 - proportion_below) / 100  # percentage to ratio.

        # If none of the constraint types match, return a default float value
        return 0.0


def compute_mirage_compliance(dose_volume: np.ndarray, structure_masks: dict):
    compliance_stats = {}

    for struct_name in sorted(structure_masks.keys()):
        struct_mask = structure_masks[struct_name]
        dose_in_struct = dose_volume[struct_mask > 0]
        if struct_name == "Chiasm":
            # Optic Chiasm: ≤55 Gy. to 0.03cc, Optic Chiasm PRV ≤55Gy to 0.03cc
            # Checking for 4 because our voxel grid is 2mmx2mmx2mm, meaning each voxel is 8mm3.
            # 0.03cc is 30mm3, which is between 3 and 4 voxels - 24mm3 and 32mm3.
            if struct_mask.sum() > 4:
                sorted_dose = np.sort(dose_in_struct)[::-1]
                calculated_dose = sorted_dose[3]
                limit_dose = 55
                if calculated_dose >= limit_dose:
                    reason = f"{struct_name} <= {limit_dose} Gy to 0.03cc violated. Dose in 0.03cc is {calculated_dose:.3f}"
                    compliance_stats[struct_name] = [
                        "Fail",
                        reason,
                        limit_dose - calculated_dose,
                    ]
                else:
                    reason = f"{struct_name} <= {limit_dose} Gy to 0.03cc achieved. Dose in 0.03cc is {calculated_dose:.3f}"
                    compliance_stats[struct_name] = [
                        "Pass",
                        reason,
                        limit_dose - calculated_dose,
                    ]
            else:
                reason = f"{struct_name} volume is smaller than 0.03cc."
                compliance_stats[struct_name] = ["NA", reason, 0]
            print(compliance_stats[struct_name])

        elif (struct_name == "Brainstem") or (struct_name == "BrainStem"):
            # Brainstem: ≤56 Gy. to 0.03cc
            if struct_mask.sum() > 4:
                sorted_dose = np.sort(dose_in_struct)[::-1]
                calculated_dose = sorted_dose[3]
                limit_dose = 56
                if calculated_dose >= limit_dose:
                    reason = f"{struct_name} <= {limit_dose} Gy to 0.03cc violated. Dose in 0.03cc is {calculated_dose:.3f}"
                    compliance_stats[struct_name] = [
                        "Fail",
                        reason,
                        limit_dose - calculated_dose,
                    ]
                else:
                    reason = f"{struct_name} <= {limit_dose} Gy to 0.03cc achieved. Dose in 0.03cc is {calculated_dose:.3f}"
                    compliance_stats[struct_name] = [
                        "Pass",
                        reason,
                        limit_dose - calculated_dose,
                    ]
            else:
                reason = f"{struct_name} volume is smaller than 0.03cc."
                compliance_stats[struct_name] = ["NA", reason, 0]
            print(compliance_stats[struct_name])

        elif "Cochlea" in struct_name:
            # Cochlea: ≤45 Gy if both sides are involved; otherwise ≤60 Gy. (Low priority OaR) to 0.03cc
            if struct_mask.sum() > 4:
                sorted_dose = np.sort(dose_in_struct)[::-1]
                calculated_dose = sorted_dose[3]
                limit_dose = 45
                if calculated_dose >= limit_dose:
                    reason = f"{struct_name} <= {limit_dose} Gy to 0.03cc violated. Dose in 0.03cc is {calculated_dose:.3f}"
                    compliance_stats[struct_name] = [
                        "Fail",
                        reason,
                        limit_dose - calculated_dose,
                    ]
                else:
                    reason = f"{struct_name} <= {limit_dose} Gy to 0.03cc achieved. Dose in 0.03cc is {calculated_dose:.3f}"
                    compliance_stats[struct_name] = [
                        "Pass",
                        reason,
                        limit_dose - calculated_dose,
                    ]
            else:
                reason = f"{struct_name} volume is smaller than 0.03cc."
                compliance_stats[struct_name] = ["NA", reason, 0]
            print(compliance_stats[struct_name])

        elif "LacrimalGland" in struct_name:
            # Lacrimal glands: <40 Gy to 0.03cc
            if struct_mask.sum() > 4:
                sorted_dose = np.sort(dose_in_struct)[::-1]
                calculated_dose = sorted_dose[3]
                limit_dose = 40
                if calculated_dose >= limit_dose:
                    reason = f"{struct_name} <= {limit_dose} Gy to 0.03cc violated. Dose in 0.03cc is {calculated_dose:.3f}"
                    compliance_stats[struct_name] = [
                        "Fail",
                        reason,
                        limit_dose - calculated_dose,
                    ]
                else:
                    reason = f"{struct_name} <= {limit_dose} Gy to 0.03cc achieved. Dose in 0.03cc is {calculated_dose:.3f}"
                    compliance_stats[struct_name] = [
                        "Pass",
                        reason,
                        limit_dose - calculated_dose,
                    ]
            else:
                reason = f"{struct_name} volume is smaller than 0.03cc."
                compliance_stats[struct_name] = ["NA", reason, 0]
            print(compliance_stats[struct_name])

        elif "OpticNerve" in struct_name:
            # Optic Nerves ≤ 56 Gy to 0.03cc, Optic Nerves PRV: ≤56 Gy. to 0.03cc
            if struct_mask.sum() > 4:
                sorted_dose = np.sort(dose_in_struct)[::-1]
                calculated_dose = sorted_dose[3]
                limit_dose = 56
                if calculated_dose >= limit_dose:
                    reason = f"{struct_name} <= {limit_dose} Gy to 0.03cc violated. Dose in 0.03cc is {calculated_dose:.3f}"
                    compliance_stats[struct_name] = [
                        "Fail",
                        reason,
                        limit_dose - calculated_dose,
                    ]
                else:
                    reason = f"{struct_name} <= {limit_dose} Gy to 0.03cc achieved. Dose in 0.03cc is {calculated_dose:.3f}"
                    compliance_stats[struct_name] = [
                        "Pass",
                        reason,
                        limit_dose - calculated_dose,
                    ]
            else:
                reason = f"{struct_name} volume is smaller than 0.03cc."
                compliance_stats[struct_name] = ["NA", reason, 0]
            print(compliance_stats[struct_name])

        elif struct_name == "Brain":
            # The dose to the normal brain minus the PTV should be kept as low as possible. The Dmean is to be ≤ 30 Gy
            limit_dose = 30
            calculated_dose = dose_in_struct.mean()
            if calculated_dose >= limit_dose:
                reason = f"{struct_name} Dmean <= {limit_dose} Gy is violated. Dmean is {calculated_dose:.3f}"
                compliance_stats[struct_name] = [
                    "Fail",
                    reason,
                    limit_dose - calculated_dose,
                ]
            else:
                reason = f"{struct_name} Dmean <= {limit_dose} Gy is achieved. Dmean is {calculated_dose:.3f}"
                compliance_stats[struct_name] = [
                    "Pass",
                    reason,
                    limit_dose - calculated_dose,
                ]
            print(compliance_stats[struct_name])

        elif "Eye" in struct_name:
            # Eye balls, retina <= 40 G to Dmax
            limit_dose = 40
            calculated_dose = dose_in_struct.max()
            if calculated_dose >= limit_dose:
                reason = f"{struct_name} Dmax <= {limit_dose} Gy is violated. Dmean is {calculated_dose:.3f}"
                compliance_stats[struct_name] = [
                    "Fail",
                    reason,
                    limit_dose - calculated_dose,
                ]
            else:
                reason = f"{struct_name} Dmax <= {limit_dose} Gy is achieved. Dmean is {calculated_dose:.3f}"
                compliance_stats[struct_name] = [
                    "Pass",
                    reason,
                    limit_dose - calculated_dose,
                ]
            print(compliance_stats[struct_name])

        elif "Lens" in struct_name:
            # Lens: 10 Gy to 0.03cc
            if struct_mask.sum() > 4:
                sorted_dose = np.sort(dose_in_struct)[::-1]
                calculated_dose = sorted_dose[3]
                limit_dose = 10
                if calculated_dose >= limit_dose:
                    reason = f"{struct_name} <= {limit_dose} Gy to 0.03cc violated. Dose in 0.03cc is {calculated_dose:.3f}"
                    compliance_stats[struct_name] = [
                        "Fail",
                        reason,
                        limit_dose - calculated_dose,
                    ]
                else:
                    reason = f"{struct_name} <= {limit_dose} Gy to 0.03cc achieved. Dose in 0.03cc is {calculated_dose:.3f}"
                    compliance_stats[struct_name] = [
                        "Pass",
                        reason,
                        limit_dose - calculated_dose,
                    ]
            else:
                reason = f"{struct_name} volume is smaller than 0.03cc."
                compliance_stats[struct_name] = ["NA", reason, 0]
            print(compliance_stats[struct_name])

        elif "PTV" in struct_name:
            # PTV: encompassed by the 95% isodose line at 60Gy
            limit_dose = 60 * 0.95
            calculated_dose = dose_in_struct.min()
            if calculated_dose <= limit_dose:
                reason = f"{struct_name} Dmin >= {limit_dose} Gy is violated. Dmin is {calculated_dose:.3f}"
                compliance_stats[struct_name] = [
                    "Fail",
                    reason,
                    limit_dose - calculated_dose,
                ]
            else:
                reason = f"{struct_name} Dmin >= {limit_dose} Gy is achieved. Dmin is {calculated_dose:.3f}"
                compliance_stats[struct_name] = [
                    "Pass",
                    reason,
                    limit_dose - calculated_dose,
                ]
            print(compliance_stats[struct_name])

        else:
            compliance_stats[struct_name] = [
                "NA",
                f"{struct_name} either has no constraints; or is not defined for both versions.",
                0,
            ]

    compliance_df = pd.DataFrame.from_dict(compliance_stats, orient="index")
    compliance_df.columns = ["Status", "Reason", "Constraint-True"]
    return compliance_df
