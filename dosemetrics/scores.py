import numpy as np
import pandas as pd


def dose_score(_pred: np.ndarray, _gt: np.ndarray, _dose_mask=None) -> np.ndarray:
    """
    DOSE_SCORE: These are modified from https://github.com/ababier/open-kbp
    :param _pred:
    :param _gt:
    :param _dose_mask:
    :return: scalar with mean average error of dose between _pred and _gt.
    """
    if _dose_mask is not None:
        _pred = _pred[_dose_mask > 0]
        _gt = _gt[_dose_mask > 0]

    return np.mean(np.abs(_pred - _gt))


def dvh_score(
    _dose: np.ndarray, _mask: np.ndarray, mode: str, spacing=None
) -> dict[str, np.ndarray]:
    """
    DVH_SCORE: These are modified from https://github.com/ababier/open-kbp
    :param _dose:
    :param _mask:
    :param mode:
    :param spacing:
    :return: dict with DVH scores.
    """
    output = {}

    if mode.lower() == "target":
        _roi_dose = _dose[_mask > 0]
        # D1
        output["D1"] = np.percentile(_roi_dose, 99)
        # D95
        output["D95"] = np.percentile(_roi_dose, 5)
        # D99
        output["D99"] = np.percentile(_roi_dose, 1)

    elif mode.upper() == "OAR":
        if spacing is None:
            raise Exception("dvh score computation requires voxel spacing information.")

        _roi_dose = _dose[_mask > 0]
        _roi_size = len(_roi_dose)
        _voxel_size = np.prod(spacing)

        # D_0.1_cc
        voxels_in_tenth_of_cc = np.maximum(1, np.round(100 / _voxel_size))
        fractional_volume_to_evaluate = 100 - voxels_in_tenth_of_cc / _roi_size * 100
        if fractional_volume_to_evaluate <= 0:
            output["D_0.1_cc"] = np.asarray(0.0)
        else:
            output["D_0.1_cc"] = np.percentile(_roi_dose, fractional_volume_to_evaluate)

        # Dmean
        output["mean"] = np.mean(_roi_dose)
    else:
        raise Exception("Unknown mode!")

    return output


def dose_summary(dose_volume, structure_masks):
    """
    DOSE_SUMMARY: summarize dose metrics for each structure.
    :param dose_volume:
    :param structure_masks:
    :return: pandas.DataFrame with dose summary.
    """
    dose_metrics = {}
    for structure in structure_masks.keys():
        dose_in_structure = dose_volume[structure_masks[structure] > 0]
        dose_metrics[structure] = {
            "Mean Dose": f"{np.mean(dose_in_structure):.3f}",
            "Max Dose": f"{np.max(dose_in_structure):.3f}",
            "Min Dose": f"{np.min(dose_in_structure):.3f}",
            "D95": f"{np.percentile(dose_in_structure, 95):.3f}",
            "D50": f"{np.percentile(dose_in_structure, 50):.3f}",
            "D5": f"{np.percentile(dose_in_structure, 5):.3f}",
        }

    df = pd.DataFrame.from_dict(dose_metrics).T
    return df
