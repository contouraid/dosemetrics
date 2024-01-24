import numpy as np
from numpy import ndarray

"""
These are modified from https://github.com/ababier/open-kbp
"""


def dose_score(_pred: np.ndarray, _gt: np.ndarray, _dose_mask=None) -> np.ndarray:
    if _dose_mask is not None:
        _pred = _pred[_dose_mask > 0]
        _gt = _gt[_dose_mask > 0]

    return np.mean(np.abs(_pred - _gt))


def dvh_score(
    _dose: np.ndarray, _mask: np.ndarray, mode: str, spacing=None
) -> dict[str, np.ndarray]:
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
