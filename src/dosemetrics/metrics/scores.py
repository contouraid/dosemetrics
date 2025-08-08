import numpy as np
import pandas as pd
import SimpleITK as sitk
import pymia.evaluation.metric as metric
import pymia.evaluation.evaluator as eval_
import pymia.evaluation.writer as writer


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


def compute_geometric_scores(a_mask_files, b_mask_files):
    """
    COMPUTE_GEOMETRIC_SCORES:
    Compute geometric scores between two sets of structure masks.
    :param a_mask_files: List of file paths for the first set of structure masks.
    :param b_mask_files: List of file paths for the second set of structure masks.
    :return: A DataFrame containing geometric scores for each structure.
    """
    a_masks = {}
    for a_file in a_mask_files:
        struct_name = a_file.split("/")[-1].split(".")[0]
        a_masks[struct_name] = a_file
    b_masks = {}
    for b_file in b_mask_files:
        struct_name = b_file.split("/")[-1].split(".")[0]
        b_masks[struct_name] = b_file

    metric_list = [
        "DSC",
        "HausdorffDistance95 (mm)",
        "HausdorffDistance100 (mm)",
        "VolumeSimilarity",
        "SurfaceDice",
        "FalseNegative (cc)",
        "SizeChange",
    ]
    metrics = [
        metric.DiceCoefficient(),
        metric.HausdorffDistance(percentile=95, metric="HDRFDST95"),
        metric.HausdorffDistance(percentile=100, metric="HDRFDST"),
        metric.VolumeSimilarity(),
        metric.SurfaceDiceOverlap(),
        metric.FalseNegative(),
    ]
    labels = {1: "FG"}

    stats = {}
    for struct_name in b_masks:
        if struct_name in a_masks:
            first_mask = sitk.ReadImage(a_masks[struct_name])
            first_mask.SetOrigin((0, 0, 0))
            last_mask = sitk.ReadImage(b_masks[struct_name])
            last_mask.SetOrigin((0, 0, 0))

            last_array = sitk.GetArrayFromImage(last_mask)
            first_array = sitk.GetArrayFromImage(first_mask)
            if last_array.sum() > first_array.sum():
                size_change = "larger"
            elif last_array.sum() == first_array.sum():
                size_change = "same"
            else:
                size_change = "smaller"

            evaluator = eval_.SegmentationEvaluator(metrics, labels)
            evaluator.evaluate(first_mask, last_mask, struct_name)
            writer.ConsoleWriter().write(evaluator.results)
            stats[struct_name] = [
                f"{evaluator.results[0].value:.3f}",
                f"{evaluator.results[1].value:.3f}",
                f"{evaluator.results[2].value:.3f}",
                f"{evaluator.results[3].value:.3f}",
                f"{evaluator.results[4].value:.3f}",
                f"{evaluator.results[5].value * 0.008:.3f}",
                size_change,
            ]

    geom_df = pd.DataFrame.from_dict(stats, orient="index")
    geom_df.columns = metric_list
    return geom_df
