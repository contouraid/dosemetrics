"""Regression tests for the explicitly defined plan-comparison metrics."""

import inspect

import numpy as np
import pytest

from dosemetrics.dose import Dose
from dosemetrics.metrics import (
    compare_body_rmse,
    compare_dvh_score,
    compare_gamma,
    compare_homogeneity_index,
    compare_oar_constraints,
    compare_oar_dvh_auc,
    compare_paddick_conformity_index,
    compare_paddick_gradient_index,
    compare_ptv_dose,
    comparison,
)
from dosemetrics.structures import OAR, Target


SPACING = (10.0, 10.0, 10.0)
ORIGIN = (0.0, 0.0, 0.0)


def make_dose(values):
    return Dose(np.asarray(values, dtype=float), SPACING, ORIGIN)


def make_target(mask, name="PTV"):
    return Target(name, np.asarray(mask, dtype=bool), SPACING, ORIGIN)


def make_oar(mask, name="OAR"):
    return OAR(name, np.asarray(mask, dtype=bool), SPACING, ORIGIN)


def test_catalog_contains_all_nine_metrics_and_five_categories():
    catalog = comparison.COMPARISON_METRICS

    assert len(catalog) == 9
    assert {metric.category for metric in catalog} == set(comparison.MetricCategory)

    prediction_count = sum(
        comparison.EvaluationTask.DOSE_PREDICTION in metric.tasks for metric in catalog
    )
    calculation_count = sum(
        comparison.EvaluationTask.DOSE_CALCULATION in metric.tasks for metric in catalog
    )
    assert prediction_count == 6
    assert calculation_count == 8


def test_catalog_exposes_a_consistent_reference_evaluated_api():
    for metric in comparison.COMPARISON_METRICS:
        assert metric.function.startswith("compare_")
        function = getattr(comparison, metric.function)
        parameters = tuple(inspect.signature(function).parameters)
        assert parameters[0].startswith("reference")
        assert parameters[1].startswith("evaluated")


def test_ptv_dose_distance_is_absolute_mean_difference():
    mask = np.ones((2, 2, 2), dtype=bool)
    ptv = make_target(mask)
    reference = make_dose(np.full((2, 2, 2), 60.0))
    evaluated = make_dose(np.full((2, 2, 2), 62.5))

    result = compare_ptv_dose(reference, evaluated, ptv)

    assert result == pytest.approx(2.5)


def test_paddick_conformity_index_distance_uses_index_difference():
    mask = np.zeros((3, 3, 3), dtype=bool)
    mask[1, 1, 1] = True
    ptv = make_target(mask)

    reference_array = np.zeros((3, 3, 3))
    reference_array[1, 1, 1] = 60.0
    evaluated_array = reference_array.copy()
    evaluated_array[1, 1, 2] = 60.0

    result = compare_paddick_conformity_index(
        make_dose(reference_array),
        make_dose(evaluated_array),
        ptv,
        prescription_dose=60.0,
    )

    # CI_reference = 1 and CI_evaluated = 1/2.
    assert result == pytest.approx(0.5)


def test_paddick_gradient_index_distance_uses_half_to_full_volume_ratio():
    reference = np.zeros((3, 3, 3))
    evaluated = np.zeros((3, 3, 3))
    reference.flat[:2] = 60.0
    reference.flat[2:6] = 30.0  # GI = 6 / 2 = 3
    evaluated.flat[:2] = 60.0
    evaluated.flat[2:8] = 30.0  # GI = 8 / 2 = 4

    result = compare_paddick_gradient_index(
        make_dose(reference), make_dose(evaluated), prescription_dose=60.0
    )

    assert result == pytest.approx(1.0)


def test_homogeneity_index_distance_uses_d2_d98_over_d50():
    mask = np.ones((5, 5, 4), dtype=bool)
    ptv = make_target(mask)
    reference_values = np.linspace(50.0, 70.0, 100).reshape(5, 5, 4)
    evaluated_values = np.linspace(55.0, 65.0, 100).reshape(5, 5, 4)

    reference_hi = (
        np.percentile(reference_values, 98) - np.percentile(reference_values, 2)
    ) / np.percentile(reference_values, 50)
    evaluated_hi = (
        np.percentile(evaluated_values, 98) - np.percentile(evaluated_values, 2)
    ) / np.percentile(evaluated_values, 50)

    result = compare_homogeneity_index(
        make_dose(reference_values), make_dose(evaluated_values), ptv
    )

    assert result == pytest.approx(abs(evaluated_hi - reference_hi))


def test_body_rmse_ignores_voxels_outside_body():
    body_mask = np.zeros((2, 2, 2), dtype=bool)
    body_mask.flat[:2] = True
    body = make_oar(body_mask, name="Body")
    reference = np.zeros((2, 2, 2))
    evaluated = np.full((2, 2, 2), 100.0)
    evaluated.flat[:2] = (3.0, 4.0)

    result = compare_body_rmse(
        make_dose(reference), make_dose(evaluated), body
    )

    assert result == pytest.approx(np.sqrt((3.0**2 + 4.0**2) / 2.0))


def test_gamma_index_passing_rate_defaults_to_three_percent_three_mm():
    body_mask = np.ones((3, 3, 3), dtype=bool)
    body = make_oar(body_mask, name="Body")
    reference = make_dose(np.full((3, 3, 3), 60.0))
    evaluated = make_dose(np.full((3, 3, 3), 60.0))

    result = compare_gamma(reference, evaluated, body=body)

    assert result == pytest.approx(100.0)


def test_full_dvh_score_combines_target_and_oar_criteria():
    mask = np.ones((2, 2, 2), dtype=bool)
    ptv = make_target(mask)
    oar = make_oar(mask)
    reference = make_dose(np.full((2, 2, 2), 10.0))
    evaluated = make_dose(np.full((2, 2, 2), 12.0))

    result = compare_dvh_score(
        reference, evaluated, targets=[ptv], oars=[oar]
    )

    # Three target errors and two OAR errors, all equal to 2 Gy.
    assert result == pytest.approx(2.0)


def test_constraint_disagreement_supports_sequences_and_mappings():
    reference = [True, True, False, False]
    evaluated = [True, False, True, False]

    assert compare_oar_constraints(
        reference, evaluated, expected_count=None
    ) == pytest.approx(0.5)
    assert compare_oar_constraints(
        dict(zip("abcd", reference)),
        dict(zip("abcd", evaluated)),
        expected_count=None,
    ) == pytest.approx(0.5)

    with pytest.raises(ValueError, match="Expected 38"):
        compare_oar_constraints(reference, evaluated)


def test_oar_dvh_area_is_absolute_difference_of_auc_not_l1_curve_area():
    mask = np.ones((2, 2, 2), dtype=bool)
    oar = make_oar(mask)
    reference_values = np.asarray([0.0] * 4 + [10.0] * 4).reshape(2, 2, 2)
    evaluated_values = np.full((2, 2, 2), 5.0)
    reference = make_dose(reference_values)
    evaluated = make_dose(evaluated_values)

    result = compare_oar_dvh_auc(reference, evaluated, oar, num_bins=100)

    bins = np.linspace(0.0, 10.0, 100)
    reference_dvh = np.asarray([np.mean(reference_values >= dose) for dose in bins])
    evaluated_dvh = np.asarray([np.mean(evaluated_values >= dose) for dose in bins])
    expected = abs(np.trapz(evaluated_dvh, bins) - np.trapz(reference_dvh, bins))
    pointwise_l1 = np.trapz(abs(evaluated_dvh - reference_dvh), bins)

    assert result == pytest.approx(expected)
    assert result != pytest.approx(pointwise_l1)


def test_geometry_mismatch_is_rejected():
    mask = np.ones((2, 2, 2), dtype=bool)
    ptv = make_target(mask)
    reference = make_dose(np.zeros((2, 2, 2)))
    evaluated = Dose(np.zeros((2, 2, 2)), (2.0, 2.0, 2.0), ORIGIN)

    with pytest.raises(ValueError, match="spacings"):
        compare_ptv_dose(reference, evaluated, ptv)
