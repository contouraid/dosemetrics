import dosemetrics
import dosemetrics.metrics
import dosemetrics.utils


def test_all_declared_public_names_exist():
    missing = [name for name in dosemetrics.__all__ if not hasattr(dosemetrics, name)]
    assert missing == []


def test_nonstandard_quality_index_is_not_public():
    assert not hasattr(dosemetrics, "quality_index")
    assert not hasattr(dosemetrics.utils, "quality_index")


def test_named_plan_comparisons_are_exported_directly():
    expected = {
        "compare_body_rmse",
        "compare_dvh_score",
        "compare_gamma",
        "compare_homogeneity_index",
        "compare_mean_oar_dvh_auc",
        "compare_oar_constraints",
        "compare_oar_dvh_auc",
        "compare_paddick_conformity_index",
        "compare_paddick_gradient_index",
        "compare_ptv_dose",
    }
    assert expected <= set(dosemetrics.metrics.__all__)
    assert all(hasattr(dosemetrics.metrics, name) for name in expected)
