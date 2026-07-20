import numpy as np

import dosemetrics
from dosemetrics import Dose, OAR
from dosemetrics.utils.compliance import quality_index


def test_all_declared_public_names_exist():
    missing = [name for name in dosemetrics.__all__ if not hasattr(dosemetrics, name)]
    assert missing == []


def test_quality_index_uses_current_dvh_module():
    dose = Dose(
        np.full((3, 3, 3), 2.0),
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
    )
    structure = OAR(
        "OAR",
        np.ones((3, 3, 3), dtype=bool),
        spacing=dose.spacing,
        origin=dose.origin,
    )

    assert quality_index(dose, structure, "mean", 10.0) == 0.8
    assert quality_index(dose, structure, "max", 10.0) == 0.8
    assert quality_index(dose, structure, "min", 1.0) == 1.0
    assert quality_index(dose, structure, "mean", 1.0) == -1.0
