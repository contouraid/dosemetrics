# Migrating to DoseMetrics 0.4

This page replaces the former v0.2→v0.3 guide. Version 0.4 completes the separation between data containers and metric functions, so intermediate v0.3 examples that call metrics on `Dose`, `Structure`, or `StructureSet` are also obsolete.

## Load data with the current API

```python
from dosemetrics import Dose, StructureType
from dosemetrics.io import load_structure, load_structure_set

dose = Dose.from_nifti("Dose.nii.gz")
structures = load_structure_set("patient_folder")
ptv = load_structure(
    "PTV.nii.gz",
    name="PTV",
    structure_type=StructureType.TARGET,
)
```

Replace removed loaders such as `read_dose_and_mask_files()`, `get_dose_and_structures_as_structure_set()`, and `Structure.from_nifti()` with `Dose.from_nifti()`, `load_structure()`, and `load_structure_set()`.

## Call metrics through domain modules

```python
from dosemetrics.metrics import conformity, dvh, homogeneity

dose_bins, volume_percent = dvh.compute_dvh(dose, ptv)
stats = dvh.compute_dose_statistics(dose, ptv)
ci = conformity.compute_conformity_index(dose, ptv, prescription_dose=60.0)
hi = homogeneity.compute_homogeneity_index(dose, ptv)
```

Do not call removed methods such as `structure.compute_dvh()`, `dose.compute_statistics()`, or `structure_set.compute_all_dvhs()`.

## Compute in bulk explicitly

```python
import pandas as pd
from dosemetrics.metrics import dvh

dvh_table = dvh.create_dvh_table(dose, structures)
statistics_table = pd.DataFrame([
    {"Structure": name, **dvh.compute_dose_statistics(dose, structure)}
    for name, structure in structures
])
```

## Compare plans with a fixed direction

```python
from dosemetrics.metrics import comparison, dose_comparison

mae = dose_comparison.compare_mae(reference, evaluated)
ptv_distance = comparison.compare_ptv_dose(reference, evaluated, ptv)
```

All `compare_*` functions now put `reference` before `evaluated`.

## Plotting moved to `dosemetrics.utils.plot`

```python
from dosemetrics.utils.plot import plot_dvh, plot_subject_dvhs

fig, ax = plot_dvh(dose, ptv)
fig, ax = plot_subject_dvhs(dose, structures)
```

The old `dosemetrics.utils.plotting` module and root-level plotting functions are not part of the 0.4 public API.

## Root namespace

The package root contains core classes and high-level I/O conveniences. Metric and utility functions should be imported from their domain modules. Avoid relying on `from dosemetrics import *`; explicit imports make API ownership clear.
