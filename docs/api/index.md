# API Reference

DoseMetrics 0.4.0 separates data representation, I/O, computations, and workflow utilities.

## Core objects

```python
from dosemetrics import Dose, OAR, Target, StructureSet, StructureType
```

- `Dose` holds a 3D dose array and its geometry.
- `OAR`, `Target`, and `AvoidanceStructure` hold binary geometry.
- `StructureSet` manages named structures but does not store dose.

[Data structure reference →](data.md)

## Loading data

```python
from dosemetrics import Dose
from dosemetrics.io import load_structure, load_structure_set, load_volume

dose = Dose.from_nifti("Dose.nii.gz")
structures = load_structure_set("patient_folder")
ptv = load_structure("PTV.nii.gz", name="PTV")
array, spacing, origin = load_volume("Dose.nii.gz")
```

[I/O reference →](io.md)

## Computing metrics

```python
from dosemetrics.metrics import dvh, conformity, homogeneity

stats = dvh.compute_dose_statistics(dose, ptv)
ci = conformity.compute_conformity_index(dose, ptv, prescription_dose=60.0)
hi = homogeneity.compute_homogeneity_index(dose, ptv)
```

The metric package exports domain modules, not a flat function namespace:

| Module | Purpose |
|---|---|
| `dvh` | DVHs, dose statistics, DVH comparisons |
| `conformity` | Coverage and conformity indices |
| `homogeneity` | Homogeneity and gradient indices |
| `geometric` | Structure overlap and surface distances |
| `gamma` | 2D/3D gamma analysis |
| `dose_comparison` | Voxel-wise image comparisons |
| `comparison` | Named plan-to-plan clinical distances |

[Metrics reference →](metrics.md)

## Utilities

```python
from dosemetrics.utils import batch, compliance, plot

fig, ax = plot.plot_dvh(dose, ptv)
constraints = compliance.get_default_constraints()
```

[Utilities reference →](utils.md)

## Comparison convention

Every `compare_*` dose function accepts `reference` first and `evaluated` second. Compatible array shape, spacing, and origin are required unless a function documents otherwise.
