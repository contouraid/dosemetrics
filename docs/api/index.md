# API Reference

DoseMetrics 0.4.0 separates data representation, I/O, computations, and workflow utilities.

## Core objects

```python
from dosemetrics import (
    AvoidanceStructure,
    Dose,
    OAR,
    Structure,
    StructureSet,
    StructureType,
    Target,
)
```

- `Dose` holds a 3D dose array and its geometry.
- `Structure` is the abstract geometry base class; `OAR`, `Target`, and
  `AvoidanceStructure` are its concrete public subclasses.
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
from dosemetrics.metrics import compare_ptv_dose, conformity, dvh, homogeneity

stats = dvh.compute_dose_statistics(dose, ptv)
ci = conformity.compute_conformity_index(dose, ptv, prescription_dose=60.0)
hi = homogeneity.compute_homogeneity_index(dose, ptv)
distance = compare_ptv_dose(reference, evaluated, ptv)
```

The metric package exports domain modules plus direct named plan comparisons:

| Module | Purpose |
|---|---|
| `conformity` | Reference-free coverage and conformity indices |
| `dose_comparison` | General voxel/image comparisons and descriptors |
| `dvh` | DVHs, dose statistics, and DVH comparisons |
| `gamma` | 2D/3D gamma maps and summaries |
| `geometric` | Structure overlap, volume, and surface metrics |
| `homogeneity` | Reference-free homogeneity and gradient indices |

The named clinical `compare_*` functions are imported directly from
`dosemetrics.metrics`; each accepts `reference` before `evaluated`.

[Metrics reference →](metrics.md)

## Utilities

```python
from dosemetrics.utils import get_default_constraints, plot_dvh

fig, ax = plot_dvh(dose, ptv)
constraints = get_default_constraints()
```

[Utilities reference →](utils.md)

## Comparison convention

Every `compare_*` dose function accepts `reference` first and `evaluated`
second. Compatible array shape, spacing, and origin are required unless a
function documents otherwise. See the [Metric Framework](../user-guide/quality-metrics.md)
for reference-free versus reference-based classification.
