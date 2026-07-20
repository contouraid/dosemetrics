# Architecture Overview

DoseMetrics 0.4.0 separates data representation, file I/O, metric computation, and workflow utilities.

## Package structure

```text
dosemetrics/
├── dose.py                 # Dose data container
├── structures.py           # Structure, OAR, Target, AvoidanceStructure
├── structure_set.py        # Named collection of structures
├── io/
│   ├── data_io.py          # Format-neutral loaders
│   ├── nifti_io.py         # NIfTI readers/writers
│   └── dicom_io.py         # DICOM-RT readers
├── metrics/
│   ├── dvh.py              # DVHs, dose statistics, DVH comparisons
│   ├── conformity.py       # Coverage and conformity
│   ├── homogeneity.py      # Homogeneity and gradients
│   ├── geometric.py        # Structure comparisons
│   ├── gamma.py            # Gamma analysis
│   ├── dose_comparison.py  # Voxel-wise dose comparisons
│   └── comparison.py       # Clinical plan-to-plan distances
└── utils/                  # Plotting, compliance, batch, cohort analysis
```

## Data containers stay lightweight

`Dose`, `Structure`, and `StructureSet` store arrays and geometry. Metric algorithms live in `dosemetrics.metrics`:

```python
from dosemetrics import Dose
from dosemetrics.io import load_structure_set
from dosemetrics.metrics import dvh

dose = Dose.from_nifti("patient/Dose.nii.gz")
structures = load_structure_set("patient")
stats = dvh.compute_dose_statistics(dose, structures["PTV"])
```

## Dose and structures are independent

A `StructureSet` never owns a dose. This permits multiple plans to share one geometry without state mutation:

```python
reference = Dose.from_nifti("reference.nii.gz")
evaluated = Dose.from_nifti("evaluated.nii.gz")
ptv = structures["PTV"]

reference_dvh = dvh.compute_dvh(reference, ptv)
evaluated_dvh = dvh.compute_dvh(evaluated, ptv)
```

## Metric naming communicates semantics

- `compute_*` characterizes one plan, one structure, or a collection.
- `compare_*` accepts `reference` before `evaluated`.
- Metrics are accessed through domain modules so similarly named concepts remain distinct.

See [Dose architecture](dose-architecture.md) and [Metrics architecture](metrics-architecture.md).
