# Architecture Overview

DoseMetrics is organized around a clean separation of three concerns: **data representation**, **metrics computation**, and **utilities**. This separation makes the library easy to extend, test, and reason about.

## Package Structure

```
dosemetrics/
├── dose.py              # Dose — 3D dose distribution data class
├── structures.py        # Structure, OAR, Target, AvoidanceStructure
├── structure_set.py     # StructureSet — collection of structures + dose
├── io/                  # File loading and saving (DICOM, NIfTI)
├── metrics/             # Computational algorithms
│   ├── dvh.py           # Dose-volume histogram functions
│   ├── statistics.py    # Dose statistics (mean, max, D95, etc.)
│   ├── conformity.py    # Conformity and coverage indices
│   ├── homogeneity.py   # Homogeneity indices
│   ├── geometric.py     # Spatial metrics (volume, centroid)
│   └── gamma.py         # Gamma index for QA
└── utils/               # Visualization, batch processing, compliance
```

## Design Principles

### 1. Data classes are pure containers

`Dose`, `Structure`, and `StructureSet` store data and expose simple properties. Heavy computation lives in the `metrics` subpackage, not in the data classes. This keeps data objects lightweight and makes algorithms independently testable.

### 2. Dose and structures are decoupled

A `Dose` object is independent of any `Structure`. This design enables a key clinical workflow: comparing multiple dose distributions (e.g., predicted vs. delivered) against the same set of structures without reloading anything.

```python
dose_planned = Dose.from_nifti("planned.nii.gz")
dose_delivered = Dose.from_nifti("delivered.nii.gz")

dvh_planned = structure.compute_dvh(dose_planned)
dvh_delivered = structure.compute_dvh(dose_delivered)
```

### 3. StructureSet for cohort workflows

`StructureSet` is the entry point for multi-structure workflows. It holds a collection of structures and a reference dose, mirrors DICOM RTSS conventions, and exposes bulk operations that eliminate manual iteration.

```python
structure_set = StructureSet.from_folder("patient_data/")
stats = structure_set.compute_all_dvhs()
```

### 4. I/O is a thin layer

The `io` subpackage translates file formats (DICOM, NIfTI) into the core data classes. It does not contain domain logic. The three-layer I/O API provides:

- **High-level loaders**: `StructureSet.from_folder()`, `Dose.from_dicom()`
- **Mid-level helpers**: `get_dose_and_structures_as_structure_set()`
- **Low-level readers**: `read_dose_nifti()`, `read_structure_dicom()`

## Key Design Decisions

For details on specific architectural decisions, see:

- [Dose Architecture](dose-architecture.md) — how the `Dose` class is structured and why dose was separated from structures
- [Metrics Architecture](metrics-architecture.md) — how the metrics subpackage is organized and the separation of data from computation
