# Dose Architecture

This document describes the design of the `Dose` class and its relationship to `Structure` and `StructureSet`, explaining the key decisions made in v0.3.0.

## The Problem with v0.2.0

In v0.2.0, dose data was stored inside each `Structure` object:

```python
# v0.2.0 — dose embedded in structure
structure.set_dose_data(dose_array, spacing, origin)
dvh = structure.compute_dvh()
```

This created three problems:

1. **Redundant storage**: Each structure held its own copy of the dose array, wasting memory in multi-structure workflows.
2. **Coupled comparison**: Comparing two dose distributions required replacing the dose inside each structure and recomputing, making it easy to introduce bugs.
3. **Poor separation of concerns**: `Structure` was responsible for both geometry and dosimetry — two conceptually distinct things.

## The v0.3.0 Solution

`Dose` is an independent first-class object. Structures are pure geometry; dosimetric computations pass a `Dose` explicitly:

```python
# v0.3.0 — dose is independent
dose = Dose.from_nifti("dose.nii.gz")
structure = Structure.from_nifti("brainstem.nii.gz")

dvh = structure.compute_dvh(dose)
```

### Dose Class

`dosemetrics.dose.Dose` holds:

| Attribute | Type | Description |
|-----------|------|-------------|
| `dose_array` | `np.ndarray` | 3D array of dose values in Gy |
| `spacing` | `tuple[float, float, float]` | Voxel spacing in mm (x, y, z) |
| `origin` | `tuple[float, float, float]` | World-space origin in mm |
| `name` | `str` | Optional label |

Key methods:

```python
# Loading
Dose.from_nifti(file_path)
Dose.from_dicom(file_path)

# Properties (simple, no heavy computation)
dose.max_dose      # float
dose.mean_dose     # float
dose.min_dose      # float
dose.shape         # tuple

# Geometry validation
dose.is_compatible_with_structure(structure)  # bool

# Dosimetric computation (requires a Structure)
dose.compute_dvh(structure, max_dose, step_size)
dose.compute_statistics(structure)
dose.compute_volume_at_dose(structure, dose_threshold)
dose.compute_dose_at_volume(structure, volume_percent)
dose.get_dose_in_structure(structure)
```

### Structure Class (Simplified)

`dosemetrics.structures.Structure` holds geometry only:

- `mask`: 3D boolean array
- `spacing`, `origin`: Geometric metadata
- `structure_type`: `StructureType.OAR | TARGET | AVOIDANCE`

Removed in v0.3.0:
- `dose_data` attribute
- `set_dose_data()` method
- `compute_dvh()` with no arguments

All dosimetric methods on `Structure` now require an explicit `Dose` argument.

## Multi-Dose Workflows

The decoupled design makes plan comparison straightforward:

```python
structure_set = StructureSet.from_folder("patient_data/")
dose_predicted = Dose.from_nifti("predicted.nii.gz")
dose_actual = Dose.from_nifti("actual.nii.gz")

ptv = structure_set["PTV"]
dvh_predicted = ptv.compute_dvh(dose_predicted)
dvh_actual    = ptv.compute_dvh(dose_actual)
```

No state mutation, no dose replacement, no risk of stale data.

## File Organization

```
src/dosemetrics/
├── dose.py            # Dose class (canonical location)
├── structures.py      # Structure, OAR, Target, AvoidanceStructure
├── structure_set.py   # StructureSet
└── io/
    ├── data_io.py     # High-level I/O helpers
    ├── dicom_io.py    # DICOM reader
    └── nifti_io.py    # NIfTI reader
```

See [Migration Guide](../user-guide/migration-v0.2-v0.3.md) for step-by-step upgrade instructions from v0.2.0.
