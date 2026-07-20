# Dose Architecture

## Data model

`Dose` is an independent 3D dose-distribution container. Structures contain geometry only, and `StructureSet` contains named structures only.

| `Dose` attribute | Meaning |
|---|---|
| `dose_array` | 3D NumPy array, conventionally in Gy |
| `spacing` | `(x, y, z)` voxel spacing in mm |
| `origin` | `(x, y, z)` world origin in mm |
| `name` | Human-readable plan identifier |
| `metadata` | Optional format-specific metadata |

## Constructing and loading dose

```python
import numpy as np
from dosemetrics import Dose

dose = Dose(
    np.zeros((64, 64, 32), dtype=float),
    spacing=(1.0, 1.0, 2.5),
    origin=(0.0, 0.0, 0.0),
    name="Synthetic",
)

nifti_dose = Dose.from_nifti("Dose.nii.gz")
dicom_dose = Dose.from_dicom("RTDOSE.dcm")
```

`Dose` exposes lightweight properties (`shape`, `min_dose`, `mean_dose`, `max_dose`) and two geometry-aware helpers:

```python
is_aligned = dose.is_compatible_with_structure(structure)
dose_values = dose.get_dose_in_structure(structure)
```

## Computing metrics

Computation does not live on `Dose` or `Structure`:

```python
from dosemetrics.metrics import dvh

dose_bins, volume_percent = dvh.compute_dvh(dose, structure)
stats = dvh.compute_dose_statistics(dose, structure)
d95 = dvh.compute_dose_at_volume(dose, structure, volume_percent=95)
v20 = dvh.compute_volume_at_dose(dose, structure, dose_threshold=20.0)
```

## Multi-plan workflow

```python
from dosemetrics.metrics import dose_comparison

reference = Dose.from_nifti("reference.nii.gz")
evaluated = Dose.from_nifti("evaluated.nii.gz")

mae = dose_comparison.compare_mae(reference, evaluated, structure)
```

This design avoids copying dose arrays into every structure and makes the reference/evaluated direction explicit.
