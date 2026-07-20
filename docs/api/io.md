# Data and I/O API

## Unified loaders

::: dosemetrics.io.data_io
    options:
      show_source: true
      heading_level: 3
      members:
        - detect_folder_format
        - load_from_folder
        - load_structure_set
        - load_volume
        - load_structure

## NIfTI helpers

::: dosemetrics.io.nifti_io
    options:
      show_source: true
      heading_level: 3

## DICOM helpers

::: dosemetrics.io.dicom_io
    options:
      show_source: true
      heading_level: 3

## Typical NIfTI workflow

```python
from dosemetrics import Dose, StructureType
from dosemetrics.io import load_structure, load_structure_set

dose = Dose.from_nifti("patient/Dose.nii.gz")
structures = load_structure_set("patient")
ptv = load_structure(
    "patient/PTV.nii.gz",
    name="PTV",
    structure_type=StructureType.TARGET,
)
```

## Typical DICOM workflow

```python
from dosemetrics import Dose
from dosemetrics.io import load_structure_set

dose = Dose.from_dicom("patient/RTDOSE/plan.dcm")
structures = load_structure_set("patient", format="dicom")
```

## Batch over patient folders

```python
from pathlib import Path
from dosemetrics import Dose
from dosemetrics.io import load_structure_set
from dosemetrics.metrics import dvh

results = []
for patient_dir in Path("data").glob("patient_*"):
    dose = Dose.from_nifti(patient_dir / "Dose.nii.gz")
    structures = load_structure_set(patient_dir)
    for name, structure in structures:
        stats = dvh.compute_dose_statistics(dose, structure)
        results.append({"patient": patient_dir.name, "structure": name, **stats})
```

## Supported formats

| Data | NIfTI | DICOM |
|---|---:|---:|
| Dose | `.nii`, `.nii.gz` | RTDOSE `.dcm` |
| Structure | binary `.nii`, `.nii.gz` | RTSTRUCT `.dcm` with reference grid |
| Image | `.nii`, `.nii.gz` | CT series |

NRRD is not currently supported by the public API.
