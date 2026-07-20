# Supported File Formats

DoseMetrics 0.4.0 reads NIfTI and DICOM-RT data. NRRD is not currently supported by the public I/O API.

## NIfTI

Load a dose and one mask:

```python
from dosemetrics import Dose, StructureType
from dosemetrics.io import load_structure

dose = Dose.from_nifti("Dose.nii.gz", name="Clinical")
ptv = load_structure(
    "PTV.nii.gz",
    name="PTV",
    structure_type=StructureType.TARGET,
)
```

Load a folder containing `Dose.nii.gz` and binary masks:

```python
from dosemetrics import Dose
from dosemetrics.io import load_structure_set

patient_dir = "patient_001"
dose = Dose.from_nifti(f"{patient_dir}/Dose.nii.gz")
structures = load_structure_set(patient_dir)
```

The dose is loaded separately from the structure set. `load_structure_set()` auto-classifies names containing `PTV`, `CTV`, `GTV`, or `TARGET` as targets; pass `structure_type_mapping` when names need explicit types.

For raw arrays and metadata:

```python
from dosemetrics.io import load_volume

array, spacing, origin = load_volume("Dose.nii.gz")
```

## DICOM-RT

Load an RTDOSE file:

```python
from dosemetrics import Dose

dose = Dose.from_dicom("RTDOSE.dcm", name="Clinical")
```

Load structures from a folder containing an RTSTRUCT and its referenced image data:

```python
from dosemetrics.io import load_structure_set

structures = load_structure_set("path/to/dicom_folder", format="dicom")
```

The format-specific module exposes lower-level readers:

```python
from dosemetrics.io import dicom_io

dose_array, spacing, origin, scaling = dicom_io.read_dicom_rtdose("RTDOSE.dcm")
roi_data = dicom_io.read_dicom_rtstruct(
    "RTSTRUCT.dcm",
    reference_image=(dose_array.shape, spacing, origin),
)
ptv_mask = roi_data["PTV"]["mask"]
```

RTSTRUCT rasterization needs a compatible reference grid. When loading a complete folder, the DICOM loader derives that grid from the available CT series or dose.

## Writing NIfTI

```python
from dosemetrics.io import nifti_io

nifti_io.write_nifti_volume(
    dose.dose_array,
    "Dose-copy.nii.gz",
    spacing=dose.spacing,
    origin=dose.origin,
)
nifti_io.write_structure_as_nifti(ptv, "PTV-copy.nii.gz")
nifti_io.write_structure_set_as_nifti(structures, "exported-structures")
```

## Spatial requirements

Before computing a metric, dose and structure data must have matching:

- array shape;
- voxel spacing;
- world-space origin.

```python
if not dose.is_compatible_with_structure(ptv):
    raise ValueError("Dose and PTV grids are not compatible")
```

DoseMetrics preserves spacing and origin but does not currently expose a public resampling function. Align grids before analysis when required.

## Format detection

```python
from dosemetrics.io import detect_folder_format

format_name = detect_folder_format("patient_001")  # "nifti" or "dicom"
```
