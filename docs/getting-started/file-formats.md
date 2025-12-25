# Supported File Formats

DoseMetrics supports multiple file formats commonly used in radiotherapy and medical imaging.

## Dose Distributions

### NIfTI (.nii, .nii.gz)

**Recommended format** for dose distributions.

```python
from dosemetrics.io import read_from_nifti

dose = read_from_nifti("dose.nii.gz")
```

**Advantages:**

- Compact with gzip compression
- Includes spatial metadata (spacing, origin, orientation)
- Wide software support
- Fast I/O

**Requirements:**

- 3D volume with dose values (typically in Gy or cGy)
- Header must contain correct voxel spacing and orientation

### DICOM-RT Dose (.dcm)

Native format from treatment planning systems.

```python
from dosemetrics.io import read_from_nifti  # Also handles DICOM

dose = read_from_nifti("RTDose.dcm")
```

**Advantages:**

- Direct export from TPS
- Contains complete metadata
- Clinical standard

**Notes:**

- DoseMetrics uses `pydicom` for DICOM reading
- Dose scaling factor is automatically applied
- Coordinate system transformations are handled

### NRRD (.nrrd)

Alternative medical imaging format.

```python
from dosemetrics.io import read_from_nifti  # Also handles NRRD

dose = read_from_nifti("dose.nrrd")
```

**Advantages:**

- Human-readable header
- Supports various data types
- Good for research workflows

## Structure Masks

### Binary NIfTI (.nii, .nii.gz)

**Recommended format** for structure masks.

```python
from dosemetrics.io import read_from_nifti

mask = read_from_nifti("ptv.nii.gz")
```

**Requirements:**

- Binary mask (0 = outside structure, 1 or 255 = inside structure)
- Must align spatially with dose grid
- Same coordinate system as dose

### DICOM-RT Structure Set (.dcm)

Contains contours from TPS.

```python
from dosemetrics import StructureSet

structures = StructureSet.from_dicom(
    "RTStruct.dcm",
    reference_image="RTDose.dcm"
)

# Extract specific structure
ptv_mask = structures.get_structure_mask("PTV")
```

**Notes:**

- Contours are converted to binary masks
- Requires reference image for grid information
- Structure names must match exactly

### NRRD Segmentation (.seg.nrrd)

Multi-label segmentation format.

```python
from dosemetrics.io import load_mask

# Load specific label
mask = load_mask("structures.seg.nrrd", label=1)
```

## Directory Structure Examples

### Example 1: Single Patient, NIfTI Format

```
patient_001/
├── dose.nii.gz
└── structures/
    ├── PTV.nii.gz
    ├── Brain.nii.gz
    ├── Brainstem.nii.gz
    └── Optic_Chiasm.nii.gz
```

```python
from dosemetrics import read_dose_and_mask_files
from pathlib import Path

patient_dir = Path("patient_001")
dose, structures = read_dose_and_mask_files(patient_dir)

for structure_name in structures.get_structure_names():
    mask = structures.get_structure_mask(structure_name)
    print(f"{structure_name}: {mask.sum()} voxels")
```

### Example 2: Multiple Plans for Comparison

```
patient_001/
├── plan_tps/
│   ├── dose.nii.gz
│   └── structures/
│       └── ...
└── plan_predicted/
    ├── dose.nii.gz
    └── structures/
        └── ...
```

```python
from dosemetrics.io import read_from_nifti

dose_tps = read_from_nifti("patient_001/plan_tps/dose.nii.gz")
dose_pred = read_from_nifti("patient_001/plan_predicted/dose.nii.gz")
```

### Example 3: DICOM Workflow

```
patient_001/
├── RTDose.dcm
├── RTStruct.dcm
└── CT_series/
    ├── CT.0001.dcm
    ├── CT.0002.dcm
    └── ...
```

```python
from dosemetrics.io import read_from_nifti
from dosemetrics import StructureSet

# Load dose
dose = read_from_nifti("patient_001/RTDose.dcm")

# Load structures
structures = StructureSet.from_dicom(
    "patient_001/RTStruct.dcm",
    reference_image="patient_001/RTDose.dcm"
)

# Get individual structure masks
ptv_mask = structures.get_structure_mask("PTV")
brainstem_mask = structures.get_structure_mask("Brainstem")
```

### Example 4: Batch Processing

```
data/
├── patient_001/
│   ├── dose.nii.gz
│   └── structures/
├── patient_002/
│   ├── dose.nii.gz
│   └── structures/
└── patient_003/
    ├── dose.nii.gz
    └── structures/
```

```python
from dosemetrics import read_dose_and_mask_files
from pathlib import Path
import pandas as pd

results = []
for patient_dir in Path("data").glob("patient_*"):
    dose, structures = read_dose_and_mask_files(patient_dir)
    
    for structure_name in structures.get_structure_names():
        mask = structures.get_structure_mask(structure_name)
        # Process and store results
        ...
```

## Data Requirements

### Spatial Alignment

**Critical:** Dose and structure masks must be spatially aligned:

- Same image spacing (voxel size)
- Same image dimensions
- Same coordinate system origin
- Same orientation

DoseMetrics will warn if alignment issues are detected.

### Dose Units

Supported dose units:

- **Gy** (Gray) - preferred
- **cGy** (centigray)

Specify units when loading if not in metadata:

```python
dose = load_dose("dose.nii.gz", dose_unit="Gy")
```

### Mask Values

Binary masks should use:

- **0** for voxels outside the structure
- **1** or **255** for voxels inside the structure

Multi-label masks:

- Each structure has a unique integer label
- 0 is reserved for background

## Format Conversion

### DICOM to NIfTI

Convert DICOM RT Dose to NIfTI:

```python
from dosemetrics.io import read_from_nifti
import nibabel as nib
import numpy as np

# Load DICOM
dose = read_from_nifti("RTDose.dcm")

# Save as NIfTI
nii = nib.Nifti1Image(dose, affine=np.eye(4))
nib.save(nii, "dose.nii.gz")
```

### Extract Structures from RT Structure Set

Convert DICOM RT Struct to individual masks:

```python
from dosemetrics import StructureSet
import nibabel as nib
from pathlib import Path

# Load structure set
structures = StructureSet.from_dicom("RTStruct.dcm", reference_image="RTDose.dcm")

# Save each structure as separate file
output_dir = Path("structures")
output_dir.mkdir(exist_ok=True)

for structure_name in structures.get_structure_names():
    mask = structures.get_structure_mask(structure_name)
    nii = nib.Nifti1Image(mask.astype(np.uint8), affine=np.eye(4))
    nib.save(nii, output_dir / f"{structure_name}.nii.gz")
```

## Best Practices

1. **Use NIfTI for storage**: Convert DICOM data to NIfTI for faster processing and smaller file sizes

2. **Organize by patient**: Keep all files for one patient in a single directory

3. **Consistent naming**: Use clear, consistent names for structures (e.g., "PTV", "Brainstem", not "ptv", "Brain_Stem")

4. **Compression**: Always use `.nii.gz` (compressed) instead of `.nii` to save space

5. **Metadata**: Preserve spatial metadata when converting between formats

## Need Help with Your Data?

If you have data in a different format or need help preparing your files:

1. Check our [interactive notebook](../notebooks/getting-started-own-data.ipynb) for detailed examples
2. Try the [live demo](https://huggingface.co/spaces/contouraid/dosemetrics) to test with sample data
3. [Open an issue](https://github.com/contouraid/dosemetrics/issues) on GitHub

[:material-rocket-launch: Try Interactive Demo](https://huggingface.co/spaces/contouraid/dosemetrics){ .md-button .md-button--primary target="_blank" }
