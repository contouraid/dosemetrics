# Data & I/O API

This module provides data structures and I/O functions for reading and writing dose distributions, structure masks, and other radiotherapy data.

## Data Structures

::: dosemetrics.io.structures
    options:
      show_source: true
      heading_level: 3
      members:
        - Structure
        - OAR
        - Target
        - AvoidanceStructure
        - StructureType

::: dosemetrics.io.structure_set
    options:
      show_source: true
      heading_level: 3
      members:
        - StructureSet

## I/O Functions

::: dosemetrics.io.data_io
    options:
      show_source: true
      heading_level: 3
      members:
        - read_from_nifti
        - read_dose_and_mask_files
        - read_file
        - read_byte_data
        - find_all_files
        - get_dose
        - get_structures

## Usage Examples

### Loading Data from NIfTI Files

```python
from dosemetrics import read_dose_and_mask_files, StructureSet

# Load dose and structures from a folder
dose, structures = read_dose_and_mask_files("path/to/patient_data")

# Access individual structures
ptv_mask = structures.get_structure_mask("PTV")
print(f"Loaded {len(structures)} structures")
```

### Creating a StructureSet

```python
from dosemetrics import StructureSet, Structure

# Create from folder
structures = StructureSet.from_folder("path/to/structures/")

# Access structure names
for name in structures.get_structure_names():
    print(f"Structure: {name}")
    mask = structures.get_structure_mask(name)
    print(f"  Volume: {mask.sum()} voxels")
```

### Reading Dose Distribution

```python
from dosemetrics.io import read_from_nifti

# Read NIfTI file
dose = read_from_nifti("path/to/dose.nii.gz")
print(f"Dose shape: {dose.shape}")
print(f"Dose range: {dose.min():.2f} - {dose.max():.2f} Gy")
```

### Working with Multiple Files

```python
from dosemetrics.io import find_all_files
from pathlib import Path

# Find all NIfTI files in directory
data_dir = Path("path/to/data")
dose_files = find_all_files(data_dir, pattern="*.nii.gz")

for dose_file in dose_files:
    print(f"Found: {dose_file}")
```

### Batch Processing

```python
from dosemetrics import read_dose_and_mask_files, compute_dvh
from pathlib import Path

# Process multiple patients
results = []
for patient_dir in Path("data").glob("patient_*"):
    # Load data
    dose, structures = read_dose_and_mask_files(patient_dir)
    
    # Compute metrics
    for structure_name in structures.get_structure_names():
        mask = structures.get_structure_mask(structure_name)
        dvh = compute_dvh(dose, mask, organ_name=structure_name)
        results.append({
            "patient": patient_dir.name,
            "structure": structure_name,
            "dvh": dvh
        })
```

## Supported Formats

### Dose Distributions

| Format | Extension | Read | Notes |
|--------|-----------|------|-------|
| NIfTI | `.nii`, `.nii.gz` | ✓ | Recommended |
| DICOM | `.dcm` | ✓ | RT Dose |
| NRRD | `.nrrd` | ✓ | Research format |

### Structure Masks

| Format | Extension | Read | Notes |
|--------|-----------|------|-------|
| NIfTI | `.nii`, `.nii.gz` | ✓ | Binary or labeled |
| DICOM | `.dcm` | ✓ | RT Structure Set |
| NRRD | `.seg.nrrd` | ✓ | Segmentation |

## Data Requirements

### Spatial Alignment

**Critical:** Dose and structure masks must be spatially aligned:

- Same image spacing (voxel size)
- Same image dimensions
- Same coordinate system origin
- Same orientation

### File Organization

Recommended structure:

```
patient_data/
├── dose.nii.gz
└── structures/
    ├── PTV.nii.gz
    ├── Brain.nii.gz
    └── Brainstem.nii.gz
```

## See Also

- [File Formats Guide](../getting-started/file-formats.md)
- [Metrics Module](metrics.md)
- [Utils Module](utils.md)
