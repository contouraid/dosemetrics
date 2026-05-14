# Migration Guide: v0.2.0 → v0.3.0

Version 0.3.0 introduces significant architectural improvements with the new `Dose` and `StructureSet` classes. This guide covers every breaking change and provides a gradual migration path.

## Breaking Changes

### 1. Dose data is no longer stored inside structures

**Before (v0.2.0):**
```python
from dosemetrics import Structure

structure = Structure.from_nifti("mask.nii.gz")
structure.set_dose_data(dose_array, spacing, origin)
dvh = structure.compute_dvh()
```

**After (v0.3.0):**
```python
from dosemetrics import Structure, Dose

structure = Structure.from_nifti("mask.nii.gz")
dose = Dose.from_nifti("dose.nii.gz")

dvh = structure.compute_dvh(dose)
```

### 2. Multi-structure workflows use StructureSet

**Before (v0.2.0):**
```python
structures = {}
for name, mask_file in mask_files.items():
    struct = Structure.from_nifti(mask_file)
    struct.set_dose_data(dose_array, spacing, origin)
    structures[name] = struct

dvhs = {name: s.compute_dvh() for name, s in structures.items()}
```

**After (v0.3.0):**
```python
from dosemetrics import StructureSet, Dose
from dosemetrics.structures import StructureType

structure_set = StructureSet.from_folder(
    folder_path="patient_data/",
    structure_types={
        "PTV": StructureType.TARGET,
        "Brainstem": StructureType.OAR,
        "SpinalCord": StructureType.OAR,
    }
)

dvhs = structure_set.compute_all_dvhs()
```

## New Recommended Patterns

### Loading data

**Option 1: StructureSet from a folder (most convenient)**
```python
from dosemetrics import StructureSet
from dosemetrics.structures import StructureType

ss = StructureSet.from_folder(
    folder_path="data/patient_001/",
    structure_types={
        "PTV": StructureType.TARGET,
        "OAR1": StructureType.OAR,
    }
)
```

**Option 2: Dose and structures loaded separately**
```python
from dosemetrics import Dose, Structure, StructureSet
from dosemetrics.structures import StructureType

dose = Dose.from_nifti("dose.nii.gz")
structures = {
    "PTV": Structure.from_nifti("ptv.nii.gz", structure_type=StructureType.TARGET),
    "Brainstem": Structure.from_nifti("brainstem.nii.gz", structure_type=StructureType.OAR),
}
ss = StructureSet(structures=structures, dose=dose)
```

**Option 3: I/O helper function**
```python
from dosemetrics.io import get_dose_and_structures_as_structure_set

ss = get_dose_and_structures_as_structure_set(
    folder_path="data/",
    structure_types={"PTV": StructureType.TARGET, "OAR1": StructureType.OAR}
)
```

### Comparing multiple dose distributions

```python
dose_predicted = Dose.from_nifti("predicted_dose.nii.gz")
dose_actual = Dose.from_nifti("actual_dose.nii.gz")

structure = ss["PTV"]
dvh_predicted = structure.compute_dvh(dose_predicted)
dvh_actual = structure.compute_dvh(dose_actual)
```

### Bulk operations

```python
dvhs   = structure_set.compute_all_dvhs()
stats  = structure_set.get_all_statistics()
oars   = structure_set.get_structures_by_type(StructureType.OAR)

for name, structure in structure_set.items():
    print(f"{name}: {structure.get_mean_dose(structure_set.dose):.2f} Gy")
```

## Backward Compatibility

Most `Structure` methods still work — just pass `dose` explicitly:

```python
structure = Structure.from_nifti("mask.nii.gz")
dose = Dose.from_nifti("dose.nii.gz")

mean_dose = structure.get_mean_dose(dose)
max_dose  = structure.get_max_dose(dose)
dvh       = structure.compute_dvh(dose)
```

## Common Migration Patterns

### Pattern 1: Single structure analysis

```python
# Before
structure = Structure.from_nifti("brainstem.nii.gz")
structure.set_dose_data(dose_array, spacing, origin)
stats = structure.get_statistics()

# After
structure = Structure.from_nifti("brainstem.nii.gz")
dose = Dose(dose_array, spacing, origin)
stats = structure.get_statistics(dose)
```

### Pattern 2: Multiple structure analysis

```python
# Before
structures = load_multiple_structures(folder)
for struct in structures.values():
    struct.set_dose_data(dose_array, spacing, origin)
results = analyze_structures(structures)

# After
structure_set = StructureSet.from_folder(folder, structure_types)
results = structure_set.compute_all_dvhs()
```

### Pattern 3: Plan comparison

```python
# Before — had to swap dose in-place (error-prone)
struct.set_dose_data(dose1_array, spacing, origin)
dvh1 = struct.compute_dvh()
struct.set_dose_data(dose2_array, spacing, origin)
dvh2 = struct.compute_dvh()

# After — clean, no mutation
struct = Structure.from_nifti("mask.nii.gz")
dvh1 = struct.compute_dvh(Dose.from_nifti("dose1.nii.gz"))
dvh2 = struct.compute_dvh(Dose.from_nifti("dose2.nii.gz"))
```

## Troubleshooting

**`AttributeError: 'Structure' object has no attribute 'set_dose_data'`**  
Remove `set_dose_data()` calls and pass a `Dose` object to the method directly.

**`TypeError: compute_dvh() missing required argument 'dose'`**  
Create a `Dose` object and pass it explicitly:
```python
dose = Dose.from_nifti("dose.nii.gz")
dvh = structure.compute_dvh(dose)
```

**`TypeError: argument of type 'dict' is not iterable`**  
Replace dictionary-based structure collections with `StructureSet`.

## Gradual Migration Strategy

You do not have to migrate everything at once:

1. **Phase 1** — Update code that creates `Dose` objects instead of setting dose data on structures
2. **Phase 2** — Replace manual structure dictionaries with `StructureSet`
3. **Phase 3** — Adopt I/O helper functions for loading
4. **Phase 4** — Leverage bulk operations and type-based filtering

Each phase is independent and can be done incrementally across your codebase.
