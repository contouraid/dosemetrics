# StructureSet: Multi-Structure Workflows

`StructureSet` is the primary entry point for workflows that involve multiple radiotherapy structures. It provides a unified object for managing structures and dose together, following DICOM RTSS conventions.

## Why StructureSet?

Before `StructureSet`, multi-structure workflows required managing plain Python dictionaries and manually passing dose data to every structure:

```python
# Old approach — manual and repetitive
dose_volume, structure_masks = dm.read_dose_and_mask_files(dose_file, mask_files)

results = []
for name, mask in structure_masks.items():
    mean_dose = dm.mean_dose(dose_volume, mask)
    max_dose  = dm.max_dose(dose_volume, mask)
    volume    = dm.volume(mask)
    results.append({"Structure": name, "Mean": mean_dose, "Max": max_dose})
stats_df = pd.DataFrame(results)
```

With `StructureSet`, the same workflow becomes:

```python
structure_set = dm.get_dose_and_structures_as_structure_set("data/patient")
stats_df = structure_set.dose_statistics_summary()
```

## Creating a StructureSet

### From a folder

```python
import dosemetrics as dm

structure_set = dm.get_dose_and_structures_as_structure_set(
    data_path="data/patient_001",
    name="Patient001"
)
```

### From existing dose and structure data

```python
dose_volume, structure_masks = dm.read_dose_and_mask_files(dose_file, mask_files)
structure_set = dm.create_structure_set_from_existing_data(
    dose_volume=dose_volume,
    structure_masks=structure_masks,
    structure_types={"PTV": "Target", "Brainstem": "OAR"},
    name="ConvertedCase"
)
```

### From file lists

```python
structure_set = dm.read_dose_and_mask_files_as_structure_set(
    dose_file="dose.nii.gz",
    mask_files=["ptv.nii.gz", "brainstem.nii.gz"],
    structure_types={"ptv": "Target", "brainstem": "OAR"}
)
```

## Common Workflows

### Dose statistics summary

```python
stats_df = structure_set.dose_statistics_summary()
print(stats_df)
#    Structure    Type  Volume_cc  Mean_Dose_Gy  Max_Dose_Gy  D95_Gy  D05_Gy
# 0       PTV  TARGET      125.3          50.1         55.2    48.5    52.1
# 1  Brainstem     OAR        3.2          12.4         25.8     5.2    22.1
```

### DVH for all structures

```python
dvh_df = structure_set.compute_bulk_dvh(max_dose=70, step_size=0.1)
```

### Constraint compliance

```python
constraints = {
    "Brainstem":   {"max_dose": 54, "unit": "Gy"},
    "Spinal_Cord": {"max_dose": 45, "unit": "Gy"},
    "PTV":         {"mean_dose": 50, "unit": "Gy"}
}
compliance_df = structure_set.compliance_check(constraints)
print(compliance_df)
#    Structure  Constraint_Type  Constraint_Value  Actual_Value  Compliant
# 0  Brainstem         Max Dose              54.0          25.8       True
# 1  Spinal_Cord       Max Dose              45.0          18.9       True
# 2  PTV               Mean Dose             50.0          50.1      False
```

### Geometric summary

```python
geom_df = structure_set.geometric_summary()
print(geom_df[["Structure", "Volume_cc", "Centroid_X", "Centroid_Y", "Centroid_Z"]])
```

## Accessing Structures

```python
# By name
brainstem = structure_set["Brainstem"]
print(f"Max dose: {brainstem.max_dose():.1f} Gy")

# By type
oars    = structure_set.get_oars()
targets = structure_set.get_targets()

# Iterate
for name, structure in structure_set:
    print(f"{name}: {structure.volume_cc():.1f} cc")

# Membership check
if "PTV" in structure_set:
    print("PTV found")
```

## Saving Results

```python
import os
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

structure_set.dose_statistics_summary().to_csv(f"{output_dir}/dose_stats.csv", index=False)
structure_set.geometric_summary().to_csv(f"{output_dir}/geometry.csv", index=False)
structure_set.compute_bulk_dvh().to_csv(f"{output_dir}/dvh.csv", index=False)
```

## API Reference

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `structures` | `dict` | Dictionary of `Structure` objects |
| `structure_names` | `list[str]` | All structure names |
| `oar_names` | `list[str]` | Names of OAR structures |
| `target_names` | `list[str]` | Names of target structures |
| `structure_count` | `int` | Number of structures |
| `has_dose` | `bool` | Whether dose is available |
| `spacing` | `tuple` | Voxel spacing in mm |
| `origin` | `tuple` | World-space origin in mm |

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `add_structure(name, mask, type)` | — | Add a structure |
| `remove_structure(name)` | — | Remove a structure |
| `get_structure(name)` | `Structure` | Get by name |
| `get_structures_by_type(type)` | `dict` | Filter by type |
| `get_oars()` | `dict` | All OAR structures |
| `get_targets()` | `dict` | All target structures |
| `dose_statistics_summary()` | `DataFrame` | Statistics for all structures |
| `geometric_summary()` | `DataFrame` | Geometry for all structures |
| `compute_bulk_dvh()` | `DataFrame` | DVH for all structures |
| `compliance_check(constraints)` | `DataFrame` | Constraint evaluation |
| `total_volume_cc()` | `float` | Total volume |

## See Also

- [Migration Guide](migration-v0.2-v0.3.md) — upgrading from v0.2.0
- [Architecture: Dose Architecture](../architecture/dose-architecture.md) — why dose and structures are decoupled
