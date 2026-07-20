# StructureSet: Multi-Structure Workflows

`StructureSet` is a named collection of structure geometry. It does not contain a dose distribution or compute dose metrics itself.

## Load a structure set

```python
from dosemetrics import Dose, StructureType
from dosemetrics.io import load_structure_set

type_mapping = {
    "PTV": StructureType.TARGET,
    "Brainstem": StructureType.OAR,
}
structures = load_structure_set(
    "patient_001",
    structure_type_mapping=type_mapping,
)
dose = Dose.from_nifti("patient_001/Dose.nii.gz")
```

## Create one in memory

```python
from dosemetrics import StructureSet, StructureType

structures = StructureSet(
    spacing=(1.0, 1.0, 2.5),
    origin=(0.0, 0.0, 0.0),
    name="Patient001",
)
structures.add_structure("PTV", ptv_mask, StructureType.TARGET)
structures.add_structure("Brainstem", brainstem_mask, StructureType.OAR)
```

## Access and iterate

```python
ptv = structures["PTV"]
brainstem = structures.get_structure("Brainstem")

for name, structure in structures:
    print(name, structure.structure_type, structure.volume_cc())

targets = structures.get_targets()
oars = structures.get_oars()
```

## Compute across all structures

```python
import pandas as pd
from dosemetrics.metrics import dvh

dvh_table = dvh.create_dvh_table(dose, structures, step_size=0.1)

statistics = []
for name, structure in structures:
    statistics.append({
        "Structure": name,
        "Type": structure.structure_type.value,
        "Volume_cc": structure.volume_cc(),
        **dvh.compute_dose_statistics(dose, structure),
    })
statistics_table = pd.DataFrame(statistics)
```

## Geometric summary

```python
geometry = structures.geometric_summary()
print(geometry[["Structure", "Type", "Volume_cc"]])
```

## Current API

| Member | Description |
|---|---|
| `structures` | Name-to-`Structure` dictionary |
| `structure_names`, `oar_names`, `target_names` | Name lists |
| `add_structure(...)`, `add_structure_object(...)` | Add geometry |
| `remove_structure(name)` | Remove geometry |
| `get_structure(name)` / `set[name]` | Retrieve geometry |
| `get_oars()`, `get_targets()` | Filter by type |
| `geometric_summary()` | Geometry table |
| `total_volume_cc()` | Sum of individual structure volumes |

Dose statistics, bulk DVHs, compliance, and plots live in `dosemetrics.metrics` and `dosemetrics.utils`, where the dose is passed explicitly.
