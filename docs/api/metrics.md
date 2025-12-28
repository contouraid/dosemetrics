# Metrics API

This module contains core functions for dose analysis, DVH computation, quality metrics, and plan comparison.

## DVH Module

::: dosemetrics.metrics.dvh
    options:
      show_source: true
      heading_level: 3

## Statistics Module

::: dosemetrics.metrics.statistics
    options:
      show_source: true
      heading_level: 3

## Conformity Module

::: dosemetrics.metrics.conformity
    options:
      show_source: true
      heading_level: 3

## Homogeneity Module

::: dosemetrics.metrics.homogeneity
    options:
      show_source: true
      heading_level: 3

## Geometric Module

::: dosemetrics.metrics.geometric
    options:
      show_source: true
      heading_level: 3

## Usage Examples

### Computing a Basic DVH

```python
from dosemetrics import Dose, Structure
from dosemetrics.metrics.dvh import compute_dvh

dose = Dose.from_nifti("dose.nii.gz")
structure = Structure.from_nifti("ptv.nii.gz", name="PTV")

dvh = compute_dvh(
    dose=dose,
    structure=structure,
    bins=1000
)
```

### Calculating Quality Metrics

```python
from dosemetrics import Dose, Structure
from dosemetrics.metrics.conformity import compute_conformity_index
from dosemetrics.metrics.homogeneity import compute_homogeneity_index
from dosemetrics.metrics.statistics import compute_dose_statistics

# Conformity Index
ci = compute_conformity_index(
    dose=dose,
    target=ptv,
    prescription_dose=60.0
)

# Homogeneity Index
hi = compute_homogeneity_index(
    dose=dose,
    target=ptv
)

# Dose Statistics
stats = compute_dose_statistics(
    dose=dose,
    structure=ptv
)
```

### Comparing Dose Distributions

```python
from dosemetrics import Dose, StructureSet
from dosemetrics.utils.comparison import compare_dose_distributions

# Compare dose distributions across multiple structures
comparison_df = compare_dose_distributions(
    dose1=dose_tps,
    dose2=dose_predicted,
    structure_set=structures,
    structure_names=["PTV", "Heart", "Lung_L"],
    output_file="comparison.pdf"
)
```

## See Also

- [I/O Module](io.md) - Loading and saving data
- [Utils Module](utils.md) - Plotting and compliance checking
- [User Guide: DVH Analysis](../user-guide/dvh-analysis.md)
- [User Guide: Quality Metrics](../user-guide/quality-metrics.md)
