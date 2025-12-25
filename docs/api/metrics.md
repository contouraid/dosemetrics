# Metrics API

This module contains core functions for dose analysis, DVH computation, quality metrics, and plan comparison.

## DVH Module

::: dosemetrics.metrics.dvh
    options:
      show_source: true
      heading_level: 3

## Scores Module

::: dosemetrics.metrics.scores
    options:
      show_source: true
      heading_level: 3

## Comparison Module

::: dosemetrics.metrics.comparison
    options:
      show_source: true
      heading_level: 3

## Usage Examples

### Computing a Basic DVH

```python
from dosemetrics.io import load_dose, load_mask
from dosemetrics.metrics.dvh import compute_dvh

dose = load_dose("dose.nii.gz")
mask = load_mask("ptv.nii.gz")

dvh = compute_dvh(
    dose=dose,
    mask=mask,
    organ_name="PTV",
    bins=1000,
    dose_unit="Gy"
)
```

### Calculating Quality Metrics

```python
from dosemetrics.metrics.scores import (
    compute_conformity_index,
    compute_homogeneity_index,
    compute_gradient_index
)

# Conformity Index
ci = compute_conformity_index(
    dose=dose,
    target_mask=ptv_mask,
    prescription_dose=60.0
)

# Homogeneity Index
hi = compute_homogeneity_index(
    dose=dose,
    target_mask=ptv_mask
)

# Gradient Index
gi = compute_gradient_index(
    dose=dose,
    target_mask=ptv_mask,
    prescription_dose=60.0
)
```

### Comparing Plans

```python
from dosemetrics.metrics.comparison import (
    compute_dose_difference,
    compute_gamma_index,
    compute_dvh_difference
)

# Absolute dose difference
diff = compute_dose_difference(
    dose1=dose_tps,
    dose2=dose_predicted,
    method="absolute"
)

# Gamma analysis
gamma = compute_gamma_index(
    reference_dose=dose_tps,
    evaluated_dose=dose_predicted,
    dose_threshold=3.0,  # 3% dose difference
    distance_threshold=3.0,  # 3mm distance
    dose_cutoff=0.1  # Ignore below 10% of max dose
)

# DVH comparison
dvh_diff = compute_dvh_difference(
    dvh1=dvh_tps,
    dvh2=dvh_predicted,
    dose_points=[50, 95, 100]  # Compare at specific dose levels
)
```

## See Also

- [I/O Module](io.md) - Loading and saving data
- [Utils Module](utils.md) - Plotting and compliance checking
- [User Guide: DVH Analysis](../user-guide/dvh-analysis.md)
- [User Guide: Quality Metrics](../user-guide/quality-metrics.md)
