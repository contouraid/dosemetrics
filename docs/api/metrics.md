# Metrics API

This module contains core functions for dose analysis, DVH computation, quality metrics, and plan comparison.

## DVH Module

::: dosemetrics.metrics.dvh
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

## Gamma Module

::: dosemetrics.metrics.gamma
    options:
      show_source: true
      heading_level: 3

## Dose Comparison Module

::: dosemetrics.metrics.dose_comparison
    options:
      show_source: true
      heading_level: 3

## Advanced DVH Module

::: dosemetrics.metrics.advanced_dvh
    options:
      show_source: true
      heading_level: 3

---

## Usage Examples

### Computing a Basic DVH

```python
from dosemetrics import Dose, Structure
from dosemetrics.metrics.dvh import compute_dvh

dose = Dose.from_nifti("dose.nii.gz")
ptv  = Structure.from_nifti("ptv.nii.gz", name="PTV")

dose_bins, volumes = compute_dvh(dose, ptv)
```

### Conformity and Homogeneity Metrics

```python
from dosemetrics.metrics.conformity import (
    compute_conformity_index,          # ICRU CI = V_target_rx / V_rx
    compute_rtog_conformity_index,     # RTOG CI = V_rx / V_target
    compute_paddick_conformity_index,  # Paddick / van't Riet CI = (V_target_rx)² / (V_target × V_rx)
    compute_conformity_number,         # van't Riet CN — same formula as Paddick CI
    compute_coverage,                  # V_target_rx / V_target
    compute_spillage,                  # (V_rx - V_target_rx) / V_rx
    compute_prescription_mae,          # mean |dose - prescription| in target
)
from dosemetrics.metrics.homogeneity import (
    compute_homogeneity_index,   # ICRU HI = (D2 - D98) / D50
    compute_gradient_index,      # Paddick-Lippitz GI = V_50% / V_100%
    compute_dose_homogeneity,    # coefficient of variation
    compute_uniformity_index,    # UI = 1 - (Dmax - Dmin) / Dref
)

prescription = 60.0  # Gy

ci_icru  = compute_conformity_index(dose, ptv, prescription)
ci_rtog  = compute_rtog_conformity_index(dose, ptv, prescription)
ci_pad   = compute_paddick_conformity_index(dose, ptv, prescription)
hi       = compute_homogeneity_index(dose, ptv)
gi       = compute_gradient_index(dose, ptv, prescription)
rx_mae   = compute_prescription_mae(dose, ptv, prescription)

print(f"ICRU CI:        {ci_icru:.3f}")
print(f"RTOG CI:        {ci_rtog:.3f}")
print(f"Paddick CI:     {ci_pad:.3f}")
print(f"HI (ICRU 83):   {hi:.3f}")
print(f"Gradient Index: {gi:.2f}")
print(f"Prescription MAE: {rx_mae:.2f} Gy")
```

### DVH Comparison Metrics

```python
from dosemetrics.metrics.dvh import compute_dvh_score, compute_dvh_auc
from dosemetrics.metrics.dose_comparison import (
    compute_normalized_mae,
    compute_variance_of_laplacian,
)

# DVH Score: average |D1|, |D95|, |D99| difference (Gy)
score = compute_dvh_score(dose_reference, dose_evaluated, ptv)

# DVH AUC: integral of DVH curve, normalised to [0, 1]
auc = compute_dvh_auc(dose, ptv, normalize=True)

# Normalized MAE with high-dose masking
n_mae = compute_normalized_mae(
    dose_reference,
    dose_evaluated,
    normalization_value=60.0,
    dose_threshold_gy=5.0,
)

# Dose sharpness (Variance of Laplacian)
vol = compute_variance_of_laplacian(dose)

print(f"DVH Score:      {score:.2f} Gy")
print(f"DVH AUC:        {auc:.3f}")
print(f"Normalized MAE: {n_mae:.4f}")
print(f"VoL (sharpness): {vol:.4f}")
```

### Dose Statistics

```python
from dosemetrics.metrics.dvh import compute_dose_statistics

stats = compute_dose_statistics(dose, ptv)
print(f"Mean dose: {stats['mean_dose']:.2f} Gy")
print(f"D95:       {stats['D95']:.2f} Gy")
print(f"D2 (near-max): {stats['D02']:.2f} Gy")
```

## See Also

- [I/O Module](io.md) — Loading and saving data
- [Utils Module](utils.md) — Plotting and compliance checking
- [User Guide: Quality Metrics](../user-guide/quality-metrics.md)
- [User Guide: DVH Analysis](../user-guide/dvh-analysis.md)
