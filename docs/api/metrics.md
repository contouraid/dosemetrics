# Metrics API

The dose-plan API is organized by meaning:

- `compute_*` functions characterize one plan (or summarize a collection).
- `compare_*` functions require a `reference` followed by an `evaluated` plan.
- Functions are accessed through their domain modules; the package does not
  flatten similarly named indices and distances into one namespace.

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

## Comparison Module

::: dosemetrics.metrics.comparison
    options:
      show_source: true
      heading_level: 3

---

## Usage Examples

### Computing a Basic DVH

```python
from dosemetrics import Dose, StructureType
from dosemetrics.io import load_structure
from dosemetrics.metrics import dvh

dose = Dose.from_nifti("dose.nii.gz")
ptv = load_structure(
    "ptv.nii.gz",
    name="PTV",
    structure_type=StructureType.TARGET,
)

dose_bins, volumes = dvh.compute_dvh(dose, ptv)
```

### Conformity and Homogeneity Metrics

```python
from dosemetrics.metrics import conformity, homogeneity

prescription = 60.0  # Gy

ci_icru = conformity.compute_conformity_index(dose, ptv, prescription)
ci_rtog = conformity.compute_rtog_conformity_index(dose, ptv, prescription)
ci_pad = conformity.compute_paddick_conformity_index(dose, ptv, prescription)
hi = homogeneity.compute_homogeneity_index(dose, ptv)
gi = homogeneity.compute_gradient_index(dose, ptv, prescription)
rx_mae = conformity.compute_prescription_mae(dose, ptv, prescription)

print(f"ICRU CI:        {ci_icru:.3f}")
print(f"RTOG CI:        {ci_rtog:.3f}")
print(f"Paddick CI:     {ci_pad:.3f}")
print(f"HI (ICRU 83):   {hi:.3f}")
print(f"Gradient Index: {gi:.2f}")
print(f"Prescription MAE: {rx_mae:.2f} Gy")
```

### DVH Comparison Metrics

```python
from dosemetrics.metrics import comparison, dose_comparison, dvh

# DVH Score: average |D1|, |D95|, |D99| difference (Gy)
score = comparison.compare_dvh_score(
    reference,
    evaluated,
    targets=[ptv],
    oars=[brainstem, spinal_cord],
)

# DVH AUC: integral of DVH curve, normalised to [0, 1]
auc = dvh.compute_dvh_auc(dose, ptv, normalize=True)

# Normalized MAE with high-dose masking
n_mae = dose_comparison.compare_normalized_mae(
    reference,
    evaluated,
    normalization_value=60.0,
    dose_threshold_gy=5.0,
)

# Dose sharpness (Variance of Laplacian)
vol = dose_comparison.compute_variance_of_laplacian(dose)

print(f"DVH Score:      {score:.2f} Gy")
print(f"DVH AUC:        {auc:.3f}")
print(f"Normalized MAE: {n_mae:.4f}")
print(f"VoL (sharpness): {vol:.4f}")
```

### Dose Statistics

```python
from dosemetrics.metrics import dvh

stats = dvh.compute_dose_statistics(dose, ptv)
print(f"Mean dose: {stats['mean_dose']:.2f} Gy")
print(f"D95:       {stats['D95']:.2f} Gy")
print(f"D2 (near-max): {stats['D02']:.2f} Gy")
```

## See Also

- [I/O Module](io.md) — Loading and saving data
- [Utils Module](utils.md) — Plotting and compliance checking
- [User Guide: Quality Metrics](../user-guide/quality-metrics.md)
- [User Guide: DVH Analysis](../user-guide/dvh-analysis.md)
