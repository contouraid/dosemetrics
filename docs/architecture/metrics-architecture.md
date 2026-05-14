# Metrics Architecture

This document describes how the `dosemetrics.metrics` subpackage is structured and the principle of separating data representation from computation.

## Separation of Concerns

The metrics subpackage applies a single governing rule: **data classes hold data; metrics functions do computation**.

| Layer | Responsibility | Location |
|-------|---------------|----------|
| Data classes | Store arrays, metadata, file paths | `dosemetrics.dose`, `dosemetrics.structures` |
| Metrics | Algorithms and formulas | `dosemetrics.metrics.*` |
| Utils | Visualization, batch processing, compliance | `dosemetrics.utils.*` |

This means no dosimetric algorithm lives inside `Dose` or `Structure`. All computation flows through the `metrics` subpackage.

## Module Breakdown

### `metrics/dvh.py` — Dose-Volume Histograms and DVH Metrics

Core DVH computation and scalar DVH-based metrics.

```python
from dosemetrics.metrics import dvh

# DVH curve
dose_bins, volumes = dvh.compute_dvh(dose, structure)

# Point dose statistics
d95  = dvh.compute_dose_at_volume(dose, structure, volume_percent=95)
v20  = dvh.compute_volume_at_dose(dose, structure, dose_threshold=20.0)
mean = dvh.compute_mean_dose(dose, structure)
stats = dvh.compute_dose_statistics(dose, structure)

# DVH comparison metrics (new in v0.4)
score = dvh.compute_dvh_score(dose_ref, dose_eval, structure)   # D1/D95/D99 difference
auc   = dvh.compute_dvh_auc(dose, structure, normalize=True)    # area under DVH curve
```

### `metrics/conformity.py` — Conformity Indices

Metrics that quantify how well the high-dose region conforms to the target volume. Implements multiple named formulations from the literature.

```python
from dosemetrics.metrics import conformity

# ICRU CI (ICRU Report 62, 1999): V_target_rx / V_rx
ci_icru = conformity.compute_conformity_index(dose, target, prescription_dose)

# van't Riet Conformation Number (1997): same formula as Paddick CI
cn = conformity.compute_conformity_number(dose, target, prescription_dose)

# RTOG CI (Shaw et al. 1993): V_rx / V_target
ci_rtog = conformity.compute_rtog_conformity_index(dose, target, prescription_dose)

# Paddick CI (Paddick 2000): (V_target_rx)² / (V_target × V_rx)
ci_pad = conformity.compute_paddick_conformity_index(dose, target, prescription_dose)

# Coverage: V_target_rx / V_target
coverage = conformity.compute_coverage(dose, target, prescription_dose)

# Spillage: (V_rx - V_target_rx) / V_rx
spillage = conformity.compute_spillage(dose, target, prescription_dose)

# Prescription MAE: mean |dose - prescription| within target (new in v0.4)
rx_mae = conformity.compute_prescription_mae(dose, target, prescription_dose)
```

All seven functions accept the same `(dose, target, prescription_dose)` signature, making it easy to compute a full conformity report in a loop.

### `metrics/homogeneity.py` — Homogeneity Indices

Metrics that quantify the uniformity of dose within a target volume.

```python
from dosemetrics.metrics import homogeneity

# ICRU HI (ICRU 83, 2010): (D2 - D98) / D50
hi = homogeneity.compute_homogeneity_index(dose, target)

# Gradient Index (Paddick & Lippitz 2006): V_50% / V_100%
gi = homogeneity.compute_gradient_index(dose, target, prescription_dose)

# Coefficient of variation: σ / μ
cv = homogeneity.compute_dose_homogeneity(dose, target)

# Uniformity Index: 1 - (Dmax - Dmin) / Dref
ui = homogeneity.compute_uniformity_index(dose, target)
```

### `metrics/geometric.py` — Spatial and Segmentation Metrics

Volume, overlap, and surface-distance metrics for comparing two structure contours.

```python
from dosemetrics.metrics import geometric

dice = geometric.compute_dice_coefficient(structure1, structure2)
iou  = geometric.compute_jaccard_index(structure1, structure2)
hd95 = geometric.compute_hausdorff_distance(structure1, structure2, percentile=95)
msd  = geometric.compute_mean_surface_distance(structure1, structure2)

df = geometric.compare_structure_sets(struct_set1, struct_set2)
```

### `metrics/gamma.py` — Gamma Index

Quantitative comparison of two dose distributions for patient-specific QA. Follows the formulation from Low et al. (1998).

```python
from dosemetrics.metrics import gamma

gamma_map  = gamma.compute_gamma_index(dose_ref, dose_eval,
                                       dose_criterion_percent=3.0,
                                       distance_criterion_mm=3.0)
pass_rate  = gamma.compute_gamma_passing_rate(gamma_map)
stats      = gamma.compute_gamma_statistics(gamma_map)
```

!!! warning "Performance"
    The 3D gamma computation is algorithmically correct but slow for large volumes (128³ takes ~53 s). For current workarounds see [Gamma Index Performance](../user-guide/gamma-performance.md). Optimization is planned.

### `metrics/dose_comparison.py` — Image-Based Dose Comparison

Pixel/voxel-level comparison metrics between two dose grids, including image-quality measures adapted for 3D dose distributions.

```python
from dosemetrics.metrics import dose_comparison

# Standard image metrics
mse  = dose_comparison.compute_mse(dose1, dose2)
mae  = dose_comparison.compute_mae(dose1, dose2)
ssim = dose_comparison.compute_ssim(dose1, dose2)
psnr = dose_comparison.compute_psnr(dose1, dose2)
ncc  = dose_comparison.compute_normalized_cross_correlation(dose1, dose2)

# Normalized MAE with optional threshold masking (new in v0.4)
n_mae = dose_comparison.compute_normalized_mae(
    dose1, dose2,
    normalization_value=60.0,
    dose_threshold_gy=5.0,
)

# Variance of Laplacian: dose sharpness / gradient complexity (new in v0.4)
vol = dose_comparison.compute_variance_of_laplacian(dose)
```

### `metrics/advanced_dvh.py` — Statistical DVH Comparisons

Advanced statistical tests and distance measures between two DVH curves.

```python
from dosemetrics.metrics import advanced_dvh

emd  = advanced_dvh.compute_dvh_wasserstein_distance(dose1, dose2, structure)
area = advanced_dvh.compute_area_between_dvh_curves(dose1, dose2, structure)
ks   = advanced_dvh.compute_dvh_ks_test(dose1, dose2, structure)
```

## Naming Conventions

Function names follow the pattern `compute_<metric_name>`. Parameters at the public API boundary are always `Dose` and `Structure` objects — raw NumPy arrays are only used internally.

## Extension Pattern

To add a new metric, create a function in the appropriate module (or a new module if the category is new):

```python
# dosemetrics/metrics/conformity.py

def compute_new_index(dose: Dose, target: Structure, prescription_dose: float) -> float:
    """Short description of what the metric measures.

    formula: ...

    Args:
        dose: Dose distribution.
        target: Target structure (PTV, CTV, etc.).
        prescription_dose: Prescription dose in Gy.

    Returns:
        Metric value (dimensionless).

    References:
        Author et al. Journal. Year.
    """
    values = dose.get_dose_in_structure(target)
    ...
```

Guidelines:

- Accept `Dose` and `Structure` objects, not raw NumPy arrays, at the public API boundary
- Return `float` for scalar metrics, `np.ndarray` for maps
- Include a docstring with the formula, parameter definitions, and a literature reference
- Add unit tests covering identical distributions, edge cases (zero dose, uniform dose), and numerical precision
- Export from `dosemetrics.metrics.__init__` so users can import via `from dosemetrics.metrics import compute_new_index`

## What Does Not Belong in Metrics

The following belong in `dosemetrics.utils`, not `dosemetrics.metrics`:

- Visualization (`plot_dvh`, `plot_dvh_score_breakdown`, `plot_dvh_auc`)
- Batch processing (`batch_compute_dvhs`)
- File I/O helpers
- Compliance checking against protocol constraints
- DataFrame assembly and CSV export
