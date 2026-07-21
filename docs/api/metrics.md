# Metrics API

`dosemetrics.metrics` exposes domain modules for reference-free computations
and direct `compare_*` functions for the named reference-based plan metrics.

| Public module | Responsibility | Dose-reference convention |
|---|---|---|
| `conformity` | Single-plan coverage and conformity quantities | Reference-free |
| `dose_comparison` | General image and voxel comparisons plus image descriptors | `compare_*` is reference-based; `compute_*` is reference-free |
| `dvh` | DVH construction, queries, summaries, and comparisons | `compute_*` is reference-free; `compare_*` is reference-based |
| `gamma` | Gamma maps, passing-rate summaries, and statistics | Gamma-map creation is reference-based |
| `geometric` | Structure overlap, volume, and surface metrics | Structure geometry; no dose reference |
| `homogeneity` | Single-plan target homogeneity and dose falloff | Reference-free |

The generated sections below are the authoritative public signatures and
defaults from the implementation. For clinical interpretation and reference
semantics, see the [Metric Framework](../user-guide/quality-metrics.md).

## Direct plan-comparison functions

Import these functions without an additional module prefix:

```python
from dosemetrics.metrics import compare_ptv_dose

distance_gy = compare_ptv_dose(reference, evaluated, ptv)
```

::: dosemetrics.metrics.compare_ptv_dose
    options:
      show_source: true
      heading_level: 3

::: dosemetrics.metrics.compare_paddick_conformity_index
    options:
      show_source: true
      heading_level: 3

::: dosemetrics.metrics.compare_paddick_gradient_index
    options:
      show_source: true
      heading_level: 3

::: dosemetrics.metrics.compare_homogeneity_index
    options:
      show_source: true
      heading_level: 3

::: dosemetrics.metrics.compare_body_rmse
    options:
      show_source: true
      heading_level: 3

::: dosemetrics.metrics.compare_gamma
    options:
      show_source: true
      heading_level: 3

::: dosemetrics.metrics.compare_oar_constraints
    options:
      show_source: true
      heading_level: 3

::: dosemetrics.metrics.compare_oar_dvh_auc
    options:
      show_source: true
      heading_level: 3

::: dosemetrics.metrics.compare_mean_oar_dvh_auc
    options:
      show_source: true
      heading_level: 3

::: dosemetrics.metrics.compare_dvh_score
    options:
      show_source: true
      heading_level: 3

The `dosemetrics.metrics.comparison` module remains importable for backward
compatibility, but direct imports are the documented invocation style.

## Comparison metadata

::: dosemetrics.metrics.comparison
    options:
      members:
        - COMPARISON_METRICS
        - EvaluationTask
        - MetricCategory
        - MetricDefinition
      show_source: false
      heading_level: 3

## `conformity`

::: dosemetrics.metrics.conformity
    options:
      show_source: true
      heading_level: 3

## `dose_comparison`

::: dosemetrics.metrics.dose_comparison
    options:
      show_source: true
      heading_level: 3

## `dvh`

::: dosemetrics.metrics.dvh
    options:
      show_source: true
      heading_level: 3

## `gamma`

::: dosemetrics.metrics.gamma
    options:
      show_source: true
      heading_level: 3

## `geometric`

::: dosemetrics.metrics.geometric
    options:
      show_source: true
      heading_level: 3

## `homogeneity`

::: dosemetrics.metrics.homogeneity
    options:
      show_source: true
      heading_level: 3
