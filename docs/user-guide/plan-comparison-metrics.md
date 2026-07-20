# Plan Metrics and Comparisons

This page defines the nine reference-based metrics exposed by
`dosemetrics.metrics.comparison`. The definitions are intentionally
explicit: names such as *conformity index*, *homogeneity index*, *dose score*,
and *DVH score* have been used for incompatible quantities in the literature.

Every comparison function starts with `compare_` and accepts
`reference` followed by `evaluated`. In contrast, functions that characterize
one plan use `compute_` and live in their clinical modules, such as
`dvh.compute_dvh_auc`, `conformity.compute_paddick_conformity_index`, and
`homogeneity.compute_homogeneity_index`.

The comparison API compares an evaluated dose with a physically realisable reference
dose. Plan-quality indices and PTV mean dose are therefore reported as
absolute distances between the two plans, not as the evaluated plan's raw
index.

## One-plan quantities and their comparisons

The raw quantity and its between-plan comparison are deliberately separate:

| One-plan quantity | API | Reference-based metric | API |
|---|---|---|---|
| Mean PTV dose | `dvh.compute_mean_dose(plan, ptv)` | PTV Dose Distance | `comparison.compare_ptv_dose(reference, evaluated, ptv)` |
| Paddick CI | `conformity.compute_paddick_conformity_index(plan, ptv, prescription)` | PCID | `comparison.compare_paddick_conformity_index(reference, evaluated, ptv, prescription)` |
| Homogeneity index | `homogeneity.compute_homogeneity_index(plan, ptv)` | HID | `comparison.compare_homogeneity_index(reference, evaluated, ptv)` |
| Paddick gradient index | `homogeneity.compute_gradient_index(plan, ptv, prescription)` | PGID | `comparison.compare_paddick_gradient_index(reference, evaluated, prescription)` |
| DVH AUC | `dvh.compute_dvh_auc(plan, structure)` | OAR DVH ABC | `comparison.compare_oar_dvh_auc(reference, evaluated, oar)` |

This prevents a raw index—whose ideal value may be 0, 1, or context
dependent—from being confused with a distance whose ideal value is 0.

## Metric catalogue

| Category | Metric | Dose prediction | Dose calculation | Better |
|---|---|:---:|:---:|---|
| Global | DVH Score | Yes | — | Lower |
| Voxel-based | Body-masked RMSE | — | Yes | Lower |
| Voxel-based | Gamma Index passing rate | — | Yes | Higher |
| PTV coverage | PTV Dose Distance | Yes | Yes | Lower |
| PTV coverage | Paddick Conformity Index Distance (PCID) | Yes | Yes | Lower |
| PTV Homogeneity | Homogeneity Index Distance (HID) | — | Yes | Lower |
| OAR sparing | Paddick Gradient Index Distance (PGID) | Yes | Yes | Lower |
| OAR sparing | OAR Constraint Disagreement | Yes | Yes | Lower |
| OAR sparing | OAR DVH Area Between Curves | Yes | Yes | Lower |

This categorisation is available programmatically:

```python
from dosemetrics.metrics import comparison

for metric in comparison.COMPARISON_METRICS:
    tasks = ", ".join(task.value for task in metric.tasks)
    print(metric.category.value, metric.name, tasks)
```

Dose calculation has one physically determined result for a fully specified
plan, so voxel-wise comparison is meaningful. Dose prediction from anatomy
alone can have multiple clinically acceptable answers; its global summary is
therefore dose-volume based. PCID, PGID, and HID should only be interpreted
when PTV Dose Distance first shows adequate target-dose agreement.

## Global: DVH Score

`comparison.compare_dvh_score` implements the complete OpenKBP definition. For targets it
uses \(D_1\), \(D_{95}\), and \(D_{99}\); for OARs it uses
\(D_{\mathrm{mean}}\) and \(D_{0.1\mathrm{cc}}\). If \(M\) is the total
number of available criteria,

$$
\mathrm{DVH\ Score}
= \frac{1}{M}\sum_{m=1}^{M}
  \left|m_{\mathrm{evaluated}}-m_{\mathrm{reference}}\right|.
$$

The result is an unweighted mean in Gy. OpenKBP defines \(D_x\) as the dose
received by at least \(x\%\) of the structure: \(D_1\), \(D_{95}\), and
\(D_{99}\) are respectively the 99th, 5th, and 1st percentiles of target
voxel dose. \(D_{0.1\mathrm{cc}}\) is the minimum dose in the hottest
0.1 cubic centimetres of an OAR.

```python
from dosemetrics.metrics import comparison

dvh_score = comparison.compare_dvh_score(
    reference,
    evaluated,
    targets=[ptv_high, ptv_low],
    oars=[brainstem, cord, parotid_left, parotid_right],
)
```

### Definitions with the same or similar name

- [Babier et al.](https://doi.org/10.1002/mp.14845) define the complete
  OpenKBP DVH score above and a separate voxel-wise MAE called *dose score*.
- Calling the complete score with one target and no OARs yields the
  target-only \(D_1/D_{95}/D_{99}\) subset. There is no second function with
  the same name and a different definition.
- [SP-DiffDose](https://arxiv.org/abs/2312.06187) reports a target-only
  \(D_1/D_{95}/D_{99}\) DVH score and separately uses a normalised relative
  dose score. It also calls dose standard deviation divided by mean dose a
  homogeneity score.
- [DoseDiff](https://arxiv.org/abs/2306.16324) uses *dose score* for more than
  one quantity. Reproducing those results therefore requires selecting the
  equation, not only the metric label.

## Voxel-based metrics

### Body-masked RMSE

`comparison.compare_body_rmse` evaluates the \(N\) voxels in the supplied body mask:

$$
\mathrm{RMSE}
= \sqrt{\frac{1}{N}\sum_{i=1}^{N}
  \left(D_{\mathrm{evaluated},i}-D_{\mathrm{reference},i}\right)^2}.
$$

It returns Gy and gives larger errors more weight than MAE.

```python
from dosemetrics.metrics import comparison

rmse = comparison.compare_body_rmse(reference, evaluated, body)
```

Do not confuse this with the OpenKBP *dose score*, which is voxel-wise MAE.
Some implementations also apply a low-dose isodose mask; those choices change
the voxel set and must be reported alongside the value.

### Gamma Index passing rate

Following [Low et al.](https://doi.org/10.1118/1.598248), for each reference
voxel \(v\),

$$
\gamma(v)
= \min_{v'}
  \sqrt{
    \frac{r^2(v,v')}{\Delta d^2}
    + \frac{\delta^2(v,v')}{\Delta D^2}
  }.
$$

Here \(r\) is physical distance and \(\delta\) is dose difference. The metric
is the percentage of evaluated voxels with \(\gamma\leq1\).
`comparison.compare_gamma` defaults to global 3%/3 mm gamma and no
low-dose exclusion. Supplying `body` restricts the reported rate to the body.

```python
from dosemetrics.metrics import comparison

passing_rate = comparison.compare_gamma(
    reference,
    evaluated,
    body=body,
)
```

Gamma results are definition-sensitive. Global versus local dose
normalisation, low-dose thresholds, interpolation, search distance, dimensional
choice, and whether \(\gamma=1\) passes must all be held constant. The generic
`dosemetrics.metrics.gamma.compare_gamma_index` defaults to a 10% low-dose
threshold; the plan-comparison wrapper deliberately defaults to 0%.

## PTV coverage

### PTV Dose Distance

For the selected high-dose PTV \(V_{\mathrm{PTV}}\),

$$
\bar D_{\mathrm{PTV}}
= \frac{1}{|V_{\mathrm{PTV}}|}
  \sum_{v\in V_{\mathrm{PTV}}}D(v),
\qquad
\mathrm{PTVDD}
= \left|
  \bar D_{\mathrm{PTV,evaluated}}
  -\bar D_{\mathrm{PTV,reference}}
  \right|.
$$

```python
from dosemetrics.metrics import comparison

ptvdd = comparison.compare_ptv_dose(reference, evaluated, ptv_high)
```

This is a between-plan distance. It is not prescription MAE, \(D_{95}\), or
the fraction of the target receiving prescription dose.

### Paddick Conformity Index Distance

Let \(V_{\mathrm{PIV}}\) be the prescription-isodose volume and
\(V_{\mathrm{PTV,PIV}}\) its intersection with the PTV. The
[Paddick index](https://doi.org/10.3171/jns.2000.93.supplement) is

$$
\mathrm{CI}_{\mathrm{Paddick}}
= \frac{V_{\mathrm{PTV,PIV}}}{V_{\mathrm{PTV}}}
  \frac{V_{\mathrm{PTV,PIV}}}{V_{\mathrm{PIV}}},
\qquad
\mathrm{PCID}
= \left|
  \mathrm{CI}_{\mathrm{evaluated}}
  -\mathrm{CI}_{\mathrm{reference}}
  \right|.
$$

```python
from dosemetrics.metrics import comparison

pcid = comparison.compare_paddick_conformity_index(
    reference, evaluated, ptv_high, prescription_dose=70.0
)
```

The conformity review by
[Feuvret et al.](https://doi.org/10.1016/j.ijrobp.2005.09.028) explains why
the label *conformity index* is insufficient. Common alternatives include:

| Definition | Formula | Main limitation |
|---|---|---|
| RTOG size ratio | \(V_{\mathrm{PIV}}/V_{\mathrm{PTV}}\) | Does not encode overlap |
| Target coverage / Lomax form | \(V_{\mathrm{PTV,PIV}}/V_{\mathrm{PTV}}\) | Can be perfect despite normal-tissue irradiation |
| Isodose purity | \(V_{\mathrm{PTV,PIV}}/V_{\mathrm{PIV}}\) | Can be perfect despite target undercoverage |
| Paddick / van't Riet conformation number | Product of coverage and purity | Used here |

The raw Paddick index is ideal at 1. PCID is instead ideal at 0 because it
measures agreement with the reference plan.

## PTV Homogeneity

### Homogeneity Index Distance

`comparison.compare_homogeneity_index` uses

$$
\mathrm{HI}=\frac{D_2-D_{98}}{D_{50}},
\qquad
\mathrm{HID}
= \left|
  \mathrm{HI}_{\mathrm{evaluated}}
  -\mathrm{HI}_{\mathrm{reference}}
  \right|.
$$

\(D_2\), \(D_{50}\), and \(D_{98}\) are the 98th, 50th, and 2nd dose
percentiles in the PTV. Both HI and HID are dimensionless; lower is better.

```python
from dosemetrics.metrics import comparison

hid = comparison.compare_homogeneity_index(
    reference, evaluated, ptv_high
)
```

[Kataria et al.](https://doi.org/10.4103/0971-6203.103606) compare five other
published or historically used definitions:

| Variant | Formula | Ideal |
|---|---|---|
| A | \(D_5/D_{95}\) | 1 |
| B | \(D_{\max}/D_{\min}\) | 1 |
| C | \((D_1-D_{98})/D_p\times100\) | 0 |
| D | \((D_5-D_{95})/D_p\times100\) | 0 |
| E / RTOG | \(D_{\max}/D_p\) | 1 |

The same article also describes the widely used
\((D_2-D_{98})/D_p\times100\) form. These variants differ in denominator,
percentiles, scaling, sensitivity to isolated voxels, and even ideal value.
They must not be substituted for \((D_2-D_{98})/D_{50}\) when computing HID.
The coefficient-of-variation definition used by SP-DiffDose,
\(\sigma_D/\bar D\), is another distinct quantity exposed in dosemetrics as
`compute_dose_homogeneity`.

## OAR sparing

### Paddick Gradient Index Distance

The [Paddick–Lippitz gradient
index](https://doi.org/10.3171/sup.2006.105.7.194) is

$$
\mathrm{GI}=\frac{V_{D_{\mathrm{Rx}}/2}}{V_{D_{\mathrm{Rx}}}},
\qquad
\mathrm{PGID}
= \left|
  \mathrm{GI}_{\mathrm{evaluated}}
  -\mathrm{GI}_{\mathrm{reference}}
  \right|.
$$

The volumes are taken over the dose grid, not only inside the PTV. A lower raw
GI means steeper dose falloff; a lower PGID means closer agreement with the
reference.

```python
from dosemetrics.metrics import comparison

pgid = comparison.compare_paddick_gradient_index(
    reference, evaluated, prescription_dose=70.0
)
```

### OAR Constraint Disagreement

For binary satisfaction states \(s_c,\hat s_c\in\{0,1\}\),

$$
\mathrm{Disagreement}
= \frac{1}{N}\sum_{c=1}^{N}
  \mathbb{I}\!\left[\hat s_c\ne s_c\right].
$$

The head-and-neck protocol uses \(N=38\) organ-dose constraints selected from
the practical compilation by
[Bisello et al. (CORSAIR)](https://doi.org/10.3390/curroncol29100552).
Because constraint applicability depends on fractionation, contour naming,
available structures, and clinical context, the function consumes resolved
pass/fail states rather than guessing them from a dose alone.

```python
from dosemetrics.metrics import comparison

# Each mapping contains the same 38 CORSAIR-derived constraint identifiers.
disagreement = comparison.compare_oar_constraints(
    reference_satisfaction,
    evaluated_satisfaction,
)
```

The function validates identical mapping keys and, by default, exactly 38
states. Pass `expected_count=None` only for a deliberately different protocol.
CORSAIR itself cautions that dose-volume constraints require expert
interpretation and are not universal automatic rules.

### OAR DVH Area Between Curves

For one OAR, cumulative DVH volume is represented as a fraction
\(V(D)\in[0,1]\). Both plans use the same 100-point dose grid:

$$
\mathrm{AUC}=\int_{D_{\min}}^{D_{\max}}V(D)\,dD,
\qquad
\mathrm{ABC}
= \left|
  \mathrm{AUC}_{\mathrm{evaluated}}
  -\mathrm{AUC}_{\mathrm{reference}}
  \right|.
$$

The trapezoidal rule approximates each AUC. The result is reported per OAR in
Gy; `comparison.compare_mean_oar_dvh_auc` provides an unweighted mean over
several OARs.

```python
from dosemetrics.metrics import comparison

brainstem_abc = comparison.compare_oar_dvh_auc(
    reference, evaluated, brainstem
)
mean_abc = comparison.compare_mean_oar_dvh_auc(
    reference, evaluated, oars
)
```

This definition is not the pointwise L1 area
\(\int|V_{\mathrm{evaluated}}(D)-V_{\mathrm{reference}}(D)|\,dD\), and it is
not the L2 curve distance. Those useful variants remain available through
`dosemetrics.metrics.dvh.compare_dvh_area`.

## Complete example

```python
from dosemetrics.metrics import comparison

results = {
    "dvh_score_gy": comparison.compare_dvh_score(
        reference, evaluated, targets=targets, oars=oars
    ),
    "rmse_gy": comparison.compare_body_rmse(
        reference, evaluated, body
    ),
    "gamma_pass_percent": comparison.compare_gamma(
        reference, evaluated, body=body
    ),
    "ptv_dose_distance_gy": comparison.compare_ptv_dose(
        reference, evaluated, ptv_high
    ),
    "pcid": comparison.compare_paddick_conformity_index(
        reference, evaluated, ptv_high, prescription
    ),
    "hid": comparison.compare_homogeneity_index(
        reference, evaluated, ptv_high
    ),
    "pgid": comparison.compare_paddick_gradient_index(
        reference, evaluated, prescription
    ),
    "constraint_disagreement": comparison.compare_oar_constraints(
        reference_satisfaction, evaluated_satisfaction
    ),
    "mean_oar_dvh_abc_gy": comparison.compare_mean_oar_dvh_auc(
        reference, evaluated, oars
    ),
}
```
