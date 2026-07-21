# Design Prompts for the Benchmark Metrics

This document is the visual specification for the evaluation framework in
`local/metrics.tex`. It contains exactly the nine metrics defined there. Do
not add alternative formulas, clinical acceptance thresholds, normalisations,
or similarly named library metrics to the generated designs.

Dose calculation has a single ground truth, so its framework includes
voxel-wise accuracy metrics. Dose prediction is ill-posed and excludes
voxel-wise metrics. Plan-quality indices and PTV mean dose are visualised as
absolute prediction–target distances, not as distance from a geometrically
ideal value, because the target is a physically realisable plan.

## Canonical notation and visual rules

- **Target** is the realisable reference or ground-truth dose and uses the
  subscript `target`.
- **Prediction** is the predicted or evaluated dose and uses the subscript
  `pred`.
- In Python examples elsewhere, `reference = target` and `evaluated = pred`.
- Use (V_{\mathrm{PTV}}) for the high-dose PTV, (V_{\mathrm{PIV}}) for
  the prescription isodose volume, and (V_{\mathrm{PTV,PIV}}) for their
  intersection.
- Use (D_{\mathrm{Rx}}) for prescribed dose. (V_{D_{\mathrm{Rx}}/2})
  and (V_{D_{\mathrm{Rx}}}) are volumes receiving at least half and all of
  the prescribed dose, respectively.
- Use a cumulative DVH, so (V(D)) is the volume fraction receiving at least
  dose (D).
- Lower is better for every distance/error/disagreement metric. Higher is
  better only for the gamma passing rate.
- PCID, PGID, and HID are valid only where PTV Dose Distance indicates
  acceptable target coverage. Every design for one of these three metrics
  must show this condition.

### Task assignment

| Metric | Dose prediction | Dose calculation | Reporting role |
|---|:---:|:---:|---|
| PTV Dose Distance | Yes | Yes | Primary |
| PCID | Yes | Yes | PTV-specific, coverage-gated |
| PGID | Yes | Yes | PTV-specific, coverage-gated |
| OAR Constraint Disagreement | Yes | Yes | Primary |
| OAR DVH Area Between Curves | Yes | Yes | Primary |
| HID | No | Yes | PTV-specific, coverage-gated |
| Body-mask RMSE | No | Yes | Primary |
| Gamma Index passing rate | No | Yes | Primary |
| DVH Score | Yes | No | Primary |

### Shared visual theme

- White background (`#FFFFFF`).
- University of Bern red (`#CC0000`) for the prediction; black for the
  target/reference; pale red (`#FFCCCC`) for prediction-related fills.
- Solid black target curve or contour; dashed red prediction curve or
  contour. Never label the black curve “prediction”.
- Light-grey guides (`#EEEEEE`), dark-grey axes, black anatomy labels, and
  LaTeX-style equations.
- Put the canonical equation visibly in every design. Use the exact symbols
  and subscripts shown below.
- Avoid example “good”, “acceptable”, or “clinical pass” cutoffs that are not
  defined in `metrics.tex`.

## 1. PTV Dose Distance — both tasks

**Canonical definition**

For the high-dose PTV region (V_{\mathrm{PTV}}),

$$
\bar D_{\mathrm{PTV}}
=\frac{1}{|V_{\mathrm{PTV}}|}
\sum_{v\in V_{\mathrm{PTV}}}D(v),
\qquad
\text{PTV Dose Distance}
=\left|\bar D_{\mathrm{PTV,pred}}
-\bar D_{\mathrm{PTV,target}}\right|.
$$

**Prompt**

> Create a clean radiotherapy evaluation diagram titled “PTV Dose
> Distance”. Show the same high-dose PTV mask for a target dose panel and a
> prediction dose panel. In each panel, use a dose heatmap only inside the PTV
> and annotate the mean as \(\bar D_{\mathrm{PTV,target}}\) or
> \(\bar D_{\mathrm{PTV,pred}}\). Between the panels, show an absolute-value
> bracket measuring the difference between those two means. Include the two
> canonical equations exactly as written above. Label it “both tasks”, “Gy”,
> and “lower is better”. Do not depict prescription-dose MAE, \(D_{95}\), or
> a coverage fraction; those are different quantities.

## 2. Paddick Conformity Index Distance (PCID) — both tasks

**Canonical definition**

$$
\mathrm{CI}
=\frac{V_{\mathrm{PTV,PIV}}}{V_{\mathrm{PTV}}}
\cdot
\frac{V_{\mathrm{PTV,PIV}}}{V_{\mathrm{PIV}}},
\qquad
\mathrm{PCID}
=\left|\mathrm{CI}_{\mathrm{pred}}
-\mathrm{CI}_{\mathrm{target}}\right|.
$$

Here (V_{\mathrm{PIV}}) is the prescription isodose volume and
(V_{\mathrm{PTV,PIV}}) is its intersection with the PTV.

**Prompt**

> Create a two-panel diagram titled “Paddick Conformity Index Distance
> (PCID)”. In each panel draw a solid black PTV contour and a prescription
> isodose contour: solid black for the target/reference plan in the left panel
> and dashed red for the prediction in the right panel. Clearly shade and
> label \(V_{\mathrm{PTV}}\), \(V_{\mathrm{PIV}}\), and their intersection
> \(V_{\mathrm{PTV,PIV}}\). Under each panel compute the same Paddick CI
> product, then connect the two values with
> \(\mathrm{PCID}=|\mathrm{CI}_{\mathrm{pred}}-\mathrm{CI}_{\mathrm{target}}|\).
> Include “both tasks”, “dimensionless”, and “lower is better”. Add a visible
> gate banner: “Interpret only when PTV Dose Distance indicates acceptable
> coverage.” Do not substitute the RTOG size ratio, coverage alone, or purity
> alone.

## 3. Paddick Gradient Index Distance (PGID) — both tasks

**Canonical definition**

$$
\mathrm{GI}=\frac{V_{D_{\mathrm{Rx}}/2}}{V_{D_{\mathrm{Rx}}}},
\qquad
\mathrm{PGID}
=\left|\mathrm{GI}_{\mathrm{pred}}
-\mathrm{GI}_{\mathrm{target}}\right|.
$$

**Prompt**

> Create a two-panel axial isodose diagram titled “Paddick Gradient Index
> Distance (PGID)”. For target and prediction separately, draw an inner full-
> prescription isodose volume labelled \(V_{D_{\mathrm{Rx}}}\) and a larger
> half-prescription volume labelled \(V_{D_{\mathrm{Rx}}/2}\). State that
> both are threshold volumes over the dose grid: dose at least
> \(D_{\mathrm{Rx}}\) and at least \(D_{\mathrm{Rx}}/2\). Show the ratio below
> each panel and the absolute prediction–target GI difference between them.
> Include “both tasks”, “dimensionless”, and “lower is better”. Add the gate
> banner: “Interpret only when PTV Dose Distance indicates acceptable
> coverage.” Do not put the volumes only inside the PTV.

## 4. OAR Constraint Disagreement — both tasks

**Canonical definition**

For the (N=38) head-and-neck organ-dose constraints from CORSAIR, let
(s_c,\hat s_c\in\{0,1\}) be target/reference and predicted satisfaction:

$$
\mathrm{Disagreement}
=\frac{1}{N}\sum_{c=1}^{N}
\mathbb I\!\left[\hat s_c\ne s_c\right],
\qquad N=38.
$$

**Prompt**

> Create a diagram titled “OAR Constraint Disagreement”. Draw 38 aligned
> rows, one per head-and-neck CORSAIR organ-dose constraint. Give each row a
> target/reference state \(s_c\) and predicted state \(\hat s_c\), each
> explicitly binary: 1 = satisfied, 0 = not satisfied. Highlight only rows
> where the two states differ and show their count divided by 38. Display the
> indicator-sum equation exactly. Include “both tasks”, “fraction in [0,1]”,
> and “lower is better”. Do not infer or draw an additional clinical
> acceptance threshold for the aggregate disagreement.

## 5. OAR DVH Area Between Curves — both tasks

**Canonical definition**

For each OAR,

$$
\mathrm{AUC}=\int_{D_{\min}}^{D_{\max}}V(D)\,dD,
\qquad
\text{OAR DVH ABC}
=\left|\mathrm{AUC}_{\mathrm{pred}}
-\mathrm{AUC}_{\mathrm{target}}\right|.
$$

Each AUC is approximated by the trapezoidal rule over 100 dose bins, and
(V(D)) is the volume fraction receiving at least dose (D).

**Prompt**

> Create a two-stage diagram titled “OAR DVH Area Between Curves”. On common
> axes from \(D_{\min}\) to \(D_{\max}\), draw the target OAR cumulative DVH
> as a solid black curve and the predicted OAR cumulative DVH as a dashed red
> curve. The y-axis is \(V(D)\), volume fraction receiving at least dose
> \(D\). Show 100 shared dose-bin markers. In separate small panels, fill the
> area under each curve and label it \(\mathrm{AUC}_{\mathrm{target}}\) or
> \(\mathrm{AUC}_{\mathrm{pred}}\), noting trapezoidal approximation. Then
> show the absolute difference of those two scalar areas. Include “per OAR”,
> “both tasks”, “Gy”, and “lower is better”. Critical: do not shade and
> integrate the pointwise absolute gap
> \(\int|V_{\mathrm{pred}}-V_{\mathrm{target}}|dD\); that is not this
> benchmark definition.

## 6. Homogeneity Index Distance (HID) — dose calculation only

**Canonical definition**

$$
\mathrm{HI}=\frac{D_2-D_{98}}{D_{50}},
\qquad
\mathrm{HID}
=\left|\mathrm{HI}_{\mathrm{pred}}
-\mathrm{HI}_{\mathrm{target}}\right|.
$$

**Prompt**

> Create a two-curve high-dose PTV DVH diagram titled “Homogeneity Index
> Distance (HID)”. Use a solid black target curve and dashed red prediction
> curve. On both curves mark \(D_2\), \(D_{50}\), and \(D_{98}\), where
> \(D_x\) is the dose received by at least \(x\%\) of PTV volume. Under each
> curve compute \((D_2-D_{98})/D_{50}\), then show the absolute difference of
> the two HI values. Include “dose calculation only”, “dimensionless”, and
> “lower is better”. Add the gate banner: “Interpret only when PTV Dose
> Distance indicates acceptable coverage.” Do not use prescription dose,
> \(D_5/D_{95}\), or dose standard deviation as the HI definition.

## 7. Body-mask RMSE — dose calculation only

**Canonical definition**

$$
\mathrm{RMSE}
=\sqrt{\frac{1}{N}\sum_{i=1}^{N}
\left(D_{\mathrm{pred},i}-D_{\mathrm{target},i}\right)^2},
$$

where (N) is the number of body-mask voxels.

**Prompt**

> Create a diagram titled “Body-mask Root Mean Squared Error”. Show aligned
> target and prediction dose heatmaps inside the same body mask. Build a third
> panel showing voxel-wise \(D_{\mathrm{pred},i}-D_{\mathrm{target},i}\),
> then visually square the differences, average them over exactly the \(N\)
> body-mask voxels, and take the square root. Put the canonical equation below.
> Include “dose calculation only”, “Gy”, “lower is better”, and “quadratic
> weighting gives heightened sensitivity to large errors”. Do not depict MAE
> or include voxels outside the body mask.

## 8. Gamma Index passing rate — dose calculation only

**Canonical definition**

For each reference voxel (v),

$$
\gamma(v)=\min_{v'}
\sqrt{
\frac{r^2(v,v')}{\Delta d^2}
+\frac{\delta^2(v,v')}{\Delta D^2}
},
\qquad
(\Delta d,\Delta D)=(3\ \mathrm{mm},3\%).
$$

Report the percentage of voxels satisfying
(\gamma(v)\leq1).

**Prompt**

> Create a diagram titled “Gamma Index Passing Rate (3 mm / 3%)”. In a
> neighbourhood around one reference voxel \(v\), show candidate prediction
> voxels \(v'\). For each candidate illustrate the spatial distance
> \(r(v,v')\) and dose difference \(\delta(v,v')\); identify the candidate
> that minimises the quadratic expression. Beside it, show a gamma map over
> reference voxels with a binary legend: pass when \(\gamma(v)\leq1\), fail
> when \(\gamma(v)>1\). Report the passing rate as a percentage. Include
> “dose calculation only” and “higher is better”. Use exactly the 3 mm / 3%
> criteria. Do not add a 95% clinical acceptance line, a low-dose exclusion,
> or a local/global normalisation claim; these are not specified by the
> canonical equation.

## 9. DVH Score — dose prediction only

**Canonical definition**

Let the complete OpenKBP set of (M) dose-volume criteria contain
(D_1,D_{95},D_{99}) for every target and
(D_{\mathrm{mean}},D_{0.1\mathrm{cc}}) for every OAR:

$$
\text{DVH Score}
=\frac{1}{M}\sum_{m=1}^{M}
\left|m_{\mathrm{pred}}-m_{\mathrm{target}}\right|.
$$

**Prompt**

> Create a diagram titled “OpenKBP DVH Score”. Split it into a target panel
> and an OAR panel. In the target panel, draw target and prediction cumulative
> DVHs and mark \(D_1\), \(D_{95}\), and \(D_{99}\) for every target. In the
> OAR panel, mark \(D_{\mathrm{mean}}\) and \(D_{0.1\mathrm{cc}}\) for every
> OAR. Collect all absolute prediction–target differences into one list of
> \(M\) criteria and show their unweighted mean using the canonical equation.
> Include “dose prediction only”, “Gy”, and “lower is better”. Do not depict
> a target-only three-point score, voxel-wise dose score, or normalised
> relative score as the complete benchmark metric.

## Final consistency check for generated designs

Before accepting any generated graphic, verify all of the following:

1. It defines one of the nine metrics above and no alternative metric.
2. `pred` always means prediction and `target` always means the realisable
   reference/ground truth.
3. The displayed task assignment matches the table above.
4. OAR DVH ABC is
   \(|\mathrm{AUC}_{\mathrm{pred}}-\mathrm{AUC}_{\mathrm{target}}|\), not a
   pointwise L1 integral.
5. DVH Score contains both the complete target and OAR criterion sets.
6. Gamma uses 3 mm / 3%, minimises over (v'), and passes at
   \(\gamma(v)\leq1\).
7. PCID, PGID, and HID carry the PTV Dose Distance validity gate.
