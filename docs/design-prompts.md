# Design Prompts for Metric Illustrations

This document contains detailed prompts for [claude.ai/design](https://claude.ai/design) to generate explanatory graphics for dosemetrics metrics that are not yet covered by the slide visuals from the Joseph Weibel MSc Thesis Defense (University of Bern).

## Visual Theme Reference

All graphics should match the aesthetic of the existing slide illustrations:

- **Background:** white (`#FFFFFF`)
- **Primary accent:** deep red (`#CC0000` / University of Bern red)
- **Secondary fills:** light red / pink (`#FFCCCC` or `rgba(204,0,0,0.15)`)
- **Structure contours:** solid black lines, 2 px
- **Prescription isodose contours:** dashed red lines
- **Text labels:** black for anatomy labels; dark red for formula annotations
- **Grid lines:** light grey (`#EEEEEE`), subtle
- **Charts:** axes in dark grey with tick marks; curve in solid red (target), dashed black (prediction/reference)
- **Formula style:** serif math font (LaTeX-style), black, below the diagram
- **Layout:** diagram on the left or centre, formula below or to the right; single title in bold sans-serif at top

---

## Group 1 — DVH Point Metrics

### 1a. Volume at Dose (VX) — `compute_volume_at_dose`

**Prompt:**

> Create a clean educational diagram explaining the "Volume at Dose" (VX) DVH metric for radiotherapy documentation.
>
> **Left panel — DVH diagram:**
> Draw a smooth cumulative dose-volume histogram curve in solid red. X-axis labelled "DOSE (Gy)", Y-axis labelled "VOLUME (%)". Mark a vertical dashed line at a specific dose value "X Gy" on the x-axis. Draw a horizontal dashed line from the intersection of the DVH curve and the vertical line across to the y-axis, landing at a value labelled "VX (%)". Shade the area to the left of the vertical line under the DVH curve in light red. Add annotation arrows pointing to: the DVH curve labelled "cumulative DVH", the intersection point labelled "read-off point", and the y-axis landing point labelled "VX".
>
> **Right panel — formula:**
> Display the formula: V_X = DVH(X Gy) = |{v ∈ V_structure : d_v ≥ X}| / |V_structure|
> Below the formula, add a one-line note: "VX is the fraction of the structure receiving at least X Gy."
>
> **Style:** white background, red curve, University of Bern red (#CC0000), sans-serif axis labels, LaTeX-style formula. Title: "Volume at Dose (VX)".

---

### 1b. Dose at Volume (DX) — `compute_dose_at_volume`

**Prompt:**

> Create a clean educational diagram explaining the "Dose at Volume" (DX) DVH metric for radiotherapy documentation.
>
> **Left panel — DVH diagram:**
> Draw a smooth cumulative DVH curve in solid red. X-axis "DOSE (Gy)", Y-axis "VOLUME (%)". Mark a horizontal dashed line at a volume fraction "X %" on the y-axis. Draw a vertical dashed line from the intersection of the DVH curve and the horizontal line down to the x-axis, landing at a value labelled "DX (Gy)". Shade the region above the horizontal line (the top X% of the volume) in light red. Add annotation arrows: DVH curve labelled "cumulative DVH"; y-axis level labelled "X% of volume"; x-axis landing point labelled "DX".
>
> **Right panel — formula:**
> DX = min{d : DVH(d) ≤ X/100}
> Below: "DX is the minimum dose received by at least X% of the structure volume."
> Add a small example table:
> | Metric | Volume fraction | Meaning |
> |--------|----------------|---------|
> | D95    | 95%            | Near-minimum (coverage dose) |
> | D50    | 50%            | Median dose |
> | D2     | 2%             | Near-maximum dose |
>
> **Style:** white background, University of Bern red (#CC0000). Title: "Dose at Volume (DX)".

---

## Group 2 — Dose Summary Statistics

### 2a. Mean, Max, Min, Median Dose

**Prompt:**

> Create a four-panel educational diagram showing four DVH-derived dose statistics (mean, max, min, median) for radiotherapy documentation.
>
> Use a single representative cumulative DVH curve (smooth S-shape in red) shown in each panel. White background with light grey grid lines.
>
> **Panel 1 (Mean Dose, D̄):** Shade the entire DVH area under the curve in light red. Mark the mean dose on the x-axis with a vertical dashed line labelled "D̄". Formula below: D̄ = (1/N) Σ dᵢ
>
> **Panel 2 (Max Dose, Dmax):** Mark the rightmost non-zero point of the DVH curve (where volume approaches 0%) with a vertical dashed red line labelled "Dmax". Annotate: "D0 — dose at 0% volume".
>
> **Panel 3 (Min Dose, Dmin):** Mark the leftmost point where the DVH drops below 100% with a vertical dashed red line labelled "Dmin". Annotate: "D100 — dose at 100% volume".
>
> **Panel 4 (Median Dose, D50):** Mark the horizontal line at 50% volume and drop a vertical dashed red line to the x-axis labelled "D50". This is identical to the DX diagram at X=50%.
>
> **Layout:** 2×2 grid of panels. Each panel has the same axes ("DOSE (Gy)" / "VOLUME (%)"). Title above each panel: "Mean Dose", "Max Dose", "Min Dose", "Median Dose". Overall title: "DVH Dose Statistics". University of Bern red (#CC0000), sans-serif.

---

### 2b. Equivalent Uniform Dose (EUD) — `compute_equivalent_uniform_dose`

**Prompt:**

> Create an educational diagram explaining Equivalent Uniform Dose (EUD) for radiotherapy documentation.
>
> **Left panel — concept illustration:**
> Draw two stacked panels:
> - Top: A pixelated/voxelised cross-section of a target structure (circle or oval). Each pixel is shaded from light to dark red representing varying dose levels (heterogeneous distribution). Label individual pixel doses: some at 55 Gy, some at 65 Gy, some at 60 Gy. Label the structure "heterogeneous dose distribution". Add a small colour bar below labelled "DOSE (Gy)".
> - Bottom: The same target structure, but uniformly shaded in a single medium red. Label it "uniform EUD distribution". Add a label "EUD = equivalent uniform dose".
> Connect the two panels with a double-headed arrow labelled "biologically equivalent".
>
> **Right panel — formula:**
> EUD = (1/N · Σ dᵢᵃ)^(1/a)
> Add a table:
> | Tissue type | Parameter a | Interpretation |
> |-------------|-------------|---------------|
> | Serial OAR  | a >> 1      | Sensitive to hot spots |
> | Tumour      | a < 0       | Sensitive to cold spots |
> | Parallel OAR| a ≈ 1       | Mean dose |
>
> **Style:** white background, University of Bern red (#CC0000). Title: "Equivalent Uniform Dose (EUD)".

---

## Group 3 — Conformity Variations

### 3a. ICRU Conformity Index — `compute_conformity_index`

**Prompt:**

> Create an educational diagram explaining the ICRU Conformity Index (CI) for radiotherapy documentation.
>
> **Diagram:**
> Draw two concentric shapes on a white background with light grey grid lines:
> - An irregular solid-black-outline shape labelled "Target (V_target)" — slightly oval.
> - Overlapping this, a dashed red boundary shape labelled "Prescription isodose (V_rx)" that is slightly offset and larger than the target on one side, smaller on another.
> - Shade the overlap region (intersection) in medium red and label it "V_target_rx (overlap)".
> - Shade the prescription isodose region outside the target in light red/pink and label it "dose outside target".
> - Shade the target region outside the prescription isodose in light grey and label it "underdosed target".
>
> **Formula below:**
> CI_ICRU = V_target_rx / V_rx
> Add interpretation: "CI = 1.0: all irradiated tissue is inside the target (ideal). CI < 1.0: dose spills outside the target."
>
> **Style:** white background, University of Bern red (#CC0000). Title: "ICRU Conformity Index (CI)".

---

### 3b. RTOG Conformity Index — `compute_rtog_conformity_index`

**Prompt:**

> Create an educational diagram explaining the RTOG Conformity Index for radiotherapy documentation.
>
> **Diagram:**
> Draw two concentric shapes on white background with grey grid:
> - Solid-black-outline oval labelled "Target (V_target)", size reference = 1.
> - Dashed red outline of a larger oval labelled "Prescription isodose (V_rx)".
> - Shade V_rx in light red. Shade V_target within it in medium red.
> - Add bidirectional arrows: one for the diameter of V_target, one for the diameter of V_rx.
> - Label the ratio: "RTOG CI = V_rx / V_target = 1.8" as an example.
>
> **Formula below:**
> RTOG CI = V_rx / V_target
> Add RTOG interpretation table:
> | RTOG CI | Grade |
> |---------|-------|
> | 1.0–2.0 | Acceptable |
> | < 1.0 or 2.0–2.5 | Minor deviation |
> | > 2.5 | Major deviation |
>
> **Style:** white background, University of Bern red (#CC0000). Title: "RTOG Conformity Index".

---

### 3c. Coverage and Spillage — `compute_coverage` / `compute_spillage`

**Prompt:**

> Create a side-by-side educational diagram showing Coverage and Spillage for radiotherapy documentation.
>
> **Left panel — Coverage:**
> Draw an irregular oval (target V_target, solid black outline). Inside the oval, shade 85% of the area in medium red and label it "covered (V_target_rx)". Leave a small white crescent along one edge labelled "underdosed (not reached by Rx isodose)". Below: Coverage = V_target_rx / V_target = 0.85
>
> **Right panel — Spillage:**
> Draw the same target oval (solid black). Surround it with a larger dashed red outline (prescription isodose V_rx). Shade the area inside V_rx but outside V_target in pink, labelled "dose spill (V_rx − V_target_rx)". Shade the overlap in medium red, labelled "V_target_rx". Below: Spillage = (V_rx − V_target_rx) / V_rx
> Add note: "Coverage + Spillage complement each other. High coverage is desired; low spillage is desired."
>
> **Style:** white background, University of Bern red (#CC0000). Overall title: "Coverage and Spillage".

---

## Group 4 — Advanced Homogeneity

### 4a. Coefficient of Variation — `compute_dose_homogeneity`

**Prompt:**

> Create an educational diagram explaining the Coefficient of Variation (CV) as a dose homogeneity metric for radiotherapy documentation.
>
> **Left panel — histogram:**
> Draw a bell-curve histogram of voxel doses within a target, x-axis "DOSE (Gy)", y-axis "NUMBER OF VOXELS". Mark the mean dose (μ) with a vertical solid red line. Mark one standard deviation (σ) on each side with dashed red lines. Shade the σ band in light red. Label μ and σ explicitly.
>
> **Right panel — formula:**
> CV = σ / μ
> where σ = standard deviation of dose in the structure, μ = mean dose.
> Add comparison table:
> | CV value | Interpretation |
> |----------|---------------|
> | < 0.03   | Highly homogeneous |
> | 0.03–0.10 | Acceptable |
> | > 0.10   | Inhomogeneous |
>
> **Style:** white background, University of Bern red (#CC0000). Title: "Coefficient of Variation (Dose Homogeneity)".

---

### 4b. Uniformity Index — `compute_uniformity_index`

**Prompt:**

> Create an educational diagram explaining the Uniformity Index (UI) for radiotherapy documentation.
>
> **Left panel — DVH diagram:**
> Draw a steep cumulative DVH curve (near-ideal homogeneous plan) in solid red. Mark on the x-axis: Dmin (leftmost point at 100% volume), Dref (median, middle), and Dmax (rightmost point at 0% volume). Draw vertical dashed lines at Dmin and Dmax. Shade the range between Dmin and Dmax in light red along the x-axis. Label the span "(Dmax − Dmin)".
>
> **Right panel — formula:**
> UI = 1 − (Dmax − Dmin) / Dref
> Add note: "UI = 1.0: perfectly uniform (Dmax = Dmin = Dref). UI < 1: dose spread within the target."
>
> **Style:** white background, University of Bern red (#CC0000). Title: "Uniformity Index (UI)".

---

## Group 5 — Geometric Overlap Metrics

### 5a. Dice Coefficient and Jaccard Index

**Prompt:**

> Create a side-by-side educational diagram explaining the Dice Similarity Coefficient (DSC) and Jaccard Index (IoU) for radiotherapy structure comparison.
>
> **Shared diagram (centre, used for both):**
> Draw two partially overlapping circles/ovals:
> - Left oval: solid black outline, labelled "Structure A (reference)", lightly shaded grey.
> - Right oval: dashed red outline, labelled "Structure B (predicted)", lightly shaded pink.
> - Overlap region: shaded in medium red, labelled "|A ∩ B|".
> - Add labels for the non-overlapping portions: "|A \ B|" (left) and "|B \ A|" (right).
> - Add total area labels "|A|" and "|B|".
>
> **Left formula panel:**
> DSC = 2|A ∩ B| / (|A| + |B|)
> "Range: 0 (no overlap) to 1 (perfect match). Ideal: DSC = 1.0."
>
> **Right formula panel:**
> Jaccard = |A ∩ B| / |A ∪ B|
> "Relation: Jaccard = DSC / (2 − DSC)."
>
> **Style:** white background, University of Bern red (#CC0000). Title: "Dice Coefficient and Jaccard Index".

---

### 5b. Volume Difference and Volume Ratio

**Prompt:**

> Create an educational diagram showing Volume Difference and Volume Ratio for structure comparison in radiotherapy.
>
> **Left panel — Volume Difference:**
> Draw two structures (ovals) side by side, one larger (reference, solid black) labelled "V_ref = 120 cc" and one smaller (predicted, dashed red) labelled "V_pred = 95 cc". Below: ΔV = |V_ref − V_pred| = 25 cc.
>
> **Right panel — Volume Ratio:**
> Draw the same two structures. Below: VR = V_pred / V_ref = 95/120 = 0.79. Add note: "VR = 1.0: same volume. VR < 1: under-segmentation. VR > 1: over-segmentation."
>
> **Style:** white background, University of Bern red (#CC0000). Title: "Volume Difference and Volume Ratio".

---

## Group 6 — Surface Distance Metrics

### 6a. Hausdorff Distance — `compute_hausdorff_distance`

**Prompt:**

> Create an educational diagram explaining the Hausdorff Distance (HD) for surface-to-surface structure comparison in radiotherapy.
>
> **Diagram:**
> Draw two irregular closed contours on a white background with grid lines:
> - Solid black contour: "Reference structure".
> - Dashed red contour: "Predicted structure" — slightly offset, with one protrusion that extends further from the reference.
> - Draw multiple arrows from points on the reference contour to the nearest point on the predicted contour (and vice versa). Show the arrows as thin grey.
> - Highlight the single longest such minimum-distance arrow in solid red and bold, labelled "Hausdorff Distance (HD) = max of all nearest-neighbour distances".
>
> **Formula below:**
> HD(A, B) = max(sup_{a∈A} inf_{b∈B} d(a,b), sup_{b∈B} inf_{a∈A} d(a,b))
> Add note: "HD is determined by the single worst-case surface mismatch. Sensitive to outliers."
>
> **Style:** white background, University of Bern red (#CC0000). Title: "Hausdorff Distance".

---

### 6b. Mean Surface Distance — `compute_mean_surface_distance`

**Prompt:**

> Create an educational diagram explaining Mean Surface Distance (MSD) for structure comparison in radiotherapy.
>
> **Diagram:**
> Draw two irregular close contours on white background with grid:
> - Solid black contour: "Reference".
> - Dashed red contour: "Predicted" — slightly offset uniformly.
> - Draw multiple evenly spaced arrows from points on the reference contour to the nearest point on the predicted contour. Label a few arrow lengths: "d₁ = 1.2 mm", "d₂ = 0.8 mm", "d₃ = 1.5 mm", etc. All arrows thin grey, equal weight (contrast with HD where one is highlighted).
> - Shade the gap between the two contours in light red.
>
> **Formula below:**
> MSD = (1/|∂A|) Σ_{a∈∂A} inf_{b∈∂B} d(a,b)
> Add note: "MSD averages all surface distances — robust to small outliers. Lower is better."
>
> **Style:** white background, University of Bern red (#CC0000). Title: "Mean Surface Distance (MSD)".

---

## Group 7 — Sensitivity and Specificity

### 7a. Sensitivity and Specificity — `compute_sensitivity` / `compute_specificity`

**Prompt:**

> Create an educational diagram explaining Sensitivity and Specificity in the context of binary structure segmentation for radiotherapy.
>
> **Left panel — 2×2 confusion matrix:**
> Draw a 2×2 grid:
> - Columns: "Reference: Positive (in structure)" | "Reference: Negative (outside)"
> - Rows: "Predicted: Positive" | "Predicted: Negative"
> - Fill cells: TP (medium red, top-left), FP (light pink, top-right), FN (light grey, bottom-left), TN (white, bottom-right).
> - Label each cell: "TP (true positive)", "FP (false positive)", "FN (false negative)", "TN (true negative)".
>
> **Right panel — formulas:**
> Sensitivity (Recall) = TP / (TP + FN)
> Specificity = TN / (TN + FP)
> Add interpretation: "Sensitivity = fraction of reference structure voxels correctly identified. Specificity = fraction of non-structure voxels correctly excluded."
>
> **Style:** white background, University of Bern red (#CC0000). Title: "Sensitivity and Specificity".

---

## Group 8 — Gamma Analysis Variants

### 8a. Gamma Passing Rate — `compute_gamma_passing_rate`

**Prompt:**

> Create an educational diagram explaining the Gamma Passing Rate for dose distribution comparison in radiotherapy.
>
> **Left panel — gamma map:**
> Draw a circular "body mask" cross-section filled with a 2D heatmap of gamma index values. Voxels with γ < 1.0 should be shaded white-to-light-red (passing). Voxels with γ ≥ 1.0 (failing) should be shaded in dark solid red. Show ~10% of voxels in dark red as failing. Add a colour bar labelled "γ value": light = 0, dark = ≥ 1.
>
> **Right panel — formula:**
> Pass Rate = (1/N) Σᵢ 1[γ(vᵢ) ≤ 1] × 100%
> Add note: "A voxel passes if its gamma index γ ≤ 1.0. The passing rate is the percentage of all evaluated voxels that pass." Add a typical clinical threshold: "Clinical criterion: Pass Rate ≥ 95% for 3%/3mm".
>
> **Style:** white background, University of Bern red (#CC0000). Title: "Gamma Passing Rate".

---

### 8b. 2D Gamma — `compare_2d_gamma`

**Prompt:**

> Create an educational diagram explaining 2D Gamma Index analysis for per-slice dose QA in radiotherapy.
>
> **Left panel — 2D slice view:**
> Draw a 2D cross-section grid (10×10 cells). Fill each cell with a colour from white (low dose) to medium red (high dose) representing a dose distribution. Overlay thin black circles at each cell centre representing the "distance-to-agreement" search radius for a specific voxel. Highlight one central voxel in bold red outline, with a larger dashed red circle showing the DTA search radius Δd. Label: "centre voxel v" and "search radius Δd".
>
> **Right panel — formula:**
> γ₂D(v) = min_{v'} √( r²(v,v') / (Δd)² + δ²(v,v') / (ΔD)² ) ≤ 1
> Note: "2D gamma is computed slice-by-slice. Faster than 3D gamma; appropriate for per-slice patient-specific QA."
>
> **Style:** white background, University of Bern red (#CC0000). Title: "2D Gamma Index".

---

## Group 9 — Voxel-Based Dose Comparison

### 9a. Mean Absolute Error (MAE) — `compare_mae`

**Prompt:**

> Create an educational diagram explaining Mean Absolute Error (MAE) for 3D dose comparison in radiotherapy.
>
> **Left panel — heatmap:**
> Draw a circular body-mask cross-section. Fill it with a 2D heatmap showing |D_pred − D_target| at each voxel — white where the two plans agree, dark red where they differ most. Add a colour bar: "0 Gy" (white) to "|ΔD| max" (dark red). Label the structure "body mask".
>
> **Right panel — formula:**
> MAE = (1/N) Σᵢ |D_pred,i − D_target,i|
> Add comparison with MSE/RMSE: "MAE treats all errors equally (linear). RMSE penalises large errors more (quadratic). Both measure voxel-wise accuracy."
>
> **Style:** white background, University of Bern red (#CC0000). Title: "Mean Absolute Error (MAE)".

---

### 9b. Structural Similarity Index (SSIM) — `compare_ssim`

**Prompt:**

> Create an educational diagram explaining the Structural Similarity Index (SSIM) for dose distribution comparison in radiotherapy.
>
> **Left panel — side-by-side dose slices:**
> Draw two small square dose maps (5×5 grid cells each):
> - Left: "Reference dose" — smooth gradient from light to dark red.
> - Right: "Predicted dose" — similar but with slight Gaussian blur/noise applied, making gradients slightly softer.
> - Label both with "DOSE (Gy)" colour bars.
>
> **Centre panel — component breakdown:**
> Show three component bars:
> - Luminance similarity (l) = comparison of mean doses
> - Contrast similarity (c) = comparison of dose variance
> - Structure similarity (s) = cross-correlation of dose patterns
> Draw each as a horizontal bar from 0 to 1, shaded in red proportional to value. Label numerical examples: l=0.98, c=0.95, s=0.91.
>
> **Right panel — formula:**
> SSIM(x, y) = l(x,y)^α · c(x,y)^β · s(x,y)^γ
> Range: [−1, 1]. SSIM = 1.0: identical distributions.
> Note: "SSIM captures perceptual similarity — two plans can have the same MAE but very different SSIM if dose gradients differ."
>
> **Style:** white background, University of Bern red (#CC0000). Title: "Structural Similarity Index (SSIM)".

---

### 9c. Dose Difference Map — `compare_dose_difference_map`

**Prompt:**

> Create an educational diagram explaining the Dose Difference Map for radiotherapy plan comparison.
>
> **Three-panel layout (left → right):**
>
> **Panel 1 — Reference dose:** A circular cross-section with smooth dose gradient, light to dark red. Title: "Target dose". Colour bar: "DOSE (Gy)".
>
> **Panel 2 — Predicted dose:** Similar cross-section with slightly different pattern. Title: "Predicted dose". Colour bar: "DOSE (Gy)".
>
> **Panel 3 — Difference map:** A diverging-colour cross-section (blue = predicted lower, white = equal, red = predicted higher). Title: "Difference (Pred − Target)". Colour bar from "−ΔD" to "+ΔD". Show some localised red hotspots and blue cold regions.
>
> **Formula below all three:**
> ΔD(v) = D_pred(v) − D_target(v)  for each voxel v
>
> **Style:** white background; for difference map use blue-white-red diverging scale with University of Bern red (#CC0000) for positive values. Title: "Dose Difference Map".

---

### 9d. Variance of Laplacian — `compute_variance_of_laplacian`

**Prompt:**

> Create an educational diagram explaining the Variance of Laplacian as a dose sharpness metric for radiotherapy.
>
> **Left panel — smooth dose plan:**
> Draw a 2D cross-section with a smooth, gently varying dose gradient (IMRT-like). Overlay the computed Laplacian as a subtle contour map with few lines. Label: "low VoL — smooth gradients". Show a small numerical example: VoL = 0.002.
>
> **Right panel — complex dose plan:**
> Draw a 2D cross-section with many sharp dose boundaries, steep falloff edges, and complex gradient patterns (VMAT with many fields). Overlay a denser Laplacian contour map with many lines. Label: "high VoL — complex gradients". Show: VoL = 0.018.
>
> **Formula below both:**
> VoL = Var[∇²D]
> where ∇²D is the discrete 3D Laplacian of the dose array.
> Note: "Higher VoL = sharper, more complex dose gradients. Useful for comparing plan complexity or modality (3DCRT vs IMRT vs VMAT)."
>
> **Style:** white background, University of Bern red (#CC0000). Title: "Variance of Laplacian (Plan Sharpness)".

---

## Group 10 — Advanced DVH Statistical Metrics

### 10a. DVH Wasserstein Distance — `compare_dvh_wasserstein`

**Prompt:**

> Create an educational diagram explaining the Wasserstein (earth-mover's) Distance between two DVH curves for radiotherapy dose comparison.
>
> **Left panel — DVH diagram:**
> Draw two smooth cumulative DVH curves on the same axes (X: "DOSE (Gy)", Y: "VOLUME (%)"):
> - Solid red: "Target DVH"
> - Dashed black: "Predicted DVH"
> - Shade the vertical gaps between the two curves at multiple dose points in light red.
> - Draw several horizontal arrows between the two curves at different volume levels, labelled "transport distance".
>
> **Right panel — analogy:**
> Draw a simple illustration of two piles of dirt (representing the two DVH distributions), with an arrow and a small cartoon shovel between them. Label: "Wasserstein = minimum work to transform one distribution into the other".
>
> **Formula below:**
> W(F, G) = ∫₀¹ |F⁻¹(p) − G⁻¹(p)| dp
> where F and G are the cumulative DVH functions.
>
> **Style:** white background, University of Bern red (#CC0000). Title: "DVH Wasserstein Distance".

---

### 10b. DVH Confidence Interval — `compute_dvh_confidence_interval`

**Prompt:**

> Create an educational diagram explaining DVH Confidence Intervals for dose distribution uncertainty quantification in radiotherapy.
>
> **Diagram:**
> Draw a central solid red DVH curve (mean DVH) with a shaded band around it representing the 95% confidence interval in light red/pink. Add two additional dashed red curves bounding the band: "95% CI upper" and "95% CI lower". X-axis: "DOSE (Gy)", Y-axis: "VOLUME (%)". Add several thin translucent grey curves inside the band representing individual sample DVHs from a population. Label the band width at D50 with a bidirectional arrow: "CI width at D50".
>
> **Formula below:**
> CI = [D̄_X − z·σ/√n, D̄_X + z·σ/√n]
> where D̄_X is the mean dose at volume fraction X, σ is the standard deviation, and n is the number of samples.
> Note: "Confidence intervals quantify how much DVH metrics vary across a patient cohort or across planning iterations."
>
> **Style:** white background, University of Bern red (#CC0000). Title: "DVH Confidence Interval".

---

### 10c. Mutual Information — `compare_mutual_information`

**Prompt:**

> Create an educational diagram explaining Mutual Information (MI) between two dose distributions in radiotherapy.
>
> **Left panel — joint histogram:**
> Draw a 2D heatmap (scatter density plot) where the x-axis is "Reference dose (Gy)" and y-axis is "Predicted dose (Gy)". Points clustered tightly around the diagonal (y = x line) in dark red; scatter in lighter red. Draw the diagonal line in black. Label: "perfect agreement → points on diagonal → high MI".
>
> **Right panel — Venn-diagram of information:**
> Draw two overlapping circles:
> - Left circle: H(X) = entropy of reference dose distribution
> - Right circle: H(Y) = entropy of predicted dose distribution
> - Overlap: I(X;Y) = mutual information, shaded in medium red.
> Label each region.
>
> **Formula below:**
> I(X; Y) = H(X) + H(Y) − H(X, Y)
> = Σ_{x,y} p(x,y) log(p(x,y) / (p(x)p(y)))
>
> **Style:** white background, University of Bern red (#CC0000). Title: "Mutual Information (MI)".

---

### 10d. KS Test and Chi-Square on DVH — `compare_dvh_ks` / `compare_dvh_chi_square`

**Prompt:**

> Create a side-by-side educational diagram explaining the Kolmogorov-Smirnov test and Chi-Square test applied to DVH curves for radiotherapy.
>
> **Left panel — KS test:**
> Draw two cumulative DVH curves (solid red = target, dashed black = predicted). Draw a vertical double-headed arrow at the dose point where the two curves are furthest apart, labelled "KS statistic D = max|F(x) − G(x)|". Shade this maximum gap in red. Below: "KS test detects any shape difference along the full DVH. p-value tests significance."
>
> **Right panel — Chi-square test:**
> Draw the same two DVH curves, but now divide the x-axis into discrete dose bins (vertical grey lines). For each bin, show a small rectangle whose height represents the observed difference in frequency between target and predicted. Shade positive differences in red, negative in pink. Below: χ² = Σ (O − E)² / E, where O = predicted bin count, E = reference bin count.
>
> **Style:** white background, University of Bern red (#CC0000). Overall title: "Statistical DVH Tests (KS and Chi-Square)".

---

## Usage Notes

- When submitting prompts to [claude.ai/design](https://claude.ai/design), specify: **"University of Bern style: white background, deep red (#CC0000), clean sans-serif typography, minimal decoration"**.
- Generated images should be saved to [images/](images/) and named after the metric function (e.g. `metric-dice-coefficient.png`).
- After adding images, embed them in the relevant documentation page with:
  ```markdown
  ![Alt text](../images/metric-name.png)
  *Caption describing what the diagram illustrates.*
  ```
- Pages that need new images:
  - [user-guide/dvh-analysis.md](user-guide/dvh-analysis.md) — Groups 1 and 2
  - [user-guide/quality-metrics.md](user-guide/quality-metrics.md) — Groups 3 and 4
  - [user-guide/geometric-analysis.md](user-guide/geometric-analysis.md) — Groups 5, 6, and 7
  - [user-guide/gamma-performance.md](user-guide/gamma-performance.md) — Group 8
  - [user-guide/quality-metrics.md](user-guide/quality-metrics.md) — Group 9
  - [user-guide/dvh-analysis.md](user-guide/dvh-analysis.md) — Group 10
