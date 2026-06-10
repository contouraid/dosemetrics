# Gamma Index: Performance and Limitations

The gamma index implementation in DoseMetrics is algorithmically correct and validated against established medical physics literature (Low et al. 1998). However, the current implementation has significant performance limitations for large volumes that users should be aware of.

## How the Gamma Index Works

![Gamma Index](../images/gamma-index.png)
*Gamma Index — for each voxel v, the index searches for the reference voxel v′ that minimises the combined spatial distance r(v, v′) / Δd and dose distance δ(v, v′) / ΔD. The voxel at 64 Gy (centre of the highlighted grid) fails because no reference neighbour within Δd = 3 mm matches its dose within ΔD. (Joseph Weibel, MSc Thesis Defense, University of Bern)*

The gamma index at a point $v$ is:

$$\gamma(v) = \min_{v'} \sqrt{\frac{r^2(v, v')}{(\Delta d)^2} + \frac{\delta^2(v, v')}{(\Delta D)^2}} \leq 1$$

where $r(v,v')$ is the Euclidean distance between voxels, $\delta(v,v')$ is the absolute dose difference, $\Delta d$ is the distance-to-agreement criterion (e.g. 3 mm), and $\Delta D$ is the dose-difference criterion (e.g. 3% of prescription). A voxel **passes** when $\gamma \leq 1$.

## Current Performance

| Volume Size | Voxels | Runtime | Clinical Target | Status |
|-------------|--------|---------|-----------------|--------|
| 32³–64³     | ≤ 262K | < 10s   | Acceptable      | OK     |
| 128³        | 2.1M   | ~53s    | < 10s           | Slow   |
| 256³        | 16.8M  | ~400s   | < 10s           | Too slow |

The bottleneck is a triple-nested Python loop that accounts for ~99% of runtime. No parallelization is used.

## What Works Well Now

The 3D gamma computation is reliable for:

- **Small volumes** (32³ to 64³): completes in < 10 seconds
- **2D gamma** (`compute_2d_gamma()`): fast for per-slice QA
- **Research and batch processing** where runtime is not interactive

The implementation handles all standard clinical criteria correctly:

- 1%/1mm (research / strict)
- 2%/2mm (tight clinical)
- 3%/3mm (standard clinical)
- 5%/3mm and 3%/5mm (lenient)

## Workarounds for Large Volumes

Until the performance issues are resolved, these approaches make the current implementation usable:

**1. Use 2D gamma for interactive QA:**
```python
from dosemetrics.metrics import gamma

gamma_2d = gamma.compute_2d_gamma(
    dose_ref_slice,
    dose_eval_slice,
    dose_criterion_percent=3.0,
    distance_criterion_mm=3.0,
    pixel_spacing=(1.0, 1.0)
)
```

**2. Crop to the region of interest:**
```python
# Crop both dose arrays to the PTV bounding box before computing gamma
```

**3. Downsample resolution** (with caution — check that clinical accuracy is maintained):
```python
# Use every other voxel for a quick QA check
dose_ref_downsampled = dose_ref.dose_array[::2, ::2, ::2]
```

**4. Run overnight for large batch jobs:** the algorithm is correct, just slow.

**5. Reduce the search radius:**
```python
gamma_map = gamma.compute_gamma_index(
    dose_ref, dose_eval,
    dose_criterion_percent=3.0,
    distance_criterion_mm=3.0,
    max_search_distance_mm=6.0,  # Default is 3× criterion = 9mm; reducing helps
)
```

## Optimization Roadmap

Performance optimization is planned in three phases:

**Phase 1 — Multi-threading** (target: 128³ < 15s)  
Add `n_jobs` parameter using `joblib` for parallel processing across CPU cores. Low effort, immediate 2–4× speedup.

**Phase 2 — Vectorization** (target: 128³ < 5s)  
Replace the Python triple loop with NumPy broadcasting and optional Numba JIT compilation. Larger engineering effort, 5–10× additional speedup.

**Phase 3 — GPU acceleration** (target: 128³ < 1s)  
CuPy-based GPU implementation with automatic CPU fallback. Requires NVIDIA GPU with CUDA.

## Algorithmic Correctness

The implementation has been validated with 24 correctness tests covering:

- Formula correctness (Low et al. 1998): `γ = sqrt((Δdose/DD)² + (distance/Dc)²)`
- Global and local normalization modes
- Dose threshold handling
- All standard clinical criteria
- Edge cases (zero dose, high-gradient regions, small volumes)
- Statistical functions (passing rate, comprehensive statistics)

The algorithm is correct. The only known issue is runtime performance for large volumes.

## References

- Low DA, et al. (1998). "A technique for the quantitative evaluation of dose distributions." *Medical Physics* 25(5):656–61.
- Wendling M, et al. (2007). "A fast algorithm for gamma evaluation in 3D." *Medical Physics* 34(5):1647–54.
- Ju T, et al. (2008). "On the feasibility of GPU-based gamma evaluation." *Medical Physics* 35(10):4431–4438.
