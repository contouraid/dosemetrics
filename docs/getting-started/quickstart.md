# Quick Start

## Load a NIfTI patient

A patient folder should contain `Dose.nii.gz` and one binary NIfTI mask per structure.

```python
from pathlib import Path
from dosemetrics import Dose
from dosemetrics.io import load_structure_set

patient_dir = Path("path/to/patient")
dose = Dose.from_nifti(patient_dir / "Dose.nii.gz", name="Clinical")
structures = load_structure_set(patient_dir)

ptv = structures["PTV"]
brainstem = structures["Brainstem"]
print(dose)
print(structures.structure_names)
```

`StructureSet` contains geometry only. Keeping the dose independent lets the same structures be reused with multiple plans.

## Compute DVHs and statistics

```python
from dosemetrics.metrics import dvh

dose_bins, volume_percent = dvh.compute_dvh(dose, ptv, step_size=0.1)
stats = dvh.compute_dose_statistics(dose, ptv)
d95 = dvh.compute_dose_at_volume(dose, ptv, volume_percent=95)
v20 = dvh.compute_volume_at_dose(dose, brainstem, dose_threshold=20.0)

print(f"PTV D95: {d95:.2f} Gy")
print(f"Brainstem V20: {v20:.1f}%")
```

For every structure at once:

```python
dvh_table = dvh.create_dvh_table(dose, structures, step_size=0.1)
dvh_table.to_csv("dvh_results.csv", index=False)
```

## Compute plan-quality metrics

```python
from dosemetrics.metrics import conformity, homogeneity

prescription = 60.0
coverage = conformity.compute_coverage(dose, ptv, prescription)
paddick_ci = conformity.compute_paddick_conformity_index(dose, ptv, prescription)
homogeneity_index = homogeneity.compute_homogeneity_index(dose, ptv)
gradient_index = homogeneity.compute_gradient_index(dose, ptv, prescription)

print(f"Coverage: {coverage:.1%}")
print(f"Paddick CI: {paddick_ci:.3f}")
print(f"Homogeneity index: {homogeneity_index:.3f}")
print(f"Gradient index: {gradient_index:.3f}")
```

## Plot structures

```python
from dosemetrics.utils.plot import plot_subject_dvhs, save_figure

fig, ax = plot_subject_dvhs(
    dose,
    structures,
    structure_names=["PTV", "Brainstem"],
)
save_figure(fig, "dvh", formats=["png", "pdf"])
```

## Compare two plans

```python
from dosemetrics import Dose
from dosemetrics.metrics import comparison, dose_comparison, gamma

reference = Dose.from_nifti("reference.nii.gz")
evaluated = Dose.from_nifti("evaluated.nii.gz")

mae_gy = dose_comparison.compare_mae(reference, evaluated)
ptv_distance_gy = comparison.compare_ptv_dose(reference, evaluated, ptv)
gamma_map = gamma.compare_gamma_index(reference, evaluated)
passing_rate = gamma.compute_gamma_passing_rate(gamma_map)

print(f"MAE: {mae_gy:.3f} Gy")
print(f"PTV mean-dose distance: {ptv_distance_gy:.3f} Gy")
print(f"3%/3 mm gamma pass rate: {passing_rate:.1f}%")
```

All comparison functions take `reference` before `evaluated` and require compatible dose geometry.

## Next steps

- [File formats](file-formats.md)
- [DVH analysis](../user-guide/dvh-analysis.md)
- [Plan comparison metrics](../user-guide/plan-comparison-metrics.md)
- [Executable notebooks](../notebooks/01-basic-usage.ipynb)
