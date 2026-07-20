# DoseMetrics

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://python.org)
[![PyPI version](https://badge.fury.io/py/dosemetrics.svg)](https://badge.fury.io/py/dosemetrics)
[![Tests](https://github.com/contouraid/dosemetrics/actions/workflows/tests.yml/badge.svg)](https://github.com/contouraid/dosemetrics/actions/workflows/tests.yml)
[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://contouraid.github.io/dosemetrics/)
[![License](https://img.shields.io/badge/license-CC%20BY--SA--NC%204.0-green.svg)](LICENSE)

DoseMetrics 0.4.0 is a Python library for radiotherapy dose analysis. It provides typed containers for dose distributions and structures, NIfTI and DICOM-RT I/O, DVH and plan-quality metrics, dose and geometry comparisons, gamma analysis, batch utilities, and publication-ready Matplotlib plots.

## Installation

```bash
pip install dosemetrics
```

For local development:

```bash
git clone https://github.com/contouraid/dosemetrics.git
cd dosemetrics
make setup
```

## Quick start

The current API keeps dose data, structure geometry, and metric computation separate:

```python
from dosemetrics import Dose
from dosemetrics.io import load_structure_set
from dosemetrics.metrics import conformity, dvh, homogeneity
from dosemetrics.utils.plot import plot_subject_dvhs

patient_dir = "path/to/patient"
dose = Dose.from_nifti(f"{patient_dir}/Dose.nii.gz", name="Clinical")
structures = load_structure_set(patient_dir)
ptv = structures["PTV"]

dose_bins, volume_percent = dvh.compute_dvh(dose, ptv)
stats = dvh.compute_dose_statistics(dose, ptv)
ci = conformity.compute_paddick_conformity_index(dose, ptv, prescription_dose=60.0)
hi = homogeneity.compute_homogeneity_index(dose, ptv)
fig, ax = plot_subject_dvhs(dose, structures)

print(f"D95: {stats['D95']:.2f} Gy; Paddick CI: {ci:.3f}; HI: {hi:.3f}")
```

A NIfTI patient folder places `Dose.nii.gz` and one binary mask per structure in the same directory. `load_structure_set()` loads only geometry; load the dose independently with `Dose.from_nifti()`.

## Comparing plans

All reference-based functions accept `reference` first and `evaluated` second:

```python
from dosemetrics import Dose
from dosemetrics.metrics import comparison, dose_comparison, gamma

reference = Dose.from_nifti("reference.nii.gz")
evaluated = Dose.from_nifti("evaluated.nii.gz")

mae_gy = dose_comparison.compare_mae(reference, evaluated)
gamma_map = gamma.compare_gamma_index(reference, evaluated)
gamma_pass = gamma.compute_gamma_passing_rate(gamma_map)
ptv_dose_distance = comparison.compare_ptv_dose(reference, evaluated, ptv)
```

## Public package layout

- `dosemetrics`: `Dose`, structure classes, `StructureSet`, and high-level loaders
- `dosemetrics.io`: unified NIfTI/DICOM loading plus format-specific modules
- `dosemetrics.metrics`: `dvh`, `conformity`, `homogeneity`, `geometric`, `gamma`, `dose_comparison`, and `comparison`
- `dosemetrics.utils`: plotting, compliance, batch processing, and cohort analysis

Metrics are intentionally exposed through domain modules rather than flattened into the package root.

## Command line

```bash
dosemetrics --version
dosemetrics dvh Dose.nii.gz patient_folder -o dvh.csv
dosemetrics statistics Dose.nii.gz patient_folder -o statistics.csv
dosemetrics conformity Dose.nii.gz PTV.nii.gz --prescription 60
dosemetrics gamma reference.nii.gz evaluated.nii.gz --dose-criteria 3 --distance-criteria 3
```

Run `dosemetrics --help` or `dosemetrics <command> --help` for all options.

## Development

```bash
make test
make docs
make app
```

The documentation notebooks use the public `contouraid/dosemetrics-data` dataset and are exercised by `tests/test_documentation.py`.

## Documentation and demo

- [Documentation](https://contouraid.github.io/dosemetrics/)
- [Interactive demo](https://huggingface.co/spaces/contouraid/dosemetrics)
- [Issue tracker](https://github.com/contouraid/dosemetrics/issues)

## Citation

```bibtex
@software{dosemetrics2026,
  author = {Kamath, Amith},
  title = {DoseMetrics: A Python Library for Radiotherapy Dose Analysis},
  url = {https://github.com/contouraid/dosemetrics},
  version = {0.4.0},
  year = {2026}
}
```

## License

Licensed under the Creative Commons Attribution-ShareAlike-NonCommercial 4.0 International License. See [LICENSE](LICENSE). Commercial use requires explicit written permission from the copyright holder.
