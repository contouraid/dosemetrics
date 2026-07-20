# DoseMetrics

DoseMetrics 0.4.0 provides data containers, I/O, metrics, comparisons, and visualization tools for radiotherapy dose analysis.

## Quick example

```python
from dosemetrics import Dose
from dosemetrics.io import load_structure_set
from dosemetrics.metrics import dvh
from dosemetrics.utils.plot import plot_dvh

patient_dir = "path/to/patient"
dose = Dose.from_nifti(f"{patient_dir}/Dose.nii.gz")
structures = load_structure_set(patient_dir)
ptv = structures["PTV"]

dose_bins, volume_percent = dvh.compute_dvh(dose, ptv)
stats = dvh.compute_dose_statistics(dose, ptv)
fig, ax = plot_dvh(dose, ptv, label="PTV")

print(f"Mean: {stats['mean_dose']:.2f} Gy; D95: {stats['D95']:.2f} Gy")
```

## What is included

- Dose-volume histograms, dose statistics, EUD, and DVH comparisons
- Conformity, coverage, spillage, homogeneity, and gradient indices
- Voxel-wise image comparisons and CPU gamma analysis
- Dice, Jaccard, surface-distance, and volume comparisons
- NIfTI and DICOM-RT loading, plus NIfTI export helpers
- Batch/cohort analysis, compliance checking, and Matplotlib plotting

## API organization

- `Dose`, `Structure`, and `StructureSet` are data containers.
- `dosemetrics.io` translates NIfTI and DICOM data into those containers.
- `dosemetrics.metrics` contains computations grouped by clinical domain.
- `dosemetrics.utils` contains plotting, compliance, batch, and cohort helpers.

Metric functions are not flattened into the package root. Import the relevant domain module, for example `from dosemetrics.metrics import dvh, conformity`.

## Start here

1. [Install DoseMetrics](getting-started/installation.md)
2. [Run the quick start](getting-started/quickstart.md)
3. [Review supported file layouts](getting-started/file-formats.md)
4. [Open the executable notebooks](notebooks/01-basic-usage.ipynb)
5. [Browse the API reference](api/index.md)

[:material-rocket-launch: Try the live demo](https://huggingface.co/spaces/contouraid/dosemetrics){ .md-button .md-button--primary target="_blank" }
