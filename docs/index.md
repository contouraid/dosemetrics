# DoseMetrics

DoseMetrics 0.4.0 provides data containers, I/O, metrics, comparisons, and visualization tools for radiotherapy dose analysis.

## Quick example

```python
from dosemetrics.metrics import dvh
from dosemetrics.utils import load_example_study, plot_dvh

dose, structures = load_example_study("test_subject")
ptv = structures["PTV"]

dose_bins, volume_percent = dvh.compute_dvh(dose, ptv, verbose=True)
stats = dvh.compute_dose_statistics(dose, ptv, verbose=True)
fig, ax = plot_dvh(dose, ptv, label="PTV")
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

Import reference-free computations from their domain module, for example
`from dosemetrics.metrics import dvh, conformity`. Named plan comparisons are
available directly, for example `from dosemetrics.metrics import compare_ptv_dose`.

## Start here

1. [Install DoseMetrics](getting-started/installation.md)
2. [Run the quick start](getting-started/quickstart.md)
3. [Review supported file layouts](getting-started/file-formats.md)
4. [Open the executable notebooks](notebooks/01-basic-usage.ipynb)
5. [Browse the API reference](api/index.md)

[:material-rocket-launch: Try the live demo](https://huggingface.co/spaces/contouraid/dosemetrics){ .md-button .md-button--primary target="_blank" }

## Project information

- [Contributing](contributing.md)
- [Changelog](changelog.md)
- [License](license.md)
