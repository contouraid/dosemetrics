---
title: DoseMetrics
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: cc-by-nc-sa-4.0
---

# DoseMetrics - Radiotherapy Dose Analysis

A Streamlit application for analyzing radiotherapy dose distributions and creating interactive visualizations for radiation therapy treatment planning.

## Features

- **Instructions**: Get started with comprehensive usage guidelines and examples
- **Dosimetric Analysis**: Comprehensive dose analysis with DVH calculations, dose statistics, and interactive visualizations
- **Geometric Comparison**: Analyze geometric differences between structure sets including dice coefficient, Hausdorff distance, and volume overlaps
- **Gamma Analysis**: Perform gamma analysis to compare dose distributions between reference and evaluated plans
- **Compliance Checking**: Evaluate treatment plans against clinical constraints and dose limits

## Usage

1. Select a task from the sidebar
2. Open **Dosimetric Analysis**
3. Load a hosted NIfTI or DICOM example, or upload your own data
4. Explore dose slices, DVHs, statistics, and target quality metrics
5. Download tabular results as CSV

Hosted examples are downloaded on demand from
[`contouraid/dosemetrics-data`](https://huggingface.co/datasets/contouraid/dosemetrics-data)
and cached by the Space.

## Data Format

The app accepts:

- **NIfTI**: one `.nii`/`.nii.gz` dose volume and one or more binary structure masks
- **DICOM-RT**: at least one RTDOSE and one RTSTRUCT; CT and RTPLAN files are optional

Uploaded DICOM files are classified by their Modality tags. Files are processed in a temporary directory that is deleted after loading.

## About

DoseMetrics provides tools for medical physicists and radiation oncologists to analyze treatment plans, including:

- Dose-volume histogram (DVH) calculation
- Treatment plan quality metrics (conformity indices, homogeneity indices)
- Compliance checking against clinical constraints
- Geometric analysis of structures
- Interactive visualizations

For more information, visit the [GitHub repository](https://github.com/contouraid/dosemetrics).

## Citation

If you use DoseMetrics in your research, please cite:

```bibtex
@software{dosemetrics,
  author = {Kamath, Amith},
  title = {DoseMetrics: Tools for Radiotherapy Dose Analysis},
  year = {2026},
  url = {https://github.com/contouraid/dosemetrics}
}
```

## License

This project is licensed under CC BY-NC-SA 4.0. See the LICENSE file for details.
