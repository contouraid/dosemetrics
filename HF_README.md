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
2. Upload your dose distribution (`.nii.gz` format) and structure masks
3. View and download the analysis results

**Note**: This app requires you to upload your own medical imaging data. No sample data files are included in the Space due to file size restrictions.

## Data Format

The app expects:
- Dose files: NIfTI format (`.nii.gz`)
- Structure masks: NIfTI format (`.nii.gz`)

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
  year = {2024},
  url = {https://github.com/contouraid/dosemetrics}
}
```

## License

This project is licensed under CC BY-NC-SA 4.0. See the LICENSE file for details.
