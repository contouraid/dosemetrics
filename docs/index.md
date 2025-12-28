# DoseMetrics

<div align="center">

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://python.org)
[![PyPI version](https://badge.fury.io/py/dosemetrics.svg)](https://badge.fury.io/py/dosemetrics)
[![Tests](https://github.com/contouraid/dosemetrics/actions/workflows/tests.yml/badge.svg)](https://github.com/contouraid/dosemetrics/actions/workflows/tests.yml)
[![License](https://img.shields.io/badge/license-CC%20BY--SA--NC%204.0-green.svg)](license.md)

**A Python library for measuring radiotherapy doses and creating interactive visualizations for radiation therapy treatment planning and analysis.**

[Get Started](getting-started/installation.md){ .md-button .md-button--primary }
[Try Live Demo](https://huggingface.co/spaces/contouraid/dosemetrics){ .md-button .md-button--secondary }
[View on GitHub](https://github.com/contouraid/dosemetrics){ .md-button .md-button--secondary }

</div>

---

## Overview

DoseMetrics provides comprehensive tools for analyzing radiation dose distributions, calculating dose-volume histograms (DVH), evaluating treatment plan quality, and creating publication-ready visualizations. This library is designed for medical physicists, radiation oncologists, and researchers working with radiotherapy treatment planning data.

## Key Features

:material-chart-line: **Dose Analysis**
:   Calculate and analyze 3D dose distributions with support for multiple file formats

:material-chart-histogram: **DVH Generation**
:   Create dose-volume histograms for organs at risk (OARs) and target volumes

:material-star-check: **Quality Metrics**
:   Compute conformity indices, homogeneity indices, and other plan quality metrics

:material-clipboard-check: **Compliance Checking**
:   Evaluate dose constraints and treatment plan compliance against clinical protocols

:material-chart-areaspline: **Interactive Visualizations**
:   Generate interactive plots using Plotly and Streamlit for exploratory analysis

:material-compare: **Comparative Analysis**
:   Compare predicted vs. actual dose distributions and analyze plan variations

:material-ruler: **Geometric Analysis**
:   Compute spatial differences and overlaps between structure sets

:material-download: **Export Capabilities**
:   Save results in various formats (CSV, PDF, PNG) for reporting and publication

## Quick Example

```python
from dosemetrics.io import load_dose, load_mask
from dosemetrics.metrics.dvh import compute_dvh
from dosemetrics.utils.plotting import plot_dvh

# Load dose and structure data
dose = load_dose("path/to/dose.nii.gz")
mask = load_mask("path/to/structure.nii.gz")

# Compute DVH
dvh_data = compute_dvh(dose, mask, organ_name="PTV")

# Create interactive visualization
plot_dvh(dvh_data, title="Target Coverage Analysis")
```

## Try It Out

!!! tip "Live Demo Available"
    Want to try DoseMetrics without installing anything? Check out our interactive demo on Hugging Face Spaces where you can upload your own data and explore the features through a web interface.
    
    **Available Features in the Live Demo:**
    
    - **Instructions**: Get started with comprehensive usage guidelines
    - **Dosimetric Analysis**: DVH calculations, dose statistics, and interactive visualizations
    - **Geometric Comparison**: Dice coefficient, Hausdorff distance, and volume overlaps
    - **Gamma Analysis**: Compare dose distributions with configurable criteria
    - **Compliance Checking**: Evaluate plans against clinical constraints
    
    [:material-rocket-launch: Launch Live Demo](https://huggingface.co/spaces/contouraid/dosemetrics){ .md-button .md-button--primary target="_blank" }

## Package Structure

The library is organized into focused modules:

- **`dosemetrics.metrics`**: Core dose calculation functions (DVH, scores, comparison metrics)
- **`dosemetrics.io`**: File I/O utilities and data structures for dose and structure management
- **`dosemetrics.utils`**: Utility functions for compliance checking, plotting, and data processing

## Installation

Install via pip:

```bash
pip install dosemetrics
```

For development with documentation tools:

```bash
pip install dosemetrics[docs]
```

See the [Installation Guide](getting-started/installation.md) for more options.

## Getting Started

New to DoseMetrics? Start here:

1. [**Installation**](getting-started/installation.md) - Set up DoseMetrics in your environment
2. [**Quick Start**](getting-started/quickstart.md) - Run your first analysis in minutes
3. [**Using Your Own Data**](notebooks/04-getting-started-own-data.ipynb) - Learn how to prepare and analyze your data
4. [**File Formats**](getting-started/file-formats.md) - Understand supported input formats

## Use Cases

DoseMetrics is used for:

- **Treatment Plan Evaluation**: Assess the quality of radiotherapy treatment plans
- **Research Studies**: Analyze large cohorts of treatment plans for clinical research
- **Quality Assurance**: Verify dose calculations and constraint compliance
- **Algorithm Development**: Develop and test dose prediction and optimization algorithms
- **Educational Purposes**: Teach radiotherapy physics and treatment planning concepts

## Citation

If you use DoseMetrics in your research, please cite:

```bibtex
@software{dosemetrics2024,
  author = {Kamath, Amith},
  title = {DoseMetrics: Tools for Radiotherapy Dose Analysis},
  year = {2024},
  url = {https://github.com/contouraid/dosemetrics}
}
```

## Support

- :material-file-document: [Documentation](https://contouraid.github.io/dosemetrics/)
- :material-bug: [Issue Tracker](https://github.com/contouraid/dosemetrics/issues)
- :material-github: [Source Code](https://github.com/contouraid/dosemetrics)
- :material-rocket-launch: [Live Demo](https://huggingface.co/spaces/contouraid/dosemetrics)

## License

DoseMetrics is licensed under the Creative Commons Attribution-ShareAlike-NonCommercial 4.0 International License. See [LICENSE](license.md) for details.
