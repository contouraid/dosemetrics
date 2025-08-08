# DoseMetrics

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://python.org)
[![PyPI version](https://badge.fury.io/py/dosemetrics.svg)](https://badge.fury.io/py/dosemetrics)
[![Tests](https://github.com/amithjkamath/dosemetrics/actions/workflows/python-app.yml/badge.svg)](https://github.com/amithjkamath/dosemetrics/actions/workflows/python-app.yml)
[![License](https://img.shields.io/badge/license-CC%20BY--SA--NC%204.0-green.svg)](LICENSE)
[![Development Status](https://img.shields.io/badge/status-alpha-orange.svg)](https://github.com/amithjkamath/dosemetrics)

A comprehensive Python library for measuring radiotherapy doses and creating interactive visualizations for radiation therapy treatment planning and analysis.

## Overview

DoseMetrics provides tools for analyzing radiation dose distributions, calculating dose-volume histograms (DVH), evaluating treatment plan quality, and creating publication-ready visualizations. This library is designed for medical physicists, radiation oncologists, and researchers working with radiotherapy treatment planning data.

## Features

- **Dose Analysis**: Calculate and analyze 3D dose distributions
- **DVH Generation**: Create dose-volume histograms for organs at risk (OARs) and targets
- **Quality Metrics**: Compute conformity indices, homogeneity indices, and other plan quality metrics
- **Compliance Checking**: Evaluate dose constraints and treatment plan compliance
- **Interactive Visualizations**: Generate interactive plots using Plotly and Streamlit
- **Comparative Analysis**: Compare predicted vs. actual dose distributions
- **Geometric Analysis**: Compute spatial differences and overlaps between structures
- **Export Capabilities**: Save results in various formats (CSV, PDF, PNG)

## Quick Start

### Installation

Install it easily using pip:

```bash
pip install dosemetrics
```

**For development**, clone the repository and install in editable mode:

```bash
git clone https://github.com/amithjkamath/dosemetrics.git
cd dosemetrics
pip install --editable .
```

### Interactive Web Application

Launch the interactive Streamlit application:

```bash
streamlit run app.py
```

This provides a user-friendly interface for uploading DICOM files, analyzing dose distributions, and generating reports.

## Usage Examples

### Basic DVH Analysis

```python
import dosemetrics as dm

# Load dose and structure data
dose_data = dm.data_utils.load_dose("path/to/dose.nii.gz")
structures = dm.data_utils.load_structures("path/to/structures/")

# Generate DVH
dvh = dm.dvh.calculate_dvh(dose_data, structures["PTV"])

# Plot DVH
dm.plot.plot_dvh(dvh, title="Target DVH")
```

### Quality Index Calculation

```python
# Calculate conformity and homogeneity indices
quality_metrics = dm.metrics.calculate_quality_indices(
    dose_data, 
    target_structure, 
    prescription_dose=50
)

print(f"Conformity Index: {quality_metrics['CI']:.3f}")
print(f"Homogeneity Index: {quality_metrics['HI']:.3f}")
```

### Compliance Checking

```python
# Define dose constraints
constraints = {
    "Brainstem": {"max_dose": 54, "unit": "Gy"},
    "Spinal_Cord": {"max_dose": 45, "unit": "Gy"},
    "Parotid_L": {"mean_dose": 26, "unit": "Gy"}
}

# Check compliance
compliance_results = dm.compliance.check_constraints(
    dose_data, structures, constraints
)
```

## Project Structure

```
dosemetrics/
├── dosemetrics/           # Core library modules
│   ├── comparison.py      # Plan comparison tools
│   ├── compliance.py      # Constraint checking
│   ├── data_utils.py      # Data loading utilities
│   ├── dvh.py            # DVH calculation
│   ├── metrics.py        # Quality metrics
│   ├── plot.py           # Visualization tools
│   └── scores.py         # Scoring algorithms
├── examples/             # Usage examples and scripts
├── test/                # Unit tests
├── data/                # Sample data for testing
└── app.py              # Streamlit web application
```

## Examples

The `examples/` directory contains comprehensive examples:

- **DVH Analysis**: Generate and compare dose-volume histograms
- **Quality Assessment**: Calculate treatment plan quality indices
- **Geometric Analysis**: Compute structure overlaps and distances
- **Interactive Plotting**: Create interactive visualizations
- **Report Generation**: Generate automated treatment plan reports

Run any example script:

```bash
python examples/plot_dvh_interactive.py
python examples/compare_quality_index.py
python examples/generate_dvh_family.py
```

## Supported Data Formats

- **DICOM**: RT Dose, RT Structure Set
- **NIfTI**: `.nii`, `.nii.gz` files
- **NRRD**: Near Raw Raster Data format
- **Text**: Eclipse DVH export files

## Development

### Running Tests

Execute the test suite to ensure everything works correctly:

```bash
python -m unittest discover -s test -p "test_*.py"
```

### Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Requirements

- Python 3.9 or higher
- See `pyproject.toml` for complete dependency list

## Documentation

For detailed API documentation and tutorials, visit our [documentation site](https://github.com/amithjkamath/dosemetrics) (coming soon).

## Citation

If you use DoseMetrics in your research, please cite:

```bibtex
@software{dosemetrics2024,
  author = {Kamath, Amith},
  title = {DoseMetrics: A Python Library for Radiotherapy Dose Analysis},
  url = {https://github.com/amithjkamath/dosemetrics},
  version = {0.2.0},
  year = {2024}
}
```

## License

This project is licensed under the Creative Commons Attribution-ShareAlike-NonCommercial 4.0 International License - see the [LICENSE](LICENSE) file for details.

**Non-Commercial Use**: This software is freely available for academic, research, and personal use. Commercial use requires explicit written permission from the copyright holder.

For commercial licensing inquiries, please contact the folks at contouraid.

## Contributors

- **Amith Kamath** - *Lead Developer* - [amithjkamath](https://github.com/amithjkamath)

## 🙏 Acknowledgments

- Medical physics community for guidance and feedback
- Open source medical imaging libraries that make this work possible
- Contributors and users who help improve the library

## Support

- **Issues**: [GitHub Issues](https://github.com/amithjkamath/dosemetrics/issues)
- **Discussions**: [GitHub Discussions](https://github.com/amithjkamath/dosemetrics/discussions)