# DoseMetrics

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://python.org)
[![PyPI version](https://badge.fury.io/py/dosemetrics.svg)](https://badge.fury.io/py/dosemetrics)
[![Tests](https://github.com/amithjkamath/dosemetrics/actions/workflows/python-app.yml/badge.svg)](https://github.com/amithjkamath/dosemetrics/actions/workflows/python-app.yml)
[![License](https://img.shields.io/badge/license-CC%20BY--SA--NC%204.0-green.svg)](LICENSE)
[![Development Status](https://img.shields.io/badge/status-alpha-orange.svg)](https://github.com/amithjkamath/dosemetrics)

A Python library for measuring radiotherapy doses and creating interactive visualizations for radiation therapy treatment planning and analysis.

## Overview

DoseMetrics provides tools for analyzing radiation dose distributions, calculating dose-volume histograms (DVH), evaluating treatment plan quality, and creating publication-ready visualizations. This library is designed for medical physicists, radiation oncologists, and researchers working with radiotherapy treatment planning data.

### Package Structure
- `dosemetrics.metrics`: Core dose calculation functions (DVH, scores, etc.)
- `dosemetrics.io`: File I/O utilities for reading dose and mask data  
- `dosemetrics.utils`: Utility functions for compliance, plotting, etc.
- App-specific code moved to separate `dosemetrics_app` package (not distributed on PyPI)

## Features

- **Dose Analysis**: Calculate and analyze 3D dose distributions
- **DVH Generation**: Create dose-volume histograms for organs at risk (OARs) and targets
- **Quality Metrics**: Compute conformity indices, homogeneity indices, and other plan quality metrics
- **Compliance Checking**: Evaluate dose constraints and treatment plan compliance
- **Interactive Visualizations**: Generate interactive plots using Plotly and Streamlit
- **Comparative Analysis**: Compare predicted vs. actual dose distributions
- **Geometric Analysis**: Compute spatial differences and overlaps between structures
- **Export Capabilities**: Save results in various formats (CSV, PDF, PNG)
- **Command Line Interface**: Basic CLI for common operations

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
cd src && streamlit run dosemetrics_app/app.py
```

Or alternatively, from the root directory:

```bash
PYTHONPATH=src streamlit run src/dosemetrics_app/app.py
```

**Note**: The application includes password authentication that is currently disabled for development/testing. If you need to enable authentication, uncomment the authentication check in `src/dosemetrics_app/app.py` and configure the `secrets.toml` file with user passwords.

This provides a user-friendly interface for uploading NIfTI files, analyzing dose distributions, and generating reports.

## Usage Examples

### Basic DVH Analysis

```python
import dosemetrics

# Load dose and structure data (using the new structure)
dose_data = dosemetrics.read_from_nifti("path/to/dose.nii.gz")
structures = {"PTV": dosemetrics.read_from_nifti("path/to/ptv.nii.gz")}

# Generate DVH
dvh_df = dosemetrics.dvh_by_structure(dose_data, structures)

# Plot DVH
dosemetrics.plot_dvh(dose_data, structures, "output.pdf")
```

### Quality Metrics Calculation

```python
# Calculate dose summary statistics
quality_metrics = dosemetrics.dose_summary(dose_data, structures)

print(f"Quality metrics: {quality_metrics}")
```

### Command Line Interface

```bash
# Generate DVH from command line
dosemetrics dvh dose.nii.gz structure1.nii.gz structure2.nii.gz -o dvh_results.csv

# Compute quality metrics  
dosemetrics quality dose.nii.gz structure1.nii.gz structure2.nii.gz -o quality_report.csv

# Check version
dosemetrics --version
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
├── src/dosemetrics/           # Core library modules (new structure)
│   ├── io/                    # Data I/O utilities
│   │   └── data_io.py        # File loading and data reading
│   ├── metrics/              # Core dose calculations
│   │   ├── dvh.py           # DVH calculation
│   │   ├── exposure.py      # Exposure metrics (formerly metrics.py)
│   │   └── scores.py        # Scoring algorithms
│   └── utils/               # Utility functions
│       ├── comparison.py    # Plan comparison tools
│       ├── compliance.py    # Constraint checking
│       └── plot.py         # Visualization tools
├── src/dosemetrics_app/      # Streamlit web application
│   ├── app.py               # Main application entry point
│   └── tabs/                # Application tab modules
│       └── variations.py    # Variations analysis tab
├── examples/                # Usage examples and scripts
├── tests/                   # Organized unit tests
│   ├── data_io/            # Tests for I/O functionality
│   ├── metrics/            # Tests for metrics calculations
│   └── utils/              # Tests for utility functions
└── data/                   # Sample data for testing
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

- **NIfTI**: `.nii`, `.nii.gz` files

## Development

### Running Tests

Execute the test suite to ensure everything works correctly:

```bash
python -m unittest discover -s tests -p "test_*.py"
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

## Acknowledgments

- Medical physics community for guidance and feedback
- Open source medical imaging libraries that make this work possible
- Contributors and users who help improve the library

## Support

- **Issues**: [GitHub Issues](https://github.com/amithjkamath/dosemetrics/issues)
- **Discussions**: [GitHub Discussions](https://github.com/amithjkamath/dosemetrics/discussions)