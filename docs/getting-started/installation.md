# Installation

## Requirements

DoseMetrics requires Python 3.9 or higher. It has been tested on:

- Python 3.9, 3.10, 3.11, and 3.12
- Linux, macOS, and Windows

## Install from PyPI

The easiest way to install DoseMetrics is via pip:

```bash
pip install dosemetrics
```

This will install the core library with all required dependencies.

## Optional Dependencies

### Documentation Tools

If you want to build the documentation locally or contribute to the docs:

```bash
pip install dosemetrics[docs]
```

This includes:

- MkDocs with Material theme
- mkdocstrings for API documentation
- mkdocs-jupyter for notebook integration

## Development Installation

For development, clone the repository and install in editable mode:

```bash
git clone https://github.com/contouraid/dosemetrics.git
cd dosemetrics
```

### Using Make (Recommended)

The repository includes a Makefile for common tasks:

```bash
make setup
```

This will:

- Create a virtual environment
- Install all dependencies (including development tools)
- Set up pre-commit hooks

### Manual Installation

Alternatively, install manually:

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode
pip install -e .

# Install development dependencies
pip install -e ".[docs]"
```

## Verify Installation

Check that DoseMetrics is installed correctly:

```python
import dosemetrics
print(dosemetrics.__version__)
```

You can also run the command-line interface:

```bash
dosemetrics --version
```

## Troubleshooting

### Import Errors

If you encounter import errors, ensure all dependencies are installed:

```bash
pip install --upgrade pip
pip install --force-reinstall dosemetrics
```

### GPU Support

DoseMetrics primarily uses CPU-based computations. However, some operations can benefit from GPU acceleration if you have compatible libraries installed (e.g., CuPy for CUDA-enabled GPUs).

### Memory Issues

For large dose distributions, ensure you have sufficient RAM. As a guideline:

- Small datasets (< 256³): 4-8 GB RAM
- Medium datasets (256³ - 512³): 8-16 GB RAM
- Large datasets (> 512³): 16+ GB RAM

## Next Steps

Now that you have DoseMetrics installed:

1. [Quick Start Guide](quickstart.md) - Run your first analysis
2. [Using Your Own Data](../notebooks/getting-started-own-data.ipynb) - Load and analyze your data
3. [File Formats](file-formats.md) - Learn about supported formats

## Try Without Installing

Don't want to install anything yet? Try our live demo:

[:material-rocket-launch: Launch Live Demo on Hugging Face](https://huggingface.co/spaces/contouraid/dosemetrics){ .md-button .md-button--primary target="_blank" }
