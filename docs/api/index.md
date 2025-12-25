# API Reference

Welcome to the DoseMetrics API documentation. This section provides detailed information about all public classes, functions, and modules.

## Modules Overview

### [:material-calculator: Metrics](metrics.md)
Core functions for dose calculations, DVH generation, quality scoring, and plan comparison.

**Key components:**

- DVH computation and analysis
- Conformity and homogeneity indices
- Dose comparison metrics
- Gamma analysis

### [:material-folder-open: Data I/O](io.md)
Data structures and I/O utilities for reading and writing dose distributions and structure masks.

**Key components:**

- Structure and StructureSet classes
- Load dose from NIfTI, DICOM, NRRD
- Load structure masks
- Save results to various formats
- DICOM RT Structure Set handling

### [:material-tools: Utils](utils.md)
Utility functions for plotting, compliance checking, and data processing.

**Key components:**

- Interactive plotting with Plotly
- Compliance checking against dose constraints
- Data transformation utilities
- Statistical analysis helpers

### [:material-database: Data Structures](data.md)
Classes for managing structure sets and dose distributions.

**Key components:**

- StructureSet class for managing multiple structures
- DoseGrid class for dose distribution handling
- Metadata management

## Quick Navigation

Looking for something specific? Here are common tasks:

**Computing DVH:**

```python
from dosemetrics.metrics.dvh import compute_dvh
```

[See metrics module documentation →](metrics.md)

**Loading Data:**

```python
from dosemetrics import read_dose_and_mask_files, StructureSet
# or
from dosemetrics.io import read_from_nifti, StructureSet
```

[See data module documentation →](io.md)

**Creating Plots:**

```python
from dosemetrics.utils.plotting import plot_dvh, plot_multiple_dvhs
```

[See utils documentation →](utils.md)

**Checking Compliance:**

```python
from dosemetrics.utils.compliance import check_constraint
```

[See utils documentation →](utils.md)

## Package Structure

```
dosemetrics/
├── metrics/          # Core calculation functions
│   ├── dvh.py       # DVH computation
│   ├── scores.py    # Quality metrics
│   └── comparison.py # Plan comparison
├── io/              # Data structures and I/O
│   ├── structures.py # Structure classes
│   ├── structure_set.py # StructureSet class
│   └── data_io.py   # File I/O functions
└── utils/           # Utilities
    ├── plotting.py  # Visualization
    ├── compliance.py # Constraint checking
    └── dose_utils.py # Data processing
```

## Usage Examples

### Basic Analysis Workflow

```python
from dosemetrics import read_dose_and_mask_files, compute_dvh
from dosemetrics.utils.plotting import plot_dvh

# Load data
dose, structures = read_dose_and_mask_files("path/to/data")
mask = structures.get_structure_mask("PTV")

# Compute DVH
dvh = compute_dvh(dose, mask, organ_name="PTV")

# Visualize
plot_dvh(dvh, title="PTV Coverage")
```

### Working with Structure Sets

```python
from dosemetrics import StructureSet, compute_dvh, dvh_by_structure
from dosemetrics.io import read_from_nifti

# Load data
dose = read_from_nifti("dose.nii.gz")
structures = StructureSet.from_folder("structures/")

# Compute DVH for all structures
dvh_results = dvh_by_structure(dose, structures)
```

## Type Hints and Return Values

All functions include comprehensive type hints for better IDE support and type checking. Example:

```python
def compute_dvh(
    dose: np.ndarray,
    mask: np.ndarray,
    organ_name: str = "Structure",
    bins: int = 1000,
    dose_unit: str = "Gy"
) -> pd.DataFrame:
    """Compute dose-volume histogram.
    
    Args:
        dose: 3D dose distribution array
        mask: Binary mask for structure
        organ_name: Name of the structure
        bins: Number of bins for histogram
        dose_unit: Unit of dose values
        
    Returns:
        DataFrame with 'dose' and 'volume' columns
    """
    ...
```

## Conventions

### Coordinate Systems

- All spatial data uses RAS+ orientation (Right, Anterior, Superior)
- Origin is typically at image corner
- Spacing is in mm

### Units

- **Dose**: Gray (Gy) or centigray (cGy)
- **Volume**: cm³ or % of total structure volume
- **Distance**: mm

### Array Shapes

- **3D volumes**: (x, y, z) where z is superior-inferior axis
- **Masks**: Binary (0/1) or labeled (0, 1, 2, ..., N)

## See Also

- [Getting Started Guide](../getting-started/quickstart.md)
- [User Guide](../user-guide/overview.md)
- [Examples](../examples/comparing-plans.md)

[:material-rocket-launch: Try Live Demo](https://huggingface.co/spaces/contouraid/dosemetrics){ .md-button .md-button--primary target="_blank" }
