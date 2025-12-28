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
from dosemetrics.utils.plot import plot_dvh, compare_dvh
```

[See utils documentation →](utils.md)

**Checking Compliance:**

```python
from dosemetrics.utils.compliance import check_compliance, quality_index
```

[See utils documentation →](utils.md)

## Package Structure

```
dosemetrics/
├── metrics/          # Core calculation functions
│   ├── dvh.py       # DVH computation
│   ├── statistics.py # Dose statistics
│   ├── conformity.py # Conformity indices
│   ├── homogeneity.py # Homogeneity indices
│   └── geometric.py # Geometric metrics
├── io/              # Data I/O
│   ├── dicom_io.py  # DICOM reading
│   └── nifti_io.py  # NIfTI I/O
├── dose.py          # Dose class
├── structures.py    # Structure classes
├── structure_set.py # StructureSet class
└── utils/           # Utilities
    ├── plot.py      # Visualization
    ├── compliance.py # Constraint checking
    ├── comparison.py # Dose comparison
    └── batch.py     # Batch processing
```

## Usage Examples

### Basic Analysis Workflow

```python
from dosemetrics import Dose, Structure
from dosemetrics.metrics.dvh import compute_dvh
from dosemetrics.utils.plot import plot_dvh

# Load data
dose = Dose.from_nifti("dose.nii.gz")
ptv = Structure.from_nifti("ptv.nii.gz", name="PTV")

# Compute DVH
dvh = compute_dvh(dose, ptv)

# Visualize
fig = plot_dvh(dvh, title="PTV Coverage")
fig.show()
```

### Working with Structure Sets

```python
from dosemetrics import Dose, StructureSet
from dosemetrics.io import load_structure_set
from dosemetrics.metrics.dvh import compute_dvh, create_dvh_table

# Load data
dose = Dose.from_nifti("dose.nii.gz")
structures = load_structure_set("structures/")

# Compute DVH for all structures
dvh_table = create_dvh_table(dose, structures)
```

## Type Hints and Return Values

All functions include comprehensive type hints for better IDE support and type checking. Example:

```python
def compute_dvh(
    dose: Dose,
    structure: Structure,
    bins: int = 1000
) -> pd.DataFrame:
    """Compute dose-volume histogram.
    
    Args:
        dose: Dose distribution object
        structure: Structure object with mask
        bins: Number of bins for histogram
        
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
