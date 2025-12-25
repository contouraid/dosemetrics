# Quick Start

This guide will help you get started with DoseMetrics in just a few minutes.

## Basic Usage

Here's a simple example that loads dose and structure data, computes a DVH, and creates a visualization:

```python
from dosemetrics import read_dose_and_mask_files, compute_dvh
from dosemetrics.utils.plotting import plot_dvh

# Load dose distribution and structures
dose, structures = read_dose_and_mask_files("path/to/patient_data")

# Get PTV mask
ptv_mask = structures.get_structure_mask("PTV")

# Compute dose-volume histogram
dvh_data = compute_dvh(dose, ptv_mask, organ_name="PTV")

# Create interactive plot
plot_dvh(dvh_data, title="PTV Coverage Analysis")
```

## Working with Multiple Structures

Analyze multiple organs at risk (OARs) and targets:

```python
from dosemetrics import read_dose_and_mask_files, compute_dvh, dvh_by_structure
from dosemetrics.utils.plotting import plot_multiple_dvhs

# Load dose and structures
dose, structures = read_dose_and_mask_files("path/to/patient_data")

# Compute DVHs for all structures
dvh_results = dvh_by_structure(dose, structures)

# Plot all DVHs together
plot_multiple_dvhs(dvh_results, title="Treatment Plan Analysis")
```

## Checking Dose Constraints

Evaluate whether your plan meets clinical constraints:

```python
from dosemetrics import read_dose_and_mask_files, compute_dvh
from dosemetrics.utils.compliance import check_constraint

# Load data
dose, structures = read_dose_and_mask_files("path/to/patient_data")
brainstem_mask = structures.get_structure_mask("Brainstem")

# Compute DVH
dvh_data = compute_dvh(dose, brainstem_mask, organ_name="Brainstem")

# Define constraint: Maximum dose to brainstem < 54 Gy
constraint = {
    "organ": "Brainstem",
    "type": "max",
    "limit": 54.0,
    "unit": "Gy"
}

# Check compliance
is_compliant = check_constraint(dvh_data, constraint)
print(f"Constraint satisfied: {is_compliant}")
```

## Computing Quality Metrics

Calculate plan quality indices:

```python
from dosemetrics import read_dose_and_mask_files
from dosemetrics.metrics.scores import compute_conformity_index, compute_homogeneity_index

# Load dose and PTV
dose, structures = read_dose_and_mask_files("path/to/patient_data")
ptv_mask = structures.get_structure_mask("PTV")

# Compute conformity index
ci = compute_conformity_index(dose, ptv_mask, prescription_dose=60.0)
print(f"Conformity Index: {ci:.3f}")

# Compute homogeneity index
hi = compute_homogeneity_index(dose, ptv_mask)
print(f"Homogeneity Index: {hi:.3f}")
```

## Comparing Two Plans

Compare predicted vs. actual dose distributions:

```python
from dosemetrics.io import read_from_nifti
from dosemetrics.metrics.comparison import compute_dose_difference, compute_gamma_index

# Load two dose distributions
dose_tps = read_from_nifti("path/to/tps_dose.nii.gz")
dose_predicted = read_from_nifti("path/to/predicted_dose.nii.gz")

# Compute absolute difference
diff = compute_dose_difference(dose_tps, dose_predicted, method="absolute")

# Compute gamma index (3%/3mm criteria)
gamma = compute_gamma_index(
    dose_tps, 
    dose_predicted, 
    dose_threshold=3.0,  # 3% dose difference
    distance_threshold=3.0,  # 3mm distance to agreement
    dose_cutoff=0.1  # Ignore doses below 10% of prescription
)

print(f"Gamma pass rate: {gamma.mean():.1f}%")
```

## Export Results

Save your analysis results:

```python
from dosemetrics import read_dose_and_mask_files, compute_dvh
import pandas as pd

# Compute DVH
dose, structures = read_dose_and_mask_files("path/to/patient_data")
mask = structures.get_structure_mask("PTV")
dvh_data = compute_dvh(dose, mask, organ_name="PTV")

# Convert to DataFrame and save
df = pd.DataFrame(dvh_data)
df.to_csv("dvh_results.csv", index=False)

# Save plot
from dosemetrics.utils.plotting import plot_dvh
fig = plot_dvh(dvh_data, title="DVH Analysis")
fig.write_html("dvh_plot.html")
fig.write_image("dvh_plot.png")
```

## Command Line Interface

DoseMetrics also provides a CLI for common operations:

```bash
# Compute DVH from command line
dosemetrics dvh \
  --dose path/to/dose.nii.gz \
  --mask path/to/mask.nii.gz \
  --output dvh_results.csv

# Check constraints
dosemetrics check-constraints \
  --dose path/to/dose.nii.gz \
  --masks path/to/masks/ \
  --constraints constraints.json \
  --output compliance_report.csv
```

## Next Steps

Now that you've seen the basics:

- [Using Your Own Data](../notebooks/getting-started-own-data.ipynb) - Detailed guide for your specific data
- [File Formats](file-formats.md) - Learn about supported file formats
- [User Guide](../user-guide/overview.md) - Comprehensive documentation
- [API Reference](../api/index.md) - Detailed API documentation

## Try the Live Demo

Want to experiment without writing code?

[:material-rocket-launch: Try Interactive Demo on Hugging Face](https://huggingface.co/spaces/contouraid/dosemetrics){ .md-button .md-button--primary target="_blank" }

Upload your data and explore DoseMetrics features through an intuitive web interface!
