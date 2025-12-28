"""
Publication-quality plotting utilities for dosemetrics.

This module provides functions for creating publication-ready plots at different levels:
- Structure-level: Plot data for individual structures (DVH, metrics box plots)
- Subject-level: Plot all structures for one subject
- Dataset-level: Population-level plots (DVH bands, violin plots, comparisons)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union, Tuple, Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from pathlib import Path
import pandas as pd

from ..dose import Dose
from ..structures import Structure
from ..structure_set import StructureSet
from ..metrics import dvh


# Color schemes
DEFAULT_COLORS = plt.cm.tab10.colors
OAR_COLOR = '#1f77b4'  # Blue
TARGET_COLOR = '#d62728'  # Red


def plot_dvh(
    dose: Dose,
    structure: Structure,
    bins: int = 1000,
    relative_volume: bool = True,
    ax: Optional[plt.Axes] = None,
    label: Optional[str] = None,
    color: Optional[str] = None,
    **plot_kwargs
) -> plt.Axes:
    """
    Plot dose-volume histogram for a single structure.
    
    Parameters
    ----------
    dose : Dose
        Dose distribution
    structure : Structure
        Structure to plot DVH for
    bins : int
        Number of bins for DVH computation
    relative_volume : bool
        If True, plot relative volume (%), else absolute volume (cc)
    ax : plt.Axes, optional
        Axis to plot on (creates new if None)
    label : str, optional
        Label for the curve (default: structure name)
    color : str, optional
        Color for the curve
    **plot_kwargs
        Additional arguments passed to plt.plot()
    
    Returns
    -------
    ax : plt.Axes
        The plot axis
    
    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from dosemetrics.utils import plot
    >>> 
    >>> fig, ax = plt.subplots()
    >>> plot.plot_dvh(dose, ptv, ax=ax, label='PTV', color='red')
    >>> plot.plot_dvh(dose, heart, ax=ax, label='Heart', color='blue')
    >>> plt.legend()
    >>> plt.show()
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Compute DVH
    # Convert bins to step_size (approximate)
    max_dose = dose.max_dose
    step_size = max_dose / bins if bins > 0 else 0.1
    dose_bins, volumes = dvh.compute_dvh(dose, structure, step_size=step_size)
    
    if not relative_volume:
        # DVH returns relative volume by default, convert to absolute if needed
        volumes = volumes / 100.0 * structure.volume_cc if hasattr(structure, 'volume_cc') else volumes
    
    # Plot
    if label is None:
        label = structure.name
    
    plot_kwargs.setdefault('linewidth', 2)
    if color:
        plot_kwargs['color'] = color
    
    ax.plot(dose_bins, volumes, label=label, **plot_kwargs)
    
    # Format axis
    ax.set_xlabel('Dose (Gy)', fontsize=12)
    if relative_volume:
        ax.set_ylabel('Volume (%)', fontsize=12)
        ax.set_ylim(0, 105)
    else:
        ax.set_ylabel('Volume (cc)', fontsize=12)
    
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    return ax


def plot_subject_dvhs(
    dose: Dose,
    structures: StructureSet,
    structure_names: Optional[List[str]] = None,
    bins: int = 1000,
    relative_volume: bool = True,
    color_by_type: bool = True,
    figsize: Tuple[float, float] = (10, 7)
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot DVHs for all structures of a subject.
    
    Parameters
    ----------
    dose : Dose
        Dose distribution
    structures : StructureSet
        Structure set
    structure_names : List[str], optional
        Specific structures to plot (default: all)
    bins : int
        Number of bins
    relative_volume : bool
        Plot relative vs absolute volume
    color_by_type : bool
        Use different colors for targets vs OARs
    figsize : Tuple[float, float]
        Figure size
    
    Returns
    -------
    fig, ax : Figure and Axes
    
    Examples
    --------
    >>> from dosemetrics.utils import plot
    >>> fig, ax = plot.plot_subject_dvhs(dose, structures)
    >>> plt.savefig('subject_dvhs.png', dpi=300, bbox_inches='tight')
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Filter structures
    if structure_names:
        struct_list = [structures.get_structure(name) for name in structure_names 
                      if name in structures.structure_names]
    else:
        struct_list = list(structures.structures.values())
    
    # Assign colors
    if color_by_type:
        from ..structures import StructureType
        colors = {}
        for s in struct_list:
            if s.structure_type == StructureType.TARGET:
                colors[s.name] = TARGET_COLOR
            else:
                colors[s.name] = OAR_COLOR
    else:
        colors = {s.name: DEFAULT_COLORS[i % len(DEFAULT_COLORS)] 
                 for i, s in enumerate(struct_list)}
    
    # Plot each DVH
    for structure in struct_list:
        plot_dvh(dose, structure, bins=bins, relative_volume=relative_volume,
                ax=ax, color=colors[structure.name])
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_title('Dose-Volume Histograms', fontsize=14, fontweight='bold')
    
    fig.tight_layout()
    
    return fig, ax


def plot_dvh_comparison(
    dose1: Dose,
    dose2: Dose,
    structure: Structure,
    labels: Tuple[str, str] = ('Dose 1', 'Dose 2'),
    bins: int = 1000,
    relative_volume: bool = True,
    figsize: Tuple[float, float] = (8, 6)
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Compare DVHs from two different dose distributions.
    
    Useful for comparing TPS vs predicted, or different treatment plans.
    
    Parameters
    ----------
    dose1, dose2 : Dose
        Dose distributions to compare
    structure : Structure
        Structure to analyze
    labels : Tuple[str, str]
        Labels for the two doses
    bins : int
        Number of bins
    relative_volume : bool
        Plot relative vs absolute volume
    figsize : Tuple[float, float]
        Figure size
    
    Returns
    -------
    fig, ax : Figure and Axes
    
    Examples
    --------
    >>> fig, ax = plot.plot_dvh_comparison(
    ...     tps_dose, pred_dose, ptv,
    ...     labels=('TPS', 'Predicted')
    ... )
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot both DVHs
    plot_dvh(dose1, structure, bins=bins, relative_volume=relative_volume,
            ax=ax, label=labels[0], color='#1f77b4', linestyle='-')
    plot_dvh(dose2, structure, bins=bins, relative_volume=relative_volume,
            ax=ax, label=labels[1], color='#ff7f0e', linestyle='--')
    
    ax.legend()
    ax.set_title(f'DVH Comparison: {structure.name}', fontsize=14, fontweight='bold')
    
    fig.tight_layout()
    
    return fig, ax


def plot_dvh_band(
    dataset: Dict[str, Dict[str, Union[Dose, StructureSet]]],
    structure_name: str,
    bins: int = 1000,
    relative_volume: bool = True,
    percentiles: Tuple[float, float] = (25, 75),
    show_median: bool = True,
    show_individual: bool = False,
    ax: Optional[plt.Axes] = None,
    color: Optional[str] = None,
    label: Optional[str] = None
) -> plt.Axes:
    """
    Plot DVH band showing population statistics.
    
    Creates a band plot showing median and interquartile range across
    multiple subjects for a single structure.
    
    Parameters
    ----------
    dataset : Dict
        Dataset dictionary from batch.load_dataset()
    structure_name : str
        Structure to plot
    bins : int
        Number of bins
    relative_volume : bool
        Plot relative vs absolute volume
    percentiles : Tuple[float, float]
        Lower and upper percentiles for band
    show_median : bool
        Whether to show median curve
    show_individual : bool
        Whether to show individual DVHs with transparency
    ax : plt.Axes, optional
        Axis to plot on
    color : str, optional
        Color for the band
    label : str, optional
        Label for the legend
    
    Returns
    -------
    ax : plt.Axes
    
    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> plot.plot_dvh_band(dataset, 'PTV', ax=ax, color='red', label='PTV')
    >>> plot.plot_dvh_band(dataset, 'Heart', ax=ax, color='blue', label='Heart')
    >>> plt.legend()
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 7))
    
    # Collect DVHs from all subjects
    all_dvhs = []
    max_dose = 0
    
    for subject_id, data in dataset.items():
        if 'dose' not in data or 'structures' not in data:
            continue
        
        dose = data['dose']
        structures = data['structures']
        structure = structures.get_structure(structure_name) if structure_name in structures else None
        
        if structure is None:
            continue
        
        try:
            max_dose_val = dose.max_dose
            step_size = max_dose_val / bins if bins > 0 else 0.1
            dose_bins, volumes = dvh.compute_dvh(dose, structure, step_size=step_size)
            
            # volumes are already in percentage (0-100)
            
            all_dvhs.append((dose_bins, volumes))
            max_dose = max(max_dose, dose_bins[-1])
            
            # Plot individual if requested
            if show_individual:
                ax.plot(dose_bins, volumes, alpha=0.1, color=color or 'gray', linewidth=1)
        
        except Exception as e:
            print(f"Warning: Error computing DVH for {subject_id}/{structure_name}: {e}")
    
    if not all_dvhs:
        print(f"No valid DVHs found for {structure_name}")
        return ax
    
    # Create common dose axis
    common_doses = np.linspace(0, max_dose, bins)
    
    # Interpolate all DVHs to common dose axis
    interpolated_dvhs = []
    for dose_bins, volumes in all_dvhs:
        interp_volumes = np.interp(common_doses, dose_bins, volumes)
        interpolated_dvhs.append(interp_volumes)
    
    dvh_array = np.array(interpolated_dvhs)
    
    # Compute statistics
    median_dvh = np.median(dvh_array, axis=0)
    lower_percentile = np.percentile(dvh_array, percentiles[0], axis=0)
    upper_percentile = np.percentile(dvh_array, percentiles[1], axis=0)
    
    # Plot band
    if color is None:
        color = DEFAULT_COLORS[0]
    
    ax.fill_between(common_doses, lower_percentile, upper_percentile,
                    alpha=0.3, color=color, label=f'{label or structure_name} (IQR)')
    
    if show_median:
        ax.plot(common_doses, median_dvh, color=color, linewidth=2,
               label=f'{label or structure_name} (median)')
    
    # Format
    ax.set_xlabel('Dose (Gy)', fontsize=12)
    if relative_volume:
        ax.set_ylabel('Volume (%)', fontsize=12)
        ax.set_ylim(0, 105)
    else:
        ax.set_ylabel('Volume (cc)', fontsize=12)
    
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    return ax


def plot_metric_boxplot(
    results: pd.DataFrame,
    metric: str,
    group_by: str = 'structure',
    figsize: Tuple[float, float] = (10, 6),
    show_points: bool = True,
    horizontal: bool = False
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create box plot for a metric across structures or subjects.
    
    Parameters
    ----------
    results : pd.DataFrame
        Results from analysis functions
    metric : str
        Metric column to plot
    group_by : str
        Column to group by ('structure' or 'subject_id')
    figsize : Tuple[float, float]
        Figure size
    show_points : bool
        Whether to show individual data points
    horizontal : bool
        Whether to make horizontal box plot
    
    Returns
    -------
    fig, ax : Figure and Axes
    
    Examples
    --------
    >>> from dosemetrics.utils import analysis, plot
    >>> results = analysis.analyze_by_dataset(dataset, metrics)
    >>> fig, ax = plot.plot_metric_boxplot(results[0], 'mean_dose')
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Prepare data
    groups = results[group_by].unique()
    data = [results[results[group_by] == g][metric].dropna() for g in groups]
    
    # Create box plot
    if horizontal:
        bp = ax.boxplot(data, labels=groups, vert=False, patch_artist=True)
        ax.set_xlabel(metric, fontsize=12)
        ax.set_ylabel(group_by.replace('_', ' ').title(), fontsize=12)
    else:
        bp = ax.boxplot(data, labels=groups, patch_artist=True)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_xlabel(group_by.replace('_', ' ').title(), fontsize=12)
        plt.xticks(rotation=45, ha='right')
    
    # Color boxes
    for patch in bp['boxes']:
        patch.set_facecolor(DEFAULT_COLORS[0])
        patch.set_alpha(0.6)
    
    # Add individual points
    if show_points:
        for i, (group, d) in enumerate(zip(groups, data)):
            x = np.random.normal(i + 1, 0.04, size=len(d))
            ax.plot(x, d, 'o', alpha=0.3, color='black', markersize=4)
    
    ax.grid(True, alpha=0.3, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    fig.tight_layout()
    
    return fig, ax


def plot_metric_comparison(
    results1: pd.DataFrame,
    results2: pd.DataFrame,
    metric: str,
    cohort_names: Tuple[str, str] = ('Cohort 1', 'Cohort 2'),
    structure_names: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (12, 6)
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Compare a metric between two cohorts.
    
    Creates side-by-side box plots for comparison.
    
    Parameters
    ----------
    results1, results2 : pd.DataFrame
        Results from two cohorts
    metric : str
        Metric to compare
    cohort_names : Tuple[str, str]
        Names for the cohorts
    structure_names : List[str], optional
        Specific structures to include
    figsize : Tuple[float, float]
        Figure size
    
    Returns
    -------
    fig, ax : Figure and Axes
    
    Examples
    --------
    >>> fig, ax = plot.plot_metric_comparison(
    ...     pre_results, post_results, 'mean_dose',
    ...     cohort_names=('Pre-treatment', 'Post-treatment')
    ... )
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Filter structures if specified
    if structure_names:
        results1 = results1[results1['structure'].isin(structure_names)]
        results2 = results2[results2['structure'].isin(structure_names)]
    
    # Get common structures
    structures1 = set(results1['structure'].unique())
    structures2 = set(results2['structure'].unique())
    common_structures = sorted(structures1 & structures2)
    
    if not common_structures:
        print("No common structures found")
        return fig, ax
    
    # Prepare data for grouped box plot
    x_pos = np.arange(len(common_structures))
    width = 0.35
    
    means1 = [results1[results1['structure'] == s][metric].mean() for s in common_structures]
    means2 = [results2[results2['structure'] == s][metric].mean() for s in common_structures]
    
    stds1 = [results1[results1['structure'] == s][metric].std() for s in common_structures]
    stds2 = [results2[results2['structure'] == s][metric].std() for s in common_structures]
    
    # Create bars
    ax.bar(x_pos - width/2, means1, width, label=cohort_names[0],
          yerr=stds1, capsize=5, alpha=0.8, color=DEFAULT_COLORS[0])
    ax.bar(x_pos + width/2, means2, width, label=cohort_names[1],
          yerr=stds2, capsize=5, alpha=0.8, color=DEFAULT_COLORS[1])
    
    # Format
    ax.set_ylabel(metric, fontsize=12)
    ax.set_xlabel('Structure', fontsize=12)
    ax.set_title(f'{metric} Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(common_structures, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    fig.tight_layout()
    
    return fig, ax


def plot_dose_slice(
    dose: Dose,
    slice_idx: Optional[int] = None,
    axis: int = 2,
    structures: Optional[StructureSet] = None,
    structure_names: Optional[List[str]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = 'viridis',
    show_colorbar: bool = True,
    figsize: Tuple[float, float] = (10, 8)
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a 2D slice of dose distribution with optional structure contours.
    
    Parameters
    ----------
    dose : Dose
        Dose distribution
    slice_idx : int, optional
        Slice index (default: middle slice)
    axis : int
        Axis to slice along (0=sagittal, 1=coronal, 2=axial)
    structures : StructureSet, optional
        Structures to overlay
    structure_names : List[str], optional
        Specific structures to show
    vmin, vmax : float, optional
        Dose value range for colormap
    cmap : str
        Colormap name
    show_colorbar : bool
        Whether to show colorbar
    figsize : Tuple[float, float]
        Figure size
    
    Returns
    -------
    fig, ax : Figure and Axes
    
    Examples
    --------
    >>> fig, ax = plot.plot_dose_slice(
    ...     dose, structures=structures,
    ...     structure_names=['PTV', 'Heart']
    ... )
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get middle slice if not specified
    if slice_idx is None:
        slice_idx = dose.dose_array.shape[axis] // 2
    
    # Extract slice
    if axis == 0:
        dose_slice = dose.dose_array[slice_idx, :, :]
    elif axis == 1:
        dose_slice = dose.dose_array[:, slice_idx, :]
    else:  # axis == 2
        dose_slice = dose.dose_array[:, :, slice_idx]
    
    # Plot dose
    im = ax.imshow(dose_slice.T, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax,
                   aspect='equal', interpolation='bilinear')
    
    # Add colorbar
    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Dose (Gy)', fontsize=12)
    
    # Overlay structure contours
    if structures:
        struct_list = [s for s in structures if s.name in structure_names] if structure_names else list(structures)
        
        for i, structure in enumerate(struct_list):
            # Get contour on this slice
            # Note: This is a simplified version - actual implementation would need
            # proper coordinate transformation and contour extraction
            color = DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
            
            # Placeholder for contour plotting
            # In practice, you'd extract the contour points for this slice
            # and plot them using ax.plot()
    
    ax.set_xlabel('X (pixels)', fontsize=12)
    ax.set_ylabel('Y (pixels)', fontsize=12)
    ax.set_title(f'Dose Distribution - Slice {slice_idx}', fontsize=14, fontweight='bold')
    
    fig.tight_layout()
    
    return fig, ax


def save_figure(
    fig: plt.Figure,
    filepath: Union[str, Path],
    dpi: int = 300,
    formats: List[str] = ['png'],
    **savefig_kwargs
) -> None:
    """
    Save figure in multiple formats with publication-quality settings.
    
    Parameters
    ----------
    fig : plt.Figure
        Figure to save
    filepath : str or Path
        Output path (without extension)
    dpi : int
        Resolution for raster formats
    formats : List[str]
        Formats to save (e.g., ['png', 'pdf', 'svg'])
    **savefig_kwargs
        Additional arguments for fig.savefig()
    
    Examples
    --------
    >>> fig, ax = plot.plot_dvh(dose, structure)
    >>> plot.save_figure(fig, 'figures/ptv_dvh', formats=['png', 'pdf'])
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    savefig_kwargs.setdefault('bbox_inches', 'tight')
    savefig_kwargs.setdefault('dpi', dpi)
    
    for fmt in formats:
        output_path = filepath.with_suffix(f'.{fmt}')
        fig.savefig(output_path, **savefig_kwargs)
        print(f"Saved: {output_path}")
