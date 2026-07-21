"""
Publication-quality plotting utilities for dosemetrics.

This module provides functions for creating publication-ready plots at different levels:
- Structure-level: Plot data for individual structures (DVH, metrics box plots)
- Subject-level: Plot all structures for one subject
- Dataset-level: Population-level plots (DVH bands, violin plots, comparisons)
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Union, Tuple
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

from ..dose import Dose
from ..structures import Structure
from ..structure_set import StructureSet
from ..metrics import dvh

# Color schemes
DEFAULT_COLORS = plt.cm.tab10.colors
OAR_COLOR = "#1f77b4"  # Blue
TARGET_COLOR = "#d62728"  # Red


def plot_dvh(
    dose: Dose,
    structure: Structure,
    bins: int = 1000,
    relative_volume: bool = True,
    ax: Optional[plt.Axes] = None,
    label: Optional[str] = None,
    color: Optional[str] = None,
    **plot_kwargs,
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
        volumes = volumes / 100.0 * structure.volume_cc()

    # Plot
    if label is None:
        label = structure.name

    plot_kwargs.setdefault("linewidth", 2)
    if color:
        plot_kwargs["color"] = color

    ax.plot(dose_bins, volumes, label=label, **plot_kwargs)

    # Format axis
    ax.set_xlabel("Dose (Gy)", fontsize=12)
    if relative_volume:
        ax.set_ylabel("Volume (%)", fontsize=12)
        ax.set_ylim(0, 105)
    else:
        ax.set_ylabel("Volume (cc)", fontsize=12)

    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return ax


def plot_subject_dvhs(
    dose: Dose,
    structures: StructureSet,
    structure_names: Optional[List[str]] = None,
    bins: int = 1000,
    relative_volume: bool = True,
    color_by_type: bool = True,
    figsize: Tuple[float, float] = (10, 7),
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
        struct_list = [
            structures.get_structure(name)
            for name in structure_names
            if name in structures.structure_names
        ]
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
        colors = {
            s.name: DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
            for i, s in enumerate(struct_list)
        }

    # Plot each DVH
    for structure in struct_list:
        plot_dvh(
            dose,
            structure,
            bins=bins,
            relative_volume=relative_volume,
            ax=ax,
            color=colors[structure.name],
        )

    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_title("Dose-Volume Histograms", fontsize=14, fontweight="bold")

    fig.tight_layout()

    return fig, ax


def plot_dvh_comparison(
    dose1: Dose,
    dose2: Dose,
    structure: Structure,
    labels: Tuple[str, str] = ("Dose 1", "Dose 2"),
    bins: int = 1000,
    relative_volume: bool = True,
    figsize: Tuple[float, float] = (8, 6),
    evaluated_structure: Optional[Structure] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Compare DVHs from two different dose distributions.

    Useful for comparing TPS vs predicted, or different treatment plans.

    Parameters
    ----------
    dose1, dose2 : Dose
        Dose distributions to compare
    structure : Structure
        Structure on the first dose grid
    labels : Tuple[str, str]
        Labels for the two doses
    bins : int
        Number of bins
    relative_volume : bool
        Plot relative vs absolute volume
    figsize : Tuple[float, float]
        Figure size
    evaluated_structure : Structure, optional
        Corresponding structure on the second dose grid. Defaults to
        ``structure`` when both doses share the same grid.

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

    evaluated_structure = evaluated_structure or structure

    # Plot both DVHs
    plot_dvh(
        dose1,
        structure,
        bins=bins,
        relative_volume=relative_volume,
        ax=ax,
        label=labels[0],
        color="#1f77b4",
        linestyle="-",
    )
    plot_dvh(
        dose2,
        evaluated_structure,
        bins=bins,
        relative_volume=relative_volume,
        ax=ax,
        label=labels[1],
        color="#ff7f0e",
        linestyle="--",
    )

    ax.legend()
    ax.set_title(f"DVH Comparison: {structure.name}", fontsize=14, fontweight="bold")

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
    label: Optional[str] = None,
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
        if "dose" not in data or "structures" not in data:
            continue

        dose = data["dose"]
        structures = data["structures"]
        structure = (
            structures.get_structure(structure_name)
            if structure_name in structures
            else None
        )

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
                ax.plot(
                    dose_bins, volumes, alpha=0.1, color=color or "gray", linewidth=1
                )

        except Exception as e:
            print(
                f"Warning: Error computing DVH for {subject_id}/{structure_name}: {e}"
            )

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

    ax.fill_between(
        common_doses,
        lower_percentile,
        upper_percentile,
        alpha=0.3,
        color=color,
        label=f"{label or structure_name} (IQR)",
    )

    if show_median:
        ax.plot(
            common_doses,
            median_dvh,
            color=color,
            linewidth=2,
            label=f"{label or structure_name} (median)",
        )

    # Format
    ax.set_xlabel("Dose (Gy)", fontsize=12)
    if relative_volume:
        ax.set_ylabel("Volume (%)", fontsize=12)
        ax.set_ylim(0, 105)
    else:
        ax.set_ylabel("Volume (cc)", fontsize=12)

    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return ax


def plot_metric_boxplot(
    results: pd.DataFrame,
    metric: str,
    group_by: str = "structure",
    figsize: Tuple[float, float] = (10, 6),
    show_points: bool = True,
    horizontal: bool = False,
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
        ax.set_ylabel(group_by.replace("_", " ").title(), fontsize=12)
    else:
        bp = ax.boxplot(data, labels=groups, patch_artist=True)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_xlabel(group_by.replace("_", " ").title(), fontsize=12)
        plt.xticks(rotation=45, ha="right")

    # Color boxes
    for patch in bp["boxes"]:
        patch.set_facecolor(DEFAULT_COLORS[0])
        patch.set_alpha(0.6)

    # Add individual points
    if show_points:
        for i, (group, d) in enumerate(zip(groups, data)):
            x = np.random.normal(i + 1, 0.04, size=len(d))
            ax.plot(x, d, "o", alpha=0.3, color="black", markersize=4)

    ax.grid(True, alpha=0.3, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    return fig, ax


def plot_metric_comparison(
    results1: pd.DataFrame,
    results2: pd.DataFrame,
    metric: str,
    cohort_names: Tuple[str, str] = ("Cohort 1", "Cohort 2"),
    structure_names: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (12, 6),
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
        results1 = results1[results1["structure"].isin(structure_names)]
        results2 = results2[results2["structure"].isin(structure_names)]

    # Get common structures
    structures1 = set(results1["structure"].unique())
    structures2 = set(results2["structure"].unique())
    common_structures = sorted(structures1 & structures2)

    if not common_structures:
        print("No common structures found")
        return fig, ax

    # Prepare data for grouped box plot
    x_pos = np.arange(len(common_structures))
    width = 0.35

    means1 = [
        results1[results1["structure"] == s][metric].mean() for s in common_structures
    ]
    means2 = [
        results2[results2["structure"] == s][metric].mean() for s in common_structures
    ]

    stds1 = [
        results1[results1["structure"] == s][metric].std() for s in common_structures
    ]
    stds2 = [
        results2[results2["structure"] == s][metric].std() for s in common_structures
    ]

    # Create bars
    ax.bar(
        x_pos - width / 2,
        means1,
        width,
        label=cohort_names[0],
        yerr=stds1,
        capsize=5,
        alpha=0.8,
        color=DEFAULT_COLORS[0],
    )
    ax.bar(
        x_pos + width / 2,
        means2,
        width,
        label=cohort_names[1],
        yerr=stds2,
        capsize=5,
        alpha=0.8,
        color=DEFAULT_COLORS[1],
    )

    # Format
    ax.set_ylabel(metric, fontsize=12)
    ax.set_xlabel("Structure", fontsize=12)
    ax.set_title(f"{metric} Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(common_structures, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    return fig, ax


def plot_dose_slice(
    dose: Dose,
    slice_idx: Optional[int] = None,
    axis: int = 0,
    structures: Optional[StructureSet] = None,
    structure_names: Optional[List[str]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "viridis",
    background: Optional[np.ndarray] = None,
    background_window: Optional[Tuple[float, float]] = None,
    min_dose: Optional[float] = None,
    prescription_dose: Optional[float] = None,
    contour_colors: Optional[Mapping[str, str]] = None,
    alpha: float = 0.82,
    show_colorbar: bool = True,
    figsize: Tuple[float, float] = (10, 8),
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a 2D slice of dose distribution with optional structure contours.

    Parameters
    ----------
    dose : Dose
        Dose distribution
    slice_idx : int, optional
        Slice index. By default, use the slice with the largest cross-section
        of the first selected structure, or the middle slice without structures.
    axis : int
        Array axis to slice along (default: 0, axial for dosemetrics arrays)
    structures : StructureSet, optional
        Structures to overlay
    structure_names : List[str], optional
        Specific structures to show
    vmin, vmax : float, optional
        Dose value range for colormap
    cmap : str
        Colormap name
    background : np.ndarray, optional
        Image volume to draw beneath the dose colorwash
    background_window : Tuple[float, float], optional
        Lower and upper display limits for the background image
    min_dose : float, optional
        Hide dose values below this threshold. Defaults to 5% of maximum dose
        when a background is supplied and zero otherwise.
    prescription_dose : float, optional
        Draw this isodose as a magenta contour
    contour_colors : Mapping[str, str], optional
        Per-structure contour colors
    alpha : float
        Dose colorwash opacity when a background is supplied
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
    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1, or 2; got {axis}")
    if background is not None and background.shape != dose.dose_array.shape:
        raise ValueError("background must have the same shape as the dose array")

    struct_list = []
    if structures is not None:
        names = structure_names or structures.structure_names
        struct_list = [structures[name] for name in names if name in structures]

    if slice_idx is None and struct_list:
        reduce_axes = tuple(index for index in range(3) if index != axis)
        slice_idx = int(np.argmax(struct_list[0].mask.sum(axis=reduce_axes)))
    elif slice_idx is None:
        slice_idx = dose.dose_array.shape[axis] // 2

    def take_slice(array: np.ndarray) -> np.ndarray:
        return np.take(array, slice_idx, axis=axis)

    fig, ax = plt.subplots(figsize=figsize)
    dose_slice = take_slice(dose.dose_array)
    if background is not None:
        background_limits = background_window or (None, None)
        ax.imshow(
            take_slice(background),
            origin="lower",
            cmap="gray",
            vmin=background_limits[0],
            vmax=background_limits[1],
        )

    threshold = min_dose
    if threshold is None:
        threshold = 0.05 * dose.max_dose if background is not None else 0.0
    visible_dose = np.ma.masked_less(dose_slice, threshold)
    im = ax.imshow(
        visible_dose,
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
        interpolation="bilinear",
        alpha=alpha if background is not None else 1.0,
    )

    # Add colorbar
    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Dose (Gy)", fontsize=12)

    colors = contour_colors or {}
    for index, structure in enumerate(struct_list):
        mask_slice = take_slice(structure.mask)
        if not np.any(mask_slice):
            continue
        color = colors.get(structure.name, DEFAULT_COLORS[index % len(DEFAULT_COLORS)])
        ax.contour(mask_slice, levels=[0.5], colors=[color], linewidths=1.8)
        ax.plot([], [], color=color, linewidth=1.8, label=structure.name)

    if prescription_dose is not None and dose_slice.min() <= prescription_dose <= dose_slice.max():
        ax.contour(
            dose_slice,
            levels=[prescription_dose],
            colors=["magenta"],
            linewidths=2.0,
        )
        ax.plot(
            [], [], color="magenta", linewidth=2.0,
            label=f"{prescription_dose:g} Gy isodose",
        )

    if struct_list or prescription_dose is not None:
        ax.legend(loc="upper right", framealpha=0.85)

    ax.set_xlabel("X (pixels)", fontsize=12)
    ax.set_ylabel("Y (pixels)", fontsize=12)
    ax.set_title(
        f"Dose Distribution - Slice {slice_idx}", fontsize=14, fontweight="bold"
    )

    fig.tight_layout()

    return fig, ax


def plot_metric_values(
    values: Mapping[str, float],
    title: str = "Metric values",
    ylabel: str = "Value",
    ylim: Optional[Tuple[float, float]] = None,
    horizontal: bool = False,
    figsize: Tuple[float, float] = (9, 5),
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot a small mapping of scalar metric names to values.

    This convenience function keeps routine metric visualization free of
    notebook-specific pandas and Matplotlib setup.
    """
    fig, ax = plt.subplots(figsize=figsize)
    labels = list(values)
    data = list(values.values())
    colors = [DEFAULT_COLORS[index % len(DEFAULT_COLORS)] for index in range(len(data))]

    if horizontal:
        bars = ax.barh(labels, data, color=colors)
        ax.bar_label(bars, fmt="%.3f", padding=3)
        ax.set_xlabel(ylabel)
    else:
        bars = ax.bar(labels, data, color=colors)
        ax.bar_label(bars, fmt="%.3f", padding=3)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=30)

    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_title(title)
    ax.grid(axis="y" if not horizontal else "x", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig, ax


def plot_dose_difference(
    reference: Dose,
    evaluated: Dose,
    structures: Optional[StructureSet] = None,
    structure_names: Optional[List[str]] = None,
    slice_idx: Optional[int] = None,
    axis: int = 0,
    percentile: float = 99.0,
    figsize: Tuple[float, float] = (9, 7),
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot an evaluated-minus-reference dose slice on a symmetric scale."""
    if reference.shape != evaluated.shape:
        raise ValueError("reference and evaluated dose arrays must have matching shapes")
    difference = evaluated.dose_array - reference.dose_array

    struct_list = []
    if structures is not None:
        names = structure_names or structures.structure_names
        struct_list = [structures[name] for name in names if name in structures]
    if slice_idx is None and struct_list:
        reduce_axes = tuple(index for index in range(3) if index != axis)
        slice_idx = int(np.argmax(struct_list[0].mask.sum(axis=reduce_axes)))
    elif slice_idx is None:
        slice_idx = difference.shape[axis] // 2

    difference_slice = np.take(difference, slice_idx, axis=axis)
    limit = float(np.percentile(np.abs(difference), percentile))
    if limit == 0:
        limit = 1.0
    fig, ax = plt.subplots(figsize=figsize)
    image = ax.imshow(
        difference_slice,
        origin="lower",
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
    )
    for index, structure in enumerate(struct_list):
        mask_slice = np.take(structure.mask, slice_idx, axis=axis)
        if np.any(mask_slice):
            color = DEFAULT_COLORS[index % len(DEFAULT_COLORS)]
            ax.contour(mask_slice, levels=[0.5], colors=[color], linewidths=1.5)
            ax.plot([], [], color=color, label=structure.name)
    if struct_list:
        ax.legend(loc="upper right")
    ax.set_title(f"Dose difference (evaluated - reference) · slice {slice_idx}")
    ax.set_xlabel("x voxel")
    ax.set_ylabel("y voxel")
    fig.colorbar(image, ax=ax, label="Dose difference (Gy)", shrink=0.82)
    fig.tight_layout()
    return fig, ax


def save_figure(
    fig: plt.Figure,
    filepath: Union[str, Path],
    dpi: int = 300,
    formats: List[str] = ["png"],
    **savefig_kwargs,
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

    savefig_kwargs.setdefault("bbox_inches", "tight")
    savefig_kwargs.setdefault("dpi", dpi)

    for fmt in formats:
        output_path = filepath.with_suffix(f".{fmt}")
        fig.savefig(output_path, **savefig_kwargs)
        print(f"Saved: {output_path}")


def plot_dvh_score_breakdown(
    dose_reference: Dose,
    dose_evaluated: Dose,
    structure: Structure,
    labels: Tuple[str, str] = ("Reference", "Evaluated"),
    bins: int = 500,
    figsize: Tuple[float, float] = (9, 6),
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot DVH comparison with D1, D95, and D99 markers for both distributions.

    Visualises the DVH Score metric by overlaying the three key dose points
    (D1, D95, D99) on paired DVH curves, making it easy to identify where the
    distributions diverge clinically.

    Parameters
    ----------
    dose_reference : Dose
        Reference dose distribution.
    dose_evaluated : Dose
        Evaluated (e.g., predicted) dose distribution.
    structure : Structure
        Structure to restrict DVH computation to.
    labels : Tuple[str, str]
        Display labels for reference and evaluated curves.
    bins : int
        Number of dose bins for DVH computation.
    figsize : Tuple[float, float]
        Figure size.

    Returns
    -------
    fig, ax : Figure and Axes

    Examples
    --------
    >>> fig, ax = plot.plot_dvh_score_breakdown(
    ...     tps_dose, predicted_dose, ptv,
    ...     labels=('TPS', 'Predicted')
    ... )
    >>> plt.show()
    """
    from ..metrics.dvh import compare_dvh_score

    fig, ax = plt.subplots(figsize=figsize)

    # Plot DVH curves
    plot_dvh(
        dose_reference,
        structure,
        bins=bins,
        ax=ax,
        label=labels[0],
        color="#1f77b4",
        linestyle="-",
    )
    plot_dvh(
        dose_evaluated,
        structure,
        bins=bins,
        ax=ax,
        label=labels[1],
        color="#ff7f0e",
        linestyle="--",
    )

    # Compute D1, D95, D99 for both
    def _dvh_point(dose_obj, vol_pct):
        vals = dose_obj.get_dose_in_structure(structure)
        return (
            float(np.percentile(vals, 100 - vol_pct)) if len(vals) > 0 else float("nan")
        )

    marker_specs = [
        (1, "D1", 95),
        (95, "D95", 5),
        (99, "D99", 1),
    ]

    for vol_pct, label_str, y_pos in marker_specs:
        d_ref = _dvh_point(dose_reference, vol_pct)
        d_eval = _dvh_point(dose_evaluated, vol_pct)

        ax.axvline(d_ref, color="#1f77b4", linestyle=":", alpha=0.6, linewidth=1.0)
        ax.axvline(d_eval, color="#ff7f0e", linestyle=":", alpha=0.6, linewidth=1.0)

        # Annotate the gap
        mid = (d_ref + d_eval) / 2.0
        diff = abs(d_ref - d_eval)
        ax.annotate(
            f"{label_str}\nΔ={diff:.1f} Gy",
            xy=(mid, y_pos),
            fontsize=8,
            ha="center",
            color="gray",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
        )

    dvh_score = compare_dvh_score(dose_reference, dose_evaluated, structure)
    ax.set_title(
        f"DVH Score: {structure.name} — DVH Score = {dvh_score:.2f} Gy",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend()
    fig.tight_layout()

    return fig, ax


def plot_dvh_auc(
    dose: Dose,
    structure: Structure,
    bins: int = 500,
    normalize: bool = True,
    ax: Optional[plt.Axes] = None,
    color: Optional[str] = None,
    label: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 6),
) -> plt.Axes:
    """
    Plot DVH with the area under the curve filled and the AUC annotated.

    Parameters
    ----------
    dose : Dose
        Dose distribution.
    structure : Structure
        Structure for DVH computation.
    bins : int
        Number of dose bins.
    normalize : bool
        If True, normalise AUC to [0, 1].
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. Creates new figure if None.
    color : str, optional
        Fill and line colour (defaults to target red).
    label : str, optional
        Curve label (defaults to structure name).
    figsize : Tuple[float, float]
        Figure size when creating a new figure.

    Returns
    -------
    ax : matplotlib.axes.Axes

    Examples
    --------
    >>> ax = plot.plot_dvh_auc(dose, ptv)
    >>> plt.show()
    """
    from ..metrics.dvh import compute_dvh_auc

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    color = color or TARGET_COLOR
    label = label or structure.name

    dose_bins, volumes = dvh.compute_dvh(dose, structure, step_size=0.5)
    ax.plot(dose_bins, volumes, color=color, linewidth=2, label=label)
    ax.fill_between(dose_bins, volumes, alpha=0.2, color=color)

    auc = compute_dvh_auc(dose, structure, num_bins=bins, normalize=normalize)
    auc_label = f"AUC = {auc:.3f}" + (" (norm.)" if normalize else " Gy·%")
    ax.text(
        0.97,
        0.97,
        auc_label,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    ax.set_xlabel("Dose (Gy)", fontsize=12)
    ax.set_ylabel("Volume (%)", fontsize=12)
    ax.set_title(f"DVH AUC: {structure.name}", fontsize=13, fontweight="bold")
    ax.legend()

    return ax
