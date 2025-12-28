"""
Multi-level analysis utilities for dosemetrics.

This module provides functions to analyze dosimetric data at different levels:
- By structure: Analyze a single structure across subjects
- By subject: Analyze all structures for a single subject  
- By dataset: Analyze entire cohorts with summary statistics
- By subset: Filter and analyze specific groups
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from pathlib import Path

from ..dose import Dose
from ..structures import Structure
from ..structure_set import StructureSet


def analyze_by_structure(
    dataset: Dict[str, Dict[str, Union[Dose, StructureSet]]],
    structure_name: str,
    metrics: Dict[str, callable]
) -> pd.DataFrame:
    """
    Analyze a single structure across all subjects.
    
    Computes specified metrics for one structure across the entire dataset,
    useful for population-level structure analysis (e.g., PTV coverage across cohort).
    
    Parameters
    ----------
    dataset : Dict
        Dataset dictionary from batch.load_dataset()
    structure_name : str
        Name of structure to analyze
    metrics : Dict[str, callable]
        Dictionary of {metric_name: metric_function}
        Each function should take (dose, structure) and return a value
    
    Returns
    -------
    results : pd.DataFrame
        DataFrame with subject_id and computed metrics
    
    Examples
    --------
    >>> from dosemetrics.metrics import dvh
    >>> from dosemetrics.utils import analysis
    >>> 
    >>> metrics = {
    ...     'mean_dose': dvh.compute_mean_dose,
    ...     'max_dose': dvh.compute_max_dose,
    ...     'D95': lambda d, s: dvh.compute_dose_at_volume(d, s, 95)
    ... }
    >>> results = analysis.analyze_by_structure(dataset, 'PTV', metrics)
    >>> print(results.describe())  # Summary statistics for PTV across subjects
    """
    results = []
    
    for subject_id, data in dataset.items():
        if 'dose' not in data or 'structures' not in data:
            continue
        
        dose = data['dose']
        structures = data['structures']
        
        # Find the structure
        try:
            structure = structures.get_structure(structure_name)
        except (ValueError, KeyError):
            continue
        
        row = {'subject_id': subject_id}
        
        # Compute all metrics
        for metric_name, metric_func in metrics.items():
            try:
                value = metric_func(dose, structure)
                row[metric_name] = value
            except Exception as e:
                print(f"Warning: Error computing {metric_name} for {subject_id}/{structure_name}: {e}")
                row[metric_name] = np.nan
        
        results.append(row)
    
    return pd.DataFrame(results)


def analyze_by_subject(
    dose: Dose,
    structures: StructureSet,
    metrics: Dict[str, callable],
    structure_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Analyze all structures for a single subject.
    
    Computes metrics for all (or selected) structures in a single subject's dataset.
    
    Parameters
    ----------
    dose : Dose
        Subject's dose distribution
    structures : StructureSet
        Subject's structure set
    metrics : Dict[str, callable]
        Dictionary of {metric_name: metric_function}
    structure_names : List[str], optional
        Specific structures to analyze (default: all)
    
    Returns
    -------
    results : pd.DataFrame
        DataFrame with structure names and computed metrics
    
    Examples
    --------
    >>> from dosemetrics.metrics import dvh
    >>> from dosemetrics.utils import analysis
    >>> 
    >>> dose = Dose.from_dicom('rtdose.dcm')
    >>> structures = StructureSet.from_dicom('rtstruct.dcm')
    >>> 
    >>> metrics = {
    ...     'mean_dose': dvh.compute_mean_dose,
    ...     'V20': lambda d, s: dvh.compute_volume_at_dose(d, s, 20)
    ... }
    >>> results = analysis.analyze_by_subject(dose, structures, metrics)
    """
    results = []
    
    # Determine which structures to analyze
    if structure_names:
        struct_list = [structures.get_structure(name) for name in structure_names 
                      if name in structures.structure_names]
    else:
        # Iterate over structure values only
        struct_list = list(structures.structures.values())
    
    for structure in struct_list:
        row = {'structure': structure.name, 'type': structure.structure_type.value}
        
        # Compute all metrics
        for metric_name, metric_func in metrics.items():
            try:
                value = metric_func(dose, structure)
                row[metric_name] = value
            except Exception as e:
                print(f"Warning: Error computing {metric_name} for {structure.name}: {e}")
                row[metric_name] = np.nan
        
        results.append(row)
    
    return pd.DataFrame(results)


def analyze_by_dataset(
    dataset: Dict[str, Dict[str, Union[Dose, StructureSet]]],
    metrics: Dict[str, callable],
    structure_names: Optional[List[str]] = None,
    summary_stats: bool = True
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Analyze entire dataset with population-level statistics.
    
    Computes metrics across all subjects and structures, with optional
    summary statistics grouped by structure.
    
    Parameters
    ----------
    dataset : Dict
        Dataset dictionary
    metrics : Dict[str, callable]
        Metrics to compute
    structure_names : List[str], optional
        Specific structures to analyze
    summary_stats : bool
        If True, return both detailed and summary dataframes
    
    Returns
    -------
    results : pd.DataFrame or Tuple[pd.DataFrame, pd.DataFrame]
        If summary_stats=False: detailed results
        If summary_stats=True: (detailed_results, summary_stats)
    
    Examples
    --------
    >>> from dosemetrics.metrics import dvh
    >>> from dosemetrics.utils import analysis
    >>> 
    >>> metrics = {
    ...     'mean_dose': dvh.compute_mean_dose,
    ...     'D95': lambda d, s: dvh.compute_dose_at_volume(d, s, 95)
    ... }
    >>> detailed, summary = analysis.analyze_by_dataset(
    ...     dataset, metrics, structure_names=['PTV', 'Heart', 'Lung_L']
    ... )
    >>> print(summary)  # Mean ± std for each metric per structure
    """
    results = []
    
    for subject_id, data in dataset.items():
        if 'dose' not in data or 'structures' not in data:
            continue
        
        dose = data['dose']
        structures = data['structures']
        
        # Determine which structures to analyze
        if structure_names:
            struct_list = [structures.get_structure(name) for name in structure_names 
                          if name in structures.structure_names]
        else:
            struct_list = list(structures.structures.values())
        
        for structure in struct_list:
            row = {
                'subject_id': subject_id,
                'structure': structure.name,
                'type': structure.structure_type.value
            }
            
            # Compute all metrics
            for metric_name, metric_func in metrics.items():
                try:
                    value = metric_func(dose, structure)
                    row[metric_name] = value
                except Exception as e:
                    print(f"Warning: Error computing {metric_name} for {subject_id}/{structure.name}: {e}")
                    row[metric_name] = np.nan
            
            results.append(row)
    
    detailed_df = pd.DataFrame(results)
    
    if not summary_stats:
        return detailed_df
    
    # Compute summary statistics grouped by structure
    metric_cols = list(metrics.keys())
    summary = detailed_df.groupby('structure')[metric_cols].agg(['mean', 'std', 'min', 'max', 'median'])
    
    return detailed_df, summary


def analyze_subset(
    dataset: Dict[str, Dict[str, Union[Dose, StructureSet]]],
    metrics: Dict[str, callable],
    subject_filter: Optional[callable] = None,
    structure_filter: Optional[callable] = None,
    **filter_kwargs
) -> pd.DataFrame:
    """
    Analyze a filtered subset of the dataset.
    
    Apply custom filters to subjects and/or structures before analysis.
    
    Parameters
    ----------
    dataset : Dict
        Dataset dictionary
    metrics : Dict[str, callable]
        Metrics to compute
    subject_filter : callable, optional
        Function that takes (subject_id, data) and returns bool
    structure_filter : callable, optional
        Function that takes (structure) and returns bool
    **filter_kwargs
        Additional filter parameters
    
    Returns
    -------
    results : pd.DataFrame
        Analysis results for filtered subset
    
    Examples
    --------
    >>> # Analyze only target structures
    >>> def target_only(structure):
    ...     return structure.structure_type == StructureType.TARGET
    >>> 
    >>> results = analysis.analyze_subset(
    ...     dataset,
    ...     metrics={'mean_dose': compute_mean_dose},
    ...     structure_filter=target_only
    ... )
    """
    results = []
    
    for subject_id, data in dataset.items():
        # Apply subject filter
        if subject_filter and not subject_filter(subject_id, data):
            continue
        
        if 'dose' not in data or 'structures' not in data:
            continue
        
        dose = data['dose']
        structures = data['structures']
        
        # Filter structures
        if structure_filter:
            struct_list = [s for s in structures.structures.values() if structure_filter(s)]
        else:
            struct_list = list(structures.structures.values())
        
        for structure in struct_list:
            row = {
                'subject_id': subject_id,
                'structure': structure.name,
                'type': structure.structure_type.value
            }
            
            # Compute metrics
            for metric_name, metric_func in metrics.items():
                try:
                    value = metric_func(dose, structure)
                    row[metric_name] = value
                except Exception as e:
                    print(f"Warning: Error computing {metric_name} for {subject_id}/{structure.name}: {e}")
                    row[metric_name] = np.nan
            
            results.append(row)
    
    return pd.DataFrame(results)


def compute_cohort_statistics(
    results: pd.DataFrame,
    metric_cols: Optional[List[str]] = None,
    group_by: str = 'structure'
) -> pd.DataFrame:
    """
    Compute cohort-level summary statistics.
    
    Parameters
    ----------
    results : pd.DataFrame
        Results from analyze_by_dataset or similar
    metric_cols : List[str], optional
        Columns to summarize (default: all numeric)
    group_by : str
        Column to group by (default: 'structure')
    
    Returns
    -------
    statistics : pd.DataFrame
        Summary statistics (mean, std, CI, etc.)
    
    Examples
    --------
    >>> results = analyze_by_dataset(dataset, metrics)
    >>> stats = compute_cohort_statistics(results[0])
    >>> print(stats)  # Population statistics per structure
    """
    if metric_cols is None:
        metric_cols = results.select_dtypes(include=[np.number]).columns.tolist()
    
    summary = results.groupby(group_by)[metric_cols].agg([
        'count',
        'mean',
        'std',
        'min',
        ('q25', lambda x: np.percentile(x, 25)),
        'median',
        ('q75', lambda x: np.percentile(x, 75)),
        'max'
    ])
    
    # Add confidence intervals
    for col in metric_cols:
        if (group_by, col, 'count') in summary.columns or (col, 'count') in summary.columns:
            n = summary[(col, 'count')] if (col, 'count') in summary.columns else summary[(group_by, col, 'count')]
            std = summary[(col, 'std')] if (col, 'std') in summary.columns else summary[(group_by, col, 'std')]
            se = std / np.sqrt(n)
            summary[(col, 'ci_95')] = 1.96 * se
    
    return summary


def compare_cohorts(
    results1: pd.DataFrame,
    results2: pd.DataFrame,
    metric_cols: Optional[List[str]] = None,
    cohort_names: Tuple[str, str] = ('Cohort1', 'Cohort2')
) -> pd.DataFrame:
    """
    Compare two cohorts statistically.
    
    Performs t-tests and computes effect sizes between two groups.
    
    Parameters
    ----------
    results1, results2 : pd.DataFrame
        Results from two different cohorts
    metric_cols : List[str], optional
        Metrics to compare
    cohort_names : Tuple[str, str]
        Names for the cohorts
    
    Returns
    -------
    comparison : pd.DataFrame
        Statistical comparison results
    
    Examples
    --------
    >>> pre_treatment = analyze_by_dataset(pre_data, metrics)
    >>> post_treatment = analyze_by_dataset(post_data, metrics)
    >>> comparison = compare_cohorts(
    ...     pre_treatment[0], post_treatment[0],
    ...     cohort_names=('Pre', 'Post')
    ... )
    """
    from scipy import stats
    
    if metric_cols is None:
        metric_cols = results1.select_dtypes(include=[np.number]).columns.tolist()
    
    comparison_results = []
    
    # Get common structures
    structures1 = set(results1['structure'].unique())
    structures2 = set(results2['structure'].unique())
    common_structures = structures1 & structures2
    
    for structure in common_structures:
        data1 = results1[results1['structure'] == structure]
        data2 = results2[results2['structure'] == structure]
        
        for metric in metric_cols:
            if metric not in data1.columns or metric not in data2.columns:
                continue
            
            values1 = data1[metric].dropna()
            values2 = data2[metric].dropna()
            
            if len(values1) < 2 or len(values2) < 2:
                continue
            
            # T-test
            t_stat, p_value = stats.ttest_ind(values1, values2)
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((len(values1)-1)*values1.std()**2 + (len(values2)-1)*values2.std()**2) / 
                                (len(values1) + len(values2) - 2))
            cohens_d = (values1.mean() - values2.mean()) / pooled_std if pooled_std > 0 else 0
            
            comparison_results.append({
                'structure': structure,
                'metric': metric,
                f'{cohort_names[0]}_mean': values1.mean(),
                f'{cohort_names[0]}_std': values1.std(),
                f'{cohort_names[1]}_mean': values2.mean(),
                f'{cohort_names[1]}_std': values2.std(),
                'difference': values1.mean() - values2.mean(),
                't_statistic': t_stat,
                'p_value': p_value,
                'cohens_d': cohens_d,
                'significant': p_value < 0.05
            })
    
    return pd.DataFrame(comparison_results)
