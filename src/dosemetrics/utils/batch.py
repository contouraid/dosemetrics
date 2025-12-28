"""
Batch processing utilities for dosemetrics.

This module provides high-level functions for processing multiple subjects,
datasets, and performing batch dosimetric analysis across entire cohorts.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Callable
import pandas as pd
import numpy as np
from collections import defaultdict

from ..dose import Dose
from ..structures import Structure
from ..structure_set import StructureSet
from ..io import load_from_folder, detect_folder_format


def load_dataset(
    root_path: Union[str, Path],
    subject_pattern: str = "*",
    dose_pattern: str = "dose*",
    structures_pattern: str = "*.nii.gz",
    auto_detect: bool = True
) -> Dict[str, Dict[str, Union[Dose, StructureSet]]]:
    """
    Load an entire dataset with multiple subjects.
    
    Automatically detects folder structure and loads all doses and structure sets.
    Supports both DICOM and NIfTI formats with automatic detection.
    
    Parameters
    ----------
    root_path : str or Path
        Root directory containing subject folders
    subject_pattern : str
        Glob pattern for subject folder names (default: "*")
    dose_pattern : str
        Pattern to identify dose files/folders
    structures_pattern : str
        Pattern to identify structure files
    auto_detect : bool
        Automatically detect DICOM vs NIfTI format
    
    Returns
    -------
    dataset : Dict[str, Dict[str, Union[Dose, StructureSet]]]
        Nested dictionary: {subject_id: {'dose': Dose, 'structures': StructureSet}}
    
    Examples
    --------
    >>> dataset = load_dataset('/data/clinical_study')
    >>> for subject_id, data in dataset.items():
    ...     dose = data['dose']
    ...     structures = data['structures']
    ...     print(f"Subject {subject_id}: {len(structures)} structures")
    """
    root_path = Path(root_path)
    dataset = {}
    
    # Find all subject folders
    subject_folders = sorted(root_path.glob(subject_pattern))
    
    for subject_folder in subject_folders:
        if not subject_folder.is_dir():
            continue
            
        subject_id = subject_folder.name
        
        try:
            # Try to load the entire folder
            if auto_detect:
                format_type = detect_folder_format(str(subject_folder))
            else:
                format_type = None
            
            # Load dose and structures
            result = load_from_folder(str(subject_folder))
            
            if result:
                dataset[subject_id] = result
                
        except Exception as e:
            print(f"Warning: Could not load subject {subject_id}: {e}")
            continue
    
    return dataset


def load_multiple_doses(
    folder_paths: List[Union[str, Path]],
    dose_names: Optional[List[str]] = None
) -> Dict[str, Dose]:
    """
    Load multiple dose distributions from different folders.
    
    Useful for comparing different treatment plans (e.g., TPS vs predicted).
    
    Parameters
    ----------
    folder_paths : List[str or Path]
        List of folders, each containing a dose distribution
    dose_names : List[str], optional
        Names for each dose (default: uses folder names)
    
    Returns
    -------
    doses : Dict[str, Dose]
        Dictionary mapping dose names to Dose objects
    
    Examples
    --------
    >>> doses = load_multiple_doses([
    ...     '/data/subject01/tps',
    ...     '/data/subject01/predicted'
    ... ], dose_names=['TPS', 'Predicted'])
    """
    doses = {}
    
    for i, folder_path in enumerate(folder_paths):
        folder_path = Path(folder_path)
        
        if dose_names and i < len(dose_names):
            name = dose_names[i]
        else:
            name = folder_path.name
        
        try:
            result = load_from_folder(str(folder_path))
            if result and 'dose' in result:
                doses[name] = result['dose']
        except Exception as e:
            print(f"Warning: Could not load dose from {folder_path}: {e}")
    
    return doses


def process_dataset_with_metric(
    dataset: Dict[str, Dict[str, Union[Dose, StructureSet]]],
    metric_func: Callable,
    structure_names: Optional[List[str]] = None,
    **metric_kwargs
) -> pd.DataFrame:
    """
    Apply a metric function across an entire dataset.
    
    Computes metrics for all subjects and all structures, returning results
    in a structured DataFrame.
    
    Parameters
    ----------
    dataset : Dict
        Dataset dictionary from load_dataset()
    metric_func : Callable
        Metric function that takes (dose, structure) and returns a value or dict
    structure_names : List[str], optional
        Specific structures to analyze (default: all structures)
    **metric_kwargs
        Additional keyword arguments passed to metric_func
    
    Returns
    -------
    results : pd.DataFrame
        DataFrame with columns: subject_id, structure_name, metric values
    
    Examples
    --------
    >>> from dosemetrics.metrics import dvh
    >>> dataset = load_dataset('/data/study')
    >>> results = process_dataset_with_metric(
    ...     dataset,
    ...     dvh.compute_mean_dose,
    ...     structure_names=['PTV', 'Heart']
    ... )
    """
    results = []
    
    for subject_id, data in dataset.items():
        if 'dose' not in data or 'structures' not in data:
            continue
        
        dose = data['dose']
        structures = data['structures']
        
        # Determine which structures to process
        if structure_names:
            struct_list = [structures.get_structure(name) for name in structure_names 
                          if name in structures.structure_names]
        else:
            struct_list = list(structures.structures.values())
        
        for structure in struct_list:
            try:
                # Call the metric function
                result = metric_func(dose, structure, **metric_kwargs)
                
                # Handle different return types
                if isinstance(result, dict):
                    row = {'subject_id': subject_id, 'structure': structure.name}
                    row.update(result)
                else:
                    row = {
                        'subject_id': subject_id,
                        'structure': structure.name,
                        'value': result
                    }
                
                results.append(row)
                
            except Exception as e:
                print(f"Warning: Error processing {subject_id}/{structure.name}: {e}")
                continue
    
    return pd.DataFrame(results)


def batch_compute_dvh(
    dataset: Dict[str, Dict[str, Union[Dose, StructureSet]]],
    structure_names: Optional[List[str]] = None,
    max_dose: Optional[float] = None,
    step_size: float = 0.1
) -> Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """
    Compute DVHs for all subjects and structures in a dataset.
    
    Parameters
    ----------
    dataset : Dict
        Dataset dictionary from load_dataset()
    structure_names : List[str], optional
        Specific structures to analyze
    max_dose : float, optional
        Maximum dose for DVH bins
    step_size : float
        DVH bin width in Gy
    
    Returns
    -------
    dvhs : Dict[str, Dict[str, Tuple]]
        Nested dict: {subject_id: {structure_name: (dose_bins, volumes)}}
    
    Examples
    --------
    >>> from dosemetrics.utils import batch
    >>> dataset = batch.load_dataset('/data/study')
    >>> dvhs = batch.batch_compute_dvh(dataset, structure_names=['PTV', 'Heart'])
    """
    from ..metrics import dvh as dvh_module
    
    dvhs = {}
    
    for subject_id, data in dataset.items():
        if 'dose' not in data or 'structures' not in data:
            continue
        
        dose = data['dose']
        structures = data['structures']
        subject_dvhs = {}
        
        # Determine which structures to process
        if structure_names:
            struct_list = [structures.get_structure(name) for name in structure_names 
                          if name in structures.structure_names]
        else:
            struct_list = list(structures.structures.values())
        
        for structure in struct_list:
            try:
                dose_bins, volumes = dvh_module.compute_dvh(
                    dose, structure, max_dose=max_dose, step_size=step_size
                )
                subject_dvhs[structure.name] = {
                    'dose_bins': dose_bins,
                    'volumes': volumes
                }
            except Exception as e:
                print(f"Warning: Error computing DVH for {subject_id}/{structure.name}: {e}")
        
        if subject_dvhs:
            dvhs[subject_id] = subject_dvhs
    
    return dvhs


def compare_doses_batch(
    dataset1: Dict[str, Dict[str, Union[Dose, StructureSet]]],
    dataset2: Dict[str, Dict[str, Union[Dose, StructureSet]]],
    comparison_func: Callable,
    structure_names: Optional[List[str]] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Compare two datasets (e.g., TPS vs predicted doses).
    
    Parameters
    ----------
    dataset1, dataset2 : Dict
        Dataset dictionaries to compare
    comparison_func : Callable
        Function that takes (dose1, dose2, structure) and returns metrics
    structure_names : List[str], optional
        Specific structures to compare
    **kwargs
        Additional arguments for comparison_func
    
    Returns
    -------
    comparison : pd.DataFrame
        Comparison results for all subjects and structures
    
    Examples
    --------
    >>> from dosemetrics.metrics import dose_comparison
    >>> tps_data = load_dataset('/data/tps')
    >>> pred_data = load_dataset('/data/predicted')
    >>> comparison = compare_doses_batch(
    ...     tps_data, pred_data,
    ...     dose_comparison.compute_mae
    ... )
    """
    results = []
    
    # Find common subjects
    common_subjects = set(dataset1.keys()) & set(dataset2.keys())
    
    for subject_id in common_subjects:
        data1 = dataset1[subject_id]
        data2 = dataset2[subject_id]
        
        if 'dose' not in data1 or 'dose' not in data2:
            continue
        
        dose1 = data1['dose']
        dose2 = data2['dose']
        
        # Get structures (use dataset1's structures)
        if 'structures' not in data1:
            continue
        
        structures = data1['structures']
        
        # Determine which structures to process
        if structure_names:
            struct_list = [structures.get_structure(name) for name in structure_names 
                          if name in structures.structure_names]
        else:
            struct_list = list(structures.structures.values())
        
        for structure in struct_list:
            try:
                result = comparison_func(dose1, dose2, structure, **kwargs)
                
                if isinstance(result, dict):
                    row = {'subject_id': subject_id, 'structure': structure.name}
                    row.update(result)
                else:
                    row = {
                        'subject_id': subject_id,
                        'structure': structure.name,
                        'value': result
                    }
                
                results.append(row)
                
            except Exception as e:
                print(f"Warning: Error comparing {subject_id}/{structure.name}: {e}")
    
    return pd.DataFrame(results)


def aggregate_results(
    results: pd.DataFrame,
    group_by: Union[str, List[str]] = 'structure',
    agg_funcs: Optional[Dict[str, Union[str, List[str]]]] = None
) -> pd.DataFrame:
    """
    Aggregate batch processing results.
    
    Compute summary statistics across subjects, structures, or other groupings.
    
    Parameters
    ----------
    results : pd.DataFrame
        Results from process_dataset_with_metric or similar
    group_by : str or List[str]
        Column(s) to group by (e.g., 'structure', 'subject_id')
    agg_funcs : Dict, optional
        Aggregation functions for each column
        Default: {'value': ['mean', 'std', 'min', 'max']}
    
    Returns
    -------
    summary : pd.DataFrame
        Aggregated statistics
    
    Examples
    --------
    >>> results = process_dataset_with_metric(dataset, compute_mean_dose)
    >>> summary = aggregate_results(results, group_by='structure')
    >>> print(summary)  # Mean dose statistics per structure
    """
    if agg_funcs is None:
        # Default aggregations for numeric columns
        numeric_cols = results.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return results.groupby(group_by).size().to_frame('count')
        
        agg_funcs = {col: ['mean', 'std', 'min', 'max', 'median'] 
                     for col in numeric_cols if col != group_by}
    
    return results.groupby(group_by).agg(agg_funcs)


def export_batch_results(
    results: pd.DataFrame,
    output_path: Union[str, Path],
    format: str = 'csv',
    **kwargs
) -> None:
    """
    Export batch processing results to file.
    
    Parameters
    ----------
    results : pd.DataFrame
        Results dataframe to export
    output_path : str or Path
        Output file path
    format : str
        Output format: 'csv', 'excel', 'json', 'parquet'
    **kwargs
        Additional arguments for the export function
    
    Examples
    --------
    >>> results = process_dataset_with_metric(dataset, compute_mean_dose)
    >>> export_batch_results(results, 'results/mean_dose.csv')
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'csv':
        results.to_csv(output_path, **kwargs)
    elif format == 'excel':
        results.to_excel(output_path, **kwargs)
    elif format == 'json':
        results.to_json(output_path, **kwargs)
    elif format == 'parquet':
        results.to_parquet(output_path, **kwargs)
    else:
        raise ValueError(f"Unsupported format: {format}")
