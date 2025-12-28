"""
Tests for dosemetrics.utils.analysis module.

Tests multi-level analysis functionality.
"""

import unittest
import numpy as np
import pandas as pd

from dosemetrics.dose import Dose
from dosemetrics.structures import Target, OAR, StructureType
from dosemetrics.structure_set import StructureSet
from dosemetrics.utils import analysis
from dosemetrics.metrics import dvh


class TestAnalysisFunctions(unittest.TestCase):
    """Test analysis functions."""
    
    def setUp(self):
        """Create test data."""
        # Create simple dose and structure data
        dose_data = np.random.rand(10, 10, 10) * 50
        spacing = (1.0, 1.0, 1.0)
        origin = (0.0, 0.0, 0.0)
        self.dose = Dose(
            dose_array=dose_data,
            spacing=spacing,
            origin=origin
        )
        
        # Create test structures
        mask1 = np.zeros((10, 10, 10), dtype=bool)
        mask1[3:7, 3:7, 3:7] = True
        structure1 = Target(
            name="PTV",
            mask=mask1,
            spacing=spacing,
            origin=origin
        )
        
        mask2 = np.zeros((10, 10, 10), dtype=bool)
        mask2[1:4, 1:4, 1:4] = True
        structure2 = OAR(
            name="OAR",
            mask=mask2,
            spacing=spacing,
            origin=origin
        )
        
        # Create StructureSet and add structures
        self.structures = StructureSet(spacing=spacing, origin=origin)
        self.structures.add_structure_object(structure1)
        self.structures.add_structure_object(structure2)
        
        # Create mock dataset
        self.dataset = {
            'subject1': {'dose': self.dose, 'structures': self.structures},
            'subject2': {'dose': self.dose, 'structures': self.structures}
        }
        
        # Test metrics
        self.metrics = {
            'mean_dose': dvh.compute_mean_dose,
            'max_dose': dvh.compute_max_dose
        }
    
    def test_analyze_by_structure(self):
        """Test structure-level analysis."""
        results = analysis.analyze_by_structure(
            self.dataset,
            'PTV',
            self.metrics
        )
        
        self.assertIsInstance(results, pd.DataFrame)
        self.assertIn('subject_id', results.columns)
        self.assertIn('mean_dose', results.columns)
        self.assertIn('max_dose', results.columns)
        self.assertEqual(len(results), 2)  # 2 subjects
    
    def test_analyze_by_subject(self):
        """Test subject-level analysis."""
        results = analysis.analyze_by_subject(
            self.dose,
            self.structures,
            self.metrics
        )
        
        self.assertIsInstance(results, pd.DataFrame)
        self.assertIn('structure', results.columns)
        self.assertIn('mean_dose', results.columns)
        self.assertEqual(len(results), 2)  # 2 structures
    
    def test_analyze_by_subject_filtered(self):
        """Test subject analysis with structure filtering."""
        results = analysis.analyze_by_subject(
            self.dose,
            self.structures,
            self.metrics,
            structure_names=['PTV']
        )
        
        self.assertEqual(len(results), 1)  # Only PTV
        self.assertEqual(results.iloc[0]['structure'], 'PTV')
    
    def test_analyze_by_dataset(self):
        """Test dataset-level analysis."""
        results = analysis.analyze_by_dataset(
            self.dataset,
            self.metrics,
            summary_stats=False
        )
        
        self.assertIsInstance(results, pd.DataFrame)
        self.assertIn('subject_id', results.columns)
        self.assertIn('structure', results.columns)
        self.assertIn('mean_dose', results.columns)
    
    def test_analyze_by_dataset_with_summary(self):
        """Test dataset analysis with summary statistics."""
        detailed, summary = analysis.analyze_by_dataset(
            self.dataset,
            self.metrics,
            summary_stats=True
        )
        
        self.assertIsInstance(detailed, pd.DataFrame)
        self.assertIsInstance(summary, pd.DataFrame)
        
        # Check summary has expected columns
        self.assertIn('mean_dose', summary.columns.get_level_values(0))
    
    def test_analyze_subset(self):
        """Test subset analysis with filtering."""
        # Filter only targets
        def target_filter(structure):
            return structure.structure_type == StructureType.TARGET
        
        results = analysis.analyze_subset(
            self.dataset,
            self.metrics,
            structure_filter=target_filter
        )
        
        self.assertIsInstance(results, pd.DataFrame)
        # Should only have PTV (target)
        unique_structures = results['structure'].unique()
        self.assertIn('PTV', unique_structures)
        self.assertNotIn('OAR', unique_structures)
    
    def test_compute_cohort_statistics(self):
        """Test cohort statistics computation."""
        # Create test dataframe
        df = pd.DataFrame({
            'subject_id': ['s1', 's1', 's2', 's2'],
            'structure': ['PTV', 'OAR', 'PTV', 'OAR'],
            'mean_dose': [50.0, 20.0, 52.0, 22.0]
        })
        
        stats = analysis.compute_cohort_statistics(df, metric_cols=['mean_dose'])
        
        self.assertIsInstance(stats, pd.DataFrame)
        # Should have statistics for each structure
        self.assertIn('PTV', stats.index)
        self.assertIn('OAR', stats.index)
    
    def test_compare_cohorts(self):
        """Test cohort comparison."""
        # Create two cohorts
        df1 = pd.DataFrame({
            'subject_id': ['s1', 's2'],
            'structure': ['PTV', 'PTV'],
            'mean_dose': [50.0, 52.0]
        })
        
        df2 = pd.DataFrame({
            'subject_id': ['s3', 's4'],
            'structure': ['PTV', 'PTV'],
            'mean_dose': [48.0, 49.0]
        })
        
        comparison = analysis.compare_cohorts(
            df1, df2,
            metric_cols=['mean_dose'],
            cohort_names=('Pre', 'Post')
        )
        
        self.assertIsInstance(comparison, pd.DataFrame)
        self.assertIn('structure', comparison.columns)
        self.assertIn('p_value', comparison.columns)
        self.assertIn('cohens_d', comparison.columns)


if __name__ == '__main__':
    unittest.main()
