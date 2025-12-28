"""
Tests for dosemetrics.utils.batch module.

Tests batch processing functionality for multi-subject datasets.
"""

import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil

from dosemetrics.dose import Dose
from dosemetrics.structures import Target, OAR, StructureType
from dosemetrics.structure_set import StructureSet
from dosemetrics.utils import batch
from dosemetrics.metrics import dvh


class TestBatchFunctions(unittest.TestCase):
    """Test batch processing functions."""
    
    def setUp(self):
        """Create test data."""
        # Create simple dose and structure data
        self.dose_data = np.random.rand(10, 10, 10) * 50
        self.spacing = (1.0, 1.0, 1.0)
        self.origin = (0.0, 0.0, 0.0)
        self.dose = Dose(
            dose_array=self.dose_data,
            spacing=self.spacing,
            origin=self.origin
        )
        
        # Create test structures
        mask1 = np.zeros((10, 10, 10), dtype=bool)
        mask1[3:7, 3:7, 3:7] = True
        self.structure1 = Target(
            name="PTV",
            mask=mask1,
            spacing=self.spacing,
            origin=self.origin
        )
        
        mask2 = np.zeros((10, 10, 10), dtype=bool)
        mask2[1:4, 1:4, 1:4] = True
        self.structure2 = OAR(
            name="OAR",
            mask=mask2,
            spacing=self.spacing,
            origin=self.origin
        )
        
        # Create StructureSet and add structures
        self.structures = StructureSet(spacing=self.spacing, origin=self.origin)
        self.structures.add_structure_object(self.structure1)
        self.structures.add_structure_object(self.structure2)
        
        # Create mock dataset
        self.dataset = {
            'subject1': {'dose': self.dose, 'structures': self.structures},
            'subject2': {'dose': self.dose, 'structures': self.structures}
        }
    
    def test_process_dataset_with_metric(self):
        """Test processing dataset with a metric function."""
        def test_metric(dose, structure):
            return dvh.compute_mean_dose(dose, structure)
        
        results = batch.process_dataset_with_metric(
            self.dataset,
            test_metric,
            structure_names=['PTV']
        )
        
        self.assertIsInstance(results, pd.DataFrame)
        self.assertIn('subject_id', results.columns)
        self.assertIn('structure', results.columns)
        self.assertIn('value', results.columns)
        self.assertEqual(len(results), 2)  # 2 subjects
    
    def test_batch_compute_dvh(self):
        """Test batch DVH computation."""
        results = batch.batch_compute_dvh(
            self.dataset,
            structure_names=['PTV'],
            step_size=5.0
        )
        
        self.assertIsInstance(results, dict)
        self.assertIn('subject1', results)
        self.assertIn('PTV', results['subject1'])
        
        # Check DVH format
        dvh_data = results['subject1']['PTV']
        self.assertIsInstance(dvh_data, dict)
        self.assertIn('dose_bins', dvh_data)
        self.assertIn('volumes', dvh_data)
        self.assertGreater(len(dvh_data['dose_bins']), 0)
    
    def test_aggregate_results(self):
        """Test result aggregation."""
        # Create test dataframe
        df = pd.DataFrame({
            'subject_id': ['s1', 's1', 's2', 's2'],
            'structure': ['PTV', 'OAR', 'PTV', 'OAR'],
            'mean_dose': [50.0, 20.0, 52.0, 22.0]
        })
        
        # Test structure-level aggregation
        structure_agg = batch.aggregate_results(df, group_by='structure')
        self.assertIsInstance(structure_agg, pd.DataFrame)
        self.assertIn('mean_dose', structure_agg.columns.get_level_values(0))
        
        # Test subject-level aggregation
        subject_agg = batch.aggregate_results(df, group_by='subject_id')
        self.assertIsInstance(subject_agg, pd.DataFrame)
    
    def test_export_batch_results(self):
        """Test exporting batch results."""
        df = pd.DataFrame({
            'subject_id': ['s1', 's2'],
            'structure': ['PTV', 'PTV'],
            'mean_dose': [50.0, 52.0]
        })
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'results.csv'
            
            # Test CSV export
            batch.export_batch_results(df, output_path, format='csv')
            self.assertTrue(output_path.exists())
            
            # Verify content
            loaded = pd.read_csv(output_path)
            self.assertEqual(len(loaded), 2)
            self.assertIn('mean_dose', loaded.columns)
    
    def test_compare_doses_batch(self):
        """Test batch dose comparison."""
        # Create second dataset with slightly different doses
        dataset2 = {
            'subject1': {'dose': self.dose, 'structures': self.structures},
            'subject2': {'dose': self.dose, 'structures': self.structures}
        }
        
        def comparison_metric(dose1, dose2, structure):
            mean1 = dvh.compute_mean_dose(dose1, structure)
            mean2 = dvh.compute_mean_dose(dose2, structure)
            return abs(mean1 - mean2)
        
        results = batch.compare_doses_batch(
            self.dataset,
            dataset2,
            comparison_metric,
            structure_names=['PTV']
        )
        
        self.assertIsInstance(results, pd.DataFrame)
        self.assertIn('subject_id', results.columns)
        self.assertIn('structure', results.columns)
        self.assertIn('value', results.columns)


if __name__ == '__main__':
    unittest.main()
