"""
Tests for dosemetrics.utils.plot module.

Tests publication-quality plotting functionality.
"""

import unittest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
import pandas as pd
import tempfile
from pathlib import Path

from dosemetrics.dose import Dose
from dosemetrics.structures import Target, OAR, StructureType
from dosemetrics.structure_set import StructureSet
from dosemetrics.utils import plot


class TestPlotFunctions(unittest.TestCase):
    """Test plotting functionality."""
    
    def setUp(self):
        """Create test data for plotting."""
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
        self.structure1 = Target(
            name="PTV",
            mask=mask1,
            spacing=spacing,
            origin=origin
        )
        
        mask2 = np.zeros((10, 10, 10), dtype=bool)
        mask2[1:4, 1:4, 1:4] = True
        self.structure2 = OAR(
            name="OAR",
            mask=mask2,
            spacing=spacing,
            origin=origin
        )
        
        # Create StructureSet and add structures
        self.structures = StructureSet(spacing=spacing, origin=origin)
        self.structures.add_structure_object(self.structure1)
        self.structures.add_structure_object(self.structure2)
        
        # Mock dataset
        self.dataset = {
            'subject1': {'dose': self.dose, 'structures': self.structures},
            'subject2': {'dose': self.dose, 'structures': self.structures}
        }
    
    def tearDown(self):
        """Close all figures after each test."""
        plt.close('all')
    
    def test_plot_dvh_basic(self):
        """Test basic DVH plotting."""
        ax = plot.plot_dvh(self.dose, self.structure1, bins=10)
        
        self.assertIsInstance(ax, plt.Axes)
        # Check axis labels
        self.assertIn('Dose', ax.get_xlabel())
        self.assertIn('Volume', ax.get_ylabel())
    
    def test_plot_dvh_with_custom_ax(self):
        """Test DVH plotting on custom axis."""
        fig, ax = plt.subplots()
        result_ax = plot.plot_dvh(self.dose, self.structure1, ax=ax, bins=10)
        
        self.assertIs(result_ax, ax)
    
    def test_plot_subject_dvhs(self):
        """Test plotting all structures for a subject."""
        fig, ax = plot.plot_subject_dvhs(
            self.dose,
            self.structures,
            bins=10
        )
        
        self.assertIsInstance(fig, plt.Figure)
        self.assertIsInstance(ax, plt.Axes)
        
        # Check that legend exists
        legend = ax.get_legend()
        self.assertIsNotNone(legend)
    
    def test_plot_dvh_comparison(self):
        """Test DVH comparison plotting."""
        # Create second dose (slightly different)
        dose2_data = self.dose.dose_array * 1.1
        dose2 = Dose(
            dose_array=dose2_data,
            spacing=self.dose.spacing,
            origin=self.dose.origin
        )
        
        fig, ax = plot.plot_dvh_comparison(
            self.dose,
            dose2,
            self.structure1,
            labels=('Dose 1', 'Dose 2'),
            bins=10
        )
        
        self.assertIsInstance(fig, plt.Figure)
        self.assertIsInstance(ax, plt.Axes)
        
        # Check legend exists with both labels
        legend = ax.get_legend()
        self.assertIsNotNone(legend)
    
    def test_plot_dvh_band(self):
        """Test DVH band plotting for population."""
        ax = plot.plot_dvh_band(
            self.dataset,
            'PTV',
            bins=10,
            show_median=True,
            show_individual=False
        )
        
        self.assertIsInstance(ax, plt.Axes)
    
    def test_plot_metric_boxplot(self):
        """Test metric box plot."""
        # Create test dataframe
        df = pd.DataFrame({
            'subject_id': ['s1', 's1', 's2', 's2'],
            'structure': ['PTV', 'OAR', 'PTV', 'OAR'],
            'mean_dose': [50.0, 20.0, 52.0, 22.0]
        })
        
        fig, ax = plot.plot_metric_boxplot(
            df,
            'mean_dose',
            group_by='structure',
            show_points=False
        )
        
        self.assertIsInstance(fig, plt.Figure)
        self.assertIsInstance(ax, plt.Axes)
    
    def test_plot_metric_comparison(self):
        """Test metric comparison plotting."""
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
        
        fig, ax = plot.plot_metric_comparison(
            df1, df2,
            'mean_dose',
            cohort_names=('Cohort1', 'Cohort2')
        )
        
        self.assertIsInstance(fig, plt.Figure)
        self.assertIsInstance(ax, plt.Axes)
    
    def test_plot_dose_slice(self):
        """Test dose slice plotting."""
        fig, ax = plot.plot_dose_slice(
            self.dose,
            slice_idx=5,
            axis=2
        )
        
        self.assertIsInstance(fig, plt.Figure)
        self.assertIsInstance(ax, plt.Axes)
    
    def test_save_figure(self):
        """Test saving figures in multiple formats."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'test_figure'
            
            # Save in PNG format
            plot.save_figure(fig, filepath, formats=['png'], dpi=100)
            
            # Check file exists
            png_file = filepath.with_suffix('.png')
            self.assertTrue(png_file.exists())


if __name__ == '__main__':
    unittest.main()
