"""
Comprehensive end-to-end tests for the Streamlit app tabs.
"""

import unittest
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

from dosemetrics import Dose, Target, OAR, StructureSet
from dosemetrics_app.utils import read_byte_data
import dosemetrics


class TestComprehensiveAnalysisTab(unittest.TestCase):
    """Test comprehensive analysis tab workflow."""

    def setUp(self):
        """Create test data."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # Create test dose distribution
        self.dose_array = np.random.rand(20, 20, 20) * 60.0
        self.dose_array[10:15, 10:15, 10:15] = 65.0  # Hot spot
        self.spacing = (1.0, 1.0, 2.5)
        self.origin = (0.0, 0.0, 0.0)
        self.dose = Dose(self.dose_array, self.spacing, self.origin)

        # Create test structures
        ptv_mask = np.zeros((20, 20, 20), dtype=bool)
        ptv_mask[8:16, 8:16, 8:16] = True
        self.ptv = Target("PTV", ptv_mask, self.spacing, self.origin)

        heart_mask = np.zeros((20, 20, 20), dtype=bool)
        heart_mask[2:8, 2:8, 2:8] = True
        self.heart = OAR("Heart", heart_mask, self.spacing, self.origin)

        lung_mask = np.zeros((20, 20, 20), dtype=bool)
        lung_mask[2:10, 12:18, 2:10] = True
        self.lung = OAR("Lung_L", lung_mask, self.spacing, self.origin)

        # Write test NIfTI files
        self.dose_path = self.temp_path / "dose.nii.gz"
        self.ptv_path = self.temp_path / "ptv.nii.gz"
        self.heart_path = self.temp_path / "heart.nii.gz"
        self.lung_path = self.temp_path / "lung.nii.gz"

        dosemetrics.nifti_io.write_nifti_volume(
            self.dose_array, str(self.dose_path), self.spacing
        )
        dosemetrics.nifti_io.write_nifti_volume(
            ptv_mask.astype(float), str(self.ptv_path), self.spacing
        )
        dosemetrics.nifti_io.write_nifti_volume(
            heart_mask.astype(float), str(self.heart_path), self.spacing
        )
        dosemetrics.nifti_io.write_nifti_volume(
            lung_mask.astype(float), str(self.lung_path), self.spacing
        )

    def tearDown(self):
        """Clean up."""
        self.temp_dir.cleanup()

    def test_load_data_workflow(self):
        """Test the complete data loading workflow."""
        # Load data as the app would
        dose, structures = read_byte_data(
            self.dose_path,
            {
                "PTV": self.ptv_path,
                "Heart": self.heart_path,
                "Lung_L": self.lung_path,
            },
        )

        # Verify dose loaded correctly
        self.assertIsInstance(dose, Dose)
        self.assertEqual(dose.shape, (20, 20, 20))
        self.assertEqual(dose.spacing, self.spacing)

        # Verify structures loaded correctly
        self.assertEqual(len(structures), 3)
        self.assertIn("PTV", structures)
        self.assertIn("Heart", structures)
        self.assertIn("Lung_L", structures)

        # Verify structure types
        self.assertIsInstance(structures["PTV"], Target)
        self.assertIsInstance(structures["Heart"], OAR)
        self.assertIsInstance(structures["Lung_L"], OAR)

    def test_structure_set_creation(self):
        """Test creating structure set from loaded structures."""
        dose, structures = read_byte_data(
            self.dose_path,
            {
                "PTV": self.ptv_path,
                "Heart": self.heart_path,
            },
        )

        # Create structure set as app does
        structure_set = StructureSet()
        structure_set.spacing = dose.spacing
        structure_set.origin = dose.origin
        for name, struct in structures.items():
            structure_set.structures[name] = struct

        # Verify structure set
        self.assertEqual(len(structure_set.structures), 2)
        self.assertIn("PTV", structure_set.structures)
        self.assertIn("Heart", structure_set.structures)
        self.assertEqual(structure_set.spacing, dose.spacing)

    def test_dvh_computation_workflow(self):
        """Test DVH computation from loaded data."""
        from dosemetrics_app.utils import dvh_by_structure

        dose, structures = read_byte_data(
            self.dose_path,
            {
                "PTV": self.ptv_path,
                "Heart": self.heart_path,
            },
        )

        # Compute DVH
        dvh_df = dvh_by_structure(dose, structures)

        # Verify DVH dataframe
        self.assertIsInstance(dvh_df, pd.DataFrame)
        self.assertIn("Dose", dvh_df.columns)
        self.assertIn("Volume", dvh_df.columns)
        self.assertIn("Structure", dvh_df.columns)

        # Check both structures present
        structure_names = dvh_df["Structure"].unique()
        self.assertIn("PTV", structure_names)
        self.assertIn("Heart", structure_names)

    def test_dose_statistics_workflow(self):
        """Test dose statistics computation workflow."""
        from dosemetrics.metrics import dvh

        dose, structures = read_byte_data(
            self.dose_path,
            {
                "PTV": self.ptv_path,
                "Heart": self.heart_path,
            },
        )

        # Create structure set
        structure_set = StructureSet()
        structure_set.spacing = dose.spacing
        structure_set.origin = dose.origin
        for name, struct in structures.items():
            structure_set.structures[name] = struct

        # Compute statistics for all structures
        results = []
        for struct in structure_set.structures.values():
            stats = {
                "Structure": struct.name,
                "Volume (cc)": struct.volume_cc,
                "Mean Dose (Gy)": dvh.compute_mean_dose(dose, struct),
                "Max Dose (Gy)": dvh.compute_max_dose(dose, struct),
                "Min Dose (Gy)": dvh.compute_min_dose(dose, struct),
            }
            results.append(stats)

        stats_df = pd.DataFrame(results)

        # Verify statistics
        self.assertEqual(len(stats_df), 2)
        self.assertIn("Structure", stats_df.columns)
        self.assertIn("Mean Dose (Gy)", stats_df.columns)

        # Check PTV stats
        ptv_stats = stats_df[stats_df["Structure"] == "PTV"].iloc[0]
        self.assertGreater(ptv_stats["Mean Dose (Gy)"], 0)
        self.assertGreater(ptv_stats["Max Dose (Gy)"], ptv_stats["Mean Dose (Gy)"])
        self.assertLess(ptv_stats["Min Dose (Gy)"], ptv_stats["Mean Dose (Gy)"])

    def test_quality_metrics_workflow(self):
        """Test quality metrics (conformity and homogeneity) workflow."""
        from dosemetrics.metrics import conformity, homogeneity

        dose, structures = read_byte_data(
            self.dose_path,
            {
                "PTV": self.ptv_path,
                "Heart": self.heart_path,
            },
        )

        # Get target structure
        target_structures = {
            name: struct
            for name, struct in structures.items()
            if any(keyword in name.upper() for keyword in ["PTV", "CTV", "GTV"])
        }

        self.assertEqual(len(target_structures), 1)
        target = target_structures["PTV"]
        prescription_dose = 60.0

        # Compute conformity metrics
        ci = conformity.compute_conformity_index(dose, target, prescription_dose)
        cn = conformity.compute_conformity_number(dose, target, prescription_dose)
        paddick_ci = conformity.compute_paddick_conformity_index(
            dose, target, prescription_dose
        )
        coverage = conformity.compute_coverage(dose, target, prescription_dose)
        spillage = conformity.compute_spillage(dose, target, prescription_dose)

        # Verify metrics are in reasonable ranges
        self.assertGreaterEqual(ci, 0)
        self.assertLessEqual(ci, 1)
        self.assertGreaterEqual(cn, 0)
        self.assertLessEqual(cn, 1)
        self.assertGreaterEqual(coverage, 0)
        self.assertLessEqual(coverage, 1)
        self.assertGreaterEqual(spillage, 0)
        self.assertLessEqual(spillage, 1)

        # Compute homogeneity
        hi = homogeneity.compute_homogeneity_index(dose, target, prescription_dose)
        self.assertGreaterEqual(hi, 0)

    def test_multiple_structures_workflow(self):
        """Test workflow with multiple OARs and targets."""
        dose, structures = read_byte_data(
            self.dose_path,
            {
                "PTV": self.ptv_path,
                "Heart": self.heart_path,
                "Lung_L": self.lung_path,
            },
        )

        # Verify all structures loaded
        self.assertEqual(len(structures), 3)

        # Create structure set
        structure_set = StructureSet()
        structure_set.spacing = dose.spacing
        structure_set.origin = dose.origin
        for name, struct in structures.items():
            structure_set.structures[name] = struct

        # Verify structure set has all structures
        self.assertEqual(len(structure_set.structures), 3)

        # Verify we can compute DVH for all
        from dosemetrics_app.utils import dvh_by_structure

        dvh_df = dvh_by_structure(dose, structures)
        structure_names = dvh_df["Structure"].unique()
        self.assertEqual(len(structure_names), 3)


class TestGeometricTab(unittest.TestCase):
    """Test geometric comparison tab workflow."""

    def setUp(self):
        """Create test data."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # Create two slightly different structure sets
        self.spacing = (1.0, 1.0, 1.0)
        self.origin = (0.0, 0.0, 0.0)

        # Structure set 1
        mask1 = np.zeros((20, 20, 20), dtype=bool)
        mask1[5:15, 5:15, 5:15] = True

        # Structure set 2 (slightly shifted)
        mask2 = np.zeros((20, 20, 20), dtype=bool)
        mask2[6:16, 6:16, 6:16] = True

        # Write files
        self.mask1_path = self.temp_path / "struct1.nii.gz"
        self.mask2_path = self.temp_path / "struct2.nii.gz"

        dosemetrics.nifti_io.write_nifti_volume(
            mask1.astype(float), str(self.mask1_path), self.spacing
        )
        dosemetrics.nifti_io.write_nifti_volume(
            mask2.astype(float), str(self.mask2_path), self.spacing
        )

    def tearDown(self):
        """Clean up."""
        self.temp_dir.cleanup()

    def test_geometric_comparison_workflow(self):
        """Test geometric comparison between two structure sets."""
        from dosemetrics.metrics import geometric

        # Load structures
        mask1_array, spacing1, origin1 = dosemetrics.load_volume(str(self.mask1_path))
        mask2_array, spacing2, origin2 = dosemetrics.load_volume(str(self.mask2_path))

        struct1 = OAR("Structure_1", mask1_array > 0.5, spacing1, origin1)
        struct2 = OAR("Structure_2", mask2_array > 0.5, spacing2, origin2)

        # Compute geometric metrics
        dice = geometric.compute_dice_coefficient(struct1, struct2)
        jaccard = geometric.compute_jaccard_index(struct1, struct2)
        hausdorff = geometric.compute_hausdorff_distance(struct1, struct2)

        # Verify metrics
        self.assertGreaterEqual(dice, 0)
        self.assertLessEqual(dice, 1)
        self.assertGreaterEqual(jaccard, 0)
        self.assertLessEqual(jaccard, 1)
        self.assertGreaterEqual(hausdorff, 0)


class TestComplianceTab(unittest.TestCase):
    """Test compliance checking tab workflow."""

    def setUp(self):
        """Create test data."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # Create test dose and structures
        dose_array = np.ones((15, 15, 15)) * 50.0
        dose_array[5:10, 5:10, 5:10] = 55.0
        self.spacing = (1.0, 1.0, 1.0)
        self.origin = (0.0, 0.0, 0.0)

        heart_mask = np.zeros((15, 15, 15), dtype=bool)
        heart_mask[3:8, 3:8, 3:8] = True

        # Write files
        self.dose_path = self.temp_path / "dose.nii.gz"
        self.heart_path = self.temp_path / "heart.nii.gz"

        dosemetrics.nifti_io.write_nifti_volume(
            dose_array, str(self.dose_path), self.spacing
        )
        dosemetrics.nifti_io.write_nifti_volume(
            heart_mask.astype(float), str(self.heart_path), self.spacing
        )

    def tearDown(self):
        """Clean up."""
        self.temp_dir.cleanup()

    def test_compliance_workflow(self):
        """Test compliance checking workflow."""
        from dosemetrics.metrics import dvh
        from dosemetrics import get_default_constraints, check_compliance

        # Load data
        dose, structures = read_byte_data(
            self.dose_path,
            {"Heart": self.heart_path},
        )

        # Create structure set
        structure_set = StructureSet()
        structure_set.spacing = dose.spacing
        structure_set.origin = dose.origin
        for name, struct in structures.items():
            structure_set.structures[name] = struct

        # Compute statistics
        stats_data = []
        for struct in structure_set.structures.values():
            stats_data.append(
                {
                    "Structure": struct.name,
                    "Mean Dose": dvh.compute_mean_dose(dose, struct),
                    "Max Dose": dvh.compute_max_dose(dose, struct),
                    "Min Dose": dvh.compute_min_dose(dose, struct),
                }
            )

        stats_df = pd.DataFrame(stats_data).set_index("Structure")

        # Create custom constraints for Heart using DataFrame
        constraints = pd.DataFrame(
            [
                {"Structure": "Heart", "Constraint Type": "max", "Level": 60.0},
            ]
        ).set_index("Structure")

        # Check compliance
        compliance_df = check_compliance(stats_df, constraints)

        # Verify compliance result
        self.assertIsInstance(compliance_df, pd.DataFrame)
        # If compliance_df is non-empty, verify it has expected columns
        if len(compliance_df) > 0:
            self.assertIn("Compliance", compliance_df.columns)


class TestAppIntegration(unittest.TestCase):
    """Integration tests for the full app."""

    def test_app_imports(self):
        """Test that all app modules can be imported."""
        # These imports should not raise errors
        from dosemetrics_app import app
        from dosemetrics_app.tabs import (
            comprehensive_analysis,
            geometric_tab,
            gamma_tab,
            compliance_tab,
            instructions,
        )
        from dosemetrics_app import utils

        # Verify key functions exist
        self.assertTrue(hasattr(utils, "read_byte_data"))
        self.assertTrue(hasattr(utils, "dvh_by_structure"))
        self.assertTrue(hasattr(utils, "infer_structure_type"))
        self.assertTrue(hasattr(comprehensive_analysis, "panel"))
        self.assertTrue(hasattr(geometric_tab, "panel"))

    def test_structure_type_inference_consistency(self):
        """Test that structure type inference is consistent."""
        from dosemetrics_app.utils import infer_structure_type

        # Test various naming patterns
        target_names = [
            "PTV_60",
            "PTV",
            "ptv_high",
            "CTV",
            "CTV_Low",
            "GTV",
            "Target",
            "Tumor",
        ]
        oar_names = [
            "Heart",
            "Lung_L",
            "Lung_R",
            "SpinalCord",
            "Bladder",
            "Rectum",
            "Structure_1",
        ]

        for name in target_names:
            self.assertEqual(
                infer_structure_type(name),
                "target",
                f"Failed for target name: {name}",
            )

        for name in oar_names:
            self.assertEqual(
                infer_structure_type(name), "oar", f"Failed for OAR name: {name}"
            )


if __name__ == "__main__":
    unittest.main()
