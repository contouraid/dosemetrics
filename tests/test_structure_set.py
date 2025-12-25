"""
Tests for the StructureSet functionality.
"""

import unittest
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock

import dosemetrics as dm
from dosemetrics.io import (
    StructureSet,
    create_structure_set_from_masks,
    StructureType,
)


class TestStructureSet(unittest.TestCase):
    """Test cases for StructureSet functionality."""

    def setUp(self):
        """Set up test data."""
        self.shape = (20, 20, 20)
        self.spacing = (2.0, 2.0, 2.0)
        self.origin = (0.0, 0.0, 0.0)

        # Create synthetic dose data
        self.dose_volume = np.random.uniform(0, 70, self.shape)

        # Create structure masks
        center = np.array(self.shape) // 2

        # Create a spherical target
        self.target_mask = np.zeros(self.shape, dtype=bool)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                for k in range(self.shape[2]):
                    distance = np.sqrt(
                        (i - center[0]) ** 2
                        + (j - center[1]) ** 2
                        + (k - center[2]) ** 2
                    )
                    if distance < 5:
                        self.target_mask[i, j, k] = True

        # Create an OAR mask
        self.oar_mask = np.zeros(self.shape, dtype=bool)
        self.oar_mask[5:15, 5:15, 5:15] = True

        self.structure_masks = {"PTV": self.target_mask, "OAR1": self.oar_mask}

        self.structure_types = {"PTV": StructureType.TARGET, "OAR1": StructureType.OAR}

    def test_create_empty_structure_set(self):
        """Test creating an empty StructureSet."""
        structure_set = StructureSet(
            spacing=self.spacing, origin=self.origin, name="TestSet"
        )

        self.assertEqual(structure_set.name, "TestSet")
        self.assertEqual(structure_set.spacing, self.spacing)
        self.assertEqual(structure_set.origin, self.origin)
        self.assertEqual(len(structure_set), 0)
        self.assertFalse(structure_set.has_dose)

    def test_create_structure_set_from_masks(self):
        """Test creating StructureSet from mask dictionaries."""
        structure_set = create_structure_set_from_masks(
            structure_masks=self.structure_masks,
            dose_volume=self.dose_volume,
            spacing=self.spacing,
            origin=self.origin,
            structure_types=self.structure_types,
            name="TestCase",
        )

        self.assertEqual(structure_set.name, "TestCase")
        self.assertEqual(len(structure_set), 2)
        self.assertTrue(structure_set.has_dose)
        self.assertIn("PTV", structure_set)
        self.assertIn("OAR1", structure_set)

        # Check structure types
        ptv = structure_set["PTV"]
        oar = structure_set["OAR1"]
        self.assertEqual(ptv.structure_type, StructureType.TARGET)
        self.assertEqual(oar.structure_type, StructureType.OAR)

        # Check dose data propagation
        self.assertTrue(ptv.has_dose)
        self.assertTrue(oar.has_dose)

    def test_add_remove_structures(self):
        """Test adding and removing structures."""
        structure_set = StructureSet(name="TestSet")

        # Add structure
        structure = structure_set.add_structure(
            "Test_OAR", self.oar_mask, StructureType.OAR
        )

        self.assertEqual(len(structure_set), 1)
        self.assertIn("Test_OAR", structure_set)
        self.assertEqual(structure.structure_type, StructureType.OAR)

        # Try to add duplicate (should raise error)
        with self.assertRaises(ValueError):
            structure_set.add_structure("Test_OAR", self.oar_mask, StructureType.OAR)

        # Remove structure
        structure_set.remove_structure("Test_OAR")
        self.assertEqual(len(structure_set), 0)
        self.assertNotIn("Test_OAR", structure_set)

        # Try to remove non-existent structure
        with self.assertRaises(ValueError):
            structure_set.remove_structure("NonExistent")

    def test_set_dose_data(self):
        """Test setting dose data for all structures."""
        structure_set = create_structure_set_from_masks(
            structure_masks=self.structure_masks,
            spacing=self.spacing,
            origin=self.origin,
            structure_types=self.structure_types,
        )

        # Initially no dose
        self.assertFalse(structure_set.has_dose)
        for structure in structure_set.structures.values():
            self.assertFalse(structure.has_dose)

        # Set dose data
        structure_set.set_dose_data(self.dose_volume)

        # Check dose propagation
        self.assertTrue(structure_set.has_dose)
        for structure in structure_set.structures.values():
            self.assertTrue(structure.has_dose)

    def test_structure_filtering(self):
        """Test filtering structures by type."""
        structure_set = create_structure_set_from_masks(
            structure_masks=self.structure_masks, structure_types=self.structure_types
        )

        oars = structure_set.get_oars()
        targets = structure_set.get_targets()

        self.assertEqual(len(oars), 1)
        self.assertEqual(len(targets), 1)
        self.assertIn("OAR1", oars)
        self.assertIn("PTV", targets)

        # Test name lists
        self.assertEqual(structure_set.oar_names, ["OAR1"])
        self.assertEqual(structure_set.target_names, ["PTV"])

    def test_dose_statistics_summary(self):
        """Test dose statistics summary generation."""
        structure_set = create_structure_set_from_masks(
            structure_masks=self.structure_masks,
            dose_volume=self.dose_volume,
            structure_types=self.structure_types,
        )

        stats_df = structure_set.dose_statistics_summary()

        self.assertEqual(len(stats_df), 2)
        self.assertIn("Structure", stats_df.columns)
        self.assertIn("Type", stats_df.columns)
        self.assertIn("Volume_cc", stats_df.columns)
        self.assertIn("Mean_Dose_Gy", stats_df.columns)
        self.assertIn("Max_Dose_Gy", stats_df.columns)

        # Check that all structures are included
        structures = set(stats_df["Structure"])
        self.assertEqual(structures, {"PTV", "OAR1"})

    def test_bulk_dvh_computation(self):
        """Test bulk DVH computation."""
        structure_set = create_structure_set_from_masks(
            structure_masks=self.structure_masks,
            dose_volume=self.dose_volume,
            structure_types=self.structure_types,
        )

        dvh_df = structure_set.compute_bulk_dvh(max_dose=70, step_size=1.0)

        self.assertIn("Dose", dvh_df.columns)
        self.assertIn("PTV", dvh_df.columns)
        self.assertIn("OAR1", dvh_df.columns)
        self.assertEqual(len(dvh_df), 71)  # 0 to 70 Gy with 1 Gy steps

    def test_compliance_checking(self):
        """Test dose constraint compliance checking."""
        structure_set = create_structure_set_from_masks(
            structure_masks=self.structure_masks,
            dose_volume=self.dose_volume,
            structure_types=self.structure_types,
        )

        constraints = {"PTV": {"mean_dose": 50}, "OAR1": {"max_dose": 30}}

        compliance_df = structure_set.compliance_check(constraints)

        self.assertEqual(len(compliance_df), 2)
        self.assertIn("Structure", compliance_df.columns)
        self.assertIn("Constraint_Type", compliance_df.columns)
        self.assertIn("Compliant", compliance_df.columns)

    def test_geometric_summary(self):
        """Test geometric summary generation."""
        structure_set = create_structure_set_from_masks(
            structure_masks=self.structure_masks, structure_types=self.structure_types
        )

        geom_df = structure_set.geometric_summary()

        self.assertEqual(len(geom_df), 2)
        self.assertIn("Structure", geom_df.columns)
        self.assertIn("Volume_cc", geom_df.columns)
        self.assertIn("Centroid_X", geom_df.columns)

        # Check that centroids are calculated
        ptv_row = geom_df[geom_df["Structure"] == "PTV"].iloc[0]
        self.assertIsNotNone(ptv_row["Centroid_X"])

    def test_structure_access_methods(self):
        """Test different ways to access structures."""
        structure_set = create_structure_set_from_masks(
            structure_masks=self.structure_masks, structure_types=self.structure_types
        )

        # Dictionary-style access
        ptv = structure_set["PTV"]
        self.assertEqual(ptv.name, "PTV")

        # Iterator access
        names_from_iter = [name for name, _ in structure_set]
        self.assertEqual(set(names_from_iter), {"PTV", "OAR1"})

        # Membership testing
        self.assertTrue("PTV" in structure_set)
        self.assertFalse("NonExistent" in structure_set)

        # Length
        self.assertEqual(len(structure_set), 2)

    def test_io_functions(self):
        """Test the new I/O functions."""
        # Test create_structure_set_from_existing_data
        structure_set = dm.create_structure_set_from_existing_data(
            dose_volume=self.dose_volume,
            structure_masks=self.structure_masks,
            structure_types={"PTV": "Target", "OAR1": "OAR"},
            name="IOTest",
        )

        self.assertEqual(structure_set.name, "IOTest")
        self.assertEqual(len(structure_set), 2)
        self.assertTrue(structure_set.has_dose)


class TestStructureSetIO(unittest.TestCase):
    """Test I/O functions for StructureSet."""

    @patch("dosemetrics.io.structure_set.read_from_nifti")
    @patch("os.path.exists")
    @patch("glob.glob")
    def test_create_structure_set_from_folder(
        self, mock_glob, mock_exists, mock_read_nifti
    ):
        """Test creating StructureSet from folder (mocked)."""
        # Setup mocks
        mock_exists.return_value = True
        mock_glob.return_value = ["/test/PTV.nii.gz", "/test/OAR1.nii.gz"]

        # Mock dose and mask data
        dose_shape = (10, 10, 10)
        mock_read_nifti.side_effect = [
            np.random.uniform(0, 70, dose_shape),  # dose
            np.random.randint(0, 2, dose_shape, dtype=bool),  # PTV
            np.random.randint(0, 2, dose_shape, dtype=bool),  # OAR1
        ]

        # Test the function
        structure_set = dm.create_structure_set_from_folder(
            data_path="/test", name="MockTest"
        )

        self.assertEqual(structure_set.name, "MockTest")
        # Note: The actual structure count depends on the mocked file list


if __name__ == "__main__":
    unittest.main()
