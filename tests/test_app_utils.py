"""
Tests for Streamlit app utilities.
"""

import unittest
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path
from io import BytesIO

from dosemetrics import Dose, Target, OAR
from dosemetrics_app.utils import (
    infer_structure_type,
    read_byte_data,
    dvh_by_structure,
)


class TestInferStructureType(unittest.TestCase):
    """Test structure type inference from names."""

    def test_target_ptv(self):
        """Test PTV is identified as target."""
        self.assertEqual(infer_structure_type("PTV_60"), "target")
        self.assertEqual(infer_structure_type("ptv"), "target")
        self.assertEqual(infer_structure_type("PTV"), "target")

    def test_target_ctv(self):
        """Test CTV is identified as target."""
        self.assertEqual(infer_structure_type("CTV_High"), "target")
        self.assertEqual(infer_structure_type("ctv"), "target")

    def test_target_gtv(self):
        """Test GTV is identified as target."""
        self.assertEqual(infer_structure_type("GTV"), "target")
        self.assertEqual(infer_structure_type("gtv_primary"), "target")

    def test_target_tumor(self):
        """Test tumor/tumour is identified as target."""
        self.assertEqual(infer_structure_type("Tumor"), "target")
        self.assertEqual(infer_structure_type("Tumour"), "target")

    def test_target_generic(self):
        """Test generic target is identified as target."""
        self.assertEqual(infer_structure_type("Target"), "target")
        self.assertEqual(infer_structure_type("TARGET_VOLUME"), "target")

    def test_oar_organs(self):
        """Test common OARs are identified correctly."""
        self.assertEqual(infer_structure_type("Heart"), "oar")
        self.assertEqual(infer_structure_type("Lung_L"), "oar")
        self.assertEqual(infer_structure_type("Lung_R"), "oar")
        self.assertEqual(infer_structure_type("Spinal_Cord"), "oar")
        self.assertEqual(infer_structure_type("Bladder"), "oar")
        self.assertEqual(infer_structure_type("Rectum"), "oar")

    def test_oar_generic(self):
        """Test generic structure names are treated as OAR."""
        self.assertEqual(infer_structure_type("Structure_1"), "oar")
        self.assertEqual(infer_structure_type("Unknown"), "oar")


class TestReadByteData(unittest.TestCase):
    """Test reading dose and structure data from various sources."""

    def setUp(self):
        """Create temporary test data."""
        # Create temporary directory
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # Create simple test dose array
        self.dose_array = np.random.rand(10, 10, 10) * 60.0
        self.spacing = (1.0, 1.0, 2.5)
        self.origin = (0.0, 0.0, 0.0)

        # Create test masks
        self.ptv_mask = np.zeros((10, 10, 10), dtype=bool)
        self.ptv_mask[3:7, 3:7, 3:7] = True

        self.heart_mask = np.zeros((10, 10, 10), dtype=bool)
        self.heart_mask[1:4, 1:4, 1:4] = True

    def tearDown(self):
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def test_read_from_paths(self):
        """Test reading from file paths (example data)."""
        # Create NIfTI files using dosemetrics
        import dosemetrics

        dose_path = self.temp_path / "dose.nii.gz"
        ptv_path = self.temp_path / "ptv.nii.gz"
        heart_path = self.temp_path / "heart.nii.gz"

        # Write test files
        dosemetrics.nifti_io.write_nifti_volume(
            self.dose_array, str(dose_path), self.spacing
        )
        dosemetrics.nifti_io.write_nifti_volume(
            self.ptv_mask.astype(float), str(ptv_path), self.spacing
        )
        dosemetrics.nifti_io.write_nifti_volume(
            self.heart_mask.astype(float), str(heart_path), self.spacing
        )

        # Read using paths
        dose, structures = read_byte_data(
            dose_path, {"PTV": ptv_path, "Heart": heart_path}
        )

        # Verify dose
        self.assertIsInstance(dose, Dose)
        self.assertEqual(dose.shape, (10, 10, 10))
        self.assertEqual(dose.spacing, self.spacing)

        # Verify structures
        self.assertEqual(len(structures), 2)
        self.assertIn("PTV", structures)
        self.assertIn("Heart", structures)

        # Verify structure types
        self.assertIsInstance(structures["PTV"], Target)
        self.assertIsInstance(structures["Heart"], OAR)

    def test_read_from_bytesio(self):
        """Test reading from BytesIO objects (uploaded files)."""
        import dosemetrics

        # Create temporary files first
        dose_path = self.temp_path / "dose.nii.gz"
        ptv_path = self.temp_path / "ptv.nii.gz"

        dosemetrics.nifti_io.write_nifti_volume(
            self.dose_array, str(dose_path), self.spacing
        )
        dosemetrics.nifti_io.write_nifti_volume(
            self.ptv_mask.astype(float), str(ptv_path), self.spacing
        )

        # Read into BytesIO
        with open(dose_path, "rb") as f:
            dose_bytes = BytesIO(f.read())
            dose_bytes.name = "dose.nii.gz"

        with open(ptv_path, "rb") as f:
            ptv_bytes = BytesIO(f.read())
            ptv_bytes.name = "ptv.nii.gz"

        # Read using BytesIO
        dose, structures = read_byte_data(dose_bytes, {"PTV": ptv_bytes})

        # Verify
        self.assertIsInstance(dose, Dose)
        self.assertEqual(len(structures), 1)
        self.assertIsInstance(structures["PTV"], Target)

    def test_read_from_list(self):
        """Test reading from a list of mask files."""
        import dosemetrics

        dose_path = self.temp_path / "dose.nii.gz"
        ptv_path = self.temp_path / "ptv.nii.gz"
        heart_path = self.temp_path / "heart.nii.gz"

        dosemetrics.nifti_io.write_nifti_volume(
            self.dose_array, str(dose_path), self.spacing
        )
        dosemetrics.nifti_io.write_nifti_volume(
            self.ptv_mask.astype(float), str(ptv_path), self.spacing
        )
        dosemetrics.nifti_io.write_nifti_volume(
            self.heart_mask.astype(float), str(heart_path), self.spacing
        )

        # Read using list (simulating uploaded files)
        with open(ptv_path, "rb") as f:
            ptv_bytes = BytesIO(f.read())
            ptv_bytes.name = "ptv.nii.gz"

        with open(heart_path, "rb") as f:
            heart_bytes = BytesIO(f.read())
            heart_bytes.name = "heart.nii.gz"

        dose, structures = read_byte_data(dose_path, [ptv_bytes, heart_bytes])

        # Verify
        self.assertIsInstance(dose, Dose)
        self.assertEqual(len(structures), 2)
        # Check structure names were extracted from filenames
        self.assertTrue(any("ptv" in name.lower() for name in structures.keys()))


class TestDVHByStructure(unittest.TestCase):
    """Test DVH computation for multiple structures."""

    def setUp(self):
        """Create test dose and structures."""
        # Create simple test dose
        dose_array = np.ones((10, 10, 10)) * 50.0
        dose_array[5:, 5:, 5:] = 60.0  # Higher dose in corner
        self.dose = Dose(dose_array, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))

        # Create test structures
        ptv_mask = np.zeros((10, 10, 10), dtype=bool)
        ptv_mask[3:7, 3:7, 3:7] = True
        self.ptv = Target("PTV", ptv_mask, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))

        heart_mask = np.zeros((10, 10, 10), dtype=bool)
        heart_mask[1:4, 1:4, 1:4] = True
        self.heart = OAR("Heart", heart_mask, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))

    def test_dvh_computation(self):
        """Test DVH computation returns DataFrame with correct structure."""
        structures = {"PTV": self.ptv, "Heart": self.heart}

        df = dvh_by_structure(self.dose, structures)

        # Check DataFrame structure
        self.assertIn("Dose", df.columns)
        self.assertIn("Volume", df.columns)
        self.assertIn("Structure", df.columns)

        # Check structures are present
        structure_names = df["Structure"].unique()
        self.assertIn("PTV", structure_names)
        self.assertIn("Heart", structure_names)

        # Check dose values are reasonable
        self.assertTrue((df["Dose"] >= 0).all())
        self.assertTrue((df["Dose"] <= self.dose.max_dose * 1.1).all())

        # Check volume values are reasonable (0-100%)
        self.assertTrue((df["Volume"] >= 0).all())
        self.assertTrue((df["Volume"] <= 100).all())

    def test_dvh_monotonic_decreasing(self):
        """Test that DVH volume decreases with increasing dose."""
        structures = {"PTV": self.ptv}

        df = dvh_by_structure(self.dose, structures)
        ptv_dvh = df[df["Structure"] == "PTV"]

        # Volume should be monotonically decreasing
        volumes = ptv_dvh["Volume"].values
        for i in range(len(volumes) - 1):
            self.assertGreaterEqual(volumes[i], volumes[i + 1])

    def test_dvh_empty_structure(self):
        """Test DVH with an empty structure."""
        # Create empty structure
        empty_mask = np.zeros((10, 10, 10), dtype=bool)
        empty_oar = OAR("Empty", empty_mask, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))

        structures = {"Empty": empty_oar}
        df = dvh_by_structure(self.dose, structures)

        # Should still return a DataFrame
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn("Empty", df["Structure"].unique())

    def test_dvh_multiple_structures(self):
        """Test DVH with multiple structures."""
        structures = {"PTV": self.ptv, "Heart": self.heart}

        df = dvh_by_structure(self.dose, structures)

        # Check each structure has data
        ptv_data = df[df["Structure"] == "PTV"]
        heart_data = df[df["Structure"] == "Heart"]

        self.assertGreater(len(ptv_data), 0)
        self.assertGreater(len(heart_data), 0)

        # Both should have reasonable number of dose bins
        self.assertGreater(len(ptv_data["Dose"].unique()), 50)
        self.assertGreater(len(heart_data["Dose"].unique()), 50)


if __name__ == "__main__":
    unittest.main()
