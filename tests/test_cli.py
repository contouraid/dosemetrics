"""
Tests for dosemetrics CLI.

Tests command-line interface functionality for all available commands.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import subprocess
import sys
import json
import pandas as pd
import numpy as np

import dosemetrics
from dosemetrics import Dose, StructureSet


class TestCLI(unittest.TestCase):
    """Test CLI functionality."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.temp_dir = Path(tempfile.mkdtemp())

        # Create test dose distribution
        dose_data = np.random.rand(64, 64, 64) * 60  # Random dose 0-60 Gy
        cls.dose = Dose(dose_data, spacing=(2.0, 2.0, 2.0), origin=(0.0, 0.0, 0.0))
        cls.dose_file = cls.temp_dir / "dose.nii.gz"
        dosemetrics.nifti_io.write_nifti_volume(
            cls.dose.dose_array, cls.dose_file, cls.dose.spacing
        )

        # Create test structures
        cls.structures_dir = cls.temp_dir / "structures"
        cls.structures_dir.mkdir()

        # Target structure (sphere in center)
        target_mask = np.zeros((64, 64, 64), dtype=bool)
        center = 32
        for i in range(64):
            for j in range(64):
                for k in range(64):
                    if (i - center) ** 2 + (j - center) ** 2 + (
                        k - center
                    ) ** 2 < 10**2:
                        target_mask[i, j, k] = True

        target_file = cls.structures_dir / "Target.nii.gz"
        dosemetrics.nifti_io.write_nifti_volume(
            target_mask.astype(float), target_file, cls.dose.spacing
        )
        cls.target_file = target_file

        # OAR structure (offset sphere)
        oar_mask = np.zeros((64, 64, 64), dtype=bool)
        center_oar = 45
        for i in range(64):
            for j in range(64):
                for k in range(64):
                    if (i - center_oar) ** 2 + (j - center) ** 2 + (
                        k - center
                    ) ** 2 < 5**2:
                        oar_mask[i, j, k] = True

        oar_file = cls.structures_dir / "Brain.nii.gz"
        dosemetrics.nifti_io.write_nifti_volume(
            oar_mask.astype(float), oar_file, cls.dose.spacing
        )

        # Create a second dose for gamma testing
        dose_data2 = dose_data + np.random.randn(64, 64, 64) * 2  # Add noise
        cls.dose2_file = cls.temp_dir / "dose2.nii.gz"
        dosemetrics.nifti_io.write_nifti_volume(
            dose_data2, cls.dose2_file, cls.dose.spacing
        )

        # Create second structure set for geometric comparison
        cls.structures_dir2 = cls.temp_dir / "structures2"
        cls.structures_dir2.mkdir()

        # Slightly different target (for geometric comparison)
        target_mask2 = np.zeros((64, 64, 64), dtype=bool)
        for i in range(64):
            for j in range(64):
                for k in range(64):
                    if (i - center - 1) ** 2 + (j - center) ** 2 + (
                        k - center
                    ) ** 2 < 10**2:
                        target_mask2[i, j, k] = True

        target_file2 = cls.structures_dir2 / "Target.nii.gz"
        dosemetrics.nifti_io.write_nifti_volume(
            target_mask2.astype(float), target_file2, cls.dose.spacing
        )

    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures."""
        shutil.rmtree(cls.temp_dir)

    def run_cli(self, *args):
        """Helper to run CLI command."""
        cmd = [sys.executable, "-m", "dosemetrics_cli"] + list(args)
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result

    def test_cli_version(self):
        """Test CLI version command."""
        result = self.run_cli("--version")
        self.assertEqual(result.returncode, 0)
        self.assertIn("dosemetrics", result.stdout)

    def test_cli_help(self):
        """Test CLI help command."""
        result = self.run_cli("--help")
        self.assertEqual(result.returncode, 0)
        self.assertIn("dosemetrics", result.stdout)
        self.assertIn("dvh", result.stdout)
        self.assertIn("statistics", result.stdout)

    def test_dvh_command(self):
        """Test DVH computation command."""
        output_file = self.temp_dir / "dvh_output.csv"

        result = self.run_cli(
            "dvh", str(self.dose_file), str(self.structures_dir), "-o", str(output_file)
        )

        self.assertEqual(result.returncode, 0, f"Command failed: {result.stderr}")
        self.assertTrue(output_file.exists())

        # Check output format
        df = pd.read_csv(output_file)
        self.assertIn("Structure", df.columns)
        self.assertIn("Dose", df.columns)
        self.assertIn("Volume", df.columns)
        self.assertTrue(len(df) > 0)

    def test_statistics_command(self):
        """Test dose statistics command."""
        output_file = self.temp_dir / "stats_output.csv"

        result = self.run_cli(
            "statistics",
            str(self.dose_file),
            str(self.structures_dir),
            "-o",
            str(output_file),
        )

        self.assertEqual(result.returncode, 0, f"Command failed: {result.stderr}")
        self.assertTrue(output_file.exists())

        # Check output format
        df = pd.read_csv(output_file)
        self.assertIn("Structure", df.columns)
        self.assertIn("Mean Dose (Gy)", df.columns)
        self.assertIn("Max Dose (Gy)", df.columns)
        self.assertIn("Min Dose (Gy)", df.columns)
        self.assertTrue(len(df) > 0)

    def test_conformity_command(self):
        """Test conformity indices command."""
        output_file = self.temp_dir / "conformity_output.json"

        result = self.run_cli(
            "conformity",
            str(self.dose_file),
            str(self.target_file),
            "--prescription",
            "60",
            "-o",
            str(output_file),
        )

        self.assertEqual(result.returncode, 0, f"Command failed: {result.stderr}")
        self.assertTrue(output_file.exists())

        # Check output format
        with open(output_file) as f:
            data = json.load(f)

        self.assertIn("conformity_index", data)
        self.assertIn("conformity_number", data)
        self.assertIn("paddick_conformity_index", data)
        self.assertIn("coverage", data)
        self.assertIn("spillage", data)
        self.assertIsInstance(data["conformity_index"], (int, float))

    def test_homogeneity_command(self):
        """Test homogeneity indices command."""
        output_file = self.temp_dir / "homogeneity_output.json"

        result = self.run_cli(
            "homogeneity",
            str(self.dose_file),
            str(self.target_file),
            "--prescription",
            "60",
            "-o",
            str(output_file),
        )

        self.assertEqual(result.returncode, 0, f"Command failed: {result.stderr}")
        self.assertTrue(output_file.exists())

        # Check output format
        with open(output_file) as f:
            data = json.load(f)

        self.assertIn("homogeneity_index", data)
        self.assertIsInstance(data["homogeneity_index"], (int, float))

    def test_geometric_command(self):
        """Test geometric comparison command."""
        output_file = self.temp_dir / "geometric_output.csv"

        result = self.run_cli(
            "geometric",
            str(self.structures_dir),
            str(self.structures_dir2),
            "-o",
            str(output_file),
        )

        self.assertEqual(result.returncode, 0, f"Command failed: {result.stderr}")
        self.assertTrue(output_file.exists())

        # Check output format
        df = pd.read_csv(output_file)
        self.assertIn("Structure", df.columns)
        self.assertIn("Dice", df.columns)
        self.assertIn("Jaccard", df.columns)
        self.assertTrue(len(df) > 0)

    def test_gamma_command(self):
        """Test gamma analysis command."""
        output_file = self.temp_dir / "gamma_map.nii.gz"
        report_file = self.temp_dir / "gamma_report.json"

        result = self.run_cli(
            "gamma",
            str(self.dose_file),
            str(self.dose2_file),
            "--dose-criteria",
            "3.0",
            "--distance-criteria",
            "3.0",
            "-o",
            str(output_file),
            "--report",
            str(report_file),
        )

        self.assertEqual(result.returncode, 0, f"Command failed: {result.stderr}")
        self.assertTrue(output_file.exists())
        self.assertTrue(report_file.exists())

        # Check report format
        with open(report_file) as f:
            data = json.load(f)

        self.assertIn("passing_rate", data)
        self.assertIn("mean_gamma", data)
        self.assertIsInstance(data["passing_rate"], (int, float))

    def test_compliance_command(self):
        """Test compliance checking command."""
        output_file = self.temp_dir / "compliance_output.csv"

        result = self.run_cli(
            "compliance",
            str(self.dose_file),
            str(self.structures_dir),
            "-o",
            str(output_file),
        )

        self.assertEqual(result.returncode, 0, f"Command failed: {result.stderr}")
        self.assertTrue(output_file.exists())

        # Check output format
        df = pd.read_csv(output_file, index_col=0)
        self.assertIn("Compliance", df.columns)
        self.assertIn("Reason", df.columns)
        self.assertTrue(len(df) > 0)

    def test_invalid_command(self):
        """Test that invalid command returns error."""
        result = self.run_cli("invalid_command")
        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
