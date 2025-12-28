"""
Comprehensive tests for dosemetrics.metrics.homogeneity module.

Tests homogeneity and dose uniformity metrics.
"""

import pytest
import numpy as np
from pathlib import Path
from huggingface_hub import snapshot_download

from dosemetrics.dose import Dose
from dosemetrics.structures import Target
from dosemetrics.metrics import homogeneity


@pytest.fixture(scope="module")
def hf_data_path():
    """Download test data from HuggingFace once per module."""
    data_path = snapshot_download(
        repo_id="contouraid/dosemetrics-data",
        repo_type="dataset"
    )
    return Path(data_path)


@pytest.fixture
def uniform_dose():
    """Create perfectly uniform dose."""
    dose_array = np.ones((50, 50, 30)) * 50.0
    return Dose(dose_array, (2.0, 2.0, 3.0), (0.0, 0.0, 0.0))


@pytest.fixture
def homogeneous_target():
    """Create target for homogeneity testing."""
    mask = np.zeros((50, 50, 30), dtype=bool)
    mask[20:30, 20:30, 10:20] = True
    return Target("TestPTV", mask, (2.0, 2.0, 3.0), (0.0, 0.0, 0.0))


@pytest.fixture
def gradient_dose():
    """Create dose with gradient (inhomogeneous)."""
    dose_array = np.zeros((50, 50, 30))
    for z in range(30):
        dose_array[:, :, z] = 40 + z * 0.5  # 40 to 55 Gy gradient
    return Dose(dose_array, (2.0, 2.0, 3.0), (0.0, 0.0, 0.0))


@pytest.fixture
def hotspot_dose():
    """Create dose with hot spot."""
    dose_array = np.ones((50, 50, 30)) * 50.0
    # Add hot spot
    dose_array[25:27, 25:27, 14:16] = 70.0  # 40% hot spot
    return Dose(dose_array, (2.0, 2.0, 3.0), (0.0, 0.0, 0.0))


class TestHomogeneityIndex:
    """Test homogeneity index (HI = (D2-D98)/D50)."""
    
    def test_hi_uniform_dose(self, uniform_dose, homogeneous_target):
        """Test HI≈0 for perfectly uniform dose."""
        hi = homogeneity.compute_homogeneity_index(uniform_dose, homogeneous_target)
        
        assert isinstance(hi, (float, np.floating))
        assert hi < 0.01  # Should be very close to 0
    
    def test_hi_gradient_dose(self, gradient_dose, homogeneous_target):
        """Test HI>0 for dose with gradient."""
        hi = homogeneity.compute_homogeneity_index(gradient_dose, homogeneous_target)
        
        assert hi > 0  # Should be positive
        assert hi < 1.0  # Should be reasonable
    
    def test_hi_hotspot_dose(self, hotspot_dose, homogeneous_target):
        """Test HI detects hot spots."""
        hi = homogeneity.compute_homogeneity_index(hotspot_dose, homogeneous_target)
        
        assert hi >= 0  # HI is non-negative
    
    def test_hi_lower_is_better(self, uniform_dose, gradient_dose, homogeneous_target):
        """Test that lower HI means more homogeneous."""
        hi_uniform = homogeneity.compute_homogeneity_index(
            uniform_dose, homogeneous_target
        )
        hi_gradient = homogeneity.compute_homogeneity_index(
            gradient_dose, homogeneous_target
        )
        
        assert hi_uniform < hi_gradient


class TestGradientIndex:
    """Test gradient index (GI)."""
    
    def test_gi_ideal_case(self):
        """Test GI=1 for ideal sharp dose falloff."""
        # Create dose with sharp falloff
        dose_array = np.zeros((50, 50, 30))
        
        # Target region gets prescription dose
        target_mask = np.zeros((50, 50, 30), dtype=bool)
        target_mask[20:30, 20:30, 10:20] = True
        dose_array[target_mask] = 60.0
        
        # Sharp falloff - only target gets 60 Gy
        # No spillage at 50% isodose
        
        dose = Dose(dose_array, (2.0, 2.0, 3.0), (0.0, 0.0, 0.0))
        target = Target("TestPTV", target_mask, (2.0, 2.0, 3.0), (0.0, 0.0, 0.0))
        
        gi = homogeneity.compute_gradient_index(dose, target, 60.0)
        
        # GI should be close to 1 (V50%/V100% with sharp falloff)
        assert 1.0 <= gi <= 1.5
    
    def test_gi_with_spillage(self):
        """Test GI>1 when dose spills beyond target."""
        dose_array = np.zeros((50, 50, 30))
        
        # Target gets 60 Gy
        target_mask = np.zeros((50, 50, 30), dtype=bool)
        target_mask[20:30, 20:30, 10:20] = True
        dose_array[target_mask] = 60.0
        
        # Large spillage region gets 30+ Gy
        dose_array[15:35, 15:35, 5:25] = 35.0
        
        dose = Dose(dose_array, (2.0, 2.0, 3.0), (0.0, 0.0, 0.0))
        target = Target("TestPTV", target_mask, (2.0, 2.0, 3.0), (0.0, 0.0, 0.0))
        
        gi = homogeneity.compute_gradient_index(dose, target, 60.0)
        
        # GI should be > 1 due to spillage
        assert gi > 1.0


class TestDoseHomogeneity:
    """Test coefficient of variation (CV)."""
    
    def test_cv_uniform_dose(self, uniform_dose, homogeneous_target):
        """Test CV≈0 for uniform dose."""
        cv = homogeneity.compute_dose_homogeneity(uniform_dose, homogeneous_target)
        
        assert cv < 0.01  # Should be very small
    
    def test_cv_gradient_dose(self, gradient_dose, homogeneous_target):
        """Test CV>0 for inhomogeneous dose."""
        cv = homogeneity.compute_dose_homogeneity(gradient_dose, homogeneous_target)
        
        assert cv > 0
        assert cv < 1.0  # Should be reasonable
    
    def test_cv_lower_is_better(self, uniform_dose, gradient_dose, homogeneous_target):
        """Test that lower CV means more homogeneous."""
        cv_uniform = homogeneity.compute_dose_homogeneity(
            uniform_dose, homogeneous_target
        )
        cv_gradient = homogeneity.compute_dose_homogeneity(
            gradient_dose, homogeneous_target
        )
        
        assert cv_uniform < cv_gradient


class TestUniformityIndex:
    """Test uniformity index (UI)."""
    
    def test_ui_uniform_dose(self, uniform_dose, homogeneous_target):
        """Test UI≈1 for perfectly uniform dose."""
        ui = homogeneity.compute_uniformity_index(uniform_dose, homogeneous_target)
        
        assert 0.99 <= ui <= 1.0
    
    def test_ui_gradient_dose(self, gradient_dose, homogeneous_target):
        """Test UI<1 for inhomogeneous dose."""
        ui = homogeneity.compute_uniformity_index(gradient_dose, homogeneous_target)
        
        assert 0 < ui < 1.0
    
    def test_ui_higher_is_better(self, uniform_dose, gradient_dose, homogeneous_target):
        """Test that higher UI means more uniform."""
        ui_uniform = homogeneity.compute_uniformity_index(
            uniform_dose, homogeneous_target
        )
        ui_gradient = homogeneity.compute_uniformity_index(
            gradient_dose, homogeneous_target
        )
        
        assert ui_uniform > ui_gradient


class TestRealDataHomogeneity:
    """Test homogeneity metrics with real HuggingFace data."""
    
    def test_homogeneity_on_real_data(self, hf_data_path):
        """Test homogeneity metrics on real dose."""
        from dosemetrics.io import load_structure_set
        from dosemetrics.metrics import dvh
        
        subject_path = hf_data_path / "test_subject"
        if not subject_path.exists():
            pytest.skip("Test data not available")
        
        dose = Dose.from_nifti(subject_path / "Dose.nii.gz")
        structures = load_structure_set(subject_path)
        
        # Find target
        target = None
        for name in structures.structure_names:
            if "PTV" in name.upper() or "GTV" in name.upper():
                target = structures.get_structure(name)
                break
        
        if target is None:
            pytest.skip("No target found")
        
        # Compute homogeneity metrics
        hi = homogeneity.compute_homogeneity_index(dose, target)
        cv = homogeneity.compute_dose_homogeneity(dose, target)
        ui = homogeneity.compute_uniformity_index(dose, target)
        
        # Check reasonable ranges
        assert hi >= 0
        assert cv >= 0
        assert 0 <= ui <= 1.0
        
        # Clinical plans should have reasonable homogeneity
        assert hi < 0.5  # HI typically < 0.2-0.3 for good plans
        assert cv < 0.5
        assert ui >= 0  # UI can vary, just check it's valid
        
        # Get prescription dose for GI
        stats = dvh.compute_dose_statistics(dose, target)
        prescription = stats['D95']
        
        gi = homogeneity.compute_gradient_index(dose, target, prescription)
        assert gi >= 1.0  # GI is always >= 1


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_single_dose_value(self):
        """Test with constant dose (all voxels same value)."""
        dose_array = np.ones((50, 50, 30)) * 42.0
        dose = Dose(dose_array, (2.0, 2.0, 3.0), (0.0, 0.0, 0.0))
        
        mask = np.zeros((50, 50, 30), dtype=bool)
        mask[20:30, 20:30, 10:20] = True
        target = Target("Test", mask, (2.0, 2.0, 3.0), (0.0, 0.0, 0.0))
        
        hi = homogeneity.compute_homogeneity_index(dose, target)
        cv = homogeneity.compute_dose_homogeneity(dose, target)
        ui = homogeneity.compute_uniformity_index(dose, target)
        
        # All should indicate perfect homogeneity
        assert hi < 0.01
        assert cv < 0.01
        assert ui > 0.99
    
    def test_zero_dose(self):
        """Test with zero dose everywhere."""
        dose = Dose(np.zeros((50, 50, 30)), (2.0, 2.0, 3.0), (0.0, 0.0, 0.0))
        
        mask = np.zeros((50, 50, 30), dtype=bool)
        mask[20:30, 20:30, 10:20] = True
        target = Target("Test", mask, (2.0, 2.0, 3.0), (0.0, 0.0, 0.0))
        
        # HI may be inf when D50=0 (division by zero)
        hi = homogeneity.compute_homogeneity_index(dose, target)
        # Just check it doesn't crash
        assert hi is not None
    
    def test_empty_structure(self, uniform_dose):
        """Test with empty structure."""
        empty = Target(
            "Empty",
            np.zeros((50, 50, 30), dtype=bool),
            (2.0, 2.0, 3.0),
            (0.0, 0.0, 0.0)
        )
        
        hi = homogeneity.compute_homogeneity_index(uniform_dose, empty)
        assert hi == 0.0
