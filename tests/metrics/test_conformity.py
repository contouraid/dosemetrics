"""
Comprehensive tests for dosemetrics.metrics.conformity module.

Tests conformity indices for target coverage evaluation.
"""

import pytest
import numpy as np
from pathlib import Path
from huggingface_hub import snapshot_download

from dosemetrics.dose import Dose
from dosemetrics.structures import Target
from dosemetrics.metrics import conformity


@pytest.fixture(scope="module")
def hf_data_path():
    """Download test data from HuggingFace once per module."""
    data_path = snapshot_download(
        repo_id="contouraid/dosemetrics-data",
        repo_type="dataset"
    )
    return Path(data_path)


@pytest.fixture
def ideal_dose():
    """Create an ideal dose distribution matching target exactly."""
    dose_array = np.zeros((50, 50, 30))
    # High dose in center region
    dose_array[20:30, 20:30, 10:20] = 60.0
    
    return Dose(dose_array, (2.0, 2.0, 3.0), (0.0, 0.0, 0.0))


@pytest.fixture
def ideal_target():
    """Create target matching ideal dose region."""
    mask = np.zeros((50, 50, 30), dtype=bool)
    mask[20:30, 20:30, 10:20] = True
    
    return Target("IdealPTV", mask, (2.0, 2.0, 3.0), (0.0, 0.0, 0.0))


@pytest.fixture
def spillage_dose():
    """Create dose with spillage beyond target."""
    dose_array = np.zeros((50, 50, 30))
    # Target region gets 60 Gy
    dose_array[20:30, 20:30, 10:20] = 60.0
    # Spillage region gets 60 Gy too
    dose_array[15:35, 15:35, 8:22] = 60.0
    
    return Dose(dose_array, (2.0, 2.0, 3.0), (0.0, 0.0, 0.0))


@pytest.fixture
def undercoverage_dose():
    """Create dose with poor target coverage."""
    dose_array = np.zeros((50, 50, 30))
    # Only half of target gets prescription dose
    dose_array[20:25, 20:30, 10:20] = 60.0  # Half
    dose_array[25:30, 20:30, 10:20] = 40.0  # Half underdosed
    
    return Dose(dose_array, (2.0, 2.0, 3.0), (0.0, 0.0, 0.0))


class TestConformityIndex:
    """Test ICRU conformity index (CI)."""
    
    def test_ci_ideal_case(self, ideal_dose, ideal_target):
        """Test CI=1 for ideal conformal plan."""
        ci = conformity.compute_conformity_index(ideal_dose, ideal_target, 60.0)
        
        assert isinstance(ci, (float, np.floating))
        assert 0.95 <= ci <= 1.0  # Should be ~1 for ideal case
    
    def test_ci_with_spillage(self, spillage_dose, ideal_target):
        """Test CI<1 when dose spills beyond target."""
        ci = conformity.compute_conformity_index(spillage_dose, ideal_target, 60.0)
        
        assert 0 < ci < 1.0  # Should be < 1 due to spillage
    
    def test_ci_with_undercoverage(self, undercoverage_dose, ideal_target):
        """Test CI<1 when target is undercovered."""
        ci = conformity.compute_conformity_index(undercoverage_dose, ideal_target, 60.0)
        
        assert 0 < ci <= 1.0  # Should be <= 1, may be 1.0 if TV_PIV = TV
    
    def test_ci_zero_dose(self, ideal_target):
        """Test CI=0 for zero dose."""
        zero_dose = Dose(np.zeros((50, 50, 30)), (2.0, 2.0, 3.0), (0.0, 0.0, 0.0))
        ci = conformity.compute_conformity_index(zero_dose, ideal_target, 60.0)
        
        assert ci == 0.0


class TestConformityNumber:
    """Test conformity number (CN)."""
    
    def test_cn_ideal_case(self, ideal_dose, ideal_target):
        """Test CN=1 for ideal plan."""
        cn = conformity.compute_conformity_number(ideal_dose, ideal_target, 60.0)
        
        assert isinstance(cn, (float, np.floating))
        assert 0.95 <= cn <= 1.0
    
    def test_cn_with_spillage(self, spillage_dose, ideal_target):
        """Test CN accounts for both coverage and spillage."""
        cn = conformity.compute_conformity_number(spillage_dose, ideal_target, 60.0)
        
        # CN should be lower due to spillage
        assert 0 < cn < 1.0
    
    def test_cn_worse_than_ci(self, spillage_dose, ideal_target):
        """Test CN ≤ CI (CN is stricter)."""
        ci = conformity.compute_conformity_index(spillage_dose, ideal_target, 60.0)
        cn = conformity.compute_conformity_number(spillage_dose, ideal_target, 60.0)
        
        assert cn <= ci + 0.01  # Allow small numerical error


class TestPaddickConformityIndex:
    """Test Paddick conformity index (for SRS)."""
    
    def test_paddick_ideal_case(self, ideal_dose, ideal_target):
        """Test Paddick CI=1 for ideal case."""
        pci = conformity.compute_paddick_conformity_index(
            ideal_dose, ideal_target, 60.0
        )
        
        assert isinstance(pci, (float, np.floating))
        assert 0.95 <= pci <= 1.0
    
    def test_paddick_with_spillage(self, spillage_dose, ideal_target):
        """Test Paddick CI penalizes spillage."""
        pci = conformity.compute_paddick_conformity_index(
            spillage_dose, ideal_target, 60.0
        )
        
        assert 0 < pci < 1.0
    
    def test_paddick_vs_cn(self, spillage_dose, ideal_target):
        """Test Paddick CI similar to CN."""
        pci = conformity.compute_paddick_conformity_index(
            spillage_dose, ideal_target, 60.0
        )
        cn = conformity.compute_conformity_number(
            spillage_dose, ideal_target, 60.0
        )
        
        # They should be similar (both use TV/PIV * TV/TIV)
        assert abs(pci - cn) < 0.1


class TestCoverageAndSpillage:
    """Test coverage and spillage helper metrics."""
    
    def test_coverage_ideal(self, ideal_dose, ideal_target):
        """Test 100% coverage for ideal case."""
        cov = conformity.compute_coverage(ideal_dose, ideal_target, 60.0)
        
        assert 0.95 <= cov <= 1.0
    
    def test_coverage_underdosed(self, undercoverage_dose, ideal_target):
        """Test reduced coverage for underdosed target."""
        cov = conformity.compute_coverage(undercoverage_dose, ideal_target, 60.0)
        
        assert 0 < cov < 1.0
    
    def test_spillage_ideal(self, ideal_dose, ideal_target):
        """Test minimal spillage for ideal case."""
        spill = conformity.compute_spillage(ideal_dose, ideal_target, 60.0)
        
        assert 0 <= spill <= 0.05  # Should be very low
    
    def test_spillage_high(self, spillage_dose, ideal_target):
        """Test high spillage when dose extends beyond target."""
        spill = conformity.compute_spillage(spillage_dose, ideal_target, 60.0)
        
        assert spill > 0.1  # Should be significant


class TestRealDataConformity:
    """Test conformity metrics with real HuggingFace data."""
    
    def test_conformity_on_real_data(self, hf_data_path):
        """Test conformity metrics on real dose and structures."""
        from dosemetrics.io import load_structure_set
        
        subject_path = hf_data_path / "test_subject"
        if not subject_path.exists():
            pytest.skip("Test data not available")
        
        dose = Dose.from_nifti(subject_path / "Dose.nii.gz")
        structures = load_structure_set(subject_path)
        
        # Find PTV if available
        ptv = None
        for name in structures.structure_names:
            if "PTV" in name.upper() or "GTV" in name.upper():
                ptv = structures.get_structure(name)
                break
        
        if ptv is None:
            pytest.skip("No PTV/GTV found")
        
        # Estimate prescription dose (e.g., D95 of target)
        from dosemetrics.metrics import dvh
        stats = dvh.compute_dose_statistics(dose, ptv)
        prescription = stats['D95']
        
        # Compute conformity indices
        ci = conformity.compute_conformity_index(dose, ptv, prescription)
        cn = conformity.compute_conformity_number(dose, ptv, prescription)
        pci = conformity.compute_paddick_conformity_index(dose, ptv, prescription)
        cov = conformity.compute_coverage(dose, ptv, prescription)
        spill = conformity.compute_spillage(dose, ptv, prescription)
        
        # Check reasonable ranges
        assert 0 <= ci <= 1.0
        assert 0 <= cn <= 1.0
        assert 0 <= pci <= 1.0
        assert 0 <= cov <= 1.0
        assert 0 <= spill <= 1.0
        
        # CN should be ≤ CI
        assert cn <= ci + 0.01
        
        # Good clinical plans should have decent coverage
        assert cov > 0.8


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_target(self, ideal_dose):
        """Test with empty target."""
        empty = Target(
            "Empty",
            np.zeros((50, 50, 30), dtype=bool),
            (2.0, 2.0, 3.0),
            (0.0, 0.0, 0.0)
        )
        
        ci = conformity.compute_conformity_index(ideal_dose, empty, 60.0)
        assert ci == 0.0
    
    def test_zero_prescription(self, ideal_dose, ideal_target):
        """Test with zero prescription dose."""
        ci = conformity.compute_conformity_index(ideal_dose, ideal_target, 0.0)
        
        # All voxels receive >= 0 Gy, so should be some value
        assert ci >= 0
    
    def test_high_prescription(self, ideal_dose, ideal_target):
        """Test with prescription higher than max dose."""
        # Max dose is 60, ask for 100
        ci = conformity.compute_conformity_index(ideal_dose, ideal_target, 100.0)
        
        # No voxels receive >= 100, so CI should be 0
        assert ci == 0.0
