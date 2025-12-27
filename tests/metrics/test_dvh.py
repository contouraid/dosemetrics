"""
Comprehensive tests for dosemetrics.metrics.dvh module.

Tests DVH computation and DVH-based metrics using new OOP API.
"""

import pytest
import numpy as np
from pathlib import Path
from huggingface_hub import snapshot_download

from dosemetrics.dose import Dose
from dosemetrics.structures import Target, OAR
from dosemetrics.metrics import dvh


@pytest.fixture(scope="module")
def hf_data_path():
    """Download test data from HuggingFace once per module."""
    data_path = snapshot_download(
        repo_id="contouraid/dosemetrics-data",
        repo_type="dataset"
    )
    return Path(data_path)


@pytest.fixture
def sample_dose():
    """Create a sample dose distribution."""
    # Create gradient dose: 0-60 Gy
    dose_array = np.zeros((50, 50, 30))
    for z in range(30):
        dose_array[:, :, z] = z * 2.0  # 0, 2, 4, ..., 58 Gy
    
    return Dose(dose_array, (2.0, 2.0, 3.0), (0.0, 0.0, 0.0), "TestDose")


@pytest.fixture
def uniform_dose():
    """Create uniform dose distribution."""
    dose_array = np.ones((50, 50, 30)) * 50.0
    return Dose(dose_array, (2.0, 2.0, 3.0), (0.0, 0.0, 0.0))


@pytest.fixture
def sample_structure():
    """Create sample target structure."""
    mask = np.zeros((50, 50, 30), dtype=bool)
    mask[20:30, 20:30, 10:20] = True
    return Target("TestPTV", mask, (2.0, 2.0, 3.0), (0.0, 0.0, 0.0))


class TestComputeDVH:
    """Test DVH computation."""
    
    def test_compute_dvh_basic(self, sample_dose, sample_structure):
        """Test basic DVH computation."""
        dose_bins, volumes = dvh.compute_dvh(sample_dose, sample_structure)
        
        assert isinstance(dose_bins, np.ndarray)
        assert isinstance(volumes, np.ndarray)
        assert len(dose_bins) == len(volumes)
        assert len(dose_bins) > 0
        
        # Check dose bins are ascending
        assert np.all(np.diff(dose_bins) > 0)
        
        # Check volumes are in valid range
        assert np.all(volumes >= 0)
        assert np.all(volumes <= 100)
        
        # DVH should be monotonically decreasing
        assert np.all(np.diff(volumes) <= 0)
        
        # First point should be ~100% (all voxels receive >= 0 Gy)
        assert volumes[0] >= 95.0
    
    def test_compute_dvh_custom_params(self, sample_dose, sample_structure):
        """Test DVH with custom max_dose and step_size."""
        dose_bins, volumes = dvh.compute_dvh(
            sample_dose,
            sample_structure,
            max_dose=40.0,
            step_size=1.0
        )
        
        # Should have 41 bins (0 to 40 inclusive)
        assert len(dose_bins) == 41
        assert dose_bins[0] == 0.0
        assert dose_bins[-1] == 40.0
        
        # Step size should be 1.0
        assert np.allclose(np.diff(dose_bins), 1.0)
    
    def test_compute_dvh_uniform_dose(self, uniform_dose, sample_structure):
        """Test DVH for uniform dose distribution."""
        dose_bins, volumes = dvh.compute_dvh(uniform_dose, sample_structure, max_dose=51.0)
        
        # For uniform 50 Gy: V>=0 = 100%, V>=50 = 100%, V>=51 = 0%
        v0 = volumes[0]  # V >= 0
        
        # Find volume at 50 Gy
        idx_50 = np.argmin(np.abs(dose_bins - 50.0))
        v50 = volumes[idx_50]
        
        # Find volume at 51 Gy
        idx_51 = np.argmin(np.abs(dose_bins - 51.0))
        v51 = volumes[idx_51]
        
        assert v0 >= 95.0  # Should be ~100%
        assert v50 >= 95.0  # Should be ~100%
        assert v51 < 5.0    # Should be ~0%
    
    def test_compute_dvh_zero_dose(self, sample_structure):
        """Test DVH for zero dose."""
        zero_dose = Dose(np.zeros((50, 50, 30)), (2.0, 2.0, 3.0), (0.0, 0.0, 0.0))
        
        dose_bins, volumes = dvh.compute_dvh(zero_dose, sample_structure)
        
        # Only first bin should have 100%, rest should be 0%
        assert volumes[0] >= 95.0
        assert np.all(volumes[1:] < 5.0)


class TestVolumeAtDose:
    """Test V_X (volume at dose) queries."""
    
    def test_volume_at_dose_basic(self, sample_dose, sample_structure):
        """Test basic V_X computation."""
        v20 = dvh.compute_volume_at_dose(sample_dose, sample_structure, 20.0)
        v40 = dvh.compute_volume_at_dose(sample_dose, sample_structure, 40.0)
        
        assert isinstance(v20, (float, np.floating))
        assert isinstance(v40, (float, np.floating))
        assert 0 <= v20 <= 100
        assert 0 <= v40 <= 100
        
        # More volume receives >= 20 Gy than >= 40 Gy
        assert v20 >= v40
    
    def test_volume_at_dose_uniform(self, uniform_dose, sample_structure):
        """Test V_X on uniform dose."""
        # Uniform 50 Gy
        v49 = dvh.compute_volume_at_dose(uniform_dose, sample_structure, 49.0)
        v50 = dvh.compute_volume_at_dose(uniform_dose, sample_structure, 50.0)
        v51 = dvh.compute_volume_at_dose(uniform_dose, sample_structure, 51.0)
        
        assert v49 >= 95.0  # Almost all voxels receive >= 49
        assert v50 >= 95.0  # Almost all voxels receive >= 50
        assert v51 < 5.0     # Almost no voxels receive >= 51
    
    def test_volume_at_dose_bounds(self, sample_dose, sample_structure):
        """Test V_X at boundary values."""
        v0 = dvh.compute_volume_at_dose(sample_dose, sample_structure, 0.0)
        v_high = dvh.compute_volume_at_dose(sample_dose, sample_structure, 100.0)
        
        assert v0 >= 95.0  # All voxels receive >= 0
        assert v_high < 5.0  # No voxels receive >= 100 (max is 58)


class TestDoseAtVolume:
    """Test D_X (dose at volume) queries."""
    
    def test_dose_at_volume_basic(self, sample_dose, sample_structure):
        """Test basic D_X computation."""
        d95 = dvh.compute_dose_at_volume(sample_dose, sample_structure, 95)
        d50 = dvh.compute_dose_at_volume(sample_dose, sample_structure, 50)
        d05 = dvh.compute_dose_at_volume(sample_dose, sample_structure, 5)
        
        assert isinstance(d95, (float, np.floating))
        assert isinstance(d50, (float, np.floating))
        assert isinstance(d05, (float, np.floating))
        
        # D95 <= D50 <= D05
        assert d95 <= d50 <= d05
    
    def test_dose_at_volume_uniform(self, uniform_dose, sample_structure):
        """Test D_X on uniform dose."""
        # All voxels get 50 Gy, so all D_X should be ~50
        d95 = dvh.compute_dose_at_volume(uniform_dose, sample_structure, 95)
        d50 = dvh.compute_dose_at_volume(uniform_dose, sample_structure, 50)
        d05 = dvh.compute_dose_at_volume(uniform_dose, sample_structure, 5)
        
        assert abs(d95 - 50.0) < 1.0
        assert abs(d50 - 50.0) < 1.0
        assert abs(d05 - 50.0) < 1.0
    
    def test_dose_at_volume_bounds(self, sample_dose, sample_structure):
        """Test D_X at boundary volumes."""
        d0 = dvh.compute_dose_at_volume(sample_dose, sample_structure, 0)    # Max dose
        d100 = dvh.compute_dose_at_volume(sample_dose, sample_structure, 100)  # Min dose
        
        # D0 should be near max, D100 near min
        assert d0 >= d100
    
    def test_dose_at_volume_invalid(self, sample_dose, sample_structure):
        """Test invalid volume percentages."""
        with pytest.raises(ValueError):
            dvh.compute_dose_at_volume(sample_dose, sample_structure, 150)
        
        with pytest.raises(ValueError):
            dvh.compute_dose_at_volume(sample_dose, sample_structure, -10)


class TestDoseAtVolumeCC:
    """Test D_Xcc (dose to absolute volume) queries."""
    
    def test_dose_at_volume_cc_basic(self, sample_dose, sample_structure):
        """Test D_Xcc computation."""
        # Structure volume in cc
        total_vol_cc = sample_structure.volume_cc()
        
        # Query dose to 1 cc
        d_1cc = dvh.compute_dose_at_volume_cc(sample_dose, sample_structure, 1.0)
        
        # Query dose to half volume
        d_half = dvh.compute_dose_at_volume_cc(
            sample_dose, sample_structure, total_vol_cc / 2
        )
        
        assert isinstance(d_1cc, (float, np.floating))
        assert isinstance(d_half, (float, np.floating))
        assert d_half >= 0
    
    def test_dose_at_volume_cc_small_volume(self, sample_dose, sample_structure):
        """Test D_0.1cc (common OAR constraint)."""
        d_01cc = dvh.compute_dose_at_volume_cc(sample_dose, sample_structure, 0.1)
        
        # Should be close to max dose
        from dosemetrics.metrics import statistics
        max_dose = statistics.compute_max_dose(sample_dose, sample_structure)
        
        assert d_01cc <= max_dose + 1.0
        assert d_01cc >= 0


class TestEquivalentUniformDose:
    """Test EUD computation."""
    
    def test_eud_basic(self, sample_dose, sample_structure):
        """Test basic EUD computation."""
        eud = dvh.compute_equivalent_uniform_dose(sample_dose, sample_structure, a_parameter=-10)
        
        assert isinstance(eud, (float, np.floating))
        assert eud > 0
    
    def test_eud_uniform_dose(self, uniform_dose, sample_structure):
        """Test EUD on uniform dose equals dose value."""
        eud = dvh.compute_equivalent_uniform_dose(uniform_dose, sample_structure, a_parameter=-10)
        
        # For uniform dose, EUD should equal the dose
        assert abs(eud - 50.0) < 1.0
    
    def test_eud_parameter_effect(self, sample_dose, sample_structure):
        """Test effect of 'a' parameter on EUD."""
        eud_neg = dvh.compute_equivalent_uniform_dose(sample_dose, sample_structure, a_parameter=-10)
        eud_pos = dvh.compute_equivalent_uniform_dose(sample_dose, sample_structure, a_parameter=10)
        
        # Both should be positive
        assert eud_neg > 0
        assert eud_pos > 0


class TestDVHTable:
    """Test DVH table creation."""
    
    def test_create_dvh_table(self, sample_dose, hf_data_path):
        """Test creating DVH table for multiple structures."""
        from dosemetrics.io import load_structure_set
        
        subject_path = hf_data_path / "test_subject"
        if not subject_path.exists():
            pytest.skip("Test data not available")
        
        dose = Dose.from_nifti(subject_path / "Dose.nii.gz")
        structures = load_structure_set(subject_path)
        
        # Create DVH table
        df = dvh.create_dvh_table(dose, structures, structure_names=structures.structure_names[:3])
        
        # Check DataFrame structure
        assert set(['Dose', 'Structure', 'Volume']).issubset(df.columns)
        
        # Check dose column is sorted within each structure
        for name in df['Structure'].unique():
            sub = df[df['Structure'] == name]
            assert (sub['Dose'].diff().dropna() >= 0).all()


class TestExtractDVHMetrics:
    """Test extracting DVH metrics from DVH curves."""
    
    def test_extract_dvh_metrics(self, sample_dose, sample_structure):
        """Test extracting standard DVH metrics."""
        metrics = dvh.extract_dvh_metrics(
            sample_dose,
            sample_structure,
            dose_thresholds=[10, 20],
            volume_percentages=[95, 50, 5]
        )
        
        # Check expected keys
        assert 'V10' in metrics and 'V20' in metrics
        assert 'D95' in metrics
        assert 'D50' in metrics
        assert 'D5' in metrics


class TestRealDataDVH:
    """Test DVH functions with real HuggingFace data."""
    
    def test_dvh_on_real_data(self, hf_data_path):
        """Test DVH computation on real dose and structures."""
        from dosemetrics.io import load_structure_set
        
        subject_path = hf_data_path / "test_subject"
        if not subject_path.exists():
            pytest.skip("Test data not available")
        
        dose = Dose.from_nifti(subject_path / "Dose.nii.gz")
        structures = load_structure_set(subject_path)
        
        # Test on all structures
        for name in structures.structure_names:
            structure = structures.get_structure(name)
            
            # Compute DVH
            dose_bins, volumes = dvh.compute_dvh(dose, structure)
            
            # Verify properties
            assert len(dose_bins) > 0
            assert len(volumes) > 0
            assert volumes[0] >= 90.0  # Should start near 100%
            assert np.all(np.diff(volumes) <= 0)  # Monotonic
    
    def test_dvh_metrics_on_real_data(self, hf_data_path):
        """Test DVH metrics on real data."""
        from dosemetrics.io import load_structure_set
        
        subject_path = hf_data_path / "test_subject"
        if not subject_path.exists():
            pytest.skip("Test data not available")
        
        dose = Dose.from_nifti(subject_path / "Dose.nii.gz")
        structures = load_structure_set(subject_path)
        
        # Test volume and dose queries
        for name in structures.structure_names:
            structure = structures.get_structure(name)
            
            # V20 query
            v20 = dvh.compute_volume_at_dose(dose, structure, 20.0)
            assert 0 <= v20 <= 100
            
            # D50 query
            d50 = dvh.compute_dose_at_volume(dose, structure, 50)
            assert d50 >= 0


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_structure(self, sample_dose):
        """Test DVH for empty structure."""
        empty = Target(
            "Empty",
            np.zeros((50, 50, 30), dtype=bool),
            (2.0, 2.0, 3.0),
            (0.0, 0.0, 0.0)
        )
        
        dose_bins, volumes = dvh.compute_dvh(sample_dose, empty)
        
        # Should handle gracefully
        assert len(dose_bins) > 0
        assert len(volumes) > 0
    
    def test_single_voxel_structure(self, sample_dose):
        """Test DVH for single-voxel structure."""
        single = Target(
            "SingleVoxel",
            np.zeros((50, 50, 30), dtype=bool),
            (2.0, 2.0, 3.0),
            (0.0, 0.0, 0.0)
        )
        single.mask[25, 25, 15] = True
        
        dose_bins, volumes = dvh.compute_dvh(sample_dose, single)
        
        # Should have step function: 100% until dose value, then 0%
        assert volumes[0] >= 95.0
    
    def test_incompatible_dose_structure(self, sample_dose):
        """Test error for incompatible dose and structure."""
        incompatible = Target(
            "Incompatible",
            np.zeros((30, 30, 20), dtype=bool),
            (1.0, 1.0, 1.0),
            (0.0, 0.0, 0.0)
        )
        
        with pytest.raises(ValueError):
            dvh.compute_dvh(sample_dose, incompatible)
