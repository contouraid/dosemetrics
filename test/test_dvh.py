import dosemetrics.dvh as dvh
import numpy as np
import unittest
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestDVH(unittest.TestCase):
    def test_compute_dvh_single_voxel(self):
        logger.info("Testing compute DVH for single voxel ...")
        zero_dose = np.zeros((10, 10, 10))
        single_voxel_struct = np.zeros((10, 10, 10))
        single_voxel_struct[5, 5, 5] = 1.0
        results = dvh.compute_dvh(zero_dose, single_voxel_struct)
        logger.info(f"DVH results: {results}")

        self.assertTrue(len(results[0]) == 650)
        self.assertTrue(len(results[1]) == 650)

        # Check that the Dose range is from 0.0 to 65 - this is default.
        for i in range(1, 650):
            self.assertTrue(results[0][i] == i * 0.1)

        # Check that single voxel is indeed 0 dose.
        self.assertTrue(results[1][0] == 100.0)
        for i in range(1, 650):
            self.assertTrue(results[1][i] == 0.0)
        logger.info("... completed testing compute DVH for single voxel.")

    def test_compute_dvh_zero_voxels(self):
        logger.info("Testing compute DVH for zero voxels ...")
        zero_dose = np.zeros((10, 10, 10))
        single_voxel_struct = np.zeros((10, 10, 10))
        single_voxel_struct[5, 5, 5] = 1.0
        max_dose = [10, 20, 30]
        step_size = [1, 2, 5]

        for md in max_dose:
            for ss in step_size:
                results = dvh.compute_dvh(
                    zero_dose, single_voxel_struct, max_dose=md, step_size=ss
                )

                self.assertTrue(len(results[0]) == md // ss)
                self.assertTrue(len(results[1]) == md // ss)

                # Check that the Dose range is from 0.0 to 65 - this is default.
                for i in range(1, md // ss):
                    self.assertTrue(results[0][i] == i * ss)

                # Check that single voxel is indeed 0 dose.
                for i in range(1, md // ss):
                    self.assertTrue(results[1][i] == 0.0)
        logger.info("... completed testing compute DVH for zero voxels.")


if __name__ == "__main__":
    unittest.main()
