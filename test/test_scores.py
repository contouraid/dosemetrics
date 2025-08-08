import dosemetrics.scores as scores
import numpy as np
import unittest
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestScores(unittest.TestCase):
    def test_dose_score(self):
        logger.info("Testing dose score ...")
        zero_dose = np.zeros((10, 10, 10))

        zero_score = scores.dose_score(zero_dose, zero_dose)
        self.assertTrue(zero_score == 0.0)

        single_voxel_struct = np.zeros((10, 10, 10))
        single_voxel_struct[5, 5, 5] = 1
        non_zero_score = scores.dose_score(zero_dose, single_voxel_struct)
        self.assertTrue(non_zero_score > 0.0)

        no_difference_score = scores.dose_score(
            single_voxel_struct, single_voxel_struct
        )
        self.assertTrue(no_difference_score == 0.0)

        commutative_score = scores.dose_score(single_voxel_struct, zero_dose)
        self.assertTrue(non_zero_score == commutative_score)

        another_single_voxel_struct = np.zeros((10, 10, 10))
        another_single_voxel_struct[5, 5, 6] = 1
        zero_mask_score = scores.dose_score(
            zero_dose, another_single_voxel_struct, single_voxel_struct
        )
        self.assertTrue(zero_mask_score == 0.0)

        large_single_voxel_struct = np.zeros((10, 10, 10))
        large_single_voxel_struct[5, 5, 5] = 10.0

        larger_score = scores.dose_score(large_single_voxel_struct, zero_dose)
        self.assertTrue(larger_score > non_zero_score)
        logger.info("... completed testing dose score.")

    def test_dvh_score(self):
        logger.info("Testing DVH score ...")
        zero_dose = np.zeros((10, 10, 10))
        single_voxel_struct = np.zeros((10, 10, 10))
        single_voxel_struct[5, 5, 5] = 1

        zero_score = scores.dvh_score(zero_dose, single_voxel_struct, "OAR", (1, 1, 1))
        self.assertTrue("D_0.1_cc" in set(zero_score.keys()))
        self.assertTrue("mean" in set(zero_score.keys()))

        self.assertTrue(zero_score["D_0.1_cc"] == 0.0)
        self.assertTrue(zero_score["mean"] == 0.0)

        non_zero_score = scores.dvh_score(zero_dose, single_voxel_struct, "Target")
        self.assertTrue("D1" in set(non_zero_score.keys()))
        self.assertTrue("D95" in set(non_zero_score.keys()))
        self.assertTrue("D99" in set(non_zero_score.keys()))
        logger.info("... completed testing DVH score.")


if __name__ == "__main__":
    unittest.main()
