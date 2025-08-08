import numpy as np
import unittest
import logging

from dosemetrics import metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestMetrics(unittest.TestCase):
    def test_create_OAR(self):
        logger.info("Testing OAR creation ...")
        # Create a zero dose structure.
        oar = metrics.OAR("test_oar")

        # Test the structure name and default volume
        self.assertTrue(oar.name == "test_oar")
        self.assertTrue(oar.volume() == 0)

        # Edit the dose list.
        oar.dose = [1, 2, 3, 4, 5]
        self.assertTrue(oar.volume() == 5)

        # Test max and mean dose
        self.assertTrue(oar.max_dose() == np.max(oar.dose))
        self.assertTrue(oar.mean_dose() == np.mean(oar.dose))
        logger.info("... completed testing OAR creation.")
