"""
METRICS: holds data structures to record geometric and dosimetric measures.
"""
import numpy as np


class Structure:
    """
    STRUCTURE: holds information about structures of interest.
    """

    def __init__(self, name):
        self.name = name
        self.dose = []

    def __print__(self):
        print("I am a ", self.name, ", with volume: ", self.volume())

    def volume(self):
        return len(self.dose)

    def mean_dose(self):
        return sum(self.dose) / self.volume()

    def max_dose(self):
        return max(self.dose)

    def dvh(self, max_dose=70, step_size=0.1):

        if max_dose is None:
            max_dose = self.max_dose()

        bins = np.arange(0, max_dose, step_size)
        total_voxels = self.volume()
        values = []

        if total_voxels == 0:
            # There's no voxels in the structure
            values = np.zeros(len(bins))
        else:
            for bin in bins:
                number = (self.dose >= bin).sum()
                value = (number / total_voxels) * 100
                values.append(value)
            values = np.asarray(values)

        return bins, values


class OAR(Structure):
    """
    OAR: specialization for organs at risk
    """

    def __init__(self, name):
        super().__init__(name)
        self.type = "oar"


class Target(Structure):
    """
    TARGET: specialization for target volumes.
    """

    def __init__(self, name):
        super().__init__(name)
        self.type = "target"
