import os
import matplotlib.pyplot as plt

from dosemetrics.data_utils import read_from_eclipse
from dosemetrics.plot import from_dataframe

plt.rcParams["figure.figsize"] = [20, 12]


def plot_dvh_from_eclipse(path_to_txt_file):
    data = read_from_eclipse(path_to_txt_file)
    from_dataframe(data, "test_subject")


if __name__ == "__main__":
    repo_root = os.path.abspath("..")
    data_file = os.path.join(repo_root, "data/test_subject.txt")
    plot_dvh_from_eclipse(data_file)
