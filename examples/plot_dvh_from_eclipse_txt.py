import os
import matplotlib.pyplot as plt

import dosemetrics

plt.rcParams["figure.figsize"] = [20, 12]


def plot_dvh_from_eclipse(path_to_txt_file: str, output_path: str = ".") -> None:
    data = dosemetrics.read_from_eclipse(path_to_txt_file)
    dosemetrics.from_dataframe(data, "test_subject", output_path)


if __name__ == "__main__":
    repo_root = os.path.abspath("..")
    data_file = os.path.join(repo_root, "data/test_subject.txt")

    results_folder = os.path.join(repo_root, "results")
    os.makedirs(results_folder, exist_ok=True)
    results_file = os.path.join(results_folder, "test_subject.png")
    plot_dvh_from_eclipse(data_file, results_file)
