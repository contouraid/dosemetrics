import glob
from dosemetrics.dvh import read_dvh_file, plot_dvh
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = [20, 12]

for subject in range(78, 101):
    cases = glob.glob(
        "/Users/amithkamath/data/DLDP/astute-dvh/" + str(subject).zfill(3) + "/*.txt"
    )
    for file in cases:
        name = file.split("\\")[-1].split(".")[0]
        data = read_dvh_file(file)
        plot_dvh(data, name)
