import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import shutil


if len(sys.argv) >= 2 and sys.argv[1] == "pull":
    print("Pulling...")
    SOURCE_DIR = "/run/user/1000/gvfs/sftp:host=aic.ufal.mff.cuni.cz,user=mayerji/home/mayerji/MastersThesis/code/experiments-data/unet"
    for name in os.listdir("data"):
        dirname = name[:-len(".csv")]
        shutil.copyfile(
            os.path.join(SOURCE_DIR, dirname, "metrics.csv"),
            os.path.join("data", name)
        )
    print("Done.")


def get_file(mode="solid"):
    for name in os.listdir("data"):
        if "skip={}".format(mode) in name:
            return name
    raise Exception("File not found")


def plot_data(mode="solid"):
    cut = slice(0, 25)
    data = np.genfromtxt(
        "data/" + get_file(mode=mode),
        delimiter=',', dtype=None, names=True
    )
    plt.plot(
        data["epoch"][cut] + 1,
        data["val_f1_score"][cut],
        label=mode,
    )
    plt.xlabel("Epoch")
    plt.ylabel("F1 score")


plot_data(mode="solid")
plot_data(mode="gated")
plot_data(mode="none")
plt.ylim([0.8, 0.98])
plt.grid(axis="y")
plt.legend(loc="lower right")
plt.title("Skip connection modes")
plt.savefig("skip.pdf")
plt.close()
