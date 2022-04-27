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


def get_file(unsup=0, ft=1):
    for name in os.listdir("data"):
        if "unsup={}".format(unsup) in name:
            if "ft={}".format(ft) in name:
                return name
    raise Exception("File not found")


def plot_data(unsup=0):
    for ft in [1, 2, 4, 8, 16, 32]:
        data = np.genfromtxt(
            "data/" + get_file(unsup=unsup, ft=ft),
            delimiter=',', dtype=None, names=True
        )
        plt.plot(
            data["epoch"] + 1,
            data["val_f1_score"],
            label="{}ft".format(ft)
        )
    plt.xlabel("Epoch")
    plt.ylabel("F1 score")


plot_data(unsup=0)
plt.legend(loc="lower right")
plt.ylim([0.8, 0.97])
plt.grid(axis="y")
plt.title("Fully-supervised")
plt.savefig("supervised.pdf")
plt.close()


plot_data(unsup=551)
plt.legend(loc="lower right")
plt.ylim([0.8, 0.97])
plt.grid(axis="y")
plt.title("Semi-supervised")
plt.savefig("semisupervised.pdf")
plt.close()
