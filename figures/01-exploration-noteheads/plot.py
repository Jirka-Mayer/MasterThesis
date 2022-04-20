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


def smooth(scalars, weight=0.5):
    last = scalars[0]
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def plot_data(sup=10, e=100, dr=0.0, ft=8, smoothing_strength=0.0):
    for unsup in [0, 5, 10, 50]:
        data = np.genfromtxt(
            "data/unet__ds=exploration,ds_seed=0,sup={},unsup={},suprep=1,seed=0,sym=notehead,e={},bs=10,val=10,ns_sz=2.0,ns_dr=0.25,l=1.0,ft={},dr={},skip=gated.csv" \
                .format(sup, unsup, e, ft, dr),
            delimiter=',', dtype=None, names=True
        )
        plt.plot(
            data["epoch"] + 1,
            smooth(data["val_f1_score"], smoothing_strength),
            label="{}:{}".format(sup, unsup)
        )
    plt.xlabel("Epoch")
    plt.ylabel("F1 score")


# plot without dropout to show how noisy the training is

plot_data(e=100, dr=0.0)
plt.legend(loc="lower right")
plt.ylim([0.5, 0.9])
plt.title("Notehead segmentation without dropout")
plt.savefig("noteheads.pdf")
plt.close()

# plot with dropout

plot_data(e=200, dr=0.5)
plt.legend(loc="lower right")
plt.ylim([0.75, 0.925])
plt.title("Notehead segmentation")
plt.savefig("noteheads-dropout.pdf")
plt.close()

plot_data(e=200, dr=0.5, smoothing_strength=0.8)
plt.legend(loc="lower right")
plt.ylim([0.75, 0.925])
plt.title("Notehead segmentation, smoothed")
plt.savefig("noteheads-dropout-smooth.pdf")
plt.close()
