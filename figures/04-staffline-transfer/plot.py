import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import shutil
import cv2


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


def get_file(unsup=0, seed=0):
    for name in os.listdir("data"):
        if "unsup={}".format(unsup) in name:
            if "suprep=1,seed={}".format(seed) in name:
                return name
    raise Exception("File not found")


def plot_data(unsup=0, label="supervised", color="r"):
    epochs = 20
    sums = np.zeros(shape=(epochs,), dtype=np.float32)
    for seed in [0, 1, 2, 3]:
        data = np.genfromtxt(
            "data/" + get_file(unsup=unsup, seed=seed),
            delimiter=',', dtype=None, names=True
        )
        sums += data["val_f1_score"][0:epochs]
        plt.plot(
            data["epoch"] + 1,
            data["val_f1_score"],
            label=label if seed == 0 else None,
            color=color,
            linewidth=1,
            alpha=0.3
        )
    plt.plot(
        range(1, epochs + 1),
        sums / 4,
        label=label + " avg.",
        color=color,
        linewidth=3,
        alpha=1
    )
    plt.xlabel("Epoch")
    plt.ylabel("F1 score")


plot_data(unsup=0, label="supervised", color="b")
plot_data(unsup=99, label="semi-supervised", color="r")
plt.legend(loc="lower left")
plt.ylim([0.0, 1.0])
plt.grid(axis="y")
plt.title("Staffline segmentation transfer")
plt.savefig("transfer.pdf")
plt.close()


#####################
# Staffline removal #
#####################

raw_img = cv2.imread("eval-0013.png", 0)
raw_img = raw_img / 255
total_h, total_w = raw_img.shape
w = (total_w - 2) // 3
input_img = raw_img[:,0:w]
segmentation_img = raw_img[:,w+1:w+1+w]
segmentation_img = (segmentation_img > 0.5).astype(np.float32) # binarize
removed_img = input_img * (1 - segmentation_img)

cv2.imwrite("staffline-removal.png", (1 - removed_img[700:1400,1400:2400]) * 255)

# plt.imshow(np.dstack([1 - removed_img]*3))
# plt.xlim(left=1400, right=2400)
# plt.ylim(bottom=1450, top=750)
# plt.axis("off")
# #plt.title("Removed stafflines by knowledge transfer")
# plt.tight_layout()
# plt.savefig("staffline-removal.pdf")
# plt.close()
