from dataset import load_independent_dataset
import matplotlib.pyplot as plt

ds = load_independent_dataset()

for f in ds.take(5):
    plt.imshow(f["image"])
    plt.show()
