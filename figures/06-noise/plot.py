import cv2
import matplotlib.pyplot as plt
import numpy as np


################################
# Reconstruction visualization #
################################

img = cv2.imread("large-noise-solid.png", 0) / 255
picked_rows = [0, 2, 5, 8]

fig, axes = plt.subplots(len(picked_rows), 3)
SCALE = 1.2
fig.set_figheight(5.0 * SCALE)
fig.set_figwidth(7.0 * SCALE)
fig.set_tight_layout(True)

def dstack(img):
    return np.dstack([1 - img] * 3)

for i, row in enumerate(picked_rows):
    h = 256 + 1
    for j, s, t in [
        (0, slice(0, 514-2), "Masked input"),
        (1, slice(514, 514*2-2), "Reconstruction"),
        (2, slice(514*2, 514*3-2), "Ground truth")
    ]:
        ax = axes[i][j]
        if i == 0:
            ax.set_title(t)
        ax.imshow(dstack(img[h*row:h*(row+1),s]))
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

plt.savefig("reconstructions.pdf")
plt.close()


#########################
# Noise size comparison #
#########################

#from ...code.app.datasets.NoiseGenerator import NoiseGenerator
import importlib.util
spec = importlib.util.spec_from_file_location(
    "app.datasets",
    "../../code/app/datasets/NoiseGenerator.py"
)
foo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(foo)
NoiseGenerator = foo.NoiseGenerator


gen = NoiseGenerator(42, 28.75*2, 0.25, True)

img = cv2.imread("large-noise-solid.png", 0)[257*2:257*3-1,513*2+4:-2] / 255

small_noise, _ = NoiseGenerator(42, 28.75*0.25, 0.25, True)._create_random_noise_mask(*img.shape)
standard_noise, _ = NoiseGenerator(42, 28.75*2, 0.25, True)._create_random_noise_mask(*img.shape)
big_noise, _ = NoiseGenerator(43, 28.75*6, 0.25, True)._create_random_noise_mask(*img.shape)

noises = [small_noise, standard_noise, big_noise]
noise_names = ["Small", "Medium", "Large"]

fig, axes = plt.subplots(2, 3)
SCALE = 1.0
fig.set_figheight(4.0 * SCALE)
fig.set_figwidth(10.0 * SCALE)
fig.set_tight_layout(True)

for i, ax in enumerate(axes[0]):
    # ax.axis("off")
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    ax.set_title(noise_names[i])
    ax.imshow(np.dstack([noises[i]] * 3))

for i, ax in enumerate(axes[1]):
    # ax.axis("off")
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    ax.imshow(np.dstack([1 - img * noises[i]] * 3))

#plt.axes("off")
#plt.show()

plt.savefig("noise-comparison.pdf")
plt.close()
