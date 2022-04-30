import cv2
import matplotlib.pyplot as plt
import numpy as np


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
noise_names = ["Small", "Ideal", "Large"]

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
