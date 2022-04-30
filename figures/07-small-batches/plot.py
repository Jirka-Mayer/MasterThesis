import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("sup-0010.png", 0) / 255
img = img[0:64,:]
img_in = img[:,0:128]
img_out = img[:,131:130*2-1]
img_ex = img[:,130*2+2:-1]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
SCALE = 1.0
fig.set_figheight(2.3 * SCALE)
fig.set_figwidth(10.0 * SCALE)
fig.set_tight_layout(True)

ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
ax1.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
ax1.set_title("Input image")
ax1.imshow(np.dstack([1 - img_in]*3))

ax2.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
ax2.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
ax2.set_title("Produced segmentation")
ax2.imshow(np.dstack([1 - img_out]*3))

ax3.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
ax3.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
ax3.set_title("Expected segmentation")
ax3.imshow(np.dstack([1 - img_ex]*3))

plt.savefig("small-batches.pdf")
plt.close()
