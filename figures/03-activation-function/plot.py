import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import cv2

#########################################
# Training process output segmentations #
#########################################

img1 = cv2.imread("sup-0000.png", 0)
img2 = cv2.imread("sup-0001.png", 0)
img3 = cv2.imread("sup-0002.png", 0)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
SCALE = 0.7
fig.set_figheight(2.2 * SCALE)
fig.set_figwidth(10.0 * SCALE)
fig.set_tight_layout(True)

ax1.imshow(np.dstack([img1, img1, img1]))
ax1.axis("off")
ax1.set_title("Untrained")

ax2.imshow(np.dstack([img2, img2, img2]))
ax2.axis("off")
ax2.set_title("First epoch")

ax3.imshow(np.dstack([img3, img3, img3]))
ax3.axis("off")
ax3.set_title("Second epoch")

plt.savefig("progression.pdf")
plt.close()


####################################
# Activation function performances #
####################################

data = np.genfromtxt("data.csv", delimiter=',', dtype=None, names=True)

plt.plot(data["epoch"], data["relu"], label="ReLU")
plt.plot(data["epoch"], data["elu"], label="ELU")
plt.plot(data["epoch"], data["leaky"], label="Leaky ReLU, 0.01")
plt.legend(loc="lower right")
plt.title("Performance of various activation functions")
plt.xlabel("Epoch")
plt.ylabel("F1 score")
#plt.show()
plt.savefig("performance.pdf")
plt.close()

##################################
# Activation function comparison #
##################################

x = np.linspace(-10, 10, 500)
y_relu = tf.keras.activations.relu(x).numpy()
y_elu = tf.keras.activations.elu(x).numpy()
y_leaky = tf.keras.activations.relu(x, alpha=0.1).numpy()

fig, (ax_relu, ax_elu, ax_leaky) = plt.subplots(1, 3)
SCALE = 0.7
fig.set_figheight(3.5 * SCALE)
fig.set_figwidth(12.0 * SCALE)
fig.set_tight_layout(True)

ax_relu.plot(x, y_relu)
ax_relu.set_title("ReLU")
ax_relu.set_xlim([-10, 10])
ax_relu.set_ylim([-10, 10])
ax_relu.set_yticks(np.linspace(-10, 10, 5))
ax_relu.set_xticks(np.linspace(-10, 10, 5))
ax_relu.grid()

ax_elu.plot(x, y_elu)
ax_elu.set_title("ELU")
ax_elu.set_xlim([-10, 10])
ax_elu.set_ylim([-10, 10])
ax_elu.set_yticks(np.linspace(-10, 10, 5))
ax_elu.set_xticks(np.linspace(-10, 10, 5))
ax_elu.set_yticklabels([])
ax_elu.grid()

ax_leaky.plot(x, y_leaky)
ax_leaky.set_title("Leaky ReLU")
ax_leaky.set_xlim([-10, 10])
ax_leaky.set_ylim([-10, 10])
ax_leaky.set_yticks(np.linspace(-10, 10, 5))
ax_leaky.set_xticks(np.linspace(-10, 10, 5))
ax_leaky.set_yticklabels([])
ax_leaky.grid()

plt.savefig("functions.pdf")
#plt.show()
plt.close()
