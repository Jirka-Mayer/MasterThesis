import tensorflow as tf
import matplotlib.pyplot as plt
from dataset import load_independent_dataset

ds = load_independent_dataset()

for item in ds:
    img = item["image"]
    noteheads = item["noteheads"]

    img = tf.image.resize(
        img[tf.newaxis, ...],
        [32, 32],
        method="area"
    )[0, ..., 0].numpy()

    noteheads = tf.image.resize(
        noteheads[tf.newaxis, ...],
        [32, 32],
        method="area"
    )[0, ..., 0].numpy()

    plt.imshow(tf.concat([img, noteheads], axis=0))
    plt.show()
