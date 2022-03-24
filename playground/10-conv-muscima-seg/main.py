#!/usr/bin/env python3
import argparse
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2") # Report only TF errors by default

import tensorflow as tf
import tensorflow_datasets as tfds

from model import Model
from dataset import load_independent_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=128, type=int, help="Batch size.")
parser.add_argument("--scatter_count", default=16, type=int, help="Images to scatter.")
parser.add_argument("--epochs", default=1000, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")

def main(args):
    tf.random.set_seed(args.seed)

    os.makedirs("fig", exist_ok=True)

    ### Prepare the dataset ###

    ds_train = load_independent_dataset(train_split=True)
    ds_test = load_independent_dataset(train_split=False)

    def tuplify(item):
        # turns the dict into a value-label pair
        return item["image"], item["noteheads"]

    def downscale(images, noteheads):
        images = tf.image.resize(images, [32, 32], method="area")
        noteheads = tf.image.resize(noteheads, [32, 32], method="area")
        return images, noteheads

    ds_train = ds_train \
        .map(tuplify, num_parallel_calls=tf.data.AUTOTUNE) \
        .map(downscale, num_parallel_calls=tf.data.AUTOTUNE) \
        .batch(args.batch_size) \
        .cache() \
        .prefetch(tf.data.AUTOTUNE)

    ds_test = ds_test \
        .map(tuplify, num_parallel_calls=tf.data.AUTOTUNE) \
        .map(downscale, num_parallel_calls=tf.data.AUTOTUNE) \
        .batch(args.scatter_count)

    for batch in ds_test:
        scatter_images, scatter_labels = batch
        break

    ### Create the model ###

    model = Model(seed=args.seed)

    model.compile(
        optimizer=tf.optimizers.Adam(0.001),
        loss=tf.losses.BinaryCrossentropy(from_logits=False)
    )

    ### Perform training ###

    # draw the initial random image
    model.visualize_reconstruction(-1, scatter_images, scatter_labels)

    model.fit(
        ds_train,
        epochs=args.epochs,
        callbacks=[
            # tf.keras.callbacks.LambdaCallback(
            #     on_epoch_end=model.generate
            # ),
            tf.keras.callbacks.LambdaCallback(
                on_epoch_end=lambda e, l: model.visualize_reconstruction(
                    e, scatter_images, scatter_labels
                )
            )
        ]
    )


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)

