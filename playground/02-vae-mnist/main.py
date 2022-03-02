#!/usr/bin/env python3
import argparse
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2") # Report only TF errors by default

import tensorflow as tf
import tensorflow_datasets as tfds

from vae_model import VaeModel

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=128, type=int, help="Batch size.")
parser.add_argument("--scatter_count", default=1024, type=int, help="Images to scatter.")
parser.add_argument("--z_dim", default=2, type=int, help="Dimension of Z.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")

def main(args):
    tf.random.set_seed(args.seed)

    os.makedirs("fig", exist_ok=True)

    ### Prepare the dataset ###

    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    def normalize_img(image, label):
        return tf.cast(image, tf.float32) / 255.0, label

    ds_train = ds_train \
        .map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE) \
        .cache() \
        .shuffle(ds_info.splits["train"].num_examples) \
        .batch(args.batch_size) \
        .prefetch(tf.data.AUTOTUNE)

    ds_test = ds_test \
        .map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE) \
        .batch(args.scatter_count) \
        .cache() \
        .prefetch(tf.data.AUTOTUNE)

    for batch in ds_test:
        scatter_images, scatter_labels = batch
        break

    ### Create the model ###

    model = VaeModel(
        seed=args.seed,
        z_dim=args.z_dim,
        image_shape=ds_info.features["image"].shape
    )

    model.compile(
        optimizer=tf.optimizers.Adam(0.001),
        loss=tf.losses.BinaryCrossentropy(from_logits=False)
    )

    ### Perform training ###

    model.fit(
        ds_train,
        epochs=args.epochs,
        callbacks=[
            tf.keras.callbacks.LambdaCallback(
                on_epoch_end=model.generate
            ),
            tf.keras.callbacks.LambdaCallback(
                on_epoch_end=lambda e, l: model.scatter(e, scatter_images, scatter_labels)
            )
        ]
    )


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
