#!/usr/bin/env python3
import argparse
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2") # Report only TF errors by default

import tensorflow as tf
import tensorflow_datasets as tfds

from unet_model import UnetModel

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=128, type=int, help="Batch size.")
parser.add_argument("--scatter_count", default=16, type=int, help="Images to scatter.")
parser.add_argument("--epochs", default=20, type=int, help="Number of epochs.")
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

    # TODO: remove the take() call
    ds_train = ds_train \
        .take(4000) \
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

    model = UnetModel(seed=args.seed)

    model.compile(
        optimizer=tf.optimizers.Adam(0.001),
        loss=tf.losses.BinaryCrossentropy(from_logits=False)
    )

    ### Perform training ###

    # draw the initial random image
    model.scatter(-1, scatter_images, scatter_labels)

    model.fit(
        ds_train,
        epochs=args.epochs,
        callbacks=[
            # tf.keras.callbacks.LambdaCallback(
            #     on_epoch_end=model.generate
            # ),
            tf.keras.callbacks.LambdaCallback(
                on_epoch_end=lambda e, l: model.scatter(e, scatter_images, scatter_labels)
            )
        ]
    )


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
