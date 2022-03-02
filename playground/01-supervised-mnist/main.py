#!/usr/bin/env python3
import argparse
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2") # Report only TF errors by default

import tensorflow as tf
import tensorflow_datasets as tfds

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=128, type=int, help="Batch size.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")

def main(args):
    tf.random.set_seed(args.seed)

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
        .batch(args.batch_size) \
        .cache() \
        .prefetch(tf.data.AUTOTUNE)

    ### Create the model ###

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(ds_info.features["image"].shape))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(
        ds_info.features["label"].num_classes,
        activation=tf.nn.softmax
    ))
    model.summary()

    model.compile(
        optimizer=tf.optimizers.Adam(0.001),
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=[tf.metrics.SparseCategoricalAccuracy("accuracy")],
    )

    ### Perform training ###

    model.fit(
        ds_train,
        epochs=args.epochs,
        validation_data=ds_test
    )


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
