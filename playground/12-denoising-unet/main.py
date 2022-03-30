#!/usr/bin/env python3
import argparse
import os
import shutil
from cv2 import threshold
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2") # Report only TF errors by default

import tensorflow as tf

from model import Model
from dataset import prepare_datasets
from metrics import F1Score

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=2, type=int, help="Batch size.")
parser.add_argument("--visualize_count", default=8, type=int, help="Images to visualize.")
parser.add_argument("--validation_split", default=0.1, type=float, help="Ratio of the dataset to use for validation.")
parser.add_argument("--no_cache_busting", default=False, action="store_true", help="Deletes the dataset cache on startup.")
parser.add_argument("--retrain", default=False, action="store_true", help="Ignore the saved model.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")

PARENT_DIR = os.path.basename(os.path.dirname(os.path.realpath(__file__)))
CACHE_PATH = os.path.expanduser("~/Datasets/Cache/" + PARENT_DIR)

def main(args):
    tf.random.set_seed(args.seed)

    if not args.no_cache_busting:
        shutil.rmtree(CACHE_PATH, ignore_errors=True)
        print("Cache busted.")

    if args.retrain:
        shutil.rmtree("checkpoints", ignore_errors=True)
        print("Deleted the saved model because '--retrain' ing")
        print("To train now, re-run the job without the flag.")
        exit(0)

    os.makedirs("fig", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    ### Prepare the dataset ###

    ds_train, ds_validate, ds_test = prepare_datasets(
        seed=args.seed,
        validation_split=args.validation_split,
        scale_factor=0.25 # TODO: testing out on smaller data
    )

    visualization_batch = ds_validate \
        .batch(args.visualize_count) \
        .take(1) \
        .get_single_element()

    ds_train = ds_train \
        .batch(args.batch_size) \
        .take(200) \
        .snapshot(path=os.path.join(CACHE_PATH, "train"), compression=None) \
        .prefetch(tf.data.AUTOTUNE)

    ds_validate = ds_validate \
        .batch(args.batch_size) \
        .snapshot(path=os.path.join(CACHE_PATH, "validate"), compression=None) \
        .prefetch(tf.data.AUTOTUNE)

    ds_test = ds_test \
        .batch(args.batch_size) \
        .snapshot(path=os.path.join(CACHE_PATH, "test"), compression=None) \
        .prefetch(tf.data.AUTOTUNE)

    ### Load model ###

    model = None
    start_from_epoch = 0
    for c in (list(sorted(os.listdir("checkpoints")))[-1:]):
        model = tf.keras.models.load_model(
            os.path.join("checkpoints", c),
            custom_objects={"Model": Model}
        )
        start_from_epoch = int(c[-4:])
    
    ### Create the model ###

    if model is None:
        model = Model(seed=args.seed)

    # (re)compile even if loaded
    model.compile(
        optimizer=tf.optimizers.Adam(0.001),
        loss=tf.losses.BinaryCrossentropy(from_logits=False),
        metrics=[F1Score()]
    )

    ### Perform training ###

    if start_from_epoch == 0:
        model.visualize(-1, visualization_batch)
    
    model.fit(
        ds_train,
        epochs=args.epochs,
        initial_epoch=start_from_epoch,
        validation_data=ds_validate,
        callbacks=[
            tf.keras.callbacks.LambdaCallback(
                on_epoch_end=lambda e, l: model.visualize(
                    e, visualization_batch
                )
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath="checkpoints/model_{epoch:04d}",
                monitor="val_loss",
                verbose=1
            ),
            tf.keras.callbacks.CSVLogger(
                "checkpoints/log.csv", separator=',', append=True
            )
        ]
    )

    ### Evaluate ###

    print("Evaluating on the test set:")
    results = model.evaluate(ds_test, return_dict=True)
    print(results)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
