import argparse
import pickle
import time
from typing import Optional
from numpy import math
import tensorflow as tf
import os
import shutil

from ..datasets.NoiseGenerator import NoiseGenerator
from .Experiment import Experiment
from ..models.DenoisingUnetModel import DenoisingUnetModel
from ..models.ModelDirectory import ModelDirectory
from ..datasets.DatasetFeeder import DatasetFeeder
from ..datasets.muscima.Muscima import Muscima
from ..datasets.SegmentationDescription import SegmentationDescription
from ..datasets.TransferDataset import TransferDataset
from ..datasets.Datasets import Datasets
from ..datasets.DatasetOptions import DatasetOptions
from ..datasets.SegmentationDescription import SegmentationDescription


class Ex_Datasets(Experiment):
    @property
    def name(self):
        return "datasets"

    def describe(self):
        return """
        Used to validate the dataset pipeline logic
        """

    def define_arguments(self, parser: argparse.ArgumentParser):
        parser.add_argument("--dataset_seed", default=0, type=int, help="Dataset-slicing random seed.")
        parser.add_argument("--dataset", default="exploration", type=str, help="Dataset name.")
        parser.add_argument("--val_pages", default=10, type=int, help="Validation page count.")
        parser.add_argument("--sup_pages", default=10, type=int, help="Supervised page count.")
        parser.add_argument("--unsup_pages", default=50, type=int, help="Unsupervised page count.")
        parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")

    def run(self, args: argparse.Namespace):
        # noise = NoiseGenerator(
        #     seed=opts.seed,
        #     max_noise_size=NOISE_SIZE,
        #     dropout_ratio=NOISE_DROPOUT,
        #     largest_tiles_only=True
        # )
        opts = DatasetOptions(
            dataset_seed=args.dataset_seed,
            tile_size_wh=(512, 256),
            validation_pages=args.val_pages,
            supervised_pages=args.sup_pages,
            unsupervised_pages=args.unsup_pages,
            batch_size=args.batch_size,
            segdesc=SegmentationDescription.NOTEHEADS, # TODO: symbol argument
            unsupervised_transformation=lambda x: tf.data.Dataset.zip((x, x)),
            verbose=True
        )
        print("OPTS:", vars(opts))
        print()
        ds_train, ds_validate, ds_test = Datasets.by_name(args.dataset, opts)

        print("Plotting...")
        import matplotlib.pyplot as plt
        for (xl, xu), (yl, yu) in ds_train.take(5).as_numpy_iterator():
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
            fig.suptitle("ds_train dataset content, batch_item 0, channel 0")
            ax1.imshow(xl[0,:,:,0]) ; ax1.set_xlabel("X_labeled")
            ax3.imshow(xu[0,:,:,0]) ; ax3.set_xlabel("X_unlabeled")
            ax2.imshow(yl[0,:,:,0]) ; ax2.set_xlabel("Y_labeled")
            ax4.imshow(yu[0,:,:,0]) ; ax4.set_xlabel("Y_unlabeled")
            plt.show()

        for x, y in ds_validate.take(5).as_numpy_iterator():
            fig, (ax1, ax2) = plt.subplots(2, 1)
            fig.suptitle("ds_validate dataset content, batch_item 0, channel 0")
            ax1.imshow(x[0,:,:,0]) ; ax1.set_ylabel("X")
            ax2.imshow(y[0,:,:,0]) ; ax2.set_ylabel("Y")
            plt.show()

        for x, y in ds_test.take(5).as_numpy_iterator():
            fig, (ax1, ax2) = plt.subplots(2, 1)
            fig.suptitle("ds_test dataset content, batch_item 0, channel 0")
            ax1.imshow(x[0,:,:,0]) ; ax1.set_ylabel("X")
            ax2.imshow(y[0,:,:,0]) ; ax2.set_ylabel("Y")
            plt.show()
