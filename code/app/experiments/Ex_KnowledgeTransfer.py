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


TILE_SIZE_WH = (512, 256)
BATCH_SIZE = 10
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10
NOISE_DROPOUT = 0.25
NOISE_SIZE = int(Muscima.DPSS * 2)


class Options:
    def __init__(self, **kwargs):
        self.seed = int(kwargs["seed"])
        self.dataset_seed = int(kwargs["dataset_seed"])
        self.symbol = str(kwargs["symbol"])
        self.unsupervised_loss_weight = float(kwargs["unsupervised_loss_weight"])

        self.val_pages = int(kwargs["val_pages"])
        self.sup_pages = int(kwargs["sup_pages"])
        self.unsup_pages = int(kwargs["unsup_pages"])


class Ex_KnowledgeTransfer(Experiment):
    @property
    def name(self):
        return "knowledge-transfer"

    def describe(self):
        return """
        TODO
        """

    def define_arguments(self, parser: argparse.ArgumentParser):
        parser.add_argument(
            "command", type=str,
            help="One of: train"
        )
        parser.add_argument("--seed", default=42, type=int, help="Random seed.")
        parser.add_argument("--symbol", default="noteheads", type=str, help="Symbol to train on.")
        parser.add_argument("--epochs", default=MAX_EPOCHS, type=int, help="Overwrites max epochs.")
        parser.add_argument("--unsupervised_loss_weight", default=1.0, type=float, help="Unsup loss weight.")

        parser.add_argument("--dataset_seed", default=42, type=int, help="Dataset-slicing random seed.")
        parser.add_argument("--val_pages", default=10, type=int, help="Validation page count.")
        parser.add_argument("--sup_pages", default=10, type=int, help="Supervised page count.")
        parser.add_argument("--unsup_pages", default=50, type=int, help="Unsupervised page count.")

        # TODO: add batch size

    def run(self, args: argparse.Namespace):
        global MAX_EPOCHS
        MAX_EPOCHS = args.epochs

        if args.command == "train":
            self.train(args)
    
    def train(self, args: argparse.Namespace):
        self.compute_single_instance(Options(
            seed=args.seed,
            symbol=args.symbol,
            unsupervised_loss_weight=args.unsupervised_loss_weight,
            dataset_seed=args.dataset_seed,
            val_pages=args.val_pages,
            sup_pages=args.sup_pages,
            unsup_pages=args.unsup_pages,
        ))

    def _symbol_name_to_segdesc(self, symbol: str) -> SegmentationDescription:
        # TODO: move this symbol naming into the segmentation description class
        if symbol == "noteheads":
            return SegmentationDescription.NOTEHEADS
        if symbol == "staffline":
            return SegmentationDescription.STAFFLINE
        if symbol == "stem":
            return SegmentationDescription.STEM
        if symbol == "beam":
            return SegmentationDescription.BEAM
        if symbol == "flags":
            return SegmentationDescription.FLAGS
        raise Exception("Unknown symbol name: " + symbol)

    def compute_single_instance(self, opts: Options) -> float:
        tf.random.set_seed(opts.seed)

        model_name = self.build_model_name(opts)
        model_directory = self.experiment_directory(model_name)

        print("#" * (len(model_name) + 4)) # print header
        print("# " + model_name + " #")
        print("#" * (len(model_name) + 4))

        # clear the model directory before running the experiment instance
        shutil.rmtree(model_directory, ignore_errors=True)
        os.makedirs(model_directory, exist_ok=True)

        noise = NoiseGenerator(
            seed=opts.seed,
            max_noise_size=NOISE_SIZE,
            dropout_ratio=NOISE_DROPOUT,
            largest_tiles_only=True
        )

        cache_dir = self.experiment_directory(
            "cache-{}-{}".format(str(os.getpid()), str(time.time()))
        )

        with DatasetFeeder(cache_dir) as feeder:
            # ds_train, ds_validate, ds_test = \
            #     TransferDataset.deepscores_to_muscima(
            #         dataset_seed=opts.dataset_seed,
            #         tile_size_wh=TILE_SIZE_WH,
            #         validation_pages=opts.val_pages,
            #         supervised_pages=opts.sup_pages,
            #         unsupervised_pages=opts.unsup_pages,
            #         batch_size=BATCH_SIZE,
            #         segdesc=self._symbol_name_to_segdesc(opts.symbol),
            #         unsupervised_transformation=noise.dataset_transformation,
            #         #output_scale_factor=0.25 # trying out, what it does to reconstruction performance
            #     )

            ds_train, ds_validate, ds_test = \
                Muscima.semisupervised_experiment_datasets(
                    dataset_seed=opts.dataset_seed,
                    validation_ratio=0.1,
                    sup_ratio=opts.sup_pages / 100,
                    unsup_ratio=opts.unsup_pages / 100,
                    batch_size=BATCH_SIZE,
                    segdesc=self._symbol_name_to_segdesc(opts.symbol),
                    tile_size_wh=TILE_SIZE_WH,
                    unsupervised_transformation=noise.dataset_transformation,
                    #output_scale_factor=0.25
                )

            ds_train = feeder(ds_train)
            ds_validate = feeder(ds_validate)
            ds_test = feeder(ds_test)

            model = DenoisingUnetModel.load_or_create(
                model_directory,
                unsup_loss_weight=opts.unsupervised_loss_weight,
                inner_features=8,
                dropout=0.5,
                skip_connection="gated"
            )
            model.perform_training(
                epochs=MAX_EPOCHS,
                ds_train=ds_train,
                ds_validate=ds_validate,
                save_checkpoints=False,
                #early_stop_after=EARLY_STOPPING_PATIENCE # TODO: debug disabled
            )
            model.perform_evaluation(ds_test)

    def build_model_name(self, opts: Options) -> str:
        # outputs: "experiment-name__foo=42_bar=baz"
        take_vars = [
            "dataset_seed", "seed", "val_pages", "sup_pages", "unsup_pages",
            "symbol", "unsupervised_loss_weight"
        ]
        opt_vars = vars(opts)
        vars_list = [v + "=" + str(opt_vars[v]) for v in take_vars]
        return "{}__{}".format(self.name, ",".join(vars_list))

    def parse_model_name(self, model_name: str) -> Options:
        items = model_name.split("__")[1].split(",")
        kwargs = {
            item.split("=")[0]: item.split("=")[1] for item in items
        }
        return Options(**kwargs)
