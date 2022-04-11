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


VALIDATION_PAGES = 10
SUPERVISED_PAGES = 60
UNSUPERVISED_PAGES = 120
UNSUP_LOSS_WEIGHT = 0.1
SEGDESC = SegmentationDescription.NOTEHEADS
TILE_SIZE_WH = (512, 256)
BATCH_SIZE = 16
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10
NOISE_DROPOUT = 0.25
NOISE_SIZE = int(Muscima.DPSS * 2)


class Options:
    def __init__(self, **kwargs):
        self.seed = int(kwargs["seed"])


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

    def run(self, args: argparse.Namespace):
        if args.command == "train":
            self.train(args)
    
    def train(self, args: argparse.Namespace):
        self.compute_single_instance(Options(
            seed=args.seed
        ))

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
            dropout_ratio=NOISE_DROPOUT
        )

        cache_dir = self.experiment_directory(
            "cache-{}-{}".format(str(os.getpid()), str(time.time()))
        )

        with DatasetFeeder(cache_dir) as feeder:
            ds_train, ds_validate, ds_test = \
                TransferDataset.deepscores_to_muscima(
                    seed=opts.seed,
                    tile_size_wh=TILE_SIZE_WH,
                    validation_pages=VALIDATION_PAGES,
                    supervised_pages=SUPERVISED_PAGES,
                    unsupervised_pages=UNSUPERVISED_PAGES,
                    batch_size=BATCH_SIZE,
                    segdesc=SEGDESC,
                    unsupervised_transformation=noise.dataset_transformation
                )

            ds_train = feeder(ds_train)
            ds_validate = feeder(ds_validate)
            ds_test = feeder(ds_test)

            model = DenoisingUnetModel.load_or_create(
                model_directory,
                unsup_loss_weight=UNSUP_LOSS_WEIGHT
            )
            model.perform_training(
                epochs=MAX_EPOCHS,
                ds_train=ds_train,
                ds_validate=ds_validate,
                save_checkpoints=False,
                early_stop_after=EARLY_STOPPING_PATIENCE
            )
            model.perform_evaluation(ds_test)

    def build_model_name(self, opts: Options) -> str:
        # outputs: "experiment-name__foo=42_bar=baz"
        take_vars = [
            "seed" # ???
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
