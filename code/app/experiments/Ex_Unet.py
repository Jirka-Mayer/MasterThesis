import argparse
import time
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
from ..datasets.DatasetOptions import DatasetOptions
from ..datasets.Datasets import Datasets


TILE_SIZE_WH = (512, 256)


class Options:
    def __init__(self, **kwargs):
        self.seed = int(kwargs["seed"])
        self.dataset_seed = int(kwargs["dataset_seed"])
        self.epochs = int(kwargs["epochs"])
        self.symbol = str(kwargs["symbol"])
        self.unsupervised_loss_weight = float(kwargs["unsupervised_loss_weight"])
        self.batch_size = int(kwargs["batch_size"])
        self.val_pages = int(kwargs["val_pages"])
        self.sup_pages = int(kwargs["sup_pages"])
        self.unsup_pages = int(kwargs["unsup_pages"])
        self.noise_size_ss = float(kwargs["noise_size_ss"])
        self.noise_dropout = float(kwargs["noise_dropout"])
        self.inner_features = int(kwargs["inner_features"])
        self.dropout = float(kwargs["dropout"])
        self.skip_connection = str(kwargs["skip_connection"])
        self.dataset = str(kwargs["dataset"])

    def dataset_options(self):
        noise = NoiseGenerator(
            seed=self.dataset_seed,
            max_noise_size=int(Muscima.DPSS * self.noise_size_ss),
            dropout_ratio=self.noise_dropout,
            largest_tiles_only=True
        )
        return DatasetOptions(
            dataset_seed=self.dataset_seed,
            tile_size_wh=TILE_SIZE_WH,
            validation_pages=self.val_pages,
            supervised_pages=self.sup_pages,
            unsupervised_pages=self.unsup_pages,
            batch_size=self.batch_size,
            segdesc=SegmentationDescription.from_name(self.symbol),
            unsupervised_transformation=noise.dataset_transformation
        )

    def model_kwargs(self):
        return {
            "unsup_loss_weight": self.unsupervised_loss_weight,
            "inner_features": self.inner_features,
            "dropout": self.dropout,
            "skip_connection": self.skip_connection
        }


class Ex_Unet(Experiment):
    @property
    def name(self):
        return "unet"

    def describe(self):
        return """
        U-Net model training and evaluation
        """

    def define_arguments(self, parser: argparse.ArgumentParser):
        parser.add_argument(
            "command", type=str,
            help="One of: train, evaluate"
        )
        parser.add_argument("--seed", default=42, type=int, help="Random seed.")
        parser.add_argument("--symbol", default="notehead", type=str, help="Symbol to train on.")
        parser.add_argument("--epochs", default=200, type=int, help="Number of training epochs.")
        parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
        
        parser.add_argument("--dataset_seed", default=42, type=int, help="Dataset-slicing random seed.")
        parser.add_argument("--dataset", default="exploration", type=str, help="Dataset name.")
        parser.add_argument("--val_pages", default=10, type=int, help="Validation page count.")
        parser.add_argument("--sup_pages", default=10, type=int, help="Supervised page count.")
        parser.add_argument("--unsup_pages", default=50, type=int, help="Unsupervised page count.")
        
        parser.add_argument("--noise_size_ss", default=2, type=float, help="Noise size in staff space multiple.")
        parser.add_argument("--noise_dropout", default=0.25, type=float, help="Noise dropout percentage.")
        
        parser.add_argument("--unsupervised_loss_weight", default=1.0, type=float, help="Unsup loss weight.")
        parser.add_argument("--inner_features", default=8, type=int, help="Model capacity.")
        parser.add_argument("--dropout", default=0.5, type=float, help="Training dropout.")
        parser.add_argument("--skip_connection", default="gated", type=str, help="Type of skip connection (none, gated, solid).")

    def run(self, args: argparse.Namespace):
        if args.command == "train":
            self.train(args)
        elif args.command == "evaluate":
            self.evaluate(args)
    
    def train(self, args: argparse.Namespace):
        self.compute_single_instance(Options(**vars(args)))

    def evaluate(self, args: argparse.Namespace):
        self.evaluate_single_instance(Options(**vars(args)))

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

        cache_dir = self.experiment_directory(
            "cache-{}-{}".format(str(os.getpid()), str(time.time()))
        )

        with DatasetFeeder(cache_dir) as feeder:
            ds_train, ds_validate, _ = Datasets.by_name(
                opts.dataset,
                opts.dataset_options()
            )

            ds_train = feeder(ds_train)
            ds_validate = feeder(ds_validate)

            model = DenoisingUnetModel.load_or_create(
                model_directory,
                **opts.model_kwargs()
            )
            model.perform_training(
                epochs=opts.epochs,
                ds_train=ds_train,
                ds_validate=ds_validate,
                save_checkpoints=True,
            )

        self.evaluate_single_instance(opts)

    def evaluate_single_instance(self, opts: Options):
        # _, _, ds_test = Datasets.by_name(
        #     opts.dataset,
        #     opts.dataset_options()
        # )
        print("TODO: implement evaluation step")
        # model -> load the best model checkpoint
        #model.perform_evaluation(ds_test)

    def build_model_name(self, opts: Options) -> str:
        # outputs: "experiment-name__foo=42_bar=baz"
        take_vars = [
            "dataset",
            "dataset_seed",
            "sup_pages",
            "unsup_pages",
            "seed",
            "symbol",
            "epochs",
            "batch_size",
            "val_pages",
            "noise_size_ss",
            "noise_dropout",
            "unsupervised_loss_weight",
            "inner_features",
            "dropout",
            "skip_connection",
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
