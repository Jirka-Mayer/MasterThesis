import argparse
import pickle
from typing import Optional
import tensorflow as tf
import os
import shutil

from ..datasets.NoiseGenerator import NoiseGenerator
from .Experiment import Experiment
from ..models.DenoisingUnetModel import DenoisingUnetModel
from ..models.ModelDirectory import ModelDirectory
from ..datasets.DatasetFeeder import DatasetFeeder
from ..datasets.muscima.Muscima import Muscima
from ..datasets.muscima.SegmentationDescription import SegmentationDescription


VALIDATION_RATIO = 0.1
SEGDESC = SegmentationDescription.NOTEHEADS
TILE_SIZE_WH = (512, 256)
BATCH_SIZE = 16
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10


class Options:
    def __init__(self, force=False, dry_run=False, **kwargs):
        self.seed = int(kwargs["seed"])
        self.sup_ratio = float(kwargs["sup_ratio"])
        self.unsup_ratio = float(kwargs["unsup_ratio"])
        self.unsup_loss_weight = float(kwargs["unsup_loss_weight"])
        self.noise_dropout = float(kwargs["noise_dropout"])
        self.max_noise_size = int(kwargs["max_noise_size"])
        self.force = force
        self.dry_run = dry_run


class Ex_HyperparamSearch(Experiment):
    @property
    def name(self):
        return "hyperparam-search"

    def describe(self):
        return """
        TODO
        """

    def define_arguments(self, parser: argparse.ArgumentParser):
        parser.add_argument(
            "command", type=str,
            help="One of: search-supervision, search-noise, gather, plot"
        )
        parser.add_argument("--seed", default=42, type=int, help="Random seed.")
        parser.add_argument("--force", default=False, action="store_true",
            help="Forces search for already computed data points.")
        parser.add_argument("--dry_run", default=False, action="store_true",
            help="Trains on only one batch for debugging purpouses.")

    def run(self, args: argparse.Namespace):
        if args.command == "search-supervision":
            self.search_supervision(args)
        elif args.command == "search-noise":
            self.search_noise(args)
        elif args.command == "gather":
            self.gather(args)
        elif args.command == "plot":
            self.plot(args)
    
    def search_supervision(self, args: argparse.Namespace):
        for unsup_ratio in [0.0, 0.05, 0.1, 0.3, 0.5]:
            for unsup_loss_weight in [8, 4, 1, 0.25, 0.125]:
                self.compute_single_instance(Options(
                    seed=args.seed,
                    sup_ratio=0.1,
                    unsup_ratio=unsup_ratio,
                    noise_dropout=0.25,
                    max_noise_size=int(Muscima.DPSS * 2),
                    force=args.force,
                    dry_run=args.dry_run,
                    unsup_loss_weight=unsup_loss_weight
                ))

    def search_noise(self, args: argparse.Namespace):
        print("Implement training...")

    def gather(self, args: argparse.Namespace):
        """
        Gathers data from all completed experiment runs into a single
        csv file that can then be moved between machines more easily
        """
        data = {}
        ls = os.listdir(self.experiment_directory())
        for model_name in os.listdir(self.experiment_directory()):
            if not model_name.startswith(self.name + "__"):
                continue
            data[model_name] = ModelDirectory(self.experiment_directory(model_name)).load_metrics_csv()
            print(model_name)
        with open(self.experiment_directory("gathered.pkl"), "wb") as f:
            pickle.dump(data, f)

    def plot(self, args: argparse.Namespace):
        with open(self.experiment_directory("gathered.pkl"), "rb") as f:
            data = pickle.load(f)
        print(data)
        print("Implement plotting...")

    def compute_single_instance(self, opts: Options) -> float:
        tf.random.set_seed(opts.seed)

        model_name = self.build_model_name(opts)
        model_directory = self.experiment_directory(model_name)
        
        print("#" * (len(model_name) + 4)) # print header
        print("# " + model_name + " #")
        print("#" * (len(model_name) + 4))

        if opts.force: # clear the model directory by force
            shutil.rmtree(model_directory, ignore_errors=True)

        if self.read_done_file(model_directory) is not None: # skip computation
            print("Skipping, already computed.")
            return

        # clear the model directory before running the experiment instance
        shutil.rmtree(model_directory, ignore_errors=True)

        noise = NoiseGenerator(
            seed=opts.seed,
            max_noise_size=opts.max_noise_size,
            dropout_ratio=opts.noise_dropout
        )

        cache_dir = self.experiment_directory("cache")

        with DatasetFeeder(cache_dir, opts.dry_run) as feeder:
            ds_train, ds_validate, ds_test = \
                Muscima.semisupervised_experiment_datasets(
                    seed=opts.seed,
                    validation_ratio=VALIDATION_RATIO,
                    sup_ratio=opts.sup_ratio,
                    unsup_ratio=opts.unsup_ratio,
                    batch_size=BATCH_SIZE,
                    segdesc=SEGDESC,
                    tile_size_wh=TILE_SIZE_WH,
                    unsupervised_transformation=noise.dataset_transformation,
                    input_scale_factor=None,
                    output_scale_factor=None
                )

            ds_train = feeder(ds_train)
            ds_validate = feeder(ds_validate)
            ds_test = feeder(ds_test)

            model = DenoisingUnetModel.load_or_create(model_directory)
            model.perform_training(
                epochs=(MAX_EPOCHS if not opts.dry_run else 2),
                ds_train=ds_train,
                ds_validate=ds_validate,
                save_checkpoints=False,
                early_stop_after=EARLY_STOPPING_PATIENCE
            )
            # model.perform_evaluation(ds_test) # NOPE, do not

            record = model.model_directory.get_best_epoch_metrics_record(
                by_metric="val_f1_score",
                search_for=max
            )
            self.write_done_file(model_directory, record["val_f1_score"])

    def build_model_name(self, opts: Options) -> str:
        # outputs: "experiment-name__foo=42_bar=baz"
        take_vars = [
            "seed",
            "sup_ratio", "unsup_ratio", "unsup_loss_weight",
            "noise_dropout", "max_noise_size"
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

    def read_done_file(self, model_directory: str) -> Optional[float]:
        try:
            with open(os.path.join(model_directory, "DONE.txt"), "r") as f:
                return float(f.readline())
        except IOError:
            return None

    def write_done_file(self, model_directory: str, value: float):
        with open(os.path.join(model_directory, "DONE.txt"), "w") as f:
            f.write(str(value) + "\n")
