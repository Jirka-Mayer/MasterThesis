import argparse
import tensorflow as tf

from ..datasets.NoiseGenerator import NoiseGenerator
from .Experiment import Experiment
from ..models.DenoisingUnetModel import DenoisingUnetModel
from ..datasets.DatasetFeeder import DatasetFeeder
from ..datasets.muscima.Muscima import Muscima
from ..datasets.muscima.SegmentationDescription import SegmentationDescription


class Options:
    def __init__(self, **kwargs):
        self.seed: int = kwargs["seed"]
        self.epochs: int = kwargs["epochs"]
        self.batch_size: int = kwargs["batch_size"]
        self.validation_ratio: float = kwargs["validation_ratio"]
        self.sup_ratio: float = kwargs["sup_ratio"]
        self.unsup_ratio: float = kwargs["unsup_ratio"]


class Ex_HyperparamSearch(Experiment):
    @property
    def name(self):
        return "hyperparam-search"

    def describe(self):
        return """
        TODO
        """

    def define_arguments(self, parser: argparse.ArgumentParser):
        parser.add_argument("--seed", default=42, type=int, help="Random seed.")

    def run(self, args: argparse.Namespace):
        opts = Options(
            seed=args.seed,
            epochs=50,
            batch_size=2,
            validation_ratio=0.1,
            sup_ratio=0.05,
            unsup_ratio=0.05
        )
        self.compute_single_instance(opts)

    def compute_single_instance(self, opts: Options) -> float:
        tf.random.set_seed(opts.seed)

        model_directory = self.experiment_directory(
            self.build_model_name(opts)
        )

        noise = NoiseGenerator(
            seed=opts.seed,
            max_noise_size=Muscima.DPSS * 2,
            dropout_ratio=0.25
        )

        with DatasetFeeder(self.experiment_directory("cache")) as feeder:
            ds_train, ds_validate, ds_test = \
                Muscima.semisupervised_experiment_datasets(
                    seed=opts.seed,
                    validation_ratio=opts.validation_ratio,
                    sup_ratio=opts.sup_ratio,
                    unsup_ratio=opts.unsup_ratio,
                    batch_size=opts.batch_size,
                    segdesc=SegmentationDescription.FLAGS,
                    tile_size_wh=(512, 256),
                    unsupervised_transformation=noise.dataset_transformation,
                    #output_scale_factor=0.25, # TODO: debug
                )

            ds_train = feeder(ds_train)
            ds_validate = feeder(ds_validate)
            ds_test = feeder(ds_test)

            model = DenoisingUnetModel.load_or_create(model_directory)
            model.perform_training(opts.epochs, ds_train, ds_validate)
            model.perform_evaluation(ds_test)

    def build_model_name(self, opts: Options) -> str:
        # outputs: "experiment-name__foo=42_bar=baz"
        take_vars = ["seed", "sup_ratio", "unsup_ratio", "batch_size"]
        opt_vars = vars(opts)
        vars_list = [v + "=" + str(opt_vars[v]) for v in take_vars]
        return "{}__{}".format(self.name, "_".join(vars_list))
