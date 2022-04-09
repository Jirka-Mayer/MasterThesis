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


class Ex01_SemisupUnet(Experiment):
    @property
    def name(self):
        return "01-semisup-unet"

    def describe(self):
        return """
        Train a denoising semi-supervised unet model on the muscima++ dataset
        on various sup-unsup training data splits with the goal of
        segmenting noteheads and measure the resulting F1 score for the
        muscima++ test set.
        """

    def define_arguments(self, parser: argparse.ArgumentParser):
        parser.add_argument("-n", "--number")

    def run(self, args: argparse.Namespace):
        """
            Possible values to search through:
            seed=[0, 1, 2, 3, 4]
            sup_ratio=0.05,
            unsup_ratio=[0.0, 0.05, 0.1, 0.3, 0.5],
            batch_size=[2, 4, 8, 16, 32]
        """

        # C (different ratios)
        for unsup_ratio in [0.0, 0.05, 0.1, 0.3, 0.5]:
            opts = Options(
                seed=0,
                epochs=75,
                batch_size=16,
                validation_ratio=0.1,
                sup_ratio=0.05,
                unsup_ratio=unsup_ratio
            )
            self.compute_single_instance(opts)
        
        # # A (small batches)
        # opts = Options(
        #     seed=42,
        #     epochs=10,
        #     batch_size=2,
        #     validation_ratio=0.1,
        #     sup_ratio=0.05,
        #     unsup_ratio=0.5
        # )
        # self.compute_single_instance(opts)

        # # B (large batches)
        # opts = Options(
        #     seed=42,
        #     epochs=50,
        #     batch_size=10,
        #     validation_ratio=0.1,
        #     sup_ratio=0.05,
        #     unsup_ratio=0.5
        # )
        # self.compute_single_instance(opts)

    def compute_single_instance(self, opts: Options) -> float:
        tf.random.set_seed(opts.seed)

        model_directory = self.experiment_directory(
            self.build_model_name(opts)
        )

        noise = NoiseGenerator(opts.seed)

        with DatasetFeeder(self.experiment_directory("cache")) as feeder:
            ds_train, ds_validate, ds_test = \
                Muscima.semisupervised_experiment_datasets(
                    seed=opts.seed,
                    validation_ratio=opts.validation_ratio,
                    sup_ratio=opts.sup_ratio,
                    unsup_ratio=opts.unsup_ratio,
                    batch_size=opts.batch_size,
                    segdesc=SegmentationDescription.NOTEHEADS,
                    tile_size_wh=(512, 256),
                    unsupervised_transformation=noise.dataset_transformation,
                    output_scale_factor=0.25, # TODO: debug
                )

            ds_train = feeder(ds_train)
            ds_validate = feeder(ds_validate)
            ds_test = feeder(ds_test)

            model = DenoisingUnetModel.load_or_create(model_directory)
            model.perform_training(opts.epochs, ds_train, ds_validate)
            model.perform_evaluation(ds_test)

    def build_model_name(self, opts: Options) -> str:
        # outputs: "experiment-name__foo=42_bar=baz"
        take_vars = ["sup_ratio", "unsup_ratio", "seed", "batch_size"]
        opt_vars = vars(opts)
        vars_list = [v + "=" + str(opt_vars[v]) for v in take_vars]
        return "{}__{}".format(self.name, "_".join(vars_list))
