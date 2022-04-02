import argparse
import os
import tensorflow as tf
from .Experiment import Experiment
from ..models.DenoisingUnetModel import DenoisingUnetModel
from ..datasets.legacy_muscima import prepare_datasets
from ..datasets.convert_to_semisup_dataset import convert_to_semisup_dataset


class Ex01_SemisupUnet(Experiment):
    @property
    def name(self):
        return "01-semisup-unet"

    def desribe(self):
        return """
        Train a denoising semi-supervised unet model on the muscima++ dataset
        on various labeled-unlabeled training data splits with the goal of
        segmenting noteheads and measure the resulting F1 score for the
        muscima++ test set.
        """

    def define_arguments(self, parser: argparse.ArgumentParser):
        parser.add_argument("-n", "--number")

    def run(self, args: argparse.Namespace):
        self.compute_single_instance(
            seed=42,
            epochs=10,
            batch_size=2,
            labeled_ratio=0.1,
            unlabeled_ratio=0.1
        )

    def compute_single_instance(
        self,
        seed: int,
        epochs: int,
        batch_size: int,
        labeled_ratio: float,
        unlabeled_ratio: float
    ) -> float:
        tf.random.set_seed(seed)

        ds_train, ds_validate, ds_test = prepare_datasets(
            seed=seed,
            validation_split=0.1,
            scale_factor=0.25 # TODO: testing out on smaller data
        )

        CACHE_PATH = os.path.expanduser("~/Datasets/Cache/my_model")

        # TODO: figure out some smart dataset caching syntax
        # TODO: figure out short dataset API

        # e.g. datasets.bust_cache()
        
        
        
        
        # test_pages = MuscimaPageList.get_independent_test_set()
        # Muscima.ds_
        
        
        
        
        
        #######################

        # slice_labeled, slice_unlabeled, slice_validation, slice_test = Muscima.get_dataset_slices(
        #     validation_ratio=0.1,
        #     labeled_ratio=labeled_ratio,
        #     unlabeled_ratio=unlabeled_ratio
        # )

        # Muscima.denoising_semisup_segmentation_datasets(
        #     tile_size=?,
        #     batch_size=?,
        #     segmentation_classes=SegmentationClasses.NOTEHEADS
        # )

        ds_train = ds_train \
            .batch(batch_size) \
            .snapshot(path=os.path.join(CACHE_PATH, "train"), compression=None) \
            .prefetch(tf.data.AUTOTUNE)

        ds_train = ds_train.take(200)
        
        ds_train = convert_to_semisup_dataset(ds_train)

        ds_validate = ds_validate \
            .batch(batch_size) \
            .snapshot(path=os.path.join(CACHE_PATH, "validate"), compression=None) \
            .prefetch(tf.data.AUTOTUNE)

        ds_test = ds_test \
            .batch(batch_size) \
            .snapshot(path=os.path.join(CACHE_PATH, "test"), compression=None) \
            .prefetch(tf.data.AUTOTUNE)

        model_directory = "my_model"

        model = DenoisingUnetModel.load_or_create(model_directory)
        model.perform_training(epochs, ds_train, ds_validate)
        model.perform_evaluation(ds_test)
