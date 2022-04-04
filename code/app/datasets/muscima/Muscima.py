from typing import Callable, Optional, Tuple
import tensorflow as tf

from .SegmentationDescription import SegmentationDescription
from ..transform_resize_images import transform_resize_images
from ..transform_sample_tiles import transform_sample_tiles
from .transform_pages_to_tile_counts import transform_pages_to_tile_counts
from .transform_pages_to_masks import transform_pages_to_masks
from .transform_pages_to_images import transform_pages_to_images
from .MuscimaPageList import MuscimaPageList
from ..create_semisup_dataset import create_semisup_dataset


class Muscima:
    """
    Represents the MUSCIMA++ dataset content
    in a more experiment-friendly way
    """

    # DPSS = dots (pixels) per staff space
    # (something like DPI, but in a more useful context)
    DPSS = 28.75

    @staticmethod
    def semisupervised_experiment_datasets(
        seed: int,
        validation_ratio: float,
        labeled_ratio: float,
        unlabeled_ratio: float,
        batch_size: int,
        segdesc: SegmentationDescription,
        tile_size_wh: Tuple[int, int],
        unsupervised_transformation: Callable[[tf.data.Dataset], tf.data.Dataset],
        input_scale_factor: Optional[float] = None,
        input_scale_method = tf.image.ResizeMethod.AREA,
        output_scale_factor: Optional[float] = None,
        output_scale_method = tf.image.ResizeMethod.AREA,
    ):
        # split up muscima pages properly
        validation_pages, labeled_pages, unlabeled_pages = \
            MuscimaPageList.get_independent_train_set() \
                .split_validation_semisup(
                    validation_ratio=validation_ratio,
                    labeled_ratio=labeled_ratio,
                    unlabeled_ratio=unlabeled_ratio,
                    seed=seed
                )
        
        test_pages = MuscimaPageList.get_independent_test_set()
        
        # training dataset
        labeled_pages_ds = labeled_pages.as_tf_dataset()
        ds_labeled = tf.data.Dataset.zip(datasets=(
            labeled_pages_ds.apply(transform_pages_to_images()),
            labeled_pages_ds.apply(transform_pages_to_masks(segdesc))
        ))
        ds_labeled = ds_labeled.apply(
            transform_resize_images(input_scale_factor, input_scale_method)
        )
        ds_labeled = ds_labeled.apply(transform_sample_tiles(
            seed=seed,
            tile_size_wh=tile_size_wh,
            tile_count_ds=labeled_pages_ds.apply(
                transform_pages_to_tile_counts(tile_size_wh)
            ),
            nonempty_channels=segdesc.nonempty_channels()
        ))
        ds_labeled = ds_labeled.shuffle(
            buffer_size=1000,
            seed=seed,
            reshuffle_each_iteration=False
        )

        unlabeled_pages_ds = unlabeled_pages.as_tf_dataset()
        ds_unlabeled = unlabeled_pages_ds.apply(transform_pages_to_images())
        ds_unlabeled = ds_unlabeled.apply(
            transform_resize_images(input_scale_factor, input_scale_method)
        )
        ds_unlabeled = ds_unlabeled.apply(unsupervised_transformation)
        ds_unlabeled = ds_unlabeled.apply(transform_sample_tiles(
            seed=seed,
            tile_size_wh=tile_size_wh,
            tile_count_ds=labeled_pages_ds.apply(
                transform_pages_to_tile_counts(tile_size_wh)
            ),
            nonempty_channels=[]
        ))
        ds_unlabeled = ds_unlabeled.shuffle(
            buffer_size=1000,
            seed=seed,
            reshuffle_each_iteration=False
        )

        ds_train = create_semisup_dataset(batch_size, ds_labeled, ds_unlabeled)
        
        # validation dataset
        validation_pages_ds = validation_pages.as_tf_dataset()
        ds_validate = tf.data.Dataset.zip(datasets=(
            validation_pages_ds.apply(transform_pages_to_images()),
            validation_pages_ds.apply(transform_pages_to_masks(segdesc))
        ))
        ds_validate = ds_validate.apply(
            transform_resize_images(input_scale_factor, input_scale_method)
        )
        ds_validate = ds_validate.batch(1)

        # testing dataset
        test_pages_ds = test_pages.as_tf_dataset()
        ds_test = tf.data.Dataset.zip(datasets=(
            test_pages_ds.apply(transform_pages_to_images()),
            test_pages_ds.apply(transform_pages_to_masks(segdesc))
        ))
        ds_test = ds_test.apply(
            transform_resize_images(input_scale_factor, input_scale_method)
        )
        ds_test = ds_test.batch(1)

        # output scaling
        ds_train = ds_train.apply(
            transform_resize_images(output_scale_factor, output_scale_method)
        )
        ds_validate = ds_validate.apply(
            transform_resize_images(output_scale_factor, output_scale_method)
        )
        ds_test = ds_test.apply(
            transform_resize_images(output_scale_factor, output_scale_method)
        )

        return ds_train, ds_validate, ds_test
