from typing import Callable, Optional, Tuple
import tensorflow as tf

from ..SegmentationDescription import SegmentationDescription
from ..transform_resize_images import transform_resize_images
from ..transform_sample_tiles import transform_sample_tiles
from .transform_mc_pages_to_tile_counts import transform_mc_pages_to_tile_counts
from .transform_mc_pages_to_masks import transform_mc_pages_to_masks
from .transform_mc_pages_to_images import transform_mc_pages_to_images
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
        dataset_seed: int,
        validation_ratio: float,
        sup_ratio: float,
        unsup_ratio: float,
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
        validation_pages, sup_pages, unsup_pages = \
            MuscimaPageList.get_independent_train_set() \
                .split_validation_semisup(
                    validation_ratio=validation_ratio,
                    sup_ratio=sup_ratio,
                    unsup_ratio=unsup_ratio,
                    seed=dataset_seed
                )
        
        test_pages = MuscimaPageList.get_independent_test_set()

        # training dataset
        sup_pages_ds = sup_pages.as_tf_dataset()
        sup_pages_ds = sup_pages_ds.shuffle(
            buffer_size=len(sup_pages_ds),
            seed=dataset_seed,
            reshuffle_each_iteration=False
        )
        ds_sup = tf.data.Dataset.zip(datasets=(
            sup_pages_ds.apply(transform_mc_pages_to_images()),
            sup_pages_ds.apply(transform_mc_pages_to_masks(segdesc))
        ))
        ds_sup = ds_sup.apply(
            transform_resize_images(input_scale_factor, input_scale_method)
        )
        ds_sup = ds_sup.apply(transform_sample_tiles(
            seed=dataset_seed,
            tile_size_wh=tile_size_wh,
            tile_count_ds=sup_pages_ds.apply(
                transform_mc_pages_to_tile_counts(tile_size_wh)
            ),
            oversample_channels=segdesc.oversampled_channel_indices()
        ))
        ds_sup = ds_sup.shuffle(
            buffer_size=500,
            seed=dataset_seed,
            reshuffle_each_iteration=False
        )

        unsup_pages_ds = unsup_pages.as_tf_dataset()
        unsup_pages_ds = unsup_pages_ds.shuffle(
            buffer_size=len(unsup_pages_ds),
            seed=dataset_seed,
            reshuffle_each_iteration=False
        )
        ds_unsup = unsup_pages_ds.apply(transform_mc_pages_to_images())
        ds_unsup = ds_unsup.apply(
            transform_resize_images(input_scale_factor, input_scale_method)
        )
        ds_unsup = ds_unsup.apply(unsupervised_transformation)
        ds_unsup = ds_unsup.apply(transform_sample_tiles(
            seed=dataset_seed,
            tile_size_wh=tile_size_wh,
            tile_count_ds=unsup_pages_ds.apply(
                transform_mc_pages_to_tile_counts(tile_size_wh)
            ),
            oversample_channels=[]
        ))
        ds_unsup = ds_unsup.shuffle(
            buffer_size=500,
            seed=dataset_seed,
            reshuffle_each_iteration=False
        )

        ds_train = create_semisup_dataset(batch_size, ds_sup, ds_unsup)
        
        # validation dataset
        validation_pages_ds = validation_pages.as_tf_dataset()
        ds_validate = tf.data.Dataset.zip(datasets=(
            validation_pages_ds.apply(transform_mc_pages_to_images()),
            validation_pages_ds.apply(transform_mc_pages_to_masks(segdesc))
        ))
        ds_validate = ds_validate.apply(
            transform_resize_images(input_scale_factor, input_scale_method)
        )
        ds_validate = ds_validate.batch(1)

        # testing dataset
        test_pages_ds = test_pages.as_tf_dataset()
        ds_test = tf.data.Dataset.zip(datasets=(
            test_pages_ds.apply(transform_mc_pages_to_images()),
            test_pages_ds.apply(transform_mc_pages_to_masks(segdesc))
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
