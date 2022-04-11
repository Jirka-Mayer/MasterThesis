import tensorflow as tf
from typing import Tuple, Callable
from .transform_sample_tiles import transform_sample_tiles
from .deepscores.DeepScores import DeepScores
from .muscima.Muscima import Muscima
from .muscima.MuscimaPageList import MuscimaPageList
from .muscima.transform_mc_pages_to_images import transform_mc_pages_to_images
from .muscima.transform_mc_pages_to_tile_counts import transform_mc_pages_to_tile_counts
from .muscima.transform_mc_pages_to_masks import transform_mc_pages_to_masks
from .deepscores.DsMetadata import DsMetadata
from .SegmentationDescription import SegmentationDescription
from .deepscores.transform_ds_pages_to_images import transform_ds_pages_to_images
from .deepscores.transform_ds_pages_to_masks import transform_ds_pages_to_masks
from .deepscores.transform_ds_pages_to_tile_counts import transform_ds_pages_to_tile_counts
from .transform_resize_images import transform_resize_images
from .create_semisup_dataset import create_semisup_dataset


class TransferDataset:
    """Dataset(s) for the knowledge transfer experiment"""
    
    @staticmethod
    def deepscores_to_muscima(
        seed: int,
        tile_size_wh: Tuple[int, int],
        validation_pages: int,
        supervised_pages: int,
        unsupervised_pages: int,
        batch_size: int,
        segdesc: SegmentationDescription,
        unsupervised_transformation: Callable[[tf.data.Dataset], tf.data.Dataset],
    ):
        meta_train = DsMetadata.from_train_set()
        deepscores_scaleup_factor = Muscima.DPSS / DeepScores.DPSS

        # setup ds pages datasets
        all_ds_pages_ds = tf.data.Dataset.range(meta_train.page_count())
        all_ds_page_count = len(all_ds_pages_ds)
        all_ds_pages_ds = all_ds_pages_ds.shuffle(
            buffer_size=all_ds_page_count,
            seed=seed,
            reshuffle_each_iteration=False
        )

        assert validation_pages + supervised_pages <= all_ds_page_count

        validation_pages_ds = all_ds_pages_ds.take(validation_pages)
        sup_pages_ds = all_ds_pages_ds.skip(validation_pages).take(supervised_pages)

        # setup muscima pages datasets
        unsup_pages_ds = MuscimaPageList.get_independent_train_set().as_tf_dataset()
        unsup_pages_ds = unsup_pages_ds.shuffle(
            buffer_size=len(unsup_pages_ds),
            seed=seed,
            reshuffle_each_iteration=False
        )
        unsup_pages_ds = unsup_pages_ds.take(unsupervised_pages)
        
        test_pages_ds = MuscimaPageList.get_independent_test_set().as_tf_dataset()

        # setup training dataset
        ds_sup = tf.data.Dataset.zip(datasets=(
            sup_pages_ds.apply(transform_ds_pages_to_images(meta_train)),
            sup_pages_ds.apply(transform_ds_pages_to_masks(meta_train, segdesc))
        ))
        ds_sup = ds_sup.apply(
            transform_resize_images(deepscores_scaleup_factor, tf.image.ResizeMethod.BILINEAR)
        )
        ds_sup = ds_sup.apply(transform_sample_tiles(
            seed=seed,
            tile_size_wh=tile_size_wh,
            tile_count_ds=sup_pages_ds.apply(
                transform_ds_pages_to_tile_counts(
                    meta_train, tile_size_wh, deepscores_scaleup_factor
                )
            ),
            oversample_channels=segdesc.oversampled_channel_indices()
        ))
        ds_sup = ds_sup.shuffle(
            buffer_size=100,
            seed=seed,
            reshuffle_each_iteration=False
        )

        ds_unsup = unsup_pages_ds.apply(transform_mc_pages_to_images())
        ds_unsup = ds_unsup.apply(unsupervised_transformation)
        ds_unsup = ds_unsup.apply(transform_sample_tiles(
            seed=seed,
            tile_size_wh=tile_size_wh,
            tile_count_ds=unsup_pages_ds.apply(
                transform_mc_pages_to_tile_counts(tile_size_wh)
            ),
            oversample_channels=[]
        ))
        ds_unsup = ds_unsup.shuffle(
            buffer_size=100,
            seed=seed,
            reshuffle_each_iteration=False
        )

        ds_train = create_semisup_dataset(batch_size, ds_sup, ds_unsup)

        # validation dataset
        ds_validate = tf.data.Dataset.zip(datasets=(
            validation_pages_ds.apply(transform_ds_pages_to_images(meta_train)),
            validation_pages_ds.apply(transform_ds_pages_to_masks(meta_train, segdesc))
        ))
        ds_validate = ds_validate.apply(
            transform_resize_images(deepscores_scaleup_factor, tf.image.ResizeMethod.BILINEAR)
        )
        ds_validate = ds_validate.batch(1)

        # testing dataset
        ds_test = tf.data.Dataset.zip(datasets=(
            test_pages_ds.apply(transform_mc_pages_to_images()),
            test_pages_ds.apply(transform_mc_pages_to_masks(segdesc))
        ))
        ds_test = ds_test.batch(1)

        return ds_train, ds_validate, ds_test





        print("loading ds meta")
        meta_train = DsMetadata.from_train_set()
        print(len(meta_train.ds["images"]))

        deepscores_scaleup_factor = Muscima.DPSS / DeepScores.DPSS
        # TODO: shuffle pages before splitting them

        ds = tf.data.Dataset.range(meta_train.page_count())
        ds = ds.shuffle(buffer_size=1000, seed=seed, reshuffle_each_iteration=False)
        
        dso = ds
        ds = transform_ds_pages_to_images(meta_train)(ds)
        ds = transform_resize_images(deepscores_scaleup_factor, tf.image.ResizeMethod.BILINEAR)(ds)
        
        tcds = transform_ds_pages_to_tile_counts(meta_train, tile_size_wh, deepscores_scaleup_factor)(dso)

        ds = transform_sample_tiles(seed, tile_size_wh, tcds, segdesc.oversampled_channel_indices())(ds)

        # ds = transform_ds_pages_to_masks(meta_train, segdesc)(ds)
        # ds = transform_resize_images(deepscores_scaleup_factor, tf.image.ResizeMethod.BILINEAR)(ds)

        print(len(ds))
        import matplotlib.pyplot as plt
        for x in ds.as_numpy_iterator():
            plt.imshow(x)
            plt.show()

        # scale deepscores up to muscima DPSS

        # return TRAIN (ds), VALIDATE (ds), TEST (mpp)
