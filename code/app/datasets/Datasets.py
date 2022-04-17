import random
from .deepscores.DeepScores import DeepScores
from .deepscores.DsMetadata import DsMetadata
from .deepscores.transform_ds_pages_to_images import transform_ds_pages_to_images
from .deepscores.transform_ds_pages_to_masks import transform_ds_pages_to_masks
from .deepscores.transform_ds_pages_to_tile_counts import transform_ds_pages_to_tile_counts
from .DatasetOptions import DatasetOptions
from .muscima.MuscimaPageList import MuscimaPageList
import tensorflow as tf
from .transform_resize_images import transform_resize_images
from .transform_sample_tiles import transform_sample_tiles
from .muscima.Muscima import Muscima
from .muscima.transform_mc_pages_to_tile_counts import transform_mc_pages_to_tile_counts
from .muscima.transform_mc_pages_to_masks import transform_mc_pages_to_masks
from .muscima.transform_mc_pages_to_images import transform_mc_pages_to_images
from .create_semisup_dataset import create_semisup_dataset
from .transform_shuffle import transform_shuffle


class Datasets:
    @staticmethod
    def exploration(opts: DatasetOptions):
        """
        Dataset containing MUSCIMA++ subsections,
        meant for parameter space exploration

        Supervised, unsupervised, validation and training sets all come from
        MUSCIMA++ in the writer-independent way (one writer is never in two sets)
        """
        remaining_pages = MuscimaPageList.get_independent_train_set() \
            .shuffle_writer_clusters(seed=opts.dataset_seed)
        
        validation_pages = remaining_pages.take(opts.validation_pages)
        remaining_pages = remaining_pages.filter_out_writers(validation_pages.get_writers())
        
        sup_pages = remaining_pages.take(opts.supervised_pages)
        remaining_pages = remaining_pages.filter_out_writers(sup_pages.get_writers())

        unsup_pages = remaining_pages.take(opts.unsupervised_pages)
        remaining_pages = remaining_pages.filter_out_writers(unsup_pages.get_writers())

        test_pages = MuscimaPageList.get_independent_test_set()

        if opts.verbose:
            print("SUP:", sup_pages)
            print("\nUNSUP:", unsup_pages)
            print("\nVALIDATION:", validation_pages)
            print("\nTEST:", test_pages)
        
        print("\nSUP: {}, UNSUP: {}, VAL: {}, TEST: {}, REMAINING: {}".format(
            len(sup_pages), len(unsup_pages), len(validation_pages),
            len(test_pages), len(remaining_pages)
        ))

        assert len(sup_pages) == opts.supervised_pages
        assert len(unsup_pages) == opts.unsupervised_pages
        assert len(validation_pages) == opts.validation_pages

        sup_pages.shuffle_in_place(seed=opts.dataset_seed)
        unsup_pages.shuffle_in_place(seed=opts.dataset_seed)

        # training dataset
        sup_pages_ds = sup_pages.as_tf_dataset()
        ds_sup = tf.data.Dataset.zip(datasets=(
            sup_pages_ds.apply(transform_mc_pages_to_images()),
            sup_pages_ds.apply(transform_mc_pages_to_masks(opts.segdesc))
        ))
        ds_sup = ds_sup.apply(transform_sample_tiles(
            seed=opts.dataset_seed,
            tile_size_wh=opts.tile_size_wh,
            tile_count_ds=sup_pages_ds.apply(
                transform_mc_pages_to_tile_counts(opts.tile_size_wh)
            ),
            oversample_channels=opts.segdesc.oversampled_channel_indices()
        ))
        ds_sup = ds_sup.shuffle(
            buffer_size=500,
            seed=opts.dataset_seed,
            reshuffle_each_iteration=False
        )

        unsup_pages_ds = unsup_pages.as_tf_dataset()
        ds_unsup = unsup_pages_ds.apply(transform_mc_pages_to_images())
        ds_unsup = ds_unsup.apply(opts.unsupervised_transformation)
        ds_unsup = ds_unsup.apply(transform_sample_tiles(
            seed=opts.dataset_seed,
            tile_size_wh=opts.tile_size_wh,
            tile_count_ds=unsup_pages_ds.apply(
                transform_mc_pages_to_tile_counts(opts.tile_size_wh)
            ),
            oversample_channels=[]
        ))
        ds_unsup = ds_unsup.shuffle(
            buffer_size=500,
            seed=opts.dataset_seed,
            reshuffle_each_iteration=False
        )

        ds_train = create_semisup_dataset(opts.batch_size, ds_sup, ds_unsup)

        # validation dataset
        validation_pages_ds = validation_pages.as_tf_dataset()
        ds_validate = tf.data.Dataset.zip(datasets=(
            validation_pages_ds.apply(transform_mc_pages_to_images()),
            validation_pages_ds.apply(transform_mc_pages_to_masks(opts.segdesc))
        ))
        ds_validate = ds_validate.batch(1)

        # testing dataset
        test_pages_ds = test_pages.as_tf_dataset()
        ds_test = tf.data.Dataset.zip(datasets=(
            test_pages_ds.apply(transform_mc_pages_to_images()),
            test_pages_ds.apply(transform_mc_pages_to_masks(opts.segdesc))
        ))
        ds_test = ds_test.batch(1)

        return ds_train, ds_validate, ds_test

    @staticmethod
    def improvement(opts: DatasetOptions):
        """
        Dataset to test model improvement by utilizing unlabeled CVC-MUSCIMA data,
        while testing on the MUSCIMA++ writer-independent test set

        Unsupervised set is CVC-MUSCIMA and all the other sets are MUSCIMA++
        """
        test_pages = MuscimaPageList.get_independent_test_set()

        remaining_pages = MuscimaPageList.get_independent_train_set() \
            .shuffle_writer_clusters(seed=opts.dataset_seed)
        
        validation_pages = remaining_pages.take(opts.validation_pages)
        remaining_pages = remaining_pages.filter_out_writers(validation_pages.get_writers())
        
        sup_pages = remaining_pages.take(opts.supervised_pages)
        remaining_sup_pages = remaining_pages.filter_out_pages(sup_pages)

        unsupable_pages = MuscimaPageList.get_entire_cvc_muscima()
        unsupable_pages = unsupable_pages.filter_out_writers(test_pages.get_writers())
        unsupable_pages = unsupable_pages.filter_out_writers(validation_pages.get_writers())
        unsupable_pages = unsupable_pages.filter_out_pages(sup_pages)

        unsup_pages = unsupable_pages.take(opts.unsupervised_pages)
        remaining_unsup_pages = unsupable_pages.filter_out_pages(unsup_pages)

        if opts.verbose:
            print("SUP:", sup_pages)
            print("\nUNSUP:", unsup_pages)
            print("\nVALIDATION:", validation_pages)
            print("\nTEST:", test_pages)
        
        print("\nSUP: {}, UNSUP: {}, VAL: {}, TEST: {}, REM. SUP: {}, REM. UNSUP: {}".format(
            len(sup_pages), len(unsup_pages), len(validation_pages),
            len(test_pages), len(remaining_sup_pages), len(remaining_unsup_pages)
        ))

        assert len(sup_pages) == opts.supervised_pages
        assert len(unsup_pages) == opts.unsupervised_pages
        assert len(validation_pages) == opts.validation_pages

        sup_pages.shuffle_in_place(seed=opts.dataset_seed)
        unsup_pages.shuffle_in_place(seed=opts.dataset_seed)

        # training dataset
        sup_pages_ds = sup_pages.as_tf_dataset()
        ds_sup = tf.data.Dataset.zip(datasets=(
            sup_pages_ds.apply(transform_mc_pages_to_images()),
            sup_pages_ds.apply(transform_mc_pages_to_masks(opts.segdesc))
        ))
        ds_sup = ds_sup.apply(transform_sample_tiles(
            seed=opts.dataset_seed,
            tile_size_wh=opts.tile_size_wh,
            tile_count_ds=sup_pages_ds.apply(
                transform_mc_pages_to_tile_counts(opts.tile_size_wh)
            ),
            oversample_channels=opts.segdesc.oversampled_channel_indices()
        ))
        ds_sup = ds_sup.shuffle(
            buffer_size=500,
            seed=opts.dataset_seed,
            reshuffle_each_iteration=False
        )

        unsup_pages_ds = unsup_pages.as_tf_dataset()
        ds_unsup = unsup_pages_ds.apply(transform_mc_pages_to_images())
        ds_unsup = ds_unsup.apply(opts.unsupervised_transformation)
        ds_unsup = ds_unsup.apply(transform_sample_tiles(
            seed=opts.dataset_seed,
            tile_size_wh=opts.tile_size_wh,
            tile_count_ds=unsup_pages_ds.apply(
                transform_mc_pages_to_tile_counts(opts.tile_size_wh)
            ),
            oversample_channels=[]
        ))
        ds_unsup = ds_unsup.shuffle(
            buffer_size=500,
            seed=opts.dataset_seed,
            reshuffle_each_iteration=False
        )

        ds_train = create_semisup_dataset(opts.batch_size, ds_sup, ds_unsup)

        # validation dataset
        validation_pages_ds = validation_pages.as_tf_dataset()
        ds_validate = tf.data.Dataset.zip(datasets=(
            validation_pages_ds.apply(transform_mc_pages_to_images()),
            validation_pages_ds.apply(transform_mc_pages_to_masks(opts.segdesc))
        ))
        ds_validate = ds_validate.batch(1)

        # testing dataset
        test_pages_ds = test_pages.as_tf_dataset()
        ds_test = tf.data.Dataset.zip(datasets=(
            test_pages_ds.apply(transform_mc_pages_to_images()),
            test_pages_ds.apply(transform_mc_pages_to_masks(opts.segdesc))
        ))
        ds_test = ds_test.batch(1)

        return ds_train, ds_validate, ds_test

    @staticmethod
    def transfer2mpp(opts: DatasetOptions):
        """
        Dataset for exploring knowledge transfer from deepscores to MUSCIMA++

        Supervised set is deepscores and all the other sets are MUSCIMA++
        """
        # setup deepscores pages
        rnd = random.Random(opts.dataset_seed)
        all_ds_pages = list(range(DeepScores.TRAIN_PAGE_COUNT))
        rnd.shuffle(all_ds_pages)

        sup_pages = all_ds_pages[:opts.supervised_pages]
        remaining_sup_pages = all_ds_pages[opts.supervised_pages:]
        sup_pages_ds = tf.data.Dataset.from_tensor_slices(sup_pages)
        
        # setup muscima pages
        remaining_pages = MuscimaPageList.get_independent_train_set() \
            .shuffle_writer_clusters(seed=opts.dataset_seed)
        
        validation_pages = remaining_pages.take(opts.validation_pages)
        remaining_pages = remaining_pages.filter_out_writers(validation_pages.get_writers())
        
        unsup_pages = remaining_pages.take(opts.unsupervised_pages)
        remaining_unsup_pages = remaining_pages.filter_out_pages(unsup_pages)

        test_pages = MuscimaPageList.get_independent_test_set()

        if opts.verbose:
            print("SUP:", sup_pages)
            print("\nUNSUP:", unsup_pages)
            print("\nVALIDATION:", validation_pages)
            print("\nTEST:", test_pages)
        
        print("\nSUP: {}, UNSUP: {}, VAL: {}, TEST: {}, REM. SUP: {}, REM. UNSUP: {}".format(
            len(sup_pages), len(unsup_pages), len(validation_pages),
            len(test_pages), len(remaining_sup_pages), len(remaining_unsup_pages)
        ))

        assert len(sup_pages) == opts.supervised_pages
        assert len(unsup_pages) == opts.unsupervised_pages
        assert len(validation_pages) == opts.validation_pages

        unsup_pages.shuffle_in_place(seed=opts.dataset_seed)

        # other useful metadata
        meta_train = DsMetadata.from_train_set(verbose=True)
        deepscores_scaleup_factor = Muscima.DPSS / DeepScores.DPSS

        # setup training dataset
        ds_sup = tf.data.Dataset.zip(datasets=(
            sup_pages_ds.apply(transform_ds_pages_to_images(meta_train)),
            sup_pages_ds.apply(transform_ds_pages_to_masks(meta_train, opts.segdesc))
        ))
        ds_sup = ds_sup.apply(
            transform_resize_images(deepscores_scaleup_factor, tf.image.ResizeMethod.BILINEAR)
        )
        ds_sup = ds_sup.apply(transform_sample_tiles(
            seed=opts.dataset_seed,
            tile_size_wh=opts.tile_size_wh,
            tile_count_ds=sup_pages_ds.apply(
                transform_ds_pages_to_tile_counts(
                    meta_train, opts.tile_size_wh, deepscores_scaleup_factor
                )
            ),
            oversample_channels=opts.segdesc.oversampled_channel_indices()
        ))
        ds_sup = ds_sup.shuffle(
            buffer_size=500,
            seed=opts.dataset_seed,
            reshuffle_each_iteration=False
        )

        unsup_pages_ds = unsup_pages.as_tf_dataset()
        ds_unsup = unsup_pages_ds.apply(transform_mc_pages_to_images())
        ds_unsup = ds_unsup.apply(opts.unsupervised_transformation)
        ds_unsup = ds_unsup.apply(transform_sample_tiles(
            seed=opts.dataset_seed,
            tile_size_wh=opts.tile_size_wh,
            tile_count_ds=unsup_pages_ds.apply(
                transform_mc_pages_to_tile_counts(opts.tile_size_wh)
            ),
            oversample_channels=[]
        ))
        ds_unsup = ds_unsup.shuffle(
            buffer_size=500,
            seed=opts.dataset_seed,
            reshuffle_each_iteration=False
        )

        ds_train = create_semisup_dataset(opts.batch_size, ds_sup, ds_unsup)

        # validation dataset
        validation_pages_ds = validation_pages.as_tf_dataset()
        ds_validate = tf.data.Dataset.zip(datasets=(
            validation_pages_ds.apply(transform_mc_pages_to_images()),
            validation_pages_ds.apply(transform_mc_pages_to_masks(opts.segdesc))
        ))
        ds_validate = ds_validate.batch(1)

        # testing dataset
        test_pages_ds = test_pages.as_tf_dataset()
        ds_test = tf.data.Dataset.zip(datasets=(
            test_pages_ds.apply(transform_mc_pages_to_images()),
            test_pages_ds.apply(transform_mc_pages_to_masks(opts.segdesc))
        ))
        ds_test = ds_test.batch(1)

        return ds_train, ds_validate, ds_test

    def by_name(name: str, opts: DatasetOptions):
        if name == "exploration":
            return Datasets.exploration(opts)
        if name == "improvement":
            return Datasets.improvement(opts)
        if name == "transfer2mpp":
            return Datasets.transfer2mpp(opts)
        raise Exception("Unknown dataset name: " + name)
