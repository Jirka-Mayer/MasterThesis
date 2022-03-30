import os
import shutil
import random
import tensorflow as tf


# copy the dataset construction file
__folder = os.path.dirname(os.path.realpath(__file__))
__source = os.path.join(__folder, "../00-datasets/01-muscima.py")
__target = os.path.join(__folder, "_copied_dataset.py")
shutil.rmtree(__target, ignore_errors=True)
shutil.copyfile(__source, __target)

# import the dataset construction file
from _copied_dataset import *


def prepare_datasets(seed, validation_split=0.1, scale_factor=None):
    # define configuration
    rnd = random.Random(seed)
    masks = {
        "noteheads": [
            "noteheadFull", "noteheadHalf", "noteheadWhole",
            "noteheadFullSmall", "noteheadHalfSmall"
        ]
    }
    nonempty_classnames = ["noteheads"]
    tile_size_wh = (512, 256)

    # setup page lists
    original_train_pages = MuscimaPageList.get_independent_train_set()
    validation_pages, train_pages = original_train_pages \
        .split(ratio=validation_split, seed=seed)
    test_pages = MuscimaPageList.get_independent_test_set()

    # create datasets
    def page_list_to_dataset(page_list):
        ds = muscimapp_images_with_masks(page_list, masks)
        ds_tc = tile_count_dataset(page_list, tile_size_wh)
        ds = sample_tiles_from(ds, ds_tc, tile_size_wh, rnd, nonempty_classnames)
        ds = add_noisy_images(ds, rnd)
        
        if scale_factor is not None:
            ds = resize_images(ds, scale_factor, tf.image.ResizeMethod.AREA)
        
        def _tuplify(item):
            """Converts the dict dataset to tuple dataset compatible with keras model"""
            return item["image_noisy"], item["image"]
        ds = ds.map(_tuplify)

        return ds
    
    return (
        page_list_to_dataset(train_pages),
        page_list_to_dataset(validation_pages),
        page_list_to_dataset(test_pages)
    )
