import tensorflow as tf
import numpy as np
from typing import List
from .DsMetadata import DsMetadata
from ..SegmentationDescription import SegmentationDescription


def transform_ds_pages_to_masks(meta: DsMetadata, segdesc: SegmentationDescription):
    def _the_transformation(pages_ds):
        def _the_generator():
            for p in pages_ds.as_numpy_iterator():
                yield np.dstack([
                    _construct_deepscores_page_mask(
                        meta, p,
                        segdesc.channel_deepscores_classes(channel_name)
                    )
                    for channel_name in segdesc.channel_names()
                ])

        ds = tf.data.Dataset.from_generator(
            _the_generator,
            output_signature=tf.TensorSpec(
                shape=[None, None, len(segdesc.channel_names())],
                dtype=tf.float32
            )
        )
        ds = ds.repeat().take(len(pages_ds)) # trick to set dataset length
        return ds
    
    return _the_transformation


def _construct_deepscores_page_mask(
    meta: DsMetadata,
    page_index: int,
    classes: List[str]
):
    # load segmentation image
    seg_path = meta.page_index_to_segmentation_path(page_index)
    data = tf.io.read_file(seg_path)
    seg_img = tf.io.decode_png(data).numpy() # int32 0-255 RGB image (0=R 1=G 2=B)

    colors_rgb = meta.classes_to_colors_rgb(classes)

    mask = None
    for c in colors_rgb:
        class_mask = np.all(seg_img == c, axis=-1)
        if mask is None:
            mask = class_mask
        else:
            mask = np.logical_or(mask, class_mask)

    return mask.astype(np.float32) # bool to float
