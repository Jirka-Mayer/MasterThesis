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

    # create color filter
    colors_rgb = meta.classes_to_colors_rgb(classes)
    colors_num = [r + g * 256 + b * 256 * 256 for r, g, b in colors_rgb]
    print(colors_num)

    # create mask
    seg_img_num = seg_img[:,:,0] + seg_img[:,:,1] * 256 + seg_img[:,:,2] * 256 * 256
    mask = np.isin(seg_img_num, colors_num)

    return mask.astype(np.float32) # bool to float
