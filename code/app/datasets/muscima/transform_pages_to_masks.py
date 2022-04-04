import tensorflow as tf
import numpy as np
import mung
from typing import List
from .MuscimaPage import MuscimaPage
from .SegmentationDescription import SegmentationDescription


def transform_pages_to_masks(segdesc: SegmentationDescription):
    def _the_transformation(pages_ds):
        def _the_generator():
            for w, p in pages_ds.as_numpy_iterator():
                page = MuscimaPage(w, p)
                yield np.dstack([
                    _construct_muscima_page_mask(
                        page,
                        segdesc.channel_mung_classes(channel_name)
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


def _construct_muscima_page_mask(page: MuscimaPage, node_classes: List[str]) -> np.ndarray:
    image = page.load_ideal_image_as_numpy()
    nodes = page.load_nodes()

    mask = np.zeros(shape=(image.shape[0], image.shape[1]), dtype=np.float32)
    for node in nodes:
        if node.class_name in node_classes:
            _print_mask_into_image(mask, node)
    return mask


def _print_mask_into_image(image: np.ndarray, node: mung.io.Node):
    yf = node.top
    yt = yf + node.height
    xf = node.left
    xt = xf + node.width
    image[yf:yt, xf:xt] = 1 - (1 - node.mask) * (1 - image[yf:yt, xf:xt]) # fuzzy OR
