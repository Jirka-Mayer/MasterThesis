from typing import Tuple
import tensorflow as tf
from .MuscimaPage import MuscimaPage


def transform_mc_pages_to_tile_counts(tile_size_wh: Tuple[int, int]):
    tile_width, tile_height = tile_size_wh
    tile_pixels = tile_width * tile_height

    def _dim_to_tiles(w, h):
        return w * h // tile_pixels

    def _the_transformation(pages_ds):
        def _the_generator():
            for w, p in pages_ds.as_numpy_iterator():
                page = MuscimaPage(w, p)
                yield _dim_to_tiles(*page.dimensions_via_magic())

        ds = tf.data.Dataset.from_generator(
            _the_generator,
            output_signature= tf.TensorSpec(shape=[], dtype=tf.int32)
        )
        ds = ds.repeat().take(len(pages_ds)) # trick to set dataset length
        return ds
    
    return _the_transformation
