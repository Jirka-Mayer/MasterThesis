from typing import Tuple
import tensorflow as tf
from .DsMetadata import DsMetadata


def transform_ds_pages_to_tile_counts(
    meta: DsMetadata,
    tile_size_wh: Tuple[int, int],
    scale_factor: float = 1.0
):
    tile_width, tile_height = tile_size_wh
    tile_pixels = tile_width * tile_height

    def _dim_to_tiles(w, h):
        return w * h // tile_pixels

    def _the_transformation(pages_ds):
        def _the_generator():
            for p in pages_ds.as_numpy_iterator():
                w, h = meta.page_index_to_size_wh(p)
                yield _dim_to_tiles(
                    int(w * scale_factor),
                    int(h * scale_factor)
                )

        ds = tf.data.Dataset.from_generator(
            _the_generator,
            output_signature= tf.TensorSpec(shape=[], dtype=tf.int32)
        )
        ds = ds.repeat().take(len(pages_ds)) # trick to set dataset length
        return ds
    
    return _the_transformation
