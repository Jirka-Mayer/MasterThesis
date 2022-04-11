import random
from numpy import dtype
import tensorflow as tf
import numpy as np
from typing import Dict, List, Tuple


def transform_sample_tiles(
    seed: int,
    tile_size_wh: Tuple[int, int],
    tile_count_ds: tf.data.Dataset,
    oversample_channels: List[int]
):
    rnd = random.Random(seed)
    tile_width, tile_height = tile_size_wh
    tile_count = tile_count_ds \
        .reduce(np.int32(0), lambda state, item: state + item) \
        .numpy()

    def _the_transformation(images_ds):
        x_channels = images_ds.element_spec[0].shape[2]
        y_channels = images_ds.element_spec[1].shape[2]
        def _the_generator():
            for (image_x, image_y), tile_count in zip(
                images_ds.as_numpy_iterator(), tile_count_ds.as_numpy_iterator()
            ):
                image = np.concatenate([image_x, image_y], axis=2)
                nonempty_image_channels = [ci + 1 for ci in oversample_channels]
                for _ in range(tile_count):
                    tile = _sample_tile_from_image(
                        image, tile_size_wh, rnd,
                        nonempty_image_channels
                    )
                    yield tile[:, :, 0:x_channels], tile[:, :, x_channels:]

        ds = tf.data.Dataset.from_generator(
            _the_generator,
            output_signature=(
                tf.TensorSpec(shape=[tile_height, tile_width, x_channels], dtype=tf.float32),
                tf.TensorSpec(shape=[tile_height, tile_width, y_channels], dtype=tf.float32)
            )
        )
        ds = ds.repeat().take(tile_count) # trick to set dataset length
        return ds
    
    return _the_transformation


def _sample_tile_from_image(
    image: np.ndarray,
    tile_size_wh: Tuple[int, int],
    rnd: random.Random,
    oversample_channels: List[int]
) -> Dict[str, np.ndarray]:
    w, h = tile_size_wh
    for attempt in range(5): # resampling attempts
        xf = rnd.randint(0, image.shape[1] - w - 1)
        xt = xf + w
        yf = rnd.randint(0, image.shape[0] - h - 1)
        yt = yf + h

        tile = image[yf:yt, xf:xt, :]

        retry = False
        for channel_index in oversample_channels:
            if np.all(tile[:, :, channel_index] < 0.1):
                retry = True

        if not retry:
            break
    
    return tile
