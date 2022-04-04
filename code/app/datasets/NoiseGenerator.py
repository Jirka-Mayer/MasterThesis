import tensorflow as tf
import numpy as np
import random


class NoiseGenerator:
    def __init__(self, seed: int):
        self._rnd = random.Random(seed)
        self._nprnd = np.random.RandomState(seed)

    def dataset_transformation(self, images_ds: tf.data.Dataset) -> tf.data.Dataset:
        def _the_generator():
            for image in images_ds.as_numpy_iterator():
                yield self._create_noisy_image(image), image # x, y
        
        ds = tf.data.Dataset.from_generator(
            _the_generator,
            output_signature=(images_ds.element_spec, images_ds.element_spec)
        )
        ds = ds.repeat().take(len(images_ds)) # trick to set dataset length
        return ds

    def _create_noisy_image(self, image: np.ndarray) -> np.ndarray:
        image = image[:, :, 0] # drop channel dim

        noisy_image = image.copy()

        noise_dim = 64
        noise = self._nprnd.rand(noise_dim, noise_dim)

        offset_x = 0
        offset_y = 0

        # some GPU shader-like code
        sz = 32
        x, y = np.indices(image.shape)
        uv_x = ((x + offset_x) // sz) % noise_dim
        uv_y = ((y + offset_y) // sz) % noise_dim

        noise_over_image = noise[uv_x, uv_y]

        noisy_image[noise_over_image > 0.75] = 0.0

        # add random mask mask
        # noise = nprnd.rand(*image.shape) * 2.0 - 1.0
        #noisy_image[mask] = 0.0

        # randomly holdout rectangles


        return noisy_image[:, :, np.newaxis] # reintroduce channel dim
