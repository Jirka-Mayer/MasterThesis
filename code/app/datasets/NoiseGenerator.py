import tensorflow as tf
import numpy as np
import random
import math


class NoiseGenerator:
    def __init__(
        self,
        seed: int,
        max_noise_size: int,
        dropout_ratio: float,
        largest_tiles_only=False
    ):
        self._rnd = random.Random(seed)
        self._nprnd = np.random.RandomState(seed)
        self._max_noise_size = max_noise_size
        self._scales = []
        self._calculate_scales()
        self._dropout_ratio = dropout_ratio
        self._largest_tiles_only = largest_tiles_only

    def _calculate_scales(self):
        self._scales = []
        s = int(self._max_noise_size)
        while s >= 1:
            self._scales.append(s)
            s = s // 2

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

    def _create_noise_mask(
        self,
        height: int,
        width: int,
        noise_scale: int,
        dropout_ratio: float
    ):
        """Creates noise mask image of given size and parameters"""
        # noise positioning
        offset_x = self._rnd.randint(0, noise_scale - 1)
        offset_y = self._rnd.randint(0, noise_scale - 1)

        # random noise that is later scaled and positioned
        noise_dim = int(((width + height) / 2) / noise_scale)
        noise = self._nprnd.rand(noise_dim, noise_dim)

        # pixel-shader that converts image coordinates
        # to noise texture uv coordinates
        x, y = np.indices((height, width))
        uv_x = ((x + offset_x) // noise_scale) % noise_dim
        uv_y = ((y + offset_y) // noise_scale) % noise_dim

        # sample the noise texture with the given coordinate matrices
        sampled_noise = noise[uv_x, uv_y]

        # binarize the noise mask
        return (sampled_noise >= dropout_ratio).astype(np.float32)

    def _create_composite_noise_mask(
        self,
        height: int,
        width: int
    ):
        """Creates noise mask of two sizes overlayed"""
        scale_bigger = 32
        scale_smaller = scale_bigger // 2
        d = 1 - math.sqrt(1 - self._dropout_ratio) # inner dropout ratio
        img_bigger = self._create_noise_mask(height, width, scale_bigger, d)
        img_smaller = self._create_noise_mask(height, width, scale_smaller, d)
        return img_bigger * img_smaller

    def _create_random_noise_mask(
        self,
        height: int,
        width: int
    ):
        """Creates a noise mask with random scale"""
        if self._largest_tiles_only:
            scale = self._scales[0]
        else:
            scale = self._rnd.choice(self._scales)

        return self._create_noise_mask(height, width, scale, self._dropout_ratio), scale

    def _create_noisy_image(self, image: np.ndarray) -> np.ndarray:
        """Creates a noisy image by masking an existing image"""
        noise, _ = self._create_random_noise_mask(
            height=image.shape[0],
            width=image.shape[1]
        )
        return image * noise[:, :, np.newaxis]


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    noise = NoiseGenerator(
        seed=42,
        max_noise_size=28.75 * 2, # two muscima staff spaces
        dropout_ratio=0.25
    )

    for _ in range(20):
        img, sz = noise._create_random_noise_mask(
            height=256,
            width=512
        )
        print()
        print("actual dropout ratio:", (img < 0.5).sum() / np.prod(img.shape))
        print("size:", sz)
        plt.imshow(img)
        plt.show()
