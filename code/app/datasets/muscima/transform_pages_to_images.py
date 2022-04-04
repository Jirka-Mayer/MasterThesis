import tensorflow as tf
from .MuscimaPage import MuscimaPage
from .transform_pages_to_image_paths import transform_pages_to_image_paths


def transform_pages_to_images():
    def _load_img(path):
        data = tf.io.read_file(path)
        img = tf.io.decode_png(data)
        img = img[:, :, 0][:, :, tf.newaxis] # set channel dimension to 1
        normalized = tf.cast(img, dtype=tf.float32) / 255.0
        return normalized

    def _the_transformation(pages_ds):
        return pages_ds \
            .apply(transform_pages_to_image_paths()) \
            .map(_load_img, num_parallel_calls=tf.data.AUTOTUNE)
    
    return _the_transformation
