import tensorflow as tf
from .DsMetadata import DsMetadata


def transform_ds_pages_to_images(meta: DsMetadata):
    def _load_img(path):
        data = tf.io.read_file(path)
        img = tf.io.decode_png(data)
        img = img[:, :, 0][:, :, tf.newaxis] # set channel dimension to 1
        normalized = tf.cast(img, dtype=tf.float32) / 255.0
        inverted = 1.0 - normalized
        return inverted

    def _the_transformation(pages_ds):
        def _the_generator():
            for p in pages_ds.as_numpy_iterator():
                yield meta.page_index_to_image_path(p)

        ds = tf.data.Dataset.from_generator(
            _the_generator,
            output_signature= tf.TensorSpec(shape=[], dtype=tf.string)
        )
        ds = ds.repeat().take(len(pages_ds)) # trick to set dataset length
        ds = ds.map(_load_img, num_parallel_calls=tf.data.AUTOTUNE)
        return ds
    
    return _the_transformation
