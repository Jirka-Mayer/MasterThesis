import tensorflow as tf
from .MuscimaPage import MuscimaPage


def transform_pages_to_image_paths():
    def _the_transformation(pages_ds):
        def _the_generator():
            for w, p in pages_ds.as_numpy_iterator():
                page = MuscimaPage(w, p)
                yield page.ideal_image_path()

        ds = tf.data.Dataset.from_generator(
            _the_generator,
            output_signature= tf.TensorSpec(shape=[], dtype=tf.string)
        )
        ds = ds.repeat().take(len(pages_ds)) # trick to set dataset length
        return ds
    
    return _the_transformation
