from typing import Optional
import tensorflow as tf


def transform_resize_images(
    scale_factor: Optional[float],
    method: tf.image.ResizeMethod
):
    def _compute_new_static_shape(input_shape, scale_factor):
        if len(input_shape) == 3:
            return [
                int(float(input_shape[0]) * scale_factor) if input_shape[0] is not None else None,
                int(float(input_shape[1]) * scale_factor) if input_shape[1] is not None else None,
                input_shape[2]
            ]
        else:
            return [
                input_shape[0],
                int(float(input_shape[1]) * scale_factor) if input_shape[1] is not None else None,
                int(float(input_shape[2]) * scale_factor) if input_shape[2] is not None else None,
                input_shape[3]
            ]

    def _resize_img(*args):
        if len(args) == 1:
            item = args[0]
        else:
            item = args
        
        if type(item) is tuple:
            return tuple(_resize_img(i) for i in item)

        new_size_3d = tf.cast(
            tf.cast(tf.shape(item)[0:2], tf.float32) * scale_factor,
            tf.int32
        )
        new_size_4d = tf.cast(
            tf.cast(tf.shape(item)[1:3], tf.float32) * scale_factor,
            tf.int32
        )
        new_size = tf.cond(
            tf.rank(item) == 3,
            lambda: new_size_3d,
            lambda: new_size_4d
        )

        result = tf.image.resize(
            images=item,
            size=new_size,
            method=method
        )
        result.set_shape(_compute_new_static_shape(item.shape, scale_factor))
        return result

    def _the_transformation(input_ds):
        if scale_factor is None or scale_factor == 1.0:
            return input_ds

        return input_ds.map(_resize_img, num_parallel_calls=tf.data.AUTOTUNE)
    
    return _the_transformation
