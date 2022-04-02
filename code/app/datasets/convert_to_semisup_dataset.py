import tensorflow as tf


def convert_to_semisup_dataset(
    ds: tf.data.Dataset,
    from_labeled=True
) -> tf.data.Dataset:
    """
    Takes an input dataset in the tuple(x, y) format and converts it to
    the semisupervised tuple(tuple(lab_x, unlab_x), tuple(lab_y, unlab_y)) format.
    The non-provided input dataset items will be populated by empty tensors.
    """
    spec_x, spec_y = ds.element_spec

    def _empty_tensor_from(spec):
        shape = list(spec.shape)
        for i in range(len(shape)):
            if i == 0 or shape[i] is None:
                shape[i] = 0
        return tf.constant(0, dtype=spec.dtype, shape=shape)

    if from_labeled:
        return ds.map(lambda x, y: (
            (x, _empty_tensor_from(spec_x)), (y, _empty_tensor_from(spec_y))
        ))
    else:
        return ds.map(lambda x, y: (
            (_empty_tensor_from(spec_x), x), (_empty_tensor_from(spec_y), y)
        ))


if __name__ == "__main__":
    labeled_x = tf.data.Dataset.from_tensor_slices([10, 20, 30, 40])
    labeled_y = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4])
    ds_labeled = tf.data.Dataset.zip((labeled_x, labeled_y))

    ds = convert_to_semisup_dataset(ds_labeled, from_labeled=False)

    for item in ds.as_numpy_iterator():
        print(item)
