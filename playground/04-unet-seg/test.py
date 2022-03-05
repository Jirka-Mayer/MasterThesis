import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2") # Report only TF errors by default

import tensorflow as tf

# images 3x2 with 3 channels, batch size 2
# shape (2, 3, 2, 3)
batch = tf.constant([
    # batch item 0 - picked channel is 2 (blue)
    [
        [[0, 1, 2], [3, 4, 5]],  # [r, g, b] is a pixel
        [[6, 7, 8], [9, 0, 1]],
        [[2, 3, 4], [5, 6, 7]],
    ],

    # batch item 1 - picked channel is 1 (green)
    [
        [[8, 9, 0], [1, 2, 3]],
        [[4, 5, 6], [7, 8, 9]],
        [[0, 1, 2], [3, 4, 5]],
    ]
])

###########

@tf.function
def build_mask_inputs_slice(batch_slice):
    class_prominences = tf.math.reduce_sum(batch_slice, axis=(0, 1))
    largest_class_index = tf.math.argmax(class_prominences)
    return batch_slice[:, :, largest_class_index]

@tf.function
def build_mask_outputs_slice(batch_slice):
    class_prominences = tf.math.reduce_sum(batch_slice, axis=(0, 1))
    largest_class_index = tf.math.argmax(class_prominences)

    batch_slice_transposed = tf.transpose(batch_slice, [2, 0, 1])
    outputs_slice_transposed = tf.tensor_scatter_nd_update(
        tf.zeros(shape=batch_slice_transposed.shape, dtype=batch_slice_transposed.dtype),
        indices=[[largest_class_index]],
        updates=[batch_slice_transposed[largest_class_index]]
    )
    outputs_slice = tf.transpose(outputs_slice_transposed, [1, 2, 0])

    return outputs_slice

###########

mask_inputs = tf.map_fn(build_mask_inputs_slice, batch)
mask_outputs = tf.map_fn(build_mask_outputs_slice, batch)

# print(mask_inputs)
# print(mask_outputs)
