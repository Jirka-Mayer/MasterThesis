import random
import tensorflow as tf


def transform_shuffle(seed: int):
    """
    Shuffles an entire dataset (because .shuffle depends also on the tf seed
    and so cannot be used for deterministic dataset splitting)
    """
    rnd = random.Random(seed)

    def _the_transformation(input_ds):
        items = [i for i in input_ds]
        rnd.shuffle(items)
        return tf.data.Dataset.from_tensor_slices(items)
    
    return _the_transformation
