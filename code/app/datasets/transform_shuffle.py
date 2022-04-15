import random
from numpy import dtype
import tensorflow as tf
import numpy as np
from typing import Dict, List, Tuple


def transform_shuffle(seed: int):
    """
    Shuffles an entire dataset (because .shuffle depends also on the tf seed
    and so cannot be used for deterministic dataset splitting)
    """
    rnd = random.Random(seed)

    def _the_transformation(input_ds):
        def _the_generator():
            items = [i for i in input_ds.as_numpy_iterator()]
            rnd.shuffle(items)
            for i in items:
                yield i

        ds = tf.data.Dataset.from_generator(
            _the_generator,
            output_signature=input_ds.element_spec
        )
        ds = ds.repeat().take(len(input_ds)) # trick to set dataset length
        return ds
    
    return _the_transformation
