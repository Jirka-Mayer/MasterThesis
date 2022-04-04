import shutil
from typing import Dict, List, Tuple
import os
import tensorflow as tf


class DatasetFeeder:
    """
    Helper class that performs the last steps of feeding a dataset into a
    training pipeline (handles caching and prefetch)
    """
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        self.known_datasets: List[Tuple[tf.data.Dataset, str]] = []

    def __enter__(self):
        self._bust_cache()
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self._bust_cache()
        return False

    def __call__(self, ds: tf.data.Dataset) -> tf.data.Dataset:
        name = self._get_dataset_name(ds)
        cache_path = os.path.join(self.cache_dir, name)
        
        os.makedirs(cache_path, exist_ok=True)

        return ds \
            .snapshot(path=cache_path, compression=None) \
            .prefetch(tf.data.AUTOTUNE)

    def _bust_cache(self):
        shutil.rmtree(self.cache_dir, ignore_errors=True)

    def _get_dataset_name(self, ds: tf.data.Dataset):
        for d, n in self.known_datasets:
            if d is ds:
                return n
        
        new_name = "ds-{:02d}".format(len(self.known_datasets))
        self.known_datasets.append((ds, new_name))
        return new_name
