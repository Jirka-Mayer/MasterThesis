import tensorflow as tf
import numpy as np


class _HelperDatasetIterator:
    def __init__(self, ds):
        self._gen = ds.as_numpy_iterator()
        self._prepulled_item = None
        self.has_more = True
        self.read_items = -1
        self.total_items = len(ds)
        self.read_item()

    def read_item(self):
        old_prepulled_item = self._prepulled_item
        
        try:
            self._prepulled_item = next(self._gen)
        except StopIteration:
            self._prepulled_item = None
            self.has_more = False
        
        self.read_items += 1
        return old_prepulled_item

    @property
    def progress(self):
        return self.read_items / self.total_items


def create_semisup_dataset(
    batch_size: int,
    ds_labeled: tf.data.Dataset,
    ds_unlabeled: tf.data.Dataset
):
    """
    Takes one labeled (supervised) and one unlabelled (unsupervised) dataset
    and combines them into one batched, semisupervised dataset. Both input
    datasets have to be in the tuple(x, y) format. The output dataset is
    in the tuple(tuple(lab_x, unlab_x), tuple(lab_y, unlab_y)) format.
    """
    spec_labeled_x, spec_labeled_y = ds_labeled.element_spec
    spec_unlabeled_x, spec_unlabeled_y = ds_unlabeled.element_spec

    def _batchify_spec(item_spec: tf.TensorSpec) -> tf.TensorSpec:
        return tf.TensorSpec(
            shape=[None] + item_spec.shape,
            dtype=item_spec.dtype
        )

    def _empty_batch_for(item_spec: tf.TensorSpec) -> tf.TensorSpec:
        return np.ndarray(
            shape=[0] + item_spec.shape,
            dtype=item_spec.dtype.as_numpy_dtype
        )

    batch_spec_labeled_x = _batchify_spec(spec_labeled_x)
    batch_spec_labeled_y = _batchify_spec(spec_labeled_y)
    batch_spec_unlabeled_x = _batchify_spec(spec_unlabeled_x)
    batch_spec_unlabeled_y = _batchify_spec(spec_unlabeled_y)

    def _the_generator():
        def _interleave_datasets():
            iter_labeled = _HelperDatasetIterator(ds_labeled)
            iter_unlabeled = _HelperDatasetIterator(ds_unlabeled)
            while iter_labeled.has_more and iter_unlabeled.has_more:
                if iter_labeled.progress < iter_unlabeled.progress:
                    yield iter_labeled.read_item(), True
                else:
                    yield iter_unlabeled.read_item(), False
            while iter_labeled.has_more:
                yield iter_labeled.read_item(), True
            while iter_unlabeled.has_more:
                yield iter_unlabeled.read_item(), True

        labeled_items = []
        unlabeled_items = []
        batched_items = 0

        def _build_batch():
            l = labeled_items
            u = unlabeled_items
            lx = np.stack([x for x, y in labeled_items]) \
                if len(l) else _empty_batch_for(spec_labeled_x)
            ly = np.stack([y for x, y in labeled_items]) \
                if len(l) else _empty_batch_for(spec_labeled_y)
            ux = np.stack([x for x, y in unlabeled_items]) \
                if len(u) else _empty_batch_for(spec_unlabeled_x)
            uy = np.stack([y for x, y in unlabeled_items]) \
                if len(u) else _empty_batch_for(spec_unlabeled_y)
            return ((lx, ux), (ly, uy))
        
        for item, is_labeled in _interleave_datasets():
            if is_labeled:
                labeled_items.append(item)
            else:
                unlabeled_items.append(item)
            batched_items += 1
            if batched_items >= batch_size:
                yield _build_batch()
                batched_items = 0
                labeled_items.clear()
                unlabeled_items.clear()
        
        if batched_items > 0:
            yield _build_batch()

    ds = tf.data.Dataset.from_generator(
        _the_generator,
        output_signature=(
            (batch_spec_labeled_x, batch_spec_unlabeled_x),
            (batch_spec_labeled_y, batch_spec_unlabeled_y)
        )
    )

    total_items = len(ds_labeled) + len(ds_unlabeled)
    total_batches = (total_items + batch_size - 1) // batch_size
    ds = ds.repeat().take(total_batches) # trick to set dataset length

    return ds


if __name__ == "__main__":
    # code that tests correctness (well "tests" by eyeballing)
    labeled_x = tf.data.Dataset.from_tensor_slices(["1", "2", "3", "4"])
    labeled_y = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4])

    unlabeled_x = tf.data.Dataset.from_tensor_slices(["u1", "u2", "u3"])
    unlabeled_y = tf.data.Dataset.from_tensor_slices([-1, -2, -3])

    ds_labeled = tf.data.Dataset.zip((labeled_x, labeled_y))
    ds_unlabeled = tf.data.Dataset.zip((unlabeled_x, unlabeled_y))

    ds = create_semisup_dataset(3, ds_labeled, ds_unlabeled)

    for item in ds.as_numpy_iterator():
        print(item)
