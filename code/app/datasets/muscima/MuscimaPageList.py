from typing import Tuple, Sequence, TypeVar
import pathlib
import random
import tensorflow as tf

from .MuscimaPage import MuscimaPage
from ..constants import *


class MuscimaPageList:
    """
    Represents a list of MuscimaPage instances
    (just a wrapper with helper methods)
    """

    Self = TypeVar("Self", bound="MuscimaPageList")

    def __init__(self, instances: Sequence[MuscimaPage]):
        self.instances = list(sorted(instances))

    def __iter__(self):
        return iter(self.instances)

    def __len__(self):
        return len(self.instances)

    def __repr__(self):
        return "MuscimaPageList(" + repr(self.instances) + ")"

    @staticmethod
    def get_muscimapp_all() -> Self:
        return MuscimaPageList([
            MuscimaPage.from_xml_filename(x) for x in
            os.listdir(MUSCIMAPP_ANNOTATIONS) if x.endswith(".xml")
        ])

    @staticmethod
    def get_independent_test_set() -> Self:
        return MuscimaPageList([
            MuscimaPage.from_xml_filename(x + ".xml") for x in
            pathlib.Path(MUSCIMAPP_TESTSET_INDEP) \
                .read_text() \
                .splitlines(keepends=False)
        ])

    @staticmethod
    def get_independent_train_set() -> Self:
        all = set(MuscimaPageList.get_muscimapp_all())
        test = set(MuscimaPageList.get_independent_test_set())
        train = all - test
        return MuscimaPageList(train)

    def filter_writers(self, writers: Sequence[int]) -> Self:
        return MuscimaPageList(
            filter(lambda i: i.writer in writers, self.instances)
        )

    def split(self, ratio=0.5, seed=None) -> Tuple[Self, Self]:
        """Split the list into two halves by writers"""
        writers = list(set([x.writer for x in self.instances]))
        
        rnd = random.Random(seed)
        rnd.shuffle(writers)

        split_index = int(len(writers) * ratio)
        first_writers = writers[:split_index]
        second_writers = writers[split_index:]

        return self.filter_writers(first_writers), self.filter_writers(second_writers)

    def split_validation_semisup(
        self,
        validation_ratio: float,
        labeled_ratio: float,
        unlabeled_ratio: float,
        seed: int
    ):
        """
        Splits the dataset into three parts:
        1) validation dataset, expressed by a ratio to the total dataset
        2) "labeled" dataset split as a ratio of the remainder
        3) "unlabeled" dataset in the same ratio unit as the labeled one
        Labeled and unlabeled ratios can be varied independently and cannot
        add up to more than 1.0 (the whole non-validation dataset part)
        """
        assert labeled_ratio + unlabeled_ratio <= 1.0

        validation_pages, train_pages = self.split(
            ratio=validation_ratio, seed=seed
        )
        used_pages, _ = train_pages.split(
            ratio=labeled_ratio + unlabeled_ratio, seed=seed
        )
        labeled_pages, unlabeled_pages = used_pages.split(
            ratio=labeled_ratio / (labeled_ratio + unlabeled_ratio), seed=seed
        )

        return validation_pages, labeled_pages, unlabeled_pages

    def as_tf_dataset(self):
        return tf.data.Dataset.from_tensor_slices(
            [tuple(p) for p in self.instances]
        )
