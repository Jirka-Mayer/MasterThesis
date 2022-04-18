from typing import List, Tuple, Sequence, TypeVar
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
        self.instances = list(instances)

    def __iter__(self):
        return iter(self.instances)

    def __len__(self):
        return len(self.instances)

    def __repr__(self):
        return "MuscimaPageList(" + repr(self.instances) + ")"

    @staticmethod
    def get_muscimapp_all() -> Self:
        return MuscimaPageList(sorted([
            MuscimaPage.from_xml_filename(x) for x in
            os.listdir(MUSCIMAPP_ANNOTATIONS) if x.endswith(".xml")
        ]))

    @staticmethod
    def get_independent_test_set() -> Self:
        return MuscimaPageList(sorted([
            MuscimaPage.from_xml_filename(x + ".xml") for x in
            pathlib.Path(MUSCIMAPP_TESTSET_INDEP) \
                .read_text() \
                .splitlines(keepends=False)
        ]))

    @staticmethod
    def get_independent_train_set() -> Self:
        all = set(MuscimaPageList.get_muscimapp_all())
        test = set(MuscimaPageList.get_independent_test_set())
        train = all - test
        return MuscimaPageList(sorted(train))

    @staticmethod
    def get_entire_cvc_muscima() -> Self:
        return MuscimaPageList(sorted([
            MuscimaPage(w, p) for w in range(1, 51) for p in range(1, 21)
        ]))

    def filter_writers(self, writers: Sequence[int]) -> Self:
        return MuscimaPageList(
            filter(lambda i: i.writer in writers, self.instances)
        )

    def filter_out_writers(self, writers: Sequence[int]) -> Self:
        return MuscimaPageList(
            filter(lambda i: i.writer not in writers, self.instances)
        )

    def filter_out_pages(self, pages: Sequence[MuscimaPage]) -> Self:
        return MuscimaPageList(
            filter(lambda i: i not in pages, self.instances)
        )

    def get_writers(self) -> List[int]:
        return list(set([x.writer for x in self.instances]))

    def shuffle_writer_clusters(self, seed: int) -> Self:
        writers = self.get_writers()

        rnd = random.Random(seed)
        rnd.shuffle(writers)

        output = []
        for w in writers:
            for i in self.instances:
                if i.writer == w:
                    output.append(i)
        return MuscimaPageList(output)

    def shuffle_in_place(self, seed: int):
        rnd = random.Random(seed)
        rnd.shuffle(self.instances)

    def repeat_in_place(self, count: int):
        if count == 1:
            return
        self.instances *= count

    def take(self, count: int) -> Self:
        return MuscimaPageList([
            p for i, p in enumerate(self.instances) if i < count
        ])

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
        sup_ratio: float,
        unsup_ratio: float,
        seed: int
    ):
        """
        Splits the dataset into three parts:
        1) validation dataset, expressed by a ratio to the total dataset
        2) "sup" dataset split as a ratio of the remainder
        3) "unsup" dataset in the same ratio unit as the sup one
        sup and unsup ratios can be varied independently and cannot
        add up to more than 1.0 (the whole non-validation dataset part)
        """
        assert sup_ratio + unsup_ratio <= 1.0

        validation_pages, remaining_pages = self.split(
            ratio=validation_ratio, seed=seed
        )
        sup_pages, remaining_pages = remaining_pages.split(
            ratio=sup_ratio, seed=seed
        )
        unsup_pages, _ = remaining_pages.split(
            ratio=unsup_ratio / (1 - sup_ratio), seed=seed
        )

        return validation_pages, sup_pages, unsup_pages

    def as_tf_dataset(self):
        return tf.data.Dataset.from_tensor_slices(
            [tuple(p) for p in self.instances]
        )
