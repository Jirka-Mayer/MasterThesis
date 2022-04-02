from typing import Tuple, Sequence, TypeVar
import pathlib
import random

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
