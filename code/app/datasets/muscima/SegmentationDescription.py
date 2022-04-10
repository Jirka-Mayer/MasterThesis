from collections import OrderedDict
from typing import List, TypeVar


class SegmentationDescription:
    """Describes a segmentation mask structure (mung classes for each mask channel)"""

    Self = TypeVar("Self", bound="SegmentationDescription")

    def __init__(self):
        self._channels = OrderedDict()
        self._nonempty_channel_names = set()

    def add_channel(self, name: str, mung_classes: List[str], nonempty=False) -> Self:
        """Fluent segmentation description builder method"""
        self._channels[name] = mung_classes
        if nonempty:
            self._nonempty_channel_names.add(name)
        return self

    def channel_names(self) -> List[str]:
        return list(self._channels.keys())

    def channel_mung_classes(self, channel_name: str):
        return self._channels[channel_name]

    def nonempty_channels(self):
        return set(
            i for i, channel_name in
            enumerate(self._channels.keys())
            if channel_name in self._nonempty_channel_names
        )

SegmentationDescription.NOTEHEADS = SegmentationDescription() \
    .add_channel("noteheads", [
        "noteheadFull", "noteheadHalf", "noteheadWhole",
        "noteheadFullSmall", "noteheadHalfSmall"
    ], nonempty=True)

SegmentationDescription.STEM = SegmentationDescription() \
    .add_channel("stem", ["stem"], nonempty=True)

SegmentationDescription.BEAM = SegmentationDescription() \
    .add_channel("beam", ["beam"], nonempty=True)

SegmentationDescription.FLAGS = SegmentationDescription() \
    .add_channel("flags", [
        "flag8thUp", "flag8thDown", "flag16thUp", "flag16thDown",
        "flag32thUp", "flag32thDown", "flag64thUp", "flag64thDown",
    ], nonempty=True)


# 16th_rest
# 8th_rest
# quarter_rest
# half_rest
# whole_rest

# c-clef
# f-clef
# g-clef

# barlines
# beam
# stem
# stafflines

# flags
# ledger_line
# noteheads

# duration-dot
# double_sharp
# flat
# natural
# sharp
