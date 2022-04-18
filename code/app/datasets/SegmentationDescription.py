from collections import OrderedDict
from typing import List, TypeVar


class _Channel:
    def __init__(
        self,
        name: str,
        mung_classes: List[str],
        deepscores_classes: List[str],
        oversample: bool
    ):
        self.name = name
        self.mung_classes = mung_classes
        self.deepscores_classes = deepscores_classes
        self.oversample = oversample


class SegmentationDescription:
    """Describes a segmentation mask structure (mung classes for each mask channel)"""

    Self = TypeVar("Self", bound="SegmentationDescription")

    def __init__(self):
        self._channels = OrderedDict()

    def add_channel(
        self,
        name: str,
        mung_classes: List[str],
        deepscores_classes: List[str],
        oversample: bool
    ) -> Self:
        """Fluent segmentation description builder method"""
        self._channels[name] = _Channel(
            name, mung_classes, deepscores_classes, oversample
        )
        return self

    def channel_names(self) -> List[str]:
        return list(self._channels.keys())

    def channel_mung_classes(self, channel_name: str):
        return self._channels[channel_name].mung_classes

    def channel_deepscores_classes(self, channel_name: str):
        return self._channels[channel_name].deepscores_classes

    def oversampled_channel_indices(self) -> List[int]:
        return list(
            i for i, channel in enumerate(self._channels.values())
            if channel.oversample
        )

    @staticmethod
    def from_name(name) -> Self:
        if name == "notehead":
            return SegmentationDescription.NOTEHEADS
        if name == "stem":
            return SegmentationDescription.STEM
        if name == "staffline":
            return SegmentationDescription.STAFFLINE
        if name == "beam":
            return SegmentationDescription.BEAM
        if name == "flag":
            return SegmentationDescription.FLAGS
        raise Exception("Unknown symbol class: " + name)

SegmentationDescription.NOTEHEADS = SegmentationDescription() \
    .add_channel("noteheads",
        mung_classes=[
            "noteheadFull", "noteheadHalf", "noteheadWhole",
            "noteheadFullSmall", "noteheadHalfSmall"
        ],
        deepscores_classes=[
            "noteheadBlackOnLine", "noteheadBlackOnLineSmall",
            "noteheadBlackInSpace", "noteheadBlackInSpaceSmall",
            "noteheadHalfOnLine", "noteheadHalfOnLineSmall",
            "noteheadHalfInSpace", "noteheadHalfInSpaceSmall",
            "noteheadWholeOnLine", "noteheadWholeOnLineSmall",
            "noteheadWholeInSpace", "noteheadWholeInSpaceSmall",
            "noteheadDoubleWholeOnLine", "noteheadDoubleWholeOnLineSmall",
            "noteheadDoubleWholeInSpace", "noteheadDoubleWholeInSpaceSmall"
        ],
        oversample=True
    )

SegmentationDescription.STEM = SegmentationDescription() \
    .add_channel("stem",
        mung_classes=["stem"],
        deepscores_classes=["stem"],
        oversample=True
    )

SegmentationDescription.STAFFLINE = SegmentationDescription() \
    .add_channel("staffLine",
        mung_classes=["staffLine"],
        deepscores_classes=["staff"],
        oversample=True
    )

SegmentationDescription.BEAM = SegmentationDescription() \
    .add_channel("beam",
        mung_classes=["beam"],
        deepscores_classes=["beam"],
        oversample=True
    )

SegmentationDescription.FLAGS = SegmentationDescription() \
    .add_channel("flags",
        mung_classes=[
            "flag8thUp", "flag8thDown", "flag16thUp", "flag16thDown",
            "flag32thUp", "flag32thDown", "flag64thUp", "flag64thDown",
        ],
        deepscores_classes=[
            "flag8thUp", "flag8thUpSmall", "flag16thUp", "flag32ndUp",
            "flag64thUp", "flag128thUp", "flag8thDown", "flag8thDownSmall",
            "flag16thDown", "flag32ndDown", "flag64thDown", "flag128thDown"
        ],
        oversample=True
    )


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
