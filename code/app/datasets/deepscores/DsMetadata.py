import json
import os
from typing import Tuple, List
from ..constants import *
from .SegmentationColors import SegmentationColors


class DsMetadata:
    def __init__(self, json_path: str, verbose=False):
        if verbose:
            print("Loading '{}' ...".format(json_path))

        with open(json_path, "r") as f:
            ds = json.load(f)
            
        self._categories = ds["categories"]
        self._image_sizes_wh = [
            (img["width"], img["height"]) for img in ds["images"]
        ]
        self._image_filenames = [
            img["filename"] for img in ds["images"]
        ]

        del ds # free up memory

        if verbose:
            print("Done.")

    @staticmethod
    def from_test_set(verbose=False):
        return DsMetadata(DEEPSCORES_TEST_ANNOTATIONS, verbose)

    @staticmethod
    def from_train_set(verbose=False):
        return DsMetadata(DEEPSCORES_TRAIN_ANNOTATIONS, verbose)

    def page_index_to_image_path(self, page_index: int):
        filename = self._image_filenames[page_index]

        return os.path.join(DEEPSCORES_PATH, "images", filename)

    def page_index_to_segmentation_path(self, page_index: int):
        filename = self._image_filenames[page_index]

        filename = filename[:-len(".png")]
        filename += "_seg.png"
        return os.path.join(DEEPSCORES_PATH, "segmentation", filename)

    def page_index_to_size_wh(self, page_index: int) -> Tuple[int, int]:
        return self._image_sizes_wh[page_index]

    def classes_to_colors_rgb(self, classes: List[str]) -> List[Tuple[int, int, int]]:
        color_indices = [
            c["color"] for c in self._categories.values()
            if c["annotation_set"] == "deepscores" and c["name"] in classes
        ]
        colors_rgb = [SegmentationColors.RGB[i] for i in color_indices]
        return colors_rgb

    def page_count(self) -> int:
        return len(self._image_filenames)
