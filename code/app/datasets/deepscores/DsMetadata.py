import json
import os
from typing import Tuple, List
from ..constants import *
from .SegmentationColors import SegmentationColors


class DsMetadata:
    def __init__(self, json_path: str):
        with open(json_path, "r") as f:
            self.ds = json.load(f)

    @staticmethod
    def from_test_set():
        return DsMetadata(DEEPSCORES_TEST_ANNOTATIONS)

    @staticmethod
    def from_train_set():
        return DsMetadata(DEEPSCORES_TRAIN_ANNOTATIONS)

    def page_index_to_image_path(self, page_index: int):
        filename = self.ds["images"][page_index]["filename"]
        return os.path.join(DEEPSCORES_PATH, "images", filename)

    def page_index_to_segmentation_path(self, page_index: int):
        filename = self.ds["images"][page_index]["filename"]
        filename = filename[:-len(".png")]
        filename += "_seg.png"
        return os.path.join(DEEPSCORES_PATH, "segmentation", filename)

    def page_index_to_size_wh(self, page_index: int) -> Tuple[int, int]:
        w = self.ds["images"][page_index]["width"]
        h = self.ds["images"][page_index]["height"]
        return w, h

    def classes_to_colors_rgb(self, classes: List[str]) -> List[Tuple[int, int, int]]:
        color_indices = [
            c["color"] for c in self.ds["categories"].values()
            if c["annotation_set"] == "deepscores" and c["name"] in classes
        ]
        colors_rgb = [SegmentationColors.RGB[i] for i in color_indices]
        return colors_rgb

    def page_count(self) -> int:
        return len(self.ds["images"])
