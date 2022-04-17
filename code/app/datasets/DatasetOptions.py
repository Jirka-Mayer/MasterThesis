import tensorflow as tf
from typing import Tuple, Optional, Callable
from .SegmentationDescription import SegmentationDescription


class DatasetOptions:
    def __init__(
        self,
        dataset_seed: int,
        tile_size_wh: Tuple[int, int],
        validation_pages: int,
        supervised_pages: int,
        unsupervised_pages: int,
        batch_size: int,
        segdesc: SegmentationDescription,
        unsupervised_transformation: Callable[[tf.data.Dataset], tf.data.Dataset],
        
        verbose=False
    ):
        self.dataset_seed = dataset_seed
        self.tile_size_wh = tile_size_wh
        self.validation_pages = validation_pages
        self.supervised_pages = supervised_pages
        self.unsupervised_pages = unsupervised_pages
        self.batch_size = batch_size
        self.segdesc = segdesc
        self.unsupervised_transformation = unsupervised_transformation

        self.verbose = verbose
