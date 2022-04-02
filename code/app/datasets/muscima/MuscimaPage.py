from typing import Tuple, List, TypeVar
import magic
import os
import mung
import mung.io
import re
import tensorflow as tf
import numpy as np

from ..constants import *


class MuscimaPage:
    """Represents a specific page of the (CVC-)MUSCIMA(++) dataset"""
    
    Self = TypeVar("Self", bound="MuscimaPage")

    def __init__(self, writer: int, page: int):
        self.writer = writer
        self.page = page

    def __iter__(self):
        yield self.writer
        yield self.page

    def __eq__(self, other):
        return isinstance(other, self.__class__) and \
            other.writer == self.writer and \
            other.page == self.page

    def __hash__(self):
        return hash(str(self.writer) + "-" + str(self.page))

    def __lt__(self, other):
        if self.writer < other.writer:
            return True
        elif self.writer == other.writer:
            if self.page < other.page:
                return True
        return False

    def __repr__(self):
        return "MuscimaPage(w{}, p{})".format(self.writer, self.page)

    @staticmethod
    def from_xml_filename(filename: str) -> Self:
        m = re.match(
            "^CVC-MUSCIMA_W-(\\d+)_N-(\\d+)_D-ideal(\\.xml)?$",
            filename
        )
        return MuscimaPage(int(m.group(1)), int(m.group(2)))

    def ideal_image_path(self) -> str:
        """Returns absolute path to the CVC-MUSCIMA ideal image"""
        return os.path.join(
            CVCMUSCIMA_IDEAL,
            "w-{:02d}/image/p{:03d}.png".format(self.writer, self.page)
        )

    def nodes_file_path(self) -> str:
        """Returns absolute path to the MUSCIMA++ 2.0 nodes xml file"""
        filename = "CVC-MUSCIMA_W-{:02d}_N-{:02d}_D-ideal.xml" \
            .format(self.writer, self.page)
        return os.path.join(MUSCIMAPP_ANNOTATIONS, filename)

    def load_nodes(self) -> List[mung.io.Node]:
        """Loads the MUSCIMA++ nodes list for the page"""
        return mung.io.read_nodes_from_file(self.nodes_file_path())

    def load_ideal_image_as_numpy(self) -> np.ndarray:
        """Loads the CVC-MUSCIMA ideal image as a numpy array"""
        path = self.ideal_image_path()
        data = tf.io.read_file(path)
        img = tf.io.decode_png(data)
        return img.numpy()

    def dimensions_via_magic(self) -> Tuple[int, int]:
        """Returns image dimensions gathered via the magic module"""
        t = magic.from_file(self.ideal_image_path())
        w, h = re.search('(\d+) x (\d+)', t).groups()
        return int(w), int(h)
