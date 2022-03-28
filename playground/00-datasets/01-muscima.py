import os
from platform import node
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2") # Report only TF errors by default

####################################
# Dataset constants and parameters #
####################################

DATASETS_PATH = os.path.expanduser("~/Datasets")

MUSCIMAPP_ANNOTATIONS = os.path.join(
    DATASETS_PATH,
    "MuscimaPlusPlus/v2.0/data/annotations"
)
MUSCIMAPP_TESTSET_INDEP = os.path.join(
    DATASETS_PATH,
    "MuscimaPlusPlus/v2.0/specifications/testset-independent.txt"
)
CVCMUSCIMA_IDEAL = os.path.join(
    DATASETS_PATH,
    "CvcMuscima_StaffRemoval/CvcMuscima-Distortions/ideal"
)

############################

from typing import Tuple, List, TypeVar, Sequence, Dict
import pathlib
import re
import random

import mung
import mung.io
import numpy as np
import tensorflow as tf


####################
# Helper datatypes #
####################


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

    def load_ideal_image_as_numpy(self) -> np.array:
        """Loads the CVC-MUSCIMA ideal image as a numpy array"""
        path = self.ideal_image_path()
        data = tf.io.read_file(path)
        img = tf.io.decode_png(data)
        return img.numpy()


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


####################
# Helper functions #
####################


def _construct_muscima_page_mask(page: MuscimaPage, node_classes: List[str]) -> np.array:
    image = page.load_ideal_image_as_numpy()
    nodes = page.load_nodes()

    mask = np.zeros(shape=(image.shape[0], image.shape[1]), dtype=np.float32)
    for node in nodes:
        if node.class_name in node_classes:
            __print_mask_into_image(mask, node)
    return mask


def __print_mask_into_image(image: np.array, node: mung.io.Node):
    yf = node.top
    yt = yf + node.height
    xf = node.left
    xt = xf + node.width
    image[yf:yt, xf:xt] = 1 - (1 - node.mask) * (1 - image[yf:yt, xf:xt]) # fuzzy OR


############################
# Tensorflow data datasets #
############################


def muscimapp_images(page_list: MuscimaPageList) -> tf.data.Dataset:
    """
    Creates a dataset of CVC-MUSCIMA ideal images, normalized to [0.0, 1.0] range,
    with one channel dimension
    """
    def _load_img(path):
        data = tf.io.read_file(path)
        img = tf.io.decode_png(data)
        normalized = tf.cast(img, dtype=tf.float32) / 255.0
        return normalized
    
    ds = tf.data.Dataset.from_tensor_slices(
        [p.ideal_image_path() for p in page_list]
    )
    ds = ds.map(_load_img, num_parallel_calls=tf.data.AUTOTUNE)
    return ds


def muscimapp_masks(
    page_list: MuscimaPageList,
    node_classes: List[str]
) -> tf.data.Dataset:
    """
    Creates a dataset of MUSCIMA++ binary masks, normalized to [0.0, 1.0] range,
    with one channel dimension
    
    node_classes: names of MuNG node classes to combine into a single mask
    """
    def _load_mask(p_tuple: tf.Tensor):
        p = MuscimaPage(*p_tuple.numpy())
        mask = _construct_muscima_page_mask(p, node_classes)
        return tf.constant(mask, dtype=tf.float32)
    
    ds = tf.data.Dataset.from_tensor_slices(
        [tuple(p) for p in page_list]
    )
    ds = ds.map(
        lambda x: tf.py_function(_load_mask, [x], tf.float32),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    return ds


def muscimapp_images_with_masks(
    page_list: MuscimaPageList,
    masks: Dict[str, List[str]]
) -> tf.data.Dataset:
    """
    Creates a dataset of dicts, where each dict contains the primary
    image under ["image"] and masks under their corresponding mask names.
    Data is normalized to [0.0, 1.0] with one channel dimension
    """
    structure = {
        mask_name: muscimapp_masks(page_list, masks[mask_name])
        for mask_name in masks.keys()
    }
    structure["image"] = muscimapp_images(page_list)
    return tf.data.Dataset.zip(structure)


# TODO: sample tiles from the dataset


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    page_list = MuscimaPageList.get_independent_train_set()
    ds = muscimapp_images_with_masks(page_list, {
        "noteheads": [
            "noteheadFull", "noteheadHalf", "noteheadWhole",
            "noteheadFullSmall", "noteheadHalfSmall"
        ]
    })
    for i, img in enumerate(ds):
        # print(img)
        # plt.imshow(img)
        # plt.show()
        print(img["image"].shape)

    exit()

############################

import PIL.Image
from typing import Tuple, List
import re
import magic

def _parse_filename(filename: str) -> Tuple[int, int]:
    """Turns "CVC-MUSCIMA_W-13_N-02_D-ideal.xml" into (13, 2)"""
    m = re.match("^CVC-MUSCIMA_W-(\\d+)_N-(\\d+)_D-ideal\\.xml$", filename)
    return int(m.group(1)), int(m.group(2))

def foo(item):
    writer, page = _parse_filename(item)
    filepath = os.path.join(
        CVCMUSCIMA_IDEAL,
        "w-{:02d}/image/p{:03d}.png".format(writer, page)
    )
    im = PIL.Image.open(filepath)
    w, h = im.size
    return item, im.size, magic.from_file(filepath)

print("START")
all_files = set([
    foo(x) for x in
    os.listdir(MUSCIMAPP_ANNOTATIONS) if x.endswith(".xml")
])
print("DONE")

print(all_files)

exit()

###############################

import tensorflow as tf

# ds = tf.data.Dataset.range(1, 4)

def mygen():
    print("GENERATOR CALLED")
    for i in range(1, 4):
        yield i

slow_ds = tf.data.Dataset.from_generator(
    mygen,
    output_signature=tf.TensorSpec(shape=(), dtype=tf.int32)
)

# ds = tf.data.Dataset.from_tensor_slices(list(slow_ds))
os.makedirs(
    os.path.expanduser("~/Datasets/Cache"),
    exist_ok=True
)
# ds = ds.cache(filename=os.path.expanduser("~/Datasets/Cache/01-muscima"))
# ds = slow_ds.snapshot(os.path.expanduser("~/Datasets/Cache/01-muscima"))

ds = slow_ds
ds = ds.scan(0, lambda s, i: (s + 1, i + 1))

# ds = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4])

# def mymap(item):
#     return tf.data.Dataset.from_tensor_slices(
#         tf.repeat([item], 5)
#     )

# ds = ds.flat_map(mymap)

for i in ds:
    print(i.numpy())
print("========")

print(ds.cardinality().numpy())
