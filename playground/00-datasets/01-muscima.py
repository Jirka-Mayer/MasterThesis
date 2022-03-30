import os
# os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2") # Report only TF errors by default

from typing import Tuple, List, TypeVar, Sequence, Dict
import pathlib
import re
import random
import magic

import mung
import mung.io
import numpy as np
import tensorflow as tf


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


def _construct_muscima_page_mask(page: MuscimaPage, node_classes: List[str]) -> np.ndarray:
    image = page.load_ideal_image_as_numpy()
    nodes = page.load_nodes()

    mask = np.zeros(shape=(image.shape[0], image.shape[1]), dtype=np.float32)
    for node in nodes:
        if node.class_name in node_classes:
            __print_mask_into_image(mask, node)
    return mask


def __print_mask_into_image(image: np.ndarray, node: mung.io.Node):
    yf = node.top
    yt = yf + node.height
    xf = node.left
    xt = xf + node.width
    image[yf:yt, xf:xt] = 1 - (1 - node.mask) * (1 - image[yf:yt, xf:xt]) # fuzzy OR


def _sample_tile_from_image(
    images: Dict[str, np.ndarray],
    tile_size_wh: Tuple[int, int],
    rnd: random.Random,
    nonempty_classnames: List[str]
) -> Dict[str, np.ndarray]:
    w, h = tile_size_wh
    for attempt in range(5): # resampling attempts
        xf = rnd.randint(0, images["image"].shape[1] - w - 1)
        xt = xf + w
        yf = rnd.randint(0, images["image"].shape[0] - h - 1)
        yt = yf + h

        mask_tiles = {
            k: images[k][yf:yt, xf:xt, :]
            for k in images.keys()
        }

        retry = False
        for mask_name in nonempty_classnames:
            if np.all(mask_tiles[mask_name] < 0.1):
                retry = True

        if not retry:
            break
    
    return mask_tiles


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
        mask = mask[:, :, np.newaxis] # add channel dimension
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


def tile_count_dataset(
    page_list: MuscimaPageList,
    tile_size_wh: Tuple[int, int]
) -> tf.data.Dataset:
    """Creates a dataset of int32 with number of tiles for each page"""
    w, h = tile_size_wh
    tile_pixels = w * h

    def _dim_to_tiles(w, h):
        return w * h // tile_pixels

    data = [_dim_to_tiles(*p.dimensions_via_magic()) for p in page_list]
    return tf.data.Dataset.from_tensor_slices(data)


def sample_tiles_from(
    images_ds: tf.data.Dataset,
    tile_count_ds: tf.data.Dataset,
    tile_size_wh: Tuple[int, int],
    rnd: random.Random,
    nonempty_classnames: List[str]
) -> tf.data.Dataset:
    assert type(images_ds.element_spec) is dict
    images_keys = list(sorted(images_ds.element_spec.keys()))
    tiles_count = tile_count_ds \
        .reduce(np.int32(0), lambda state, item: state + item) \
        .numpy()

    def _generate_tiles():
        for image_tensor, tile_count in zip(images_ds, tile_count_ds):
            images = {
                k: image_tensor[k].numpy()
                for k in images_keys
            }
            for _ in range(tile_count.numpy()):
                tile = _sample_tile_from_image(
                    images, tile_size_wh, rnd, nonempty_classnames
                )
                tile_tensor = {
                    k: tf.constant(tile[k], dtype=tf.float32)
                    for k in images_keys
                }
                yield tile_tensor
    
    ds = tf.data.Dataset.from_generator(
        _generate_tiles,
        output_signature={
            k: tf.TensorSpec(shape=[None, None, None], dtype=tf.float32)
            for k in images_keys
        }
    )
    ds = ds.repeat().take(tiles_count) # trick to set dataset length
    return ds


def resize_images(
    input_ds: tf.data.Dataset,
    scale_factor: float,
    method: tf.image.ResizeMethod
) -> tf.data.Dataset:
    """Resizes images of given dataset (assumes dict to be a data item)"""
    def _resize_img(item):
        return {
            k: tf.image.resize(
                images=item[k][tf.newaxis, :, :, :],
                size=tf.cast(
                    tf.cast(tf.shape(item[k])[0:2], tf.float32) * scale_factor,
                    tf.int32
                ),
                method=method
            )[0, :, :, :]
            for k in item.keys()
        }
    
    return input_ds.map(_resize_img, num_parallel_calls=tf.data.AUTOTUNE)


def _create_noisy_image(image: np.ndarray, rnd: random.Random) -> np.ndarray:
    noisy_image = image.copy()

    nprnd = np.random.RandomState(seed=rnd.randint(0, 1000000))

    # add random noise mask
    noise = nprnd.rand(*image.shape) * 2.0 - 1.0
    noisy_image += noise

    # randomly holdout rectangles


    return noisy_image


def add_noisy_images(input_ds: tf.data.Dataset, rnd: random.Random) -> tf.data.Dataset:
    assert type(input_ds.element_spec) is dict
    images_keys = list(sorted(input_ds.element_spec.keys()))
    dataset_length = len(input_ds)

    def _generate_items():
        for item in input_ds:
            image = item["image"].numpy()
            noisy_image = _create_noisy_image(image, rnd)

            item_out = {k: item[k] for k in item.keys()}
            item_out["image_noisy"] = tf.constant(noisy_image, dtype=tf.float32)
            yield item_out
    
    ds = tf.data.Dataset.from_generator(
        _generate_items,
        output_signature={
            k: tf.TensorSpec(shape=[None, None, None], dtype=tf.float32)
            for k in images_keys + ["image_noisy"]
        }
    )
    ds = ds.repeat().take(dataset_length) # trick to set dataset length
    return ds


#########################
# Main (debugging code) #
#########################


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import tqdm
    
    page_list = MuscimaPageList.get_independent_train_set()
    masks = {
        "noteheads": [
            "noteheadFull", "noteheadHalf", "noteheadWhole",
            "noteheadFullSmall", "noteheadHalfSmall"
        ]
    }
    nonempty_classnames = ["noteheads"]
    rnd = random.Random(42)
    tile_size_wh = (512, 256)

    ds = muscimapp_images_with_masks(page_list, masks)
    ds = sample_tiles_from(
        ds,
        tile_count_dataset(page_list, tile_size_wh),
        tile_size_wh,
        rnd,
        nonempty_classnames
    )
    ds = add_noisy_images(ds, rnd)
    ds = resize_images(ds, 0.25, tf.image.ResizeMethod.AREA)

    # Debugging loop code:
    for img in ds.take(5):
        # print(img["image"])
        plt.imshow(img["image_noisy"])
        plt.show()
        # print(img["image"].shape)
    exit()

    # Dummy computation to time dataset iteration:
    # 40s without anything (all times are without image resizing)
    # 1:40 building cache
    # 26s pulling from cache
    # 1:10 building snapshot
    # 16s with snapshot
    # 4s with snapshot if the file gets cached by the OS in RAM on subsequent runs
    # SNAPSHOT:
    ds = ds.snapshot(
        path=os.path.expanduser("~/Datasets/Cache/01-muscima-debug"),
        compression=None # important for performance
    )
    # CACHE:
    # ds = ds.cache(os.path.expanduser("~/Datasets/Cache/01-muscima-debug"))
    for _ in range(5): # multiple runs
        total_pixels = 0
        for img in tqdm.tqdm(ds, total=len(ds)):
            w, h, _ = img["image"].shape
            total_pixels += w * h
        print("Total pixels:", total_pixels)
