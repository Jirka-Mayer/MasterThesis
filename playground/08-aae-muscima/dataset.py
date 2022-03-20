import os
import pathlib
import re
import numpy as np
import cv2
import random
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import mung
import mung.io


DATASETS_PATH = os.path.expanduser("~/Datasets")
PLAYGROUND_DATASETS_PATH = os.path.join(DATASETS_PATH, "MyPlaygroundDatasets")
DATASET_NAME = "01-notehead-segmentation"

MASKS = {
    # segmentation mask name => list of muscima classes to aggregate
    "noteheads": [
        "noteheadFull", "noteheadHalf", "noteheadWhole",
        "noteheadFullSmall", "noteheadHalfSmall"
    ]

    # NOTE: "image" mask is not a mask, but the input image itself
}
TILE_SHAPE = (128, 128)

# displays a given segmentation mask
DEBUG_MASK = None # e.g.: "noteheads"

# display tile coverage visualization
DEBUG_TILE_COVERAGE = False

MUSCIMAPP_ANNOTATIONS = os.path.join(
    DATASETS_PATH,
    "MuscimaPlusPlus/v2.0/data/annotations"
)
MUSCIMAPP_TESTSET_INDEP = os.path.join(
    DATASETS_PATH,
    "MuscimaPlusPlus/v2.0/specifications/testset-independent.txt"
)
MUSCIMAPP_TESTSET_DEP = os.path.join(
    DATASETS_PATH,
    "MuscimaPlusPlus/v2.0/specifications/testset-dependent.txt"
)
CVCMUSCIMA_IDEAL = os.path.join(
    DATASETS_PATH,
    "CvcMuscima_StaffRemoval/CvcMuscima-Distortions/ideal"
)


def load_independent_dataset(train_split=True):
    import tensorflow as tf

    if train_split:
        dataset_path = os.path.join(
            PLAYGROUND_DATASETS_PATH, DATASET_NAME + "--train-independent"
        )
    else:
        dataset_path = os.path.join(
            PLAYGROUND_DATASETS_PATH, DATASET_NAME + "--test-independent"
        )
    
    classnames = ["image"] + list(MASKS.keys())

    def path_to_data(file_path):
        filename = tf.strings.split(file_path, os.sep)[-1]

        data = {}
        for c in classnames:
            image_filepath = tf.strings.join([dataset_path, "image", filename], os.sep)
            image = tf.io.decode_png(tf.io.read_file(image_filepath))
            normalized = tf.cast(image, dtype=tf.float32) / 255.0
            data[c] = normalized

        return data

    return tf.data.Dataset.list_files(
        os.path.join(dataset_path, "image", "*")
    ).map(path_to_data)


def build_the_dataset():
    # independent = test on handwriting never seen during training
    # dependent = same writers are both in train and test splits
    __build_dataset(
        DATASET_NAME + "--train-independent",
        __gather_filenames(is_dependent=False, is_train=True)
    )
    __build_dataset(
        DATASET_NAME + "--test-independent",
        __gather_filenames(is_dependent=False, is_train=False)
    )


def __build_dataset(name: str, filenames: List[str]):
    print("Building the '{}' dataset...".format(name))

    os.makedirs(
        os.path.join(PLAYGROUND_DATASETS_PATH, name),
        exist_ok=True
    )
    os.makedirs(
        os.path.join(PLAYGROUND_DATASETS_PATH, name, "image"),
        exist_ok=True
    )
    for mask_name in MASKS.keys():
        os.makedirs(
            os.path.join(PLAYGROUND_DATASETS_PATH, name, mask_name),
            exist_ok=True
        )

    total_tile_count = 0
    try:
        for i, f in enumerate(filenames):
            print("Processing '{}'... ({}/{})".format(f, i+1, len(filenames)))
            tile_count = __process_image(name, f)
            print("Sampled {} tiles".format(tile_count))
            total_tile_count += tile_count
    finally:
        print("Sampled {} tiles in total".format(total_tile_count))


def __process_image(dataset_name: str, filename: str):
    ### Load input data ###
    writer, page = __parse_filename(filename)
    input_image = cv2.imread(
        os.path.join(
            CVCMUSCIMA_IDEAL,
            "w-{:02d}/image/p{:03d}.png".format(writer, page)
        ),
        cv2.IMREAD_GRAYSCALE
    ) / 255 # normalize
    nodes = mung.io.read_nodes_from_file(
        os.path.join(MUSCIMAPP_ANNOTATIONS, filename)
    )

    ### Create segmentation masks ###
    mask_images = {}
    for mask_name in MASKS.keys():
        mask_image = np.zeros(shape=input_image.shape, dtype=np.float32)
        node_classes = MASKS[mask_name]

        for node in nodes:
            if node.class_name in node_classes:
                __print_mask_into_image(mask_image, node)
        
        mask_images[mask_name] = mask_image

        if DEBUG_MASK == mask_name:
            plt.imshow(
                np.concatenate([input_image, mask_image], axis=0)
            )
            plt.show()

    ### Sample training tiles ###
    tile_count = int(
        (input_image.shape[0] / TILE_SHAPE[0]) * (input_image.shape[1] / TILE_SHAPE[1])
    )
    coords_list = []
    for r in range(tile_count):
        tile_masks, coords = __sample_tile(input_image, mask_images)
        coords_list.append(coords)
        __save_tile(tile_masks, coords, writer, page, dataset_name)

    if DEBUG_TILE_COVERAGE:
        __display_tile_coverage(input_image, coords_list)

    return tile_count


def __display_tile_coverage(input_image, coords_list):
    img = np.copy(input_image)
    for xf, xt, yf, yt in coords_list:
        img[yf:yt, xf:xt] += 0.05
        img[yf:yt, xf] = 1
        img[yf:yt, xt] = 1
        img[yf, xf:xt] = 1
        img[yt, xf:xt] = 1
    plt.imshow(img)
    plt.show()


def __sample_tile(input_image, mask_images):
    for attempt in range(5): # resampling attempts
        xf = random.randint(0, input_image.shape[1] - TILE_SHAPE[1] - 1)
        xt = xf + TILE_SHAPE[1]
        yf = random.randint(0, input_image.shape[0] - TILE_SHAPE[0] - 1)
        yt = yf + TILE_SHAPE[0]
        
        coords = (xf, xt, yf, yt)

        mask_tiles = {
            mask_name: mask_images[mask_name][yf:yt, xf:xt]
            for mask_name in mask_images.keys()
        }
        mask_tiles["image"] = input_image[yf:yt, xf:xt]

        # WARNING: Currently checking for all classes!
        retry = False
        for mask_name in mask_tiles.keys():
            if np.all(mask_tiles[mask_name] < 0.1):
                retry = True

        if not retry:
            break
    
    return mask_tiles, coords


def __save_tile(tile_masks, coords, writer, page, dataset_name):
    for mask_name in tile_masks.keys():
        tile_filepath = os.path.join(
            PLAYGROUND_DATASETS_PATH,
            dataset_name,
            mask_name,
            "w{}.p{}.x{}-{}.y{}-{}.png".format(writer, page, *coords)
        )
        cv2.imwrite(
            tile_filepath,
            tile_masks[mask_name] * 255
        )


def __print_mask_into_image(image, node: mung.io.Node):
    # faster implementation (fuzzy "OR")
    yf = node.top
    yt = yf + node.height
    xf = node.left
    xt = xf + node.width
    image[yf:yt, xf:xt] = 1 - (1 - node.mask) * (1 - image[yf:yt, xf:xt])

    # slow, per-pixel implementation
    # for i in range(node.height):
    #     for j in range(node.width):
    #         if node.mask[i, j] != 0:
    #             image[node.top + i, node.left + j] = node.mask[i, j]


def __parse_filename(filename: str) -> Tuple[int, int]:
    """Turns "CVC-MUSCIMA_W-13_N-02_D-ideal.xml" into (13, 2)"""
    m = re.match("^CVC-MUSCIMA_W-(\\d+)_N-(\\d+)_D-ideal\\.xml$", filename)
    return int(m.group(1)), int(m.group(2))


def __gather_filenames(is_dependent: bool, is_train: bool) -> List[str]:
    """
    Returns a list of "CVC-MUSCIMA_W-13_N-02_D-ideal.xml" filenames.
    Writer dependent/independent train/test split has to be specified.
    """
    all_files = set([
        x for x in
        os.listdir(MUSCIMAPP_ANNOTATIONS) if x.endswith(".xml")
    ])
    independent_testset = set([
        x + ".xml" for x in
        pathlib.Path(MUSCIMAPP_TESTSET_INDEP).read_text().splitlines(keepends=False)
    ])
    dependent_testset = set([
        x + ".xml" for x in
        pathlib.Path(MUSCIMAPP_TESTSET_DEP).read_text().splitlines(keepends=False)
    ])
    if is_train:
        if is_dependent:
            return list(sorted(all_files - dependent_testset))
        else:
            return list(sorted(all_files - independent_testset))
    else:
        if is_dependent:
            return list(sorted(dependent_testset))
        else:
            return list(sorted(independent_testset))


if __name__ == "__main__":
    build_the_dataset()
