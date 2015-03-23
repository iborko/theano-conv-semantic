import random
import logging
import os
from itertools import imap
from file_helper import get_file_list, load_sample

logger = logging.getLogger(__name__)

# image extension
IMAGE_EXT = 'bmp'
# image segmentation extension
SEGM_IMAGE_EXT = 'bmp'
# images folder
IMAGE_FOLDER = 'Images'
# image segmentations folder
SEGM_IMAGE_FOLDER = 'GroundTruth'


def join_img_segm_paths(image_names, segm_names):
    """
    Checks if a image path has a belonging path of a segmented image.
    Returns a list of tuples (image_path, segm_path).
    """
    joined_names = []
    for image_name in image_names:
        segm_path = filter(
            lambda x: x.startswith(image_name.split('.')[0]),
            segm_names)
        if len(segm_path) == 0:
            logger.warning("Image %s has no it's segmented pair, "
                           "skipping", image_name)
        joined_names.append((image_name, segm_path[0]))
    return joined_names


def load_dataset(path, shuffle=False):
    """
    Returns a list of Sample objects, parsed dataset.
    """

    image_names = get_file_list(os.path.join(path, IMAGE_FOLDER),
                                IMAGE_EXT)
    segm_names = get_file_list(os.path.join(path, SEGM_IMAGE_FOLDER),
                               SEGM_IMAGE_EXT)

    joined_filenames = join_img_segm_paths(image_names, segm_names)
    if shuffle:
        random.seed(23455)
        random.shuffle(joined_filenames)

    return imap(lambda x: load_sample(
        os.path.join(path, IMAGE_FOLDER), x[0],
        os.path.join(path, SEGM_IMAGE_FOLDER), x[1]
    ),
        joined_filenames
    )
