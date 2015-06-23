import random
import logging
import os
from itertools import imap
from file_helper import get_file_list, load_sample_rgbd

logger = logging.getLogger(__name__)

# image extension
IMAGE_EXT = 'png'
# image segmentation extension
SEGM_IMAGE_EXT = 'png'
# images folder
IMAGE_FOLDER = 'RGB/left'
IMAGE_D_FOLDER = 'RGB/depth/spsstereo/depth_norm_hist'
# image segmentations folder
SEGM_IMAGE_FOLDER = 'GT'


def join_img_segm_paths(image_names, d_names, segm_names):
    """
    Checks if a image path has a belonging path of a segmented image.
    Returns a list of tuples (image_path, segm_path).
    """
    joined_names = []
    for image_name in image_names:
        only_img_name = image_name.split('.')[0]
        segm_path = filter(
            lambda x: x.startswith(only_img_name),
            segm_names)
        if len(segm_path) == 0:
            logger.warning("Image %s has no it's segmented pair, "
                           "skipping", image_name)

        d_path = filter(
            lambda x: x.startswith(only_img_name),
            d_names)
        if len(d_path) == 0:
            logger.warning("image %s has no it's depth pair, "
                           "skipping", image_name)

        joined_names.append((image_name, d_path[0], segm_path[0]))
    return joined_names


def load_dataset(path, shuffle=False):
    """
    returns a list of sample objects, parsed dataset.
    """

    image_names = get_file_list(os.path.join(path, IMAGE_FOLDER),
                                IMAGE_EXT)
    d_names = get_file_list(os.path.join(path, IMAGE_D_FOLDER),
                            IMAGE_EXT)
    segm_names = get_file_list(os.path.join(path, SEGM_IMAGE_FOLDER),
                               SEGM_IMAGE_EXT)

    joined_filenames = join_img_segm_paths(image_names, d_names, segm_names)
    if shuffle:
        random.seed(23456)
        random.shuffle(joined_filenames)

    return imap(lambda x: load_sample_rgbd(
        os.path.join(path, IMAGE_FOLDER), x[0],
        os.path.join(path, IMAGE_D_FOLDER), x[1],
        os.path.join(path, SEGM_IMAGE_FOLDER), x[2]
    ),
        joined_filenames
    )
