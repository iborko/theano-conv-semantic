import os
import logging
import cv2
from sample import Sample

logger = logging.getLogger(__name__)


def load_image_grayscale(dirname, fname):
    return cv2.imread(os.path.join(dirname, fname), cv2.IMREAD_GRAYSCALE)


def load_image_rgb(dirname, fname):
    return cv2.imread(os.path.join(dirname, fname))


def load_sample(img_dirname, img_name, segm_dirname, segm_name):
    """
    Load image and segmented image into Sample object.
    Returns sample object.

    img_dirname: string
        image directory
    img_name: string
        image filename
    segm_dirname: string
        segmented image directory
    segm_name: string
        segmented filename
    """
    image = load_image_rgb(img_dirname, img_name)
    segmented_image = load_image_rgb(segm_dirname, segm_name)
    name = img_name.split('.')[0]

    return Sample(name, image, segmented_image)


def get_file_list(path, extension):
    """ Returns list of files in folder PATH whos extension is EXTENSION """

    extension = extension.lower()
    files = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        files.extend(filenames)
        break  # because only top level is wanted

    return [f for f in files if f.lower().endswith(extension)]
