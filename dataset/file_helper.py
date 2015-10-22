import os
import logging
import cv2
import numpy as np
from sample import Sample

logger = logging.getLogger(__name__)


def load_image_grayscale(dirname, fname):
    path = os.path.join(dirname, fname)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print "Cant load image", path
        exit(1)
    return img


def load_image_rgb(dirname, fname):
    path = os.path.join(dirname, fname)
    img = cv2.imread(os.path.join(dirname, fname))
    if img is None:
        print "Cant load image", path
        exit(1)
    return img


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


def load_sample_rgbd(img_dirname, img_name,
                     d_dirname, d_name,
                     segm_dirname, segm_name):
    """
    Load image and segmented image into Sample object.
    Returns sample object.

    img_dirname: string
        image directory
    img_name: string
        image filename
    d_dirname: string
        depth image directory
    d_name: string
        depth image filename
    segm_dirname: string
        segmented image directory
    segm_name: string
        segmented filename
    """
    image = load_image_rgb(img_dirname, img_name)
    depth = load_image_grayscale(d_dirname, d_name)
    segmented_image = load_image_rgb(segm_dirname, segm_name)
    name = img_name.split('.')[0]

    img_rgbd = np.zeros((image.shape[0], image.shape[1], 4), dtype='uint8')
    img_rgbd[:, :, 0:3] = image
    depth = depth[:image.shape[0], :image.shape[1]]
    img_rgbd[:depth.shape[0], :depth.shape[1], 3] = depth

    return Sample(name, img_rgbd, segmented_image)


def get_file_list(path, extension):
    """ Returns list of files in folder PATH whos extension is EXTENSION """

    extension = extension.lower()
    files = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        files.extend(filenames)
        break  # because only top level is wanted

    return [f for f in files if f.lower().endswith(extension)]
