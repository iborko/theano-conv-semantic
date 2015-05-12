"""
Module containing functions for processing of input images
"""

import logging
import cv2
from preprocessing.transform_in import crop_to_shape
from preprocessing.transform_in import fill_to_shape

logger = logging.getLogger(__name__)


def resize_marked_image(img, requested_shape):
    '''
    Resizes marked image (image with class markings) with no interpolation.

    img: numpy array
        image
    requested_shape: 2-tuple
        shape of the output image
    '''
    assert(len(requested_shape) == 2)
    return cv2.resize(
        img, (requested_shape[1], requested_shape[0]),
        interpolation=cv2.INTER_NEAREST)


def process_out(img, cc, requested_shape):
    '''
    Replaces RGB colors of every pixel with class number.

    img: numpy array
        image
    cc: ClassCounter object
        counting of class markings
    requested_shape: 2-tuple
        requested dimension of output shape
    '''

    logger.debug("Image has shape %s", img.shape)
    # rotate image for 90, if in portrait orientation
    if img.shape[0] > img.shape[1]:
        img = cv2.transpose(img)

    img = crop_to_shape(img, requested_shape)

    img = fill_to_shape(img, requested_shape)

    # gives (r, g, b) for every pixel, where r=g=b=class_index
    img = cc.count_matrix(img)
    assert(img.ndim == 2)

    return img


def process_iccv(img, requested_shape):
    '''
    Process iccv marked image

    img: numpy array
        image
    requested_shape: 2-tuple
        requested dimension of output shape
    '''

    logger.debug("Image has shape %s", img.shape)
    # rotate image for 90, if in portrait orientation
    if img.shape[0] > img.shape[1]:
        img = cv2.transpose(img)

    img[img < 0] = 8  # replace -1 with 8
    img += 1  # shift all by 1, leave room for class(0), empty parts

    img = crop_to_shape(img, requested_shape)

    img = fill_to_shape(img, requested_shape)

    assert(img.ndim == 2)

    return img
