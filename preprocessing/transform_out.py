"""
Module containing functions for processing of input images
"""

import logging
import cv2

logger = logging.getLogger(__name__)


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

    # crop to requested shape
    img = img[:requested_shape[0], :requested_shape[1], :]

    fill_y = requested_shape[0] - img.shape[0]
    fill_x = requested_shape[1] - img.shape[1]
    if fill_x < 0 or fill_y < 0:
        logger.error("Image shape not valid %s", img.shape)
        exit(1)
    img = cv2.copyMakeBorder(img, fill_x, fill_y, 0, 0,
                             cv2.BORDER_CONSTANT, 0)
    # gives (r, g, b) for every pixel, where r=g=b=class_index
    img = cc.count_matrix(img)
    assert(img.ndim == 2)

    return img
