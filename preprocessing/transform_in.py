"""
Module containing functions for processing input images
"""

import logging
import cv2
# import pylab
import numpy as np

logger = logging.getLogger(__name__)


def get_laplacian_pyramid_layer(img, n):
    '''Returns the n-th layer of the laplacian pyramid'''
    currImg, i = img, 0
    while i < n:
        down = cv2.pyrDown(currImg)
        up = cv2.pyrUp(down)
        lap = currImg - up
        currImg = down
        i += 1
    return lap


def normalize_block(block):
    mean = np.mean(block, dtype=np.float64)
    block -= mean

    stddev = np.std(block, dtype=np.float64)
    if stddev > 0.000001:
        block /= stddev


def normalize(img):
    ''' Normalizes (mean=0, stdev=1) image in blocks of 15x15 '''
    b_x, b_y = 15, 15
    for x in xrange(0, img.shape[0], b_x):
        for y in xrange(0, img.shape[1], b_y):
            normalize_block(img[x:(x + b_x), y:(y + b_y)])


def crop_to_shape(img, requested_shape):
    ''' If image shape is larger than requested, crop it '''
    return img[:requested_shape[0], :requested_shape[1]]


def fill_to_shape(img, requested_shape):
    ''' Fill image with black, output shape will be as defined in
    requested shape '''
    fill_y = requested_shape[0] - img.shape[0]
    fill_x = requested_shape[1] - img.shape[1]
    if fill_x < 0 or fill_y < 0:
        logger.error("Image shape not valid %s", img.shape)
        exit(1)
    return cv2.copyMakeBorder(img, fill_x, fill_y, 0, 0,
                              cv2.BORDER_CONSTANT, 0)


def yuv_laplacian_norm(img, requested_shape, n_layers=1):
    '''
    Image is cropped and filled to static shape size, then
    YUV transformation, laplacian pyramid and block normalization
    are applied to an image.
    '''
    # requested_shape = (216, 320)

    '''
    for j in xrange(3):
        pylab.subplot(2, 3, 0 * 3 + j + 1)
        pylab.gray()
        pylab.axis('off')
        pylab.imshow(img[:, :, j])
    '''

    logger.debug("Image has shape %s", img.shape)
    #   rotate image for 90, if in portrait orientation
    if img.shape[0] > img.shape[1]:
        img = cv2.transpose(img)

    #   crop and fill to shape
    img = crop_to_shape(img, requested_shape)
    img = fill_to_shape(img, requested_shape)

    #   convert to YUV (inplace)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    float_img = img.astype('float32')
    #   generate laplacian pyramid
    pyr_levels = []
    for i in xrange(n_layers):
        pyr_levels.append(get_laplacian_pyramid_layer(float_img, i+1))

    #   normalize from 0-255 to 0-1
    for i in xrange(n_layers):
        pyr_levels[i] /= 255.0

    '''
    #   normalize blocks of every channel
    for i in xrange(n_layers):
        normalize(pyr_levels[i][:, :, 0])  # Y
        normalize(pyr_levels[i][:, :, 1])  # U
        normalize(pyr_levels[i][:, :, 2])  # V
    '''

    # print "Before axis swap\n", pyr_levels[0][:5, :5, 0]
    for i in xrange(n_layers):
        pyr_levels[i] = np.rollaxis(pyr_levels[i], 2, 0)
    # print "After axis swap\n", pyr_levels[0][0, :5, :5]

    '''
    #   debug output
    for j in xrange(3):
        pylab.subplot(2, 3, 1 * 3 + j + 1)
        pylab.gray()
        pylab.axis('off')
        pylab.imshow(pyr_levels[0][j, :, :])
    pylab.show()
    '''

    return pyr_levels


def yuv(img, requested_shape):
    '''
    Image is cropped and filled to static shape size, then
    YUV transformation, laplacian pyramid and block normalization
    are applied to an image.
    '''

    logger.debug("Image has shape %s", img.shape)
    #   rotate image for 90, if in portrait orientation
    if img.shape[0] > img.shape[1]:
        img = cv2.transpose(img)

    #   crop and fill to shape
    img = crop_to_shape(img, requested_shape)
    img = fill_to_shape(img, requested_shape)

    #   convert to YUV (inplace)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    float_img = img.astype('float32')
    float_img = float_img / 255.0
    float_img = np.rollaxis(float_img, 2, 0)

    return float_img


def rgb_mean(img, requested_shape):
    '''
    Image is cropped and filled to static shape size,
    values are scaled to 0-1 range and mean is subtracted.
    '''

    logger.debug("Image has shape %s", img.shape)
    #   rotate image for 90, if in portrait orientation
    if img.shape[0] > img.shape[1]:
        img = cv2.transpose(img)

    #   crop and fill to shape
    img = crop_to_shape(img, requested_shape)
    img = fill_to_shape(img, requested_shape)

    float_img = img.astype('float32')
    float_img /= 255.0
    float_img -= float_img.mean()
    float_img = np.rollaxis(float_img, 2, 0)

    return float_img
