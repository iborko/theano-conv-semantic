"""
Module containing functions for processing input images
"""

import logging
import cv2
# import pylab
import numpy as np

logger = logging.getLogger(__name__)


def resize(img, out_shape, inter=1):
    """
    Resize image (using OpenCV2)

    img: numpy.array
        img to resize
    out_shape: 2-tuple
        requested size
    inter: int
        interpolation: 0-nearest_neighbour, 1-linear, 2-cubic
    """
    assert(len(out_shape) == 2)
    ret_img = cv2.resize(img, (out_shape[1], out_shape[0]),
                         interpolation=inter)
    return ret_img


def zca_whiten(x):
    """
    Applies ZCA whitening to the data (X)
    http://xcorr.net/2011/05/27/whiten-a-matrix-matlab-code/

    x: numpy 2d array
        input data, rows are data points, columns are features

    Returns: ZCA whitened 2d array
    """
    assert(x.ndim == 2)
    EPS = 10e-5

    X = x - x.mean(axis=0)

    #   covariance matrix
    cov = np.dot(X.T, X)
    #   d = (lambda1, lambda2, ..., lambdaN)
    d, E = np.linalg.eigh(cov)
    #   D = diag(d) ^ (-1/2)
    D = np.diag(1. / np.sqrt(d + EPS))
    #   W_zca = E * D * E.T
    W = np.dot(np.dot(E, D), E.T)

    X_white = np.dot(X, W)

    return X_white


def calc_hog(img):
    """
    Calculate HOG descriptors of and image
    """
    raise Exception('not implemented')
    return None


def get_laplacian_pyramid_layer(img, n):
    '''Returns the n-th layer of the laplacian pyramid'''
    currImg, i = img, 0
    while i < n:
        down = cv2.pyrDown(currImg)
        up = cv2.pyrUp(down)
        shp = currImg.shape
        lap = currImg - up[:shp[0], :shp[1]]
        currImg = down
        i += 1
    return lap


def normalize_block(block):
    mean = np.mean(block, dtype=np.float64)
    block -= mean

    stddev = np.std(block, dtype=np.float64)
    if stddev > 0.000001:
        block /= stddev


def normalize_blocks(img):
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
    img[:, :, :3] = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2YUV)

    float_img = img.astype('float32')
    #   generate laplacian pyramid
    pyr_levels = []
    for i in xrange(n_layers):
        pyr_levels.append(get_laplacian_pyramid_layer(float_img, i+1))

    #   normalize from 0-255 to 0-1
    for i in xrange(n_layers):
        pyr_levels[i] /= 255.0

    #   normalize blocks
    #    of every layer
    for i in range(n_layers):
        #   of every channel
        for j in range(pyr_levels[i].shape[2]):
            # normalize_blocks(pyr_levels[i][:, :, j])
            pyr_levels[i][:, :, j] -= pyr_levels[i][:, :, j].mean()

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
