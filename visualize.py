"""
Visualization of convolutional filters
"""

import numpy as np
import pylab
import logging


logger = logging.getLogger(__name__)


def construct_stacked_array(array):
    """
    Constructs h-stacked array. Array must have 3 dimensions.
    Extend first dimension through second.

    Example: if array has shape (2, 3, 3), output will we (3, 2*3)
    """

    assert(array.ndim == 3)
    array_shape = array.shape
    filter_shape = (array.shape[1], array.shape[2])

    # there is array_shape[0] filters
    # +1 because of black borders
    stacked_shape = (array_shape[1], array_shape[0] * (array_shape[2] + 1))
    #  logger.debug("Stacked array shape %s", stacked_shape)

    stacked = np.zeros(stacked_shape)

    for i in xrange(array.shape[0]):
        stacked[:, i*filter_shape[1]+i:(i+1)*filter_shape[1]+i] = array[i]

    return stacked


def construct_stacked_matrix(array):
    """
    Constructs matrix from 4d array, stacked first horizontaly,
    then verticaly.

    Example: if array has shape (2, 3, 5, 5), output will be (2*5, 3*5)
    """

    assert(array.ndim == 4)
    a_shape = array.shape
    filter_shape = (a_shape[2], a_shape[3])

    stacked_shape = (a_shape[0] * (a_shape[2] + 1),
                     a_shape[1] * (a_shape[3] + 1))
    #  logger.debug("Stacked array shape %s", stacked_shape)

    stacked = np.zeros(stacked_shape)

    for i in xrange(array.shape[0]):
        stacked[i*filter_shape[0]+i:(i+1)*filter_shape[0]+i, :] =\
            construct_stacked_array(array[i])

    return stacked


def normalize(array):
    """ Makes array copy, normalizes it to range [0, 1] and returns """

    a = np.array(array, copy=True, dtype='float32')
    a -= a.min()
    a /= max(a.max(), 0.00001)  # beware of zero division
    return a


def visualize_array(array, title='Image', show=True, write=False):
    """ Visualize 3d and 4d array as image. filters (shape[2], shape[3])
    are stacked first horizontaly, then verticaly """

    assert(array.ndim == 3 or array.ndim == 4)
    array = normalize(array)  # this makes a copy

    if array.ndim == 3:
        array = construct_stacked_array(array)
    elif array.ndim == 4:
        array = construct_stacked_matrix(array)
    else:
        raise NotImplementedError()

    cm = pylab.gray()
    if show:
        fig = pylab.gcf()
        fig.canvas.set_window_title(title)
        pylab.axis('off')
        pylab.imshow(array, interpolation='nearest', cmap=cm)
        pylab.show()
        pylab.draw()

    if write:
        pylab.imsave(title + '.png', array, cmap=cm)


def show_out_image(img, title='Image', show=True, write=False):
    """ Plots image representing pixel classes """

    cm = pylab.get_cmap('gnuplot')
    if show:
        pylab.axis('off')
        pylab.imshow(img, interpolation='nearest', cmap=cm)
        pylab.show()
        pylab.draw()

    if write:
        pylab.imsave(title + '.png', img, cmap=cm)


def test():
    a = np.arange(98).reshape((2, 7, 7))
    print "A\n", a
    visualize_array(a)

    a2 = np.array([a, a[::-1]])
    visualize_array(a2, title="dupla", write=True)

    classified = np.zeros((240, 320), dtype='int32')
    classified[130:180, 200:220] = 1
    classified[20:40, 100:120] = 2
    classified[60:70, 150:180] = 3
    show_out_image(classified)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test()
