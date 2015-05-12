"""
Inception Layer (from Going Deeper With Convolutions paper)
"""
import logging
import theano.tensor as T
from theano.tensor.signal import downsample

from helpers.layers.conv import ConvLayer
from helpers.build_multiscale import upsample

log = logging.getLogger(__name__)

lReLU = lambda x: T.maximum(x, 1./5 * x)  # leaky ReLU


def build_inception(
        rng, input, image_shape,
        n_1, n_3red, n_3, n_5red, n_5, n_poolred):
    """
    Create layers contained in Inception layer

    :type rng: numpy.random.RandomState
    :param rng: a random number generator use to init weights

    :type input: theano.tensor.dtensor4
    :param input: symbolic image tensor, of shape image_shape

    :type image_shape: tuple or list of length 4
    :param image_shape: (batch_size, num_in_feat_maps, height, width)

    :type n_1: int
    :param n_1: number of feature maps made by 1x1 filters

    :type n_3red: int
    :param n_3red: number of reduced feature maps before 3x3 filter

    :type n_3: int
    :param n_3: number of feature maps made by 3x3 filters

    :type n_5red: int
    :param n_5red: number of reduced feature maps before 5x5 filter

    :type n_5: int
    :param n_5: number of feature maps made by 5x5 filters

    :type n_poolred: int
    :param n_poolred: number to reduce feature maps to after pooling
    """
    log.info("Building inception layer %s", (n_1, n_3, n_5, n_poolred))
    activation = lReLU
    bias = 0.

    # number of feature maps in previous layer
    n_prev = image_shape[1]
    # image dimensions
    img_y = image_shape[2]
    img_x = image_shape[3]

    #   first stage
    l1 = ConvLayer(
        rng, input,
        filter_shape=(n_1, n_prev, 1, 1),
        image_shape=image_shape,
        activation=activation, bias=bias,
        border_mode='valid')
    l1_out = l1.output

    l3_red = ConvLayer(
        rng, input,
        filter_shape=(n_3red, n_prev, 1, 1),
        image_shape=image_shape,
        activation=activation, bias=bias,
        border_mode='valid')
    l3_red_out = l3_red.output

    l5_red = ConvLayer(
        rng, input,
        filter_shape=(n_5red, n_prev, 1, 1),
        image_shape=image_shape,
        activation=activation, bias=bias,
        border_mode='valid')
    l5_red_out = l5_red.output

    l_pool = downsample.max_pool_2d(
        input=input,
        ds=(2, 2))

    #   second stage
    l3 = ConvLayer(
        rng, l3_red_out,
        filter_shape=(n_3, n_3red, 3, 3),
        image_shape=(image_shape[0], n_3red, img_y, img_x),
        activation=activation, bias=bias,
        border_mode='same')
    l3_out = l3.output

    l5 = ConvLayer(
        rng, l5_red_out,
        filter_shape=(n_5, n_5red, 5, 5),
        image_shape=(image_shape[0], n_5red, img_y, img_x),
        activation=activation, bias=bias,
        border_mode='same')
    l5_out = l5.output

    l_pool_red = ConvLayer(
        rng, l_pool,
        filter_shape=(n_poolred, n_prev, 1, 1),
        image_shape=(image_shape[0], n_prev, img_y // 2, img_x // 2),
        activation=activation, bias=bias,
        border_mode='valid')
    l_pool_out = upsample(l_pool_red.output, 2)

    out = T.concatenate(
        [l1_out, l3_out, l5_out, l_pool_out], axis=1)
    # number of output feature maps (concatenated layers)
    n_maps_out = n_1 + n_3 + n_5 + n_poolred

    out_shp = (image_shape[0], n_maps_out, img_y, img_x)
    log.info("Inception layer output has size %s", out_shp)

    layers = [l_pool_red, l5, l3, l5_red, l3_red, l1]

    return layers, out, out_shp
