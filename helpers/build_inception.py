"""
Build a network using Inception layer from GoogLeNet.
"""
import numpy
import theano.tensor as T
from theano.tensor.signal import downsample

from helpers.layers.conv import ConvLayer
from helpers.layers.log_reg import LogisticRegression
from helpers.layers.inception import build_inception
from helpers.build_multiscale import reduce_img_dim, upsample
from helpers.layers.dropout import DropoutLayer

lReLU = lambda x: T.maximum(x, 1./5 * x)  # leaky ReLU


def build_net(x, y, batch_size, classes, image_shape):
    """
    Build model for conv network for segmentation

    x: symbolic theano variable, 4d tensor
        input data (or symbol representing it)
    y: symbolic theano variable, imatrix
        output data (or symbol representing it)
    batch_size: int
        size of batch
    classes: int
        number of classes
    image_shape: tuple
        image dimensions

    returns: list
        list of all layers, first layer is actually the last (log reg)
    """

    rng = numpy.random.RandomState(23455)

    # inception parameters
    incep0 = (64, 64, 128, 16, 32, 32)
    incep1 = (128, 128, 192, 32, 64, 64)

    activation = lReLU
    bias = 0.

    layer0 = ConvLayer(
        rng,
        input=x,
        image_shape=(batch_size, 3, image_shape[0], image_shape[1]),
        filter_shape=(32, 3, 7, 7),
        activation=activation, bias=bias, border_mode='same')
    layer0_out = downsample.max_pool_2d(
        input=layer0.output,
        ds=(2, 2))
    img_shp1 = reduce_img_dim(image_shape)

    layer1 = ConvLayer(
        rng,
        input=layer0_out,
        image_shape=(batch_size, 32, img_shp1[0], img_shp1[1]),
        filter_shape=(128, 32, 5, 5),
        activation=activation, bias=bias, border_mode='same')
    layer1_out = downsample.max_pool_2d(
        input=layer1.output,
        ds=(2, 2))
    img_shp2 = reduce_img_dim(img_shp1)

    lrs0, out0, incep_shp0 = build_inception(
        rng, input=layer1_out,
        image_shape=(batch_size, 128, img_shp2[0], img_shp2[1]),
        n_1=incep0[0],
        n_3red=incep0[1], n_3=incep0[2],
        n_5red=incep0[3], n_5=incep0[4],
        n_poolred=incep0[5])

    lrs1, out1, incep_shp1 = build_inception(
        rng, input=out0,
        image_shape=incep_shp0,
        n_1=incep1[0],
        n_3red=incep1[1], n_3=incep1[2],
        n_5red=incep1[3], n_5=incep1[4],
        n_poolred=incep1[5])

    conc = T.concatenate([upsample(out1, 2), layer1.output], axis=1)
    drop = DropoutLayer(conc, conc.shape, 0.6)
    logreg_in = drop.output.dimshuffle(0, 2, 3, 1).\
        reshape((-1, conc.shape[1]))

    # classify the values of the fully-connected sigmoidal layer
    layer_last = LogisticRegression(input=logreg_in,
                                    n_in=576,
                                    n_out=classes)

    # list of all layers
    layers = [layer_last, drop] + lrs1 + lrs0 + [layer1, layer0]
    return layers, img_shp1
