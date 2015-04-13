import logging
import numpy
import theano.tensor as T

from helpers.layers.log_reg import LogisticRegression
from helpers.layers.conv import ConvPoolLayer
from helpers.layers.hidden_dropout import HiddenLayerDropout


logger = logging.getLogger(__name__)


def reduce_image_dim(image_shape, filter_size, pooling_size):
    """
    Helper function for calculation of image_dimensions

    First reduces dimension by filter_size (x - filer_size + 1)
    then divides it by pooling factor  ( x // 2)

    image_shape; tuple
    filter_size: int
    pooling_size: int
    """

    return map(lambda x: (x - filter_size + 1) // pooling_size, image_shape)


def upsample(x, factor):
    """
    Upsamples last two dimensions of symbolic theano tensor.

    x: symbolic theano tensor
        variable to upsample
    factor: int
        upsampling factor
    """
    shp = x.shape
    x_1 = T.extra_ops.repeat(x, factor, axis=len(shp)-2)
    x_2 = T.extra_ops.repeat(x_1, factor, axis=len(shp)-1)
    return x_2


def build_scale(x, batch_size, image_shape, nkerns, nfilters, sparse,
                activation, bias, rng, layers):
    """
    x: symbolic theano variable
        input data
    batch_size: int
        size of minibatch
    image_size: 2-tuple
        size of input image
    nkerns: list of ints
        kernels dimension (x and y dimension are the same)
    nfilters: list of ints
        number of filter banks per layer
    sparse: boolean
        is network sparse?
    activation: theano symbolic expression
        activation function
    bias: float
        bias amount
    rng: numpy random generator
        random generator used for generating weights
    layers: list of layer objects (ConvLayers)
        list of layers used to build layers of current scale
    """
    assert(len(nkerns) == len(nfilters))

    layer0_Y_input = x[:, [0]]  # Y channel
    layer0_UV_input = x[:, [1, 2]]  # U, V channels

    ws = [None] * (len(nfilters) + 1)
    bs = [None] * len(ws)
    if layers is not None:
        for i, layer in enumerate(layers[::-1]):
            ws[i] = layer.W
            bs[i] = layer.b

    # Construct the first convolutional pooling layer
    #  layer0 has 20 filter maps for U and V channel
    layer0_Y = ConvPoolLayer(
        rng,
        input=layer0_Y_input,
        image_shape=(batch_size, 1, image_shape[0], image_shape[1]),
        filter_shape=(20, 1, nfilters[0], nfilters[0]),
        activation=activation, bias=bias,
        W=ws[0], b=bs[0],
    )

    #  layer0 has 12 filter maps for U and V channel
    layer0_UV = ConvPoolLayer(
        rng,
        input=layer0_UV_input,
        image_shape=(batch_size, 2, image_shape[0], image_shape[1]),
        filter_shape=(12, 2, nfilters[0], nfilters[0]),
        activation=activation, bias=bias,
        W=ws[1], b=bs[1],
    )

    # stack outputs from Y, U, V channel layer
    layer0_output = T.concatenate([layer0_Y.output,
                                   layer0_UV.output], axis=1)

    # Construct the second convolutional pooling layer
    image_shape1 = reduce_image_dim(image_shape, nfilters[0], 2)
    filters_to_use1 = nkerns[0] // 2 if sparse else nkerns[0]
    layer1 = ConvPoolLayer(
        rng,
        input=layer0_output,
        image_shape=(batch_size, nkerns[0], image_shape1[0], image_shape1[1]),
        filter_shape=(nkerns[1], filters_to_use1, nfilters[1], nfilters[1]),
        activation=activation, bias=bias,
        W=ws[2], b=bs[2],
    )

    # Construct the third convolutional pooling layer
    image_shape2 = reduce_image_dim(image_shape1, nfilters[1], 2)
    filters_to_use2 = nkerns[1] // 2 if sparse else nkerns[1]
    layer2 = ConvPoolLayer(
        rng,
        input=layer1.output,
        image_shape=(batch_size, nkerns[1], image_shape2[0], image_shape2[1]),
        filter_shape=(nkerns[2], filters_to_use2, nfilters[2], nfilters[2]),
        activation=activation, bias=bias,
        poolsize=(1, 1),
        W=ws[3], b=bs[3],
    )

    # Construct the 4th convolutional pooling layer
    image_shape3 = reduce_image_dim(image_shape2, nfilters[2], 1)
    filters_to_use3 = nkerns[2] // 2 if sparse else nkerns[2]
    layer3 = ConvPoolLayer(
        rng,
        input=layer2.output,
        image_shape=(batch_size, nkerns[2], image_shape3[0], image_shape3[1]),
        filter_shape=(nkerns[3], filters_to_use3, nfilters[3], nfilters[3]),
        activation=activation, bias=bias,
        W=ws[4], b=bs[4],
    )
    image_shape4 = reduce_image_dim(image_shape3, nfilters[3], 2)

    return [layer3, layer2, layer1, layer0_UV, layer0_Y], image_shape4


def build_multiscale(x0, x2, x4, y, batch_size, classes, image_shape,
                     nkerns, sparse=False, activation=T.tanh, bias=0.0):
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
    layers: list
        list of kernel_dimensions

    returns: list
        list of all layers, first layer is actually the last (log reg)
    """
    # net has 4 conv layers
    assert(len(nkerns) == 4)
    # this version has to have 16 filters in first layer
    assert(nkerns[0] == 32)

    # convolution kernel size
    nfilters = [5, 5, 3, 3]

    logger.info('... building the model')

    rng = numpy.random.RandomState(23455)
    scale0, img_shp = build_scale(
        x0, batch_size, image_shape, nkerns, nfilters,
        sparse, activation, bias, rng, None)
    scale2, _ = build_scale(
        x2, batch_size, image_shape, nkerns, nfilters,
        sparse, activation, bias, rng, scale0)
    scale4, _ = build_scale(
        x4, batch_size, image_shape, nkerns, nfilters,
        sparse, activation, bias, rng, scale2)

    scale0_out = scale0[0].output.dimshuffle(0, 2, 3, 1).\
        reshape((-1, nkerns[3]))
    scale2_out = upsample(scale2[0].output, 2).dimshuffle(0, 2, 3, 1).\
        reshape((-1, nkerns[3]))
    scale4_out = upsample(scale4[0].output, 4).dimshuffle(0, 2, 3, 1).\
        reshape((-1, nkerns[3]))

    layer4_input = T.concatenate([scale0_out, scale2_out, scale4_out], axis=1)

    # classify the values of the fully-connected sigmoidal layer
    layer_last = LogisticRegression(input=layer4_input,
                                    n_in=nkerns[3] * 3,  # 3 scales
                                    n_out=classes)

    # list of all layers
    layers = [layer_last] + scale0
    return layers, tuple(img_shp)


def extend_net_w2l(layers, classes, nkerns,
                   activation=T.tanh, bias=0.0):
    """
    Extends net with hidden layers.
    """
    assert(len(nkerns) == 2)
    rng = numpy.random.RandomState(23456)
    DROPOUT_RATE = 0.5

    input = layers[1].output.dimshuffle(0, 2, 3, 1).reshape((-1, 256))

    layer_h0 = HiddenLayerDropout(
        rng=rng,
        input=input,
        n_in=256,
        n_out=nkerns[0],
        activation=activation, bias=bias,
        dropout_p=DROPOUT_RATE
    )

    layer_h1 = HiddenLayerDropout(
        rng=rng,
        input=layer_h0.output,
        n_in=nkerns[0],
        n_out=nkerns[1],
        activation=activation, bias=bias,
        dropout_p=DROPOUT_RATE
    )

    # classify the values of the fully-connected sigmoidal layer
    layer_h2 = LogisticRegression(input=layer_h1.output,
                                  n_in=nkerns[1],
                                  n_out=classes)

    new_layers = [layer_h2, layer_h1, layer_h0]
    all_layers = new_layers + layers[1:]
    return all_layers, new_layers


def extend_net_w1l(layers, classes, nkerns,
                   activation=T.tanh, bias=0.0):
    """
    Extends net with hidden layers.
    """
    assert(len(nkerns) == 1)
    rng = numpy.random.RandomState(23456)
    DROPOUT_RATE = 0.5

    input = layers[1].output.dimshuffle(0, 2, 3, 1).reshape((-1, 256))

    layer_h0 = HiddenLayerDropout(
        rng=rng,
        input=input,
        n_in=256,
        n_out=nkerns[0],
        activation=activation, bias=bias,
        dropout_p=DROPOUT_RATE
    )

    # classify the values of the fully-connected sigmoidal layer
    layer_h1 = LogisticRegression(input=layer_h0.output,
                                  n_in=nkerns[0],
                                  n_out=classes)

    new_layers = [layer_h1, layer_h0]
    all_layers = new_layers + layers[1:]
    return all_layers, new_layers
