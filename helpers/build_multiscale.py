import logging
import numpy
import theano.tensor as T

from helpers.layers.log_reg import LogisticRegression
from helpers.layers.conv import ConvPoolLayer
from helpers.layers.hidden_dropout import HiddenLayerDropout


logger = logging.getLogger(__name__)


def upsample(x, factor):
    """
    Upsamples last two dimensions of symbolic theano tensor.

    x: symbolic theano tensor
        variable to upsample
    factor: int
        upsampling factor
    """
    x_1 = T.extra_ops.repeat(x, factor, axis=x.ndim-2)
    x_2 = T.extra_ops.repeat(x_1, factor, axis=x.ndim-1)
    return x_2


def div_tuple(tup, n):
    """ Divide every element of tuple by n """
    return tuple([x/n for x in tup])


def reduce_img_dim(img_shp, ignore_border=True):
    """ Reduces img dim by factor 2, if ignore_border is False,
    round to higher value, otherwise on lower """
    to_add = 1 if ignore_border is False else 0
    new_shp = [(x+to_add) // 2 for x in img_shp]
    return tuple(new_shp)


def crop_to_size(x, img_size, filter_size):
    """
    Crops last two dims of theano tensor x to img_size.
    Filter_size is parameter so theano can optimize expression at compile time

    x: theano 4d tensor
        Input tensor
    img_size: 2d tuple
        Image size (last two dims of tensor)
    filter_size: int
        Size of conv filter

    Returns: theano 4d tensor
        cropped theano tensor
    """
    assert(type(img_size) is tuple)
    assert(type(filter_size) is int)
    assert(len(img_size) == 2)
    assert((filter_size - 1) % 2 == 0)

    start_ind = (filter_size - 1) / 2
    return x[:, :, start_ind:(start_ind + img_size[0]),
             start_ind:(start_ind + img_size[1])]


def build_scale_3l(x, batch_size, image_shape, nkerns, nfilters, sparse,
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
        activation=activation, bias=bias, border_mode='full',
        ignore_border_pool=False,
        W=ws[0], b=bs[0],
    )

    #  layer0 has 12 filter maps for U and V channel
    layer0_UV = ConvPoolLayer(
        rng,
        input=layer0_UV_input,
        image_shape=(batch_size, 2, image_shape[0], image_shape[1]),
        filter_shape=(12, 2, nfilters[0], nfilters[0]),
        activation=activation, bias=bias, border_mode='full',
        ignore_border_pool=False,
        W=ws[1], b=bs[1],
    )

    # stack outputs from Y, U, V channel layer
    layer0_output = T.concatenate([layer0_Y.output,
                                   layer0_UV.output], axis=1)

    image_shape1 = reduce_img_dim(image_shape, False)
    layer1_input = crop_to_size(layer0_output, image_shape1, nfilters[0])

    # Construct the second convolutional pooling layer
    filters_to_use1 = nkerns[0] // 2 if sparse else nkerns[0]
    layer1 = ConvPoolLayer(
        rng,
        input=layer1_input,
        image_shape=(batch_size, nkerns[0], image_shape1[0], image_shape1[1]),
        filter_shape=(nkerns[1], filters_to_use1, nfilters[1], nfilters[1]),
        activation=activation, bias=bias, border_mode='full',
        ignore_border_pool=False,
        W=ws[2], b=bs[2],
    )

    image_shape2 = reduce_img_dim(image_shape1, False)
    layer2_input = crop_to_size(layer1.output, image_shape2, nfilters[1])

    # Construct the third convolutional pooling layer
    filters_to_use2 = nkerns[1] // 2 if sparse else nkerns[1]
    layer2 = ConvPoolLayer(
        rng,
        input=layer2_input,
        image_shape=(batch_size, nkerns[1], image_shape2[0], image_shape2[1]),
        filter_shape=(nkerns[2], filters_to_use2, nfilters[2], nfilters[2]),
        activation=activation, bias=bias, border_mode='full',
        poolsize=(1, 1), ignore_border_pool=False,
        only_conv=True,
        W=ws[3], b=bs[3],
    )

    image_shape3 = image_shape2
    layer2_output = crop_to_size(layer2.output, image_shape3, nfilters[2])

    logger.info("Scale output has size of %s", image_shape3)

    layers = [layer2, layer1, layer0_UV, layer0_Y]
    return layers, image_shape3, layer2_output


def build_scale_4l(x, batch_size, image_shape, nkerns, nfilters, sparse,
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
        activation=activation, bias=bias, border_mode='full',
        ignore_border_pool=False,
        W=ws[0], b=bs[0],
    )

    #  layer0 has 12 filter maps for U and V channel
    layer0_UV = ConvPoolLayer(
        rng,
        input=layer0_UV_input,
        image_shape=(batch_size, 2, image_shape[0], image_shape[1]),
        filter_shape=(12, 2, nfilters[0], nfilters[0]),
        activation=activation, bias=bias, border_mode='full',
        ignore_border_pool=False,
        W=ws[1], b=bs[1],
    )

    # stack outputs from Y, U, V channel layer
    layer0_output = T.concatenate([layer0_Y.output,
                                   layer0_UV.output], axis=1)

    image_shape1 = reduce_img_dim(image_shape, False)
    layer1_input = crop_to_size(layer0_output, image_shape1, nfilters[0])

    # Construct the second convolutional pooling layer
    filters_to_use1 = nkerns[0] // 2 if sparse else nkerns[0]
    layer1 = ConvPoolLayer(
        rng,
        input=layer1_input,
        image_shape=(batch_size, nkerns[0], image_shape1[0], image_shape1[1]),
        filter_shape=(nkerns[1], filters_to_use1, nfilters[1], nfilters[1]),
        activation=activation, bias=bias, border_mode='full',
        ignore_border_pool=False,
        W=ws[2], b=bs[2],
    )

    image_shape2 = reduce_img_dim(image_shape1, False)
    layer2_input = crop_to_size(layer1.output, image_shape2, nfilters[1])

    # Construct the third convolutional pooling layer
    filters_to_use2 = nkerns[1] // 2 if sparse else nkerns[1]
    layer2 = ConvPoolLayer(
        rng,
        input=layer2_input,
        image_shape=(batch_size, nkerns[1], image_shape2[0], image_shape2[1]),
        filter_shape=(nkerns[2], filters_to_use2, nfilters[2], nfilters[2]),
        activation=activation, bias=bias, border_mode='full',
        poolsize=(1, 1), ignore_border_pool=False,
        W=ws[3], b=bs[3],
    )

    image_shape3 = image_shape2
    layer3_input = crop_to_size(layer2.output, image_shape3, nfilters[2])

    # Construct the 4th convolutional pooling layer
    filters_to_use3 = nkerns[2] // 2 if sparse else nkerns[2]
    layer3 = ConvPoolLayer(
        rng,
        input=layer3_input,
        image_shape=(batch_size, nkerns[2], image_shape3[0], image_shape3[1]),
        filter_shape=(nkerns[3], filters_to_use3, nfilters[3], nfilters[3]),
        activation=activation, bias=bias, border_mode='full',
        ignore_border_pool=False,
        W=ws[4], b=bs[4],
    )
    image_shape4 = reduce_img_dim(image_shape3, False)
    layer3_output = crop_to_size(layer3.output, image_shape4, nfilters[3])
    logger.info("Scale output has size of %s", image_shape4)

    layers = [layer3, layer2, layer1, layer0_UV, layer0_Y]
    return layers, image_shape4, layer3_output


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
    # net has 3 conv layers
    assert(len(nkerns) == 3)
    # this version has to have 16 filters in first layer
    assert(nkerns[0] == 32)

    # convolution kernel size
    nfilters = [7, 7, 5]

    logger.info('... building the model')

    rng = numpy.random.RandomState(23455)
    layers0, img_shp, out0 = build_scale_3l(
        x0, batch_size, image_shape, nkerns, nfilters,
        sparse, activation, bias, rng, None)
    image_shape_s2 = div_tuple(image_shape, 2)
    layers2, _, out2 = build_scale_3l(
        x2, batch_size, image_shape_s2, nkerns, nfilters,
        sparse, activation, bias, rng, layers0)
    image_shape_s4 = div_tuple(image_shape_s2, 2)
    layers4, _, out4 = build_scale_3l(
        x4, batch_size, image_shape_s4, nkerns, nfilters,
        sparse, activation, bias, rng, layers2)

    scale0_out = out0.dimshuffle(0, 2, 3, 1).\
        reshape((-1, nkerns[-1]))
    scale2_out = upsample(out2, 2).dimshuffle(0, 2, 3, 1).\
        reshape((-1, nkerns[-1]))
    scale4_out = upsample(out4, 4).dimshuffle(0, 2, 3, 1).\
        reshape((-1, nkerns[-1]))

    layer4_input = T.concatenate([scale0_out, scale2_out, scale4_out], axis=1)

    # classify the values of the fully-connected sigmoidal layer
    layer_last = LogisticRegression(input=layer4_input,
                                    n_in=nkerns[-1] * 3,  # 3 scales
                                    n_out=classes)

    # list of all layers
    layers = [layer_last] + layers0
    return layers, img_shp, layer4_input


def extend_net_w2l(input, layers, classes, nkerns,
                   activation=T.tanh, bias=0.0):
    """
    Extends net with hidden layers.
    """
    assert(len(nkerns) == 2)
    rng = numpy.random.RandomState(23454)
    DROPOUT_RATE = 0.5

    layer_h0 = HiddenLayerDropout(
        rng=rng,
        input=input,
        n_in=256 * 3,
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


def extend_net_w1l(input, layers, classes, nkerns,
                   activation=T.tanh, bias=0.0):
    """
    Extends net with hidden layers.
    """
    assert(len(nkerns) == 1)
    rng = numpy.random.RandomState(23456)
    DROPOUT_RATE = 0.5

    layer_h0 = HiddenLayerDropout(
        rng=rng,
        input=input,
        n_in=256 * 3,
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
