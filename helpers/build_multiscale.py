import logging
import numpy
import theano.tensor as T

from helpers.layers.log_reg import LogisticRegression
from helpers.layers.conv import ConvPoolLayer
# from helpers.layers.hidden_dropout import HiddenLayerDropout
from helpers.layers.hidden import HiddenLayer
from helpers.layers.dropout import DropoutLayer


logger = logging.getLogger(__name__)


def get_net_builder(name):
    """
    Return builder function by name

    name: string
        net builder name (function that builds theano net)
    """
    return globals()[name]


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


def gen_mat_indices(shp):
    """
    shp: 2d tuple
        Matrix shape

    Returns tensor of two matrices. Element in first matrix is tensor
    row number and element in second is column number.
    """
    assert(type(shp) is tuple)
    assert(len(shp) == 2)

    x = T.repeat(
        T.arange(shp[0]).reshape((-1, 1)),
        shp[1],
        axis=1)
    y = T.repeat(
        T.arange(shp[1]).reshape((1, -1)),
        shp[0],
        axis=0)

    return T.stack(x, y)


def build_scale_3l_rgbd1(x, batch_size, image_shape, nkerns, nfilters, sparse,
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

    layer0_input = x

    ws = [None] * (len(nfilters) + 1)  # +1 additional first layer
    bs = [None] * len(ws)
    if layers is not None:
        for i, layer in enumerate(layers[::-1]):
            ws[i] = layer.W
            bs[i] = layer.b

    # Construct the first convolutional pooling layer
    layer0 = ConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 4, image_shape[0], image_shape[1]),
        filter_shape=(nkerns[0] / 2, 4, nfilters[0], nfilters[0]),
        activation=activation, bias=bias, border_mode='same',
        ignore_border_pool=False,
        W=ws[0], b=bs[0],
    )

    layer0_1 = ConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 4, image_shape[0], image_shape[1]),
        filter_shape=(nkerns[0] / 2, 4, 5, 5),
        activation=activation, bias=bias, border_mode='same',
        ignore_border_pool=False,
        W=ws[1], b=bs[1],
    )

    # stack outputs from Y, U, V channel layer
    layer0_output = T.concatenate([layer0.output, layer0_1.output], axis=1)

    image_shape1 = reduce_img_dim(image_shape, False)

    # Construct the second convolutional pooling layer
    filters_to_use1 = nkerns[0] // 2 if sparse else nkerns[0]
    layer1 = ConvPoolLayer(
        rng,
        input=layer0_output,
        image_shape=(batch_size, nkerns[0], image_shape1[0], image_shape1[1]),
        filter_shape=(nkerns[1], filters_to_use1, nfilters[1], nfilters[1]),
        activation=activation, bias=bias, border_mode='same',
        ignore_border_pool=False,
        W=ws[2], b=bs[2],
    )

    image_shape2 = reduce_img_dim(image_shape1, False)

    # Construct the third convolutional pooling layer
    filters_to_use2 = nkerns[1] // 2 if sparse else nkerns[1]
    layer2 = ConvPoolLayer(
        rng,
        input=layer1.output,
        image_shape=(batch_size, nkerns[1], image_shape2[0], image_shape2[1]),
        filter_shape=(nkerns[2], filters_to_use2, nfilters[2], nfilters[2]),
        activation=activation, bias=bias, border_mode='same',
        poolsize=(1, 1), ignore_border_pool=False,
        only_conv=True,
        W=ws[3], b=bs[3],
    )

    image_shape3 = image_shape2
    layer2_output = layer2.output

    logger.info("Scale output has size of %s", image_shape3)

    layers = [layer2, layer1, layer0_1, layer0]
    return layers, image_shape3, layer2_output


def build_scale_3l_rgbd(x, batch_size, image_shape, nkerns, nfilters, sparse,
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

    layer0_input = x

    ws = [None] * len(nfilters)
    bs = [None] * len(ws)
    if layers is not None:
        for i, layer in enumerate(layers[::-1]):
            ws[i] = layer.W
            bs[i] = layer.b

    # Construct the first convolutional pooling layer
    layer0 = ConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 4, image_shape[0], image_shape[1]),
        filter_shape=(nkerns[0], 4, nfilters[0], nfilters[0]),
        activation=activation, bias=bias, border_mode='same',
        ignore_border_pool=False,
        W=ws[0], b=bs[0],
    )

    # stack outputs from Y, U, V channel layer
    layer0_output = layer0.output

    image_shape1 = reduce_img_dim(image_shape, False)

    # Construct the second convolutional pooling layer
    filters_to_use1 = nkerns[0] // 2 if sparse else nkerns[0]
    layer1 = ConvPoolLayer(
        rng,
        input=layer0_output,
        image_shape=(batch_size, nkerns[0], image_shape1[0], image_shape1[1]),
        filter_shape=(nkerns[1], filters_to_use1, nfilters[1], nfilters[1]),
        activation=activation, bias=bias, border_mode='same',
        ignore_border_pool=False,
        W=ws[1], b=bs[1],
    )

    image_shape2 = reduce_img_dim(image_shape1, False)

    # Construct the third convolutional pooling layer
    filters_to_use2 = nkerns[1] // 2 if sparse else nkerns[1]
    layer2 = ConvPoolLayer(
        rng,
        input=layer1.output,
        image_shape=(batch_size, nkerns[1], image_shape2[0], image_shape2[1]),
        filter_shape=(nkerns[2], filters_to_use2, nfilters[2], nfilters[2]),
        activation=activation, bias=bias, border_mode='same',
        poolsize=(1, 1), ignore_border_pool=False,
        only_conv=True,
        W=ws[2], b=bs[2],
    )

    image_shape3 = image_shape2
    layer2_output = layer2.output

    logger.info("Scale output has size of %s", image_shape3)

    layers = [layer2, layer1, layer0]
    return layers, image_shape3, layer2_output


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
    #  layer0 has 10 filter maps for U and V channel
    layer0_Y = ConvPoolLayer(
        rng,
        input=layer0_Y_input,
        image_shape=(batch_size, 1, image_shape[0], image_shape[1]),
        filter_shape=(10, 1, nfilters[0], nfilters[0]),
        activation=activation, bias=bias, border_mode='same',
        ignore_border_pool=False,
        W=ws[0], b=bs[0],
    )

    #  layer0 has 6 filter maps for U and V channel
    layer0_UV = ConvPoolLayer(
        rng,
        input=layer0_UV_input,
        image_shape=(batch_size, 2, image_shape[0], image_shape[1]),
        filter_shape=(6, 2, nfilters[0], nfilters[0]),
        activation=activation, bias=bias, border_mode='same',
        ignore_border_pool=False,
        W=ws[1], b=bs[1],
    )

    # stack outputs from Y, U, V channel layer
    layer0_output = T.concatenate([layer0_Y.output,
                                   layer0_UV.output], axis=1)

    image_shape1 = reduce_img_dim(image_shape, False)

    # Construct the second convolutional pooling layer
    filters_to_use1 = nkerns[0] // 2 if sparse else nkerns[0]
    layer1 = ConvPoolLayer(
        rng,
        input=layer0_output,
        image_shape=(batch_size, nkerns[0], image_shape1[0], image_shape1[1]),
        filter_shape=(nkerns[1], filters_to_use1, nfilters[1], nfilters[1]),
        activation=activation, bias=bias, border_mode='same',
        ignore_border_pool=False,
        W=ws[2], b=bs[2],
    )

    image_shape2 = reduce_img_dim(image_shape1, False)

    # Construct the third convolutional pooling layer
    filters_to_use2 = nkerns[1] // 2 if sparse else nkerns[1]
    layer2 = ConvPoolLayer(
        rng,
        input=layer1.output,
        image_shape=(batch_size, nkerns[1], image_shape2[0], image_shape2[1]),
        filter_shape=(nkerns[2], filters_to_use2, nfilters[2], nfilters[2]),
        activation=activation, bias=bias, border_mode='same',
        poolsize=(1, 1), ignore_border_pool=False,
        only_conv=True,
        W=ws[3], b=bs[3],
    )

    image_shape3 = image_shape2
    layer2_output = layer2.output

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
        activation=activation, bias=bias, border_mode='same',
        ignore_border_pool=False,
        W=ws[0], b=bs[0],
    )

    #  layer0 has 12 filter maps for U and V channel
    layer0_UV = ConvPoolLayer(
        rng,
        input=layer0_UV_input,
        image_shape=(batch_size, 2, image_shape[0], image_shape[1]),
        filter_shape=(12, 2, nfilters[0], nfilters[0]),
        activation=activation, bias=bias, border_mode='same',
        ignore_border_pool=False,
        W=ws[1], b=bs[1],
    )

    # stack outputs from Y, U, V channel layer
    layer0_output = T.concatenate([layer0_Y.output,
                                   layer0_UV.output], axis=1)

    image_shape1 = reduce_img_dim(image_shape, False)

    # Construct the second convolutional pooling layer
    filters_to_use1 = nkerns[0] // 2 if sparse else nkerns[0]
    layer1 = ConvPoolLayer(
        rng,
        input=layer0_output,
        image_shape=(batch_size, nkerns[0], image_shape1[0], image_shape1[1]),
        filter_shape=(nkerns[1], filters_to_use1, nfilters[1], nfilters[1]),
        activation=activation, bias=bias, border_mode='same',
        ignore_border_pool=False,
        W=ws[2], b=bs[2],
    )

    image_shape2 = reduce_img_dim(image_shape1, False)

    # Construct the third convolutional pooling layer
    filters_to_use2 = nkerns[1] // 2 if sparse else nkerns[1]
    layer2 = ConvPoolLayer(
        rng,
        input=layer1.output,
        image_shape=(batch_size, nkerns[1], image_shape2[0], image_shape2[1]),
        filter_shape=(nkerns[2], filters_to_use2, nfilters[2], nfilters[2]),
        activation=activation, bias=bias, border_mode='same',
        poolsize=(1, 1), ignore_border_pool=False,
        W=ws[3], b=bs[3],
    )

    image_shape3 = image_shape2

    # Construct the 4th convolutional pooling layer
    filters_to_use3 = nkerns[2] // 2 if sparse else nkerns[2]
    layer3 = ConvPoolLayer(
        rng,
        input=layer2.output,
        image_shape=(batch_size, nkerns[2], image_shape3[0], image_shape3[1]),
        filter_shape=(nkerns[3], filters_to_use3, nfilters[3], nfilters[3]),
        activation=activation, bias=bias, border_mode='same',
        ignore_border_pool=False,
        W=ws[4], b=bs[4],
    )
    image_shape4 = reduce_img_dim(image_shape3, False)
    layer3_output = layer3.output
    logger.info("Scale output has size of %s", image_shape4)

    layers = [layer3, layer2, layer1, layer0_UV, layer0_Y]
    return layers, image_shape4, layer3_output


def build_multiscale_rgbd_withloc(
        x0, x2, x4, y, batch_size, classes, image_shape,
        nkerns, seed, sparse=False,
        activation=T.tanh, bias=0.0):
    """
    Build model for conv network for segmentation
    (last layers uses not only features from conv net, but its indices too)

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
    nkerns: list of ints
        list of kernel_dimensions
    seed: int
        seed for random generator

    returns: list
        list of all layers, first layer is actually the last (log reg)
    """
    assert(type(seed) is int)

    # net has 3 conv layers
    assert(len(nkerns) == 3)

    # convolution kernel size
    nfilters = [7, 7, 7]

    logger.info('... building the model')

    rng = numpy.random.RandomState(seed)
    layers0, img_shp, out0 = build_scale_3l_rgbd(
        x0, batch_size, image_shape, nkerns, nfilters,
        sparse, activation, bias, rng, None)
    image_shape_s2 = div_tuple(image_shape, 2)
    layers2, _, out2 = build_scale_3l_rgbd(
        x2, batch_size, image_shape_s2, nkerns, nfilters,
        sparse, activation, bias, rng, layers0)
    image_shape_s4 = div_tuple(image_shape_s2, 2)
    layers4, _, out4 = build_scale_3l_rgbd(
        x4, batch_size, image_shape_s4, nkerns, nfilters,
        sparse, activation, bias, rng, layers2)

    scale0_out = out0
    scale2_out = upsample(out2, 2)
    scale4_out = upsample(out4, 4)

    indices = gen_mat_indices(img_shp)
    ind_ext = T.repeat(indices.dimshuffle('x', 0, 1, 2), batch_size, axis=0)

    n_features = nkerns[-1] * 3 + 2  # +2 for index of every pixel

    conc = T.concatenate([scale0_out, scale2_out, scale4_out, ind_ext], axis=1)
    layer4_in = conc.dimshuffle(0, 2, 3, 1).\
        reshape((-1, n_features))

    # classify the values of the fully-connected sigmoidal layer
    layer_last = LogisticRegression(input=layer4_in,
                                    n_in=n_features,
                                    n_out=classes)

    # list of all layers
    layers = [layer_last] + layers0
    return layers, img_shp, layer4_in


def build_multiscale_rgbd(x0, x2, x4, y, batch_size, classes, image_shape,
                          nkerns, seed, sparse=False,
                          activation=T.tanh, bias=0.0):
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
    nkerns: list of ints
        list of kernel_dimensions
    seed: int
        seed for random generator

    returns: list
        list of all layers, first layer is actually the last (log reg)
    """
    assert(type(seed) is int)

    # net has 3 conv layers
    assert(len(nkerns) == 3)

    # convolution kernel size
    nfilters = [7, 7, 7]

    logger.info('... building the model')

    rng = numpy.random.RandomState(seed)
    layers0, img_shp, out0 = build_scale_3l_rgbd1(
        x0, batch_size, image_shape, nkerns, nfilters,
        sparse, activation, bias, rng, None)
    image_shape_s2 = div_tuple(image_shape, 2)
    layers2, _, out2 = build_scale_3l_rgbd1(
        x2, batch_size, image_shape_s2, nkerns, nfilters,
        sparse, activation, bias, rng, layers0)
    image_shape_s4 = div_tuple(image_shape_s2, 2)
    layers4, _, out4 = build_scale_3l_rgbd1(
        x4, batch_size, image_shape_s4, nkerns, nfilters,
        sparse, activation, bias, rng, layers2)

    scale0_out = out0
    scale2_out = upsample(out2, 2)
    scale4_out = upsample(out4, 4)

    conc = T.concatenate([scale0_out, scale2_out, scale4_out], axis=1)
    layer4_in = conc.dimshuffle(0, 2, 3, 1).\
        reshape((-1, nkerns[-1] * 3))

    # classify the values of the fully-connected sigmoidal layer
    layer_last = LogisticRegression(input=layer4_in,
                                    n_in=nkerns[-1] * 3,  # 3 scales
                                    n_out=classes)

    # list of all layers
    layers = [layer_last] + layers0
    return layers, img_shp, layer4_in


def build_multiscale(x0, x2, x4, y, batch_size, classes, image_shape,
                     nkerns, seed, sparse=False,
                     activation=T.tanh, bias=0.0):
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
    nkerns: list of ints
        list of kernel_dimensions
    seed: int
        seed for random generator

    returns: list
        list of all layers, first layer is actually the last (log reg)
    """
    assert(type(seed) is int)

    # net has 3 conv layers
    assert(len(nkerns) == 3)
    # this version has to have 16 filters in first layer
    assert(nkerns[0] == 16)

    # convolution kernel size
    nfilters = [7, 7, 7]

    logger.info('... building the model')

    rng = numpy.random.RandomState(seed)
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

    scale0_out = out0
    scale2_out = upsample(out2, 2)
    scale4_out = upsample(out4, 4)

    conc = T.concatenate([scale0_out, scale2_out, scale4_out], axis=1)
    layer4_in = conc.dimshuffle(0, 2, 3, 1).\
        reshape((-1, nkerns[-1] * 3))

    # classify the values of the fully-connected sigmoidal layer
    layer_last = LogisticRegression(input=layer4_in,
                                    n_in=nkerns[-1] * 3,  # 3 scales
                                    n_out=classes)

    # list of all layers
    layers = [layer_last] + layers0
    return layers, img_shp, layer4_in


def extend_net_w2l(input, n_in, layers, classes, nkerns, seed,
                   activation=T.tanh, bias=0.0):
    """
    Extends net with hidden layers.
    """
    assert(len(nkerns) == 2)
    rng = numpy.random.RandomState(seed)
    DROPOUT_RATE = 0.5

    layer_h0 = DropoutLayer(input, input.shape, DROPOUT_RATE)
    layer_h1 = HiddenLayer(
        rng=rng,
        input=layer_h0.output,
        n_in=n_in,
        n_out=nkerns[0],
        activation=activation, bias=bias
    )

    layer_h2 = DropoutLayer(layer_h1.output, layer_h1.output.shape,
                            DROPOUT_RATE)
    layer_h3 = HiddenLayer(
        rng=rng,
        input=layer_h2.output,
        n_in=nkerns[0],
        n_out=nkerns[1],
        activation=activation, bias=bias
    )

    # classify the values of the fully-connected sigmoidal layer
    layer_h4 = LogisticRegression(input=layer_h3.output,
                                  n_in=nkerns[1],
                                  n_out=classes)

    new_layers = [layer_h4, layer_h3, layer_h2, layer_h1, layer_h0]
    all_layers = new_layers + layers[1:]
    return all_layers, new_layers


def extend_net_w1l_drop(input, n_in, layers, classes, nkerns, seed,
                        activation=T.tanh, bias=0.0):
    """
    Extends net with one hidden layers and dropout.
    """
    assert(len(nkerns) == 1)
    rng = numpy.random.RandomState(seed)
    DROPOUT_RATE = 0.5

    layer_h0 = HiddenLayer(
        rng=rng,
        input=input,
        n_in=n_in,
        n_out=nkerns[0],
        activation=activation, bias=bias
    )

    lh1_in = layer_h0.output
    layer_h1 = DropoutLayer(lh1_in, lh1_in.shape, DROPOUT_RATE)

    # classify the values of the fully-connected sigmoidal layer
    layer_h2 = LogisticRegression(input=layer_h1.output,
                                  n_in=nkerns[0],
                                  n_out=classes)

    new_layers = [layer_h2, layer_h1, layer_h0]
    all_layers = new_layers + layers[1:]
    return all_layers, new_layers


def extend_net_w1l(input, n_in, layers, classes, nkerns, seed,
                   activation=T.tanh, bias=0.0):
    """
    Extends net with hidden layers.
    """
    assert(len(nkerns) == 1)
    rng = numpy.random.RandomState(seed)

    layer_h0 = HiddenLayer(
        rng=rng,
        input=input,
        n_in=n_in,
        n_out=nkerns[0],
        activation=activation, bias=bias
    )

    # classify the values of the fully-connected sigmoidal layer
    layer_h1 = LogisticRegression(input=layer_h0.output,
                                  n_in=nkerns[0],
                                  n_out=classes)

    new_layers = [layer_h1, layer_h0]
    all_layers = new_layers + layers[1:]
    return all_layers, new_layers
