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


def build_net(x, y, batch_size, classes, image_shape, nkerns, sparse=False,
              activation=T.tanh, bias=0.0):
    """
    Build model for Farabet, Pami (2013) conv network for segmentation

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
    # farabet pami net has 3 conv layers
    assert(len(nkerns) == 3)
    # this version has to have 16 filters in first layer
    assert(nkerns[0] == 16)

    # convolution kernel size
    filter_size = 5

    logger.info('... building the model')

    rng = numpy.random.RandomState(23455)

    # Reshape matrix of rasterized images to a 4D tensor
    # resulting shape is (batch_size, 1, height, width)
    layer0_Y_input = x[:, [0]]  # Y channel
    layer0_UV_input = x[:, [1, 2]]  # U, V channels

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (216-7+1 , 320-7+1) = (210, 314)
    # maxpooling reduces this further to (210/2, 314/2) = (105, 157)
    # 4D output tensor is thus of shape (batch_size, 10, 105, 157)
    #  layer0 has 10 filter maps for Y channel
    layer0_Y = ConvPoolLayer(
        rng,
        input=layer0_Y_input,
        image_shape=(batch_size, 1, image_shape[0], image_shape[1]),
        filter_shape=(10, 1, filter_size, filter_size),
        activation=activation, bias=bias,
    )

    #  layer0 has 6 filter maps for U and V channel
    layer0_UV = ConvPoolLayer(
        rng,
        input=layer0_UV_input,
        image_shape=(batch_size, 2, image_shape[0], image_shape[1]),
        filter_shape=(6, 2, filter_size, filter_size),
        activation=activation, bias=bias,
    )

    # stack outputs from Y, U, V channel layer
    layer0_output = T.concatenate([layer0_Y.output,
                                   layer0_UV.output], axis=1)

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (105-7+1, 157-7+1) = (99, 151)
    # maxpooling reduces this further to (99/2, 151/2) = (49, 75)
    # 4D output tensor is thus of shape (nkerns[0], nkerns[1], 49, 75)
    image_shape1 = reduce_image_dim(image_shape, filter_size, 2)
    filters_to_use1 = nkerns[0] // 2 if sparse else nkerns[0]
    layer1 = ConvPoolLayer(
        rng,
        input=layer0_output,
        image_shape=(batch_size, nkerns[0], image_shape1[0], image_shape1[1]),
        filter_shape=(nkerns[1], filters_to_use1, filter_size, filter_size),
        activation=activation, bias=bias,
    )  # create 64 feature maps from 8 fmaps

    # Construct the third convolutional pooling layer
    # filtering reduces the image size to (49-7+1, 75-7+1) = (43, 69)
    # no maxpooling
    image_shape2 = reduce_image_dim(image_shape1, filter_size, 2)
    filters_to_use2 = nkerns[1] // 2 if sparse else nkerns[1]
    layer2 = ConvPoolLayer(
        rng,
        input=layer1.output,
        image_shape=(batch_size, nkerns[1], image_shape2[0], image_shape2[1]),
        filter_shape=(nkerns[2], filters_to_use2, filter_size, filter_size),
        activation=activation, bias=bias,
        poolsize=(1, 1),  # no pooling
        only_conv=True,  # this layer has only bank of filters
    )  # create 256 feature maps from 32 fmaps

    # Logistic regression operates on (batch_size, pixel_count, feature_size)
    # matrices
    #
    # output from previous layer (Conv layer) is matrix of size (batch_size,
    # feature_size, height, width) first, we dimshuffle this to (batch_size,
    # height, width, feature_size) so that we have features of every pixel
    # of every feature map. Then we transform it to (batch_size * height *
    # * width, feature_size) so that Linear regression can do pixel-wise
    # classification
    layer3_input = layer2.output.dimshuffle(0, 2, 3, 1).reshape((-1, nkerns[2]))
    image_shape3 = reduce_image_dim(image_shape2, filter_size, 1)

    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(input=layer3_input,
                                n_in=nkerns[2],
                                n_out=classes)

    # list of all layers
    layers = [layer3, layer2, layer1, layer0_Y, layer0_UV]
    return layers, tuple(image_shape3)


def build_net1(x, y, batch_size, classes, image_shape, nkerns, sparse=False,
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
    layers: list
        list of kernel_dimensions

    returns: list
        list of all layers, first layer is actually the last (log reg)
    """
    # net has 4 conv layers
    assert(len(nkerns) == 4)
    # this version has to have 16 filters in first layer
    assert(nkerns[0] == 32)

    DROPOUT_RATE = None

    # convolution kernel size
    filter_size = 7

    logger.info('... building the model')

    rng = numpy.random.RandomState(23455)

    layer0_Y_input = x[:, [0]]  # Y channel
    layer0_UV_input = x[:, [1, 2]]  # U, V channels

    # Construct the first convolutional pooling layer
    layer0_Y = ConvPoolLayer(
        rng,
        input=layer0_Y_input,
        image_shape=(batch_size, 1, image_shape[0], image_shape[1]),
        filter_shape=(20, 1, filter_size, filter_size),
        activation=activation, bias=bias,
    )

    #  layer0 has 6 filter maps for U and V channel
    layer0_UV = ConvPoolLayer(
        rng,
        input=layer0_UV_input,
        image_shape=(batch_size, 2, image_shape[0], image_shape[1]),
        filter_shape=(12, 2, filter_size, filter_size),
        activation=activation, bias=bias,
    )

    # stack outputs from Y, U, V channel layer
    layer0_output = T.concatenate([layer0_Y.output,
                                   layer0_UV.output], axis=1)

    # Construct the second convolutional pooling layer
    image_shape1 = reduce_image_dim(image_shape, filter_size, 2)
    filters_to_use1 = nkerns[0] // 2 if sparse else nkerns[0]
    layer1 = ConvPoolLayer(
        rng,
        input=layer0_output,
        image_shape=(batch_size, nkerns[0], image_shape1[0], image_shape1[1]),
        filter_shape=(nkerns[1], filters_to_use1, filter_size, filter_size),
        activation=activation, bias=bias,
    )  # create 64 feature maps from 8 fmaps

    # Construct the third convolutional pooling layer
    image_shape2 = reduce_image_dim(image_shape1, filter_size, 2)
    filters_to_use2 = nkerns[1] // 2 if sparse else nkerns[1]
    layer2 = ConvPoolLayer(
        rng,
        input=layer1.output,
        image_shape=(batch_size, nkerns[1], image_shape2[0], image_shape2[1]),
        filter_shape=(nkerns[2], filters_to_use2, filter_size, filter_size),
        activation=activation, bias=bias,
        poolsize=(1, 1),  # no pooling
    )

    layer3_input = layer2.output.dimshuffle(0, 2, 3, 1).reshape((-1, nkerns[2]))
    image_shape3 = reduce_image_dim(image_shape2, filter_size, 1)

    # construct a dropout hidden layer
    layer3 = HiddenLayerDropout(
        rng=rng,
        input=layer3_input,
        n_in=nkerns[2],
        n_out=nkerns[3],
        activation=activation, bias=bias,
        dropout_p=DROPOUT_RATE
    )

    # classify the values of the fully-connected sigmoidal layer
    layer4 = LogisticRegression(input=layer3.output,
                                n_in=nkerns[3],
                                n_out=classes)

    # list of all layers
    layers = [layer4, layer3, layer2, layer1, layer0_Y, layer0_UV]
    return layers, tuple(image_shape3)


def build_net2(x, y, batch_size, classes, image_shape, nkerns, sparse=False,
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
    nfilters = [7, 5, 3, 3]

    logger.info('... building the model')

    rng = numpy.random.RandomState(23455)

    layer0_Y_input = x[:, [0]]  # Y channel
    layer0_UV_input = x[:, [1, 2]]  # U, V channels

    # Construct the first convolutional pooling layer
    layer0_Y = ConvPoolLayer(
        rng,
        input=layer0_Y_input,
        image_shape=(batch_size, 1, image_shape[0], image_shape[1]),
        filter_shape=(20, 1, nfilters[0], nfilters[0]),
        activation=activation, bias=bias,
    )

    #  layer0 has 6 filter maps for U and V channel
    layer0_UV = ConvPoolLayer(
        rng,
        input=layer0_UV_input,
        image_shape=(batch_size, 2, image_shape[0], image_shape[1]),
        filter_shape=(12, 2, nfilters[0], nfilters[0]),
        activation=activation, bias=bias,
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
        only_conv=True,
        poolsize=(1, 1),  # no pooling
    )

    layer4_input = layer3.output.dimshuffle(0, 2, 3, 1).reshape((-1, nkerns[3]))
    image_shape4 = reduce_image_dim(image_shape3, nfilters[3], 1)

    # classify the values of the fully-connected sigmoidal layer
    layer4 = LogisticRegression(input=layer4_input,
                                n_in=nkerns[3],
                                n_out=classes)

    # list of all layers
    layers = [layer4, layer3, layer2, layer1, layer0_Y, layer0_UV]
    return layers, tuple(image_shape4)


def extend_net(layers, classes, nkerns,
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


def extend_net1(layers, classes, nkerns,
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
