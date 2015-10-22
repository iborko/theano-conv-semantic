"""
Convolutional layer for an artifical neural network.
Supports sparse connectivity between input and output features.
"""
import logging

import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv


logger = logging.getLogger(__name__)


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


class ConvLayer(object):
    """
    Convolutional layer of a artificial neural network.
    """

    def __init__(self, rng, input, filter_shape, image_shape,
                 activation=T.tanh, bias=0.0, stride=(1, 1),
                 border_mode='valid',
                 W=None, b=None):
        """
        Allocate layer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)
        """
        logger.info("Creating conv layer with filter shape %r,"
                    "image shape: %r", filter_shape, image_shape)

        #   filter_shape[1] is number of maps used from layer before
        maps_to_use = filter_shape[1]
        assert image_shape[1] == maps_to_use
        self.input = input

        #   there are "num input feature maps * filter height * filter width"
        #   inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        #   each unit in the lower layer receives a gradient from:
        #   "num output feature maps * filter height * filter width" /
        #     pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]))
        #           numpy.prod(poolsize))
        #   initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        W_values = rng.uniform(low=-W_bound, high=W_bound,
                               size=filter_shape)

        if W is None:
            self.W = theano.shared(
                numpy.asarray(
                    W_values,
                    dtype=theano.config.floatX),
                borrow=True)
        else:
            self.W = W

        if b is None:
            #   the bias is a 1D tensor -- one bias per output feature map
            b_values = numpy.zeros((filter_shape[0],),
                                   dtype=theano.config.floatX)
            b_values += bias
            self.b = theano.shared(value=b_values, borrow=True)
        else:
            self.b = b

        b_mode = 'full' if border_mode == 'same' else border_mode
        # convolve input feature maps with filters
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape,
            subsample=stride,
            border_mode=b_mode
        )
        if border_mode == 'same':
            assert(filter_shape[-1] == filter_shape[-2])
            img_shape = (image_shape[2], image_shape[3])
            conv_out = crop_to_size(conv_out, img_shape, filter_shape[-1])
            # print "---> cropped image"

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = activation(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]

    def get_weights(self):
        return (self.W.get_value(), self.b.get_value())

    def set_weights(self, weights):
        W, b = weights
        self.W.set_value(W)
        self.b.set_value(b)

    def __getstate__(self):
        return (self.W.get_value(), self.b.get_value(),
                self.input, self.output)

    def __setstate__(self, state):
        W, b, input, output = state
        self.W = theano.shared(W, borrow=True)
        self.b = theano.shared(W, borrow=True)
        self.input = input
        self.output = output


class ConvPoolLayer(object):
    """
    Convolutional and pooling layer of a artificial neural network.
    Supports sparse connectivity between layers.
    """

    def __init__(self, rng, input, filter_shape, image_shape,
                 poolsize=(2, 2), only_conv=False, activation=T.tanh,
                 bias=0.0, stride=(1, 1), ignore_border_pool=True,
                 border_mode='valid',
                 W=None, b=None):
        """
        Allocate layer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """
        logger.info("Creating conv layer with filter shape %r,"
                    "image shape: %r", filter_shape, image_shape)

        #   filter_shape[1] is number of maps used from layer before
        maps_to_use = filter_shape[1]
        #   get random filter maps
        #   from each image randomly select MAPS_TO_USE filter maps
        n_in_feature_maps = image_shape[1]
        n_out_feature_maps = filter_shape[0]

        #   filter shape
        #   if layer is sparse, half of values will be zeroed
        filter_shape_sparse = list(filter_shape)
        filter_shape_sparse[1] = n_in_feature_maps

        assert image_shape[1] >= maps_to_use
        self.input = input

        #   there are "num input feature maps * filter height * filter width"
        #   inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        #   each unit in the lower layer receives a gradient from:
        #   "num output feature maps * filter height * filter width" /
        #     pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        #   initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        W_values = rng.uniform(low=-W_bound, high=W_bound,
                               size=filter_shape_sparse)

        if maps_to_use == n_in_feature_maps:
            logger.info("Layer fully connected")
        else:
            logger.info("Layer sparse (random) connected")

            #   list of feature indices ([0, 1, ..., n_features_in])
            feature_indices = numpy.tile(
                numpy.arange(n_in_feature_maps, dtype='int32'),
                (n_out_feature_maps, 1))
            #   shuffle every row
            map(lambda row: rng.shuffle(row), feature_indices)
            #   prepare the indices which will be zeroed out
            to_zero_out = feature_indices[:, maps_to_use:]
            feature_indices = feature_indices[:, :maps_to_use]
            logger.debug('Feature indices:\n%r', feature_indices)
            logger.debug('Will zero out:\n%r', to_zero_out)

            #   zero out unused weights
            for ind, tzo_row in enumerate(to_zero_out):
                W_values[ind, tzo_row] = 0.

        if W is None:
            self.W = theano.shared(
                numpy.asarray(
                    W_values,
                    dtype=theano.config.floatX),
                borrow=True)
        else:
            self.W = W

        if b is None:
            #   the bias is a 1D tensor -- one bias per output feature map
            b_values = numpy.zeros((filter_shape[0],),
                                   dtype=theano.config.floatX)
            b_values += bias
            self.b = theano.shared(value=b_values, borrow=True)
        else:
            self.b = b

        # convolve input feature maps with filters
        b_mode = 'full' if border_mode == 'same' else border_mode
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape_sparse,
            image_shape=image_shape,
            subsample=stride,
            border_mode=b_mode
        )
        if border_mode == 'same':
            assert(filter_shape[-1] == filter_shape[-2])
            img_shape = (image_shape[2], image_shape[3])
            conv_out = crop_to_size(conv_out, img_shape, filter_shape[-1])
            # print "---> cropped image"

        # mode where this layer is just a bank of filters
        if only_conv:
            logger.debug("--Only conv mode")
            self.output = conv_out + self.b.dimshuffle('x', 0, 'x', 'x')

        else:
            logger.debug("--Pooling full mode")
            # add the bias term. Since the bias is a vector (1D array), we first
            # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
            # thus be broadcasted across mini-batches and feature map
            # width & height
            tanh_out = activation(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))

            # downsample each feature map individually, using maxpooling
            pooled_out = downsample.max_pool_2d(
                input=tanh_out,
                ds=poolsize,
                ignore_border=ignore_border_pool,
            )

            self.output = pooled_out

        # store parameters of this layer
        self.params = [self.W, self.b]

    def get_weights(self):
        return (self.W.get_value(), self.b.get_value())

    def set_weights(self, weights):
        W, b = weights
        self.W.set_value(W)
        self.b.set_value(b)

    def __getstate__(self):
        return (self.W.get_value(), self.b.get_value(),
                self.input, self.output)

    def __setstate__(self, state):
        W, b, input, output = state
        self.W = theano.shared(W, borrow=True)
        self.b = theano.shared(W, borrow=True)
        self.input = input
        self.output = output


def test_layer():

    rng = numpy.random.RandomState(23455)

    input_vals = numpy.zeros((1, 3, 3, 3))
    input_vals[0, 0] = numpy.arange(9).reshape((3, 3))
    input_vals[0, 1] = input_vals[0, 0][::-1]
    input_vals[0, 2] = numpy.arange(8, -1, -1).reshape((3, 3))
    input_vals = input_vals.astype(theano.config.floatX) / 10

    input = theano.shared(value=input_vals, borrow=True)
    print 'Input'
    print input.get_value()

    layer = ConvPoolLayer(
        rng,
        input=input,
        image_shape=(1, 3, 3, 3),
        filter_shape=(2, 2, 2, 2)
    )

    result = layer.output.eval()
    print 'Result'
    print result


def test_gradient():

    from logistic_sgd import LogisticRegression

    rng = numpy.random.RandomState(23455)

    input_vals = numpy.zeros((1, 3, 3, 3))
    input_vals[0, 0] = numpy.arange(9).reshape((3, 3))
    input_vals[0, 1] = input_vals[0, 0][::-1]
    input_vals[0, 2] = numpy.arange(8, -1, -1).reshape((3, 3))
    input_vals = input_vals.astype(theano.config.floatX) / 10
    output_vals = numpy.array([1], dtype='int32')

    input = theano.shared(value=input_vals, borrow=True)
    out = theano.shared(value=output_vals, borrow=True)

    print 'Input'
    print input.get_value()

    x = T.tensor4('x')
    y = T.ivector('y')

    layer0 = ConvPoolLayer(
        rng,
        input=x,
        image_shape=(1, 3, 3, 3),
        filter_shape=(2, 2, 2, 2)
    )

    layer1 = LogisticRegression(layer0.output.flatten(), 2, 2)

    compute_conv_out = theano.function(
        [],
        layer0.output,
        givens={x: input}
    )

    # create a list of all model parameters to be fit by gradient descent
    params = layer1.params + layer0.params

    # the cost we minimize during training is the NLL of the model
    # and L2 regularization (lamda * L2-norm)
    cost = layer1.negative_log_likelihood(y)

    # create a list of gradients for all model parameters
    # cost - 0-d tensor scalar with respect to which we are differentiating
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - 0.5 * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        [],
        cost,
        updates=updates,
        givens={x: input, y: out}
    )

    test_model = theano.function(
        [],
        layer1.errors(y),
        givens={x: input, y: out}
    )

    print 'Conv output'
    print compute_conv_out()

    print ('Test error of %.2f %%' % (100.0 * test_model()))

    print 'Weights before train'
    print layer0.W.get_value()

    print 'Train ouput', train_model()

    print 'Weights after train1'
    print layer0.W.get_value()

    print 'Train ouput', train_model()

    print 'Weights after train2'
    print layer0.W.get_value()

    print ('Test error of %.2f %%' % test_model())


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    test_layer()
    # test_gradient()
