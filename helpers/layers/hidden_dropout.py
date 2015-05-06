"""
Hidden layer for a neural network with dropout
and autoencoding.
"""
import logging
import theano
from theano.ifelse import ifelse
import theano.tensor as T
import numpy as np

log = logging.getLogger(__name__)


class HiddenLayerDropout(object):

    """
    Artificial neural network hidden layer implementation
    that provides dropout and autoencoding capabilities.

    Probability factor dropout_p determines output neuron retention
    (neuron output is zeroes out with probability (1 - p)).
    During evaluation / use activations are not zeroed out but
    multiplied with dropout_p to ensure that the expected input
    into the next layer remains the same.
    """

    def __init__(self, n_in, n_out, dropout_p=None,
                 activation=T.nnet.sigmoid, bias=0.0,
                 w=None, b=None, input=None, rng=None):
        """
        Creates the hidden layer for an artifical neural network.

        :type n_in: int
        :param n_in: Number of inputs to this layer.

        :type n_out: int
        :param n_out: Number of units (neurons) in this layer.

        :type dropout_p: float
        :param dropout_p: Probability of retaining a neuron when using
            dropout. If None, dropout is not used.

        :type activation: Theano symbolic function.
        :param activation: Activation function for output features.
            Defaults to T.nnet.sigmoid.

        :type w: theano.matrix
        :param w: Weights variable, typically a shared variable
            backed by an ndarray of shape (n_in, n_out). If None it is
            randomly initialzied.

        :type w: theano.vector
        :param w: Bias variable, typically a shared variable
            backed by an ndarray of shape (n_out, ). If None it is
            randomly initialzied.

        :param input: Symbolic Theano variable for data input.
            If None, a matrix variable is created.

        :type rng: numpy.random.RandomState
        :param rng: A random number generator used for all randomness.
        """
        log.info("Creating hidden layer, n_in=%d, n_out=%d, drop_p=%r",
                 n_in, n_out, dropout_p)

        if input is None:
            input = T.matrix("input")
        self.input = input

        #   our randomness generators
        if rng is None:
            rng = np.random.RandomState()

        #   this flag controls whether dropout is used in training mode
        #   (in which neuron activations are droppped), or in eval/running
        #   mode (in which activations are multiplied with p)
        self.training_mode = theano.shared(np.array(1, dtype='int8'))

        #   init weights and biases if necessary
        if w is None:
            w_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == T.nnet.sigmoid:
                w_values *= 4

            w = theano.shared(value=w_values, name='W', borrow=True)
        else:
            assert w.get_value().shape == (n_in, n_out)
        self.W = w

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b_values += bias
            b = theano.shared(value=b_values, name='b', borrow=True)
        else:
            assert b.get_value().size == n_out
        self.b = b

        #   calculate input and output for the layer
        output = activation(T.dot(input, self.W) + self.b)

        #   ******************
        #   DROPOUT CODE START
        #   ******************

        if dropout_p is not None:
            #   we are using dropout
            assert dropout_p > 0. and dropout_p < 1.
            rand = theano.tensor.shared_randomstreams.RandomStreams()

            #   create different outputs for training and evalu
            output_train = output * rand.binomial(
                size=output.shape, p=dropout_p, dtype=theano.config.floatX)
            output_eval = output * dropout_p
            output = ifelse(self.training_mode, output_train, output_eval)

        #   ****************
        #   DROPOUT CODE END
        #   ****************

        self.output = output
        # parameters of the model
        self.params = [self.W, self.b]

    def get_weights(self):
        return (self.W.get_value(), self.b.get_value())

    def set_weights(self, weights):
        W, b = weights
        self.W.set_value(W)
        self.b.set_value(b)
