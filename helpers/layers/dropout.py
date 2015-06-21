import numpy as np
import theano
import logging
from theano.ifelse import ifelse
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

log = logging.getLogger(__name__)


class DropoutLayer(object):
    def __init__(self, input, in_shp, dropout_p=0.5):
        """
        Creates the dropout layer for an artifical neural network.

        :param input: Symbolic Theano variable for data input.
            If None, a matrix variable is created.

        :type in_shp: tuple
        :param in_shp: Shape of input tensor

        :type dropout_p: float
        :param dropout_p: Probability of retaining a neuron when using
            dropout. If None, dropout is not used.
        """
        log.info("Creating dropout layer, drop_p=%r", dropout_p)

        self.input = input

        #   this flag controls whether dropout is used in training mode
        #   (in which neuron activations are droppped), or in eval/running
        #   mode (in which activations are multiplied with p)
        self.training_mode = theano.shared(np.array(1, dtype='int8'))

        assert dropout_p > 0. and dropout_p < 1.
        rand = RandomStreams()

        #   create different outputs for training and evalu
        output_train = input * rand.binomial(
            size=in_shp, p=dropout_p, dtype=theano.config.floatX)
        output_test = input * dropout_p

        self.output = ifelse(self.training_mode, output_train, output_test)

        # parameters of the model
        self.params = None

    def get_weights(self):
        return None

    def set_weights(self, weights):
        pass
