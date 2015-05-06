"""
Implementation of hidden layer for an artificial neural network
From:
    http://deeplearning.net/tutorial/
"""
import numpy

import theano
import theano.tensor as T


class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh, bias=0.0):
        """
        Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer

        :type bias: float
        :param bias: initial bias
        """
        self.input = input

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)
        else:
            assert W.get_value().shape == (n_in, n_out)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b_values += bias
            b = theano.shared(value=b_values, name='b', borrow=True)
        else:
            assert b.get_value().size == n_out

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
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
