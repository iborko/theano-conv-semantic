"""
Implementation of multiclass logistic regression layer with softmax.
From:
    http://deeplearning.net/tutorial/
"""

import numpy

import theano
import theano.tensor as T


def build_loss(log_reg_layer, func_name, *loss_args):
    """
    Build loss function
    log_reg_layer: helpers.layers.log_reg.LogisticRegression object
        classification layer of a net
    func_name: string
        loss function name
    """
    assert(type(log_reg_layer) is LogisticRegression)

    loss_function = LogisticRegression.__dict__[func_name]
    n_args = loss_function.func_code.co_argcount - 1  # minus self
    selected_args = loss_args[:n_args]
    return loss_function(log_reg_layer, *selected_args)


class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        self.n_classes = n_out

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyper plain for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of hyper
        # plain-k
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y, p_c=None, care_classes=None):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        p_correct_classes = self.p_y_given_x[T.arange(y.shape[0]), y]
        if care_classes is not None:
            p_correct_classes = T.set_subtensor(
                p_correct_classes[T.eq(care_classes[y], 0).nonzero()],
                T.cast(1.0, 'float32'))
        return -T.mean(T.log(p_correct_classes))

    def boost_negative_log_likelihood(self, y, alpha):
        """
        Using boosted cross entropy for log-likelihood,
        http://research.microsoft.com/pubs/230081/IS140944.pdf

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        :type alpha: int
        :param alpha: boosting order (default set to 1, 2)
        """
        p_correct_classes = self.p_y_given_x[T.arange(y.shape[0]), y]
        return -T.mean(T.pow(1 - p_correct_classes, alpha)
                       * T.log(p_correct_classes))

    def bayesian_nll_ds(self, y, p_c, care_classes=None):
        """
        Bayesian negative log likelihood (uses class apriors to calc loss)
        Class priors calculated per dataset.

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        :type p_c: theano.tensor.TensorType
        :param y: aprior class probabilities in train dataset

        :type care_classes: theano.tensor.TensorType
        :param care_classes: indices of classes whose gradient we track,
                             other gradients are set to zero
        """
        p_correct_classes = self.p_y_given_x[T.arange(y.shape[0]), y]
        if care_classes is not None:
            p_correct_classes = T.set_subtensor(
                p_correct_classes[T.eq(care_classes[y], 0).nonzero()],
                T.cast(1.0, 'float32'))
        return -T.mean(T.log(p_correct_classes) / p_c[y])

    def bayesian_nll(self, y):
        """
        Bayesian negative log likelihood (uses class apriors to calc loss)
        Class priors calculated per minibatch.
        """
        # TODO check if training loss in inf
        p_c = T.zeros((self.n_classes), dtype='float32')
        total = T.prod(y.shape, dtype='float32')
        for i in range(self.n_classes):
            p_c = T.set_subtensor(
                p_c[i],
                T.sum(T.eq(y, i), dtype='float32') / total)
        p_correct_classes = self.p_y_given_x[T.arange(y.shape[0]), y]
        return -1. / self.n_classes *\
            T.mean(T.log(p_correct_classes) / p_c[y])

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y), dtype='float32')
        else:
            raise NotImplementedError()

    def accurate_pixels_class(self, y):
        """
        Returns number of correctly classified pixels per class
        and total number of pixels per class.
        (pair of numpy 1d arrays)

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """
        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if not y.dtype.startswith('int'):
            raise NotImplementedError()

        correct = T.zeros((self.n_classes), dtype='int32')
        total = T.zeros((self.n_classes), dtype='int32')
        for i in range(self.n_classes):
            correct = T.set_subtensor(
                correct[i],
                T.switch(
                    T.any(T.eq(y, i)),
                    T.sum(T.eq(y[T.eq(y, i).nonzero()],
                               self.y_pred[T.eq(y, i).nonzero()])),
                    0)
                )
            total = T.set_subtensor(total[i], T.sum(T.eq(y, i)))
        return correct, total

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
