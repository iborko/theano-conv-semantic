import theano
import theano.tensor as T
import numpy as np
import logging


logger = logging.getLogger(__name__)


class UpdateParameters(object):
    """
    Holds parameters for updating weights, e.g. learning rate
    """

    def __init__(self, updates, learning_rate):
        """
        updates: list of tuples
            list of pairs of theano update rules
        learning_rate: theano shared variable
            shared variable containing learning rate
        """
        self.updates = updates
        self.learning_rate = learning_rate

    def lower_rate_by_factor(self, factor):
        """
        Updates learning rate by factor:
            rate = rate * factor

        factor: float
            a factor that reduces value of learning rate
        """
        self.learning_rate.set_value(self.learning_rate.get_value() * factor)
        logger.info("Learning rate lowered to %f",
                    self.learning_rate.get_value())


def gradient_updates_SGD(cost, params, learning_rate):
    '''
    Compute updates for gradient descent with momentum

    :parameters:
        - cost : theano.tensor.var.TensorVariable
            Theano cost function to minimize
        - params : list of theano.tensor.var.TensorVariable
            Parameters to compute gradient against
        - learning_rate : float
            Gradient descent learning rate
        - decrease_rate : float
            Rate at which learning rate is decreasing. If left 1.0, there
            will be no decrease. rate *= decrease_Rate

    :returns:
        UpdateParameters object
    '''
    # List of update steps for each parameter
    updates = []

    # shared variable for learning rate
    rate = theano.shared(np.array(learning_rate,
                         dtype=theano.config.floatX))

    # create a list of gradients for all model parameters
    # cost - 0-d tensor scalar with respect to which we are differentiating
    grads = T.grad(cost, params)

    # Just gradient descent on cost
    for param, grad in zip(params, grads):

        # Each parameter is updated by taking a step in the direction of the
        #  gradient.
        updates.append((param, param - learning_rate * grad))

    return UpdateParameters(updates, rate)


def gradient_updates_momentum(cost, params, learning_rate, momentum):
    '''
    Compute updates for gradient descent with momentum

    :parameters:
        - cost : theano.tensor.var.TensorVariable
            Theano cost function to minimize
        - params : list of theano.tensor.var.TensorVariable
            Parameters to compute gradient against
        - learning_rate : float
            Gradient descent learning rate
        - momentum : float
            Momentum parameter, should be at least 0 (standard gradient
            descent) and less than 1

    :returns:
        UpdateParameters object
    '''
    # Make sure momentum is a sane value
    assert momentum < 1 and momentum >= 0

    # shared variable for learning rate
    rate = theano.shared(np.array(learning_rate,
                         dtype=theano.config.floatX))

    # create a list of gradients for all model parameters
    # cost - 0-d tensor scalar with respect to which we are differentiating
    grads = T.grad(cost, params)

    # List of update steps for each parameter
    updates = []
    for param, grad in zip(params, grads):

        # For each parameter, we'll create a param_update shared variable.
        # This variable will keep track of the parameter's update step across
        #  iterations.
        # We initialize it to 0
        param_update = theano.shared(param.get_value() * 0.,
                                     broadcastable=param.broadcastable)

        # Each parameter is updated by taking a step in the direction of the
        #  gradient. However, we also "mix in" the previous step according to
        #  the given momentum value. Note that when updating param_update, we
        #  are using its old value and also the new gradient step.
        updates.append((param, param - rate * param_update))

        updates.append((param_update,
                        momentum * param_update + (1. - momentum) * grad))
    return UpdateParameters(updates, rate)


def gradient_updates_rprop(cost, params, learning_rate, increase=1.2, decrease=0.5):
    '''
    The Rprop method uses the same general strategy as SGD (both methods are
    make small parameter adjustments using local derivative information). The
    difference is that in Rprop, only the signs of the partial derivatives are
    taken into account when making parameter updates. That is, the step size
    for each parameter is independent of the magnitude of the gradient for
    that parameter.

    To accomplish this, Rprop maintains a separate learning rate for every
    parameter in the model, and adjusts this learning rate based on the
    consistency of the sign of the gradient of the loss with respect to that
    parameter over time. Whenever two consecutive gradients for a parameter
    have the same sign, the learning rate for that parameter increases, and
    whenever the signs disagree, the learning rate decreases. This has a
    similar effect to momentum-based SGD methods but effectively maintains
    parameter-specific momentum values.

    The implementation here actually uses the "iRprop-" variant of Rprop
    described in Algorithm 4 from Igel and Huesken (2000), "Improving the
    Rprop Learning Algorithm." This variant resets the running gradient
    estimates to zero in cases where the previous and current gradients
    have switched signs.

    :parameters:
        - cost : theano.tensor.var.TensorVariable
            Theano cost function to minimize
        - params : list of theano.tensor.var.TensorVariable
            Parameters to compute gradient against
        - learning_rate : float
            learning rate
        - increase : float
            Rprop learning rate increase
        - decrease : float
            Rprop learning rate decrease

    :returns:
        updates : list
            List of updates, one for each parameter
    '''
    updates = []

    # rprop constants
    min_step = 1e-6
    max_step = 50

    # shared variable for learning rate
    rate = theano.shared(np.array(learning_rate,
                         dtype=theano.config.floatX))

    # create a list of gradients for all parameters
    grads = T.grad(cost, params)

    for param, grad in zip(params, grads):
        # shared variable for previous gradients
        grad_prev = theano.shared(np.zeros_like(param.get_value()),
                                  name='{}_{}'.format(param.name, 'grad'))
        # shared variable for previous learning rates
        step_prev = theano.shared(np.zeros_like(param.get_value()) + rate,
                                  name='{}_{}'.format(param.name, 'step'))
        test = grad * grad_prev
        same = T.gt(test, 0)
        diff = T.lt(test, 0)
        step = T.minimum(max_step,
                         T.maximum(min_step, step_prev * (
                                   T.eq(test, 0) +
                                   same * increase +
                                   diff * decrease)))
        grad = grad - diff * grad

        updates.append((param, param - T.sgn(grad) * step))
        updates.append((grad_prev, grad))
        updates.append((step_prev, step))

    return UpdateParameters(updates, rate)


def gradient_updates_rms(cost, params, learning_rate, momentum, halflife=7):
    '''
    RmsProp trains neural network models using scaled SGD.
    The RmsProp method uses the same general strategy as SGD (both methods are
    make small parameter adjustments using local derivative information). The
    difference here is that as gradients are computed during each parameter
    update, an exponential moving average of squared gradient magnitudes is
    maintained as well. At each update, the EMA is used to compute the
    root-mean-square (RMS) gradient value that's been seen in the recent past.
    The actual gradient is normalized by this RMS scale before being applied
    to update the parameters.

    Like Rprop, this learning method effectively maintains a sort of
    parameter-specific momentum value, but the difference here is that only
    the magnitudes of the gradients are taken into account, rather than the
    signs. The weight parameter for the EMA window is taken from the
    "momentum" keyword argument. If this weight is set to a low value,
    the EMA will have a short memory and will be prone to changing quickly.
    If the momentum parameter is set close to 1, the EMA will have a long
    history and will change slowly.
    The implementation here is modeled after Graves (2013), "Generating
    Sequences With Recurrent Neural Networks," http://arxiv.org/abs/1308.0850.
    '''

    ewma = float(np.exp(-np.log(2) / halflife))
    logger.debug('RMS prop, EWMA parameter is %.3f' % ewma)
    logger.debug('RMS prop, learning rate is %f' % learning_rate)
    logger.debug('RMS prop, momentum is %.2f' % momentum)

    # list of gradients for all params
    grads = T.grad(cost, params)

    # shared variable for learning rate
    rate = theano.shared(np.array(learning_rate,
                         dtype=theano.config.floatX))

    updates = []
    for param, grad in zip(params, grads):
        # previous values
        g1_prev = theano.shared(np.zeros_like(param.get_value()),
                                name='{}_{}'.format(param.name, 'g1_ewma'))
        g2_prev = theano.shared(np.zeros_like(param.get_value()),
                                name='{}_{}'.format(param.name, 'g2_ewma'))
        vel_prev = theano.shared(np.zeros_like(param.get_value()),
                                 name='{}_{}'.format(param.name, 'vel'))

        g1_t = ewma * g1_prev + (1 - ewma) * grad
        g2_t = ewma * g2_prev + (1 - ewma) * grad * grad
        rms = T.sqrt(g2_t - g1_t * g1_t + 1e-4)
        vel_t = momentum * vel_prev - grad * rate / rms

        updates.append((g1_prev, g1_t))
        updates.append((g2_prev, g2_t))
        updates.append((vel_prev, vel_t))
        updates.append((param, param + vel_t))

    return UpdateParameters(updates, rate)


def gradient_updates_domkorms(cost, params, learning_rate):

    # list of gradients for all params
    grads = T.grad(cost, params)

    eps = 1e-12

    # shared variable for learning rate
    rate = theano.shared(np.array(learning_rate,
                         dtype=theano.config.floatX))
    updates = []
    for param, grad in zip(params, grads):
        # previous values
        mss = theano.shared(np.ones_like(param.get_value()),
                            name='{}_{}'.format(param.name, 'mss'))

        updates.append((mss, mss * 0.9 + 0.1 * grad * grad))
        updates.append((param, param - rate * grad / T.sqrt(mss + eps)))

    return UpdateParameters(updates, rate)
