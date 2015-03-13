"""
Functions for testing conv net, training and testing on just one image
Version that works with Conv that uses randomization on beginning
"""
import os
import sys
import time
import logging

import numpy
import visualize

import theano
import theano.tensor as T

from helpers.data_helper import shared_dataset, load_data
from helpers.build_net import build_net
from helpers.weight_updates import gradient_updates_rms
from helpers.eval import eval_model


# DATA_PATH = "/media/Win/Data/MSRC_images/theano_datasets/"
DATA_PATH = "/media/Win/Data/"
logger = logging.getLogger(__name__)


def evaluate_conv(n_epochs=200, batch_size=1):
    """ Evaluates Farabet-like conv network

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer
    """
    logger.info("... loading data")
    logger.debug("Theano.config.floatX is %s" % theano.config.floatX)

    train_set_x, train_set_y = \
        shared_dataset(load_data(DATA_PATH + 'x_train.bin',
                                 DATA_PATH + 'y_train.bin'))
    test_set_x, test_set_y = \
        shared_dataset(load_data(DATA_PATH + 'x_test.bin',
                                 DATA_PATH + 'y_test.bin'))

    logger.info('Train set has %d images' %
                train_set_x.get_value().shape[0])
    logger.info('Input  train set has shape of %s ',
                train_set_x.get_value().shape)
    # print 'Output train set has shape of',\
    #     train_set_y.get_value().shape

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]

    n_train_batches /= batch_size
    n_test_batches /= batch_size
    # n_test_batches = 1

    logger.debug("Number of train batches %d" % n_train_batches)
    logger.debug("Number of test batches %d" % n_test_batches)

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # input is presented as (batch, channel, x, y)
    x = T.tensor4('x')
    # matrix row - batch index, column label of pixel
    # every column is a list of pixel labels (image matrix reshaped to list)
    y = T.imatrix('y')

    # create all layers
    layers = build_net(x, y, batch_size, 9, (216, 320), [16, 64, 256], True)
    # last layer, log reg
    log_reg_layer = layers[0]

    y_data = y.flatten(1)

    ###############
    # BUILD MODEL #
    ###############

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        (log_reg_layer.errors(y_data), log_reg_layer.negative_log_likelihood(y_data)),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    predict_image = theano.function(
        [index],
        log_reg_layer.y_pred.reshape((43, 69)),
        givens={x: train_set_x[index:index+1]}
    )

    # create a list of all model parameters to be fit by gradient descent
    params = [p for l in layers for p in l.params]
    # list of Ws through all layers
    weights = [l.params[0] for l in layers]

    assert(len(weights) == len(params)/2)

    # the cost we minimize during training is the NLL of the model
    #  and L2 regularization (lamda * L2-norm)
    # L2-norm is sum of squared params (using only W, not b)
    #  params has Ws on even locations
    cost = log_reg_layer.negative_log_likelihood(y_data)\
        + 10**-5 * T.sum([T.sum(w ** 2) for w in weights])

    # train_model is a function that updates the model parameters
    train_model = theano.function(
        [index],
        cost,
        updates=gradient_updates_rms(cost, params, 0.0001, 0.8),
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    ###############
    # TRAIN MODEL #
    ###############
    start_time = time.clock()
    best_validation_loss, best_iter, best_params = eval_model(
        n_epochs, train_model, test_model, n_train_batches, n_test_batches,
        layers)
    end_time = time.clock()

    # load the best params
    for params, layer in zip(best_params, layers):
        layer.set_weights(params)

    #   write to file images classified with best params
    n_trainset = train_set_x.get_value(borrow=True).shape[0]
    [visualize.show_out_image(predict_image(i), title="image" + str(i),
                              show=False, write=True)
     for i in xrange(n_trainset)]

    #   write to file visualization of the best filters
    # visualize.visualize_array(layer0_Y.W.get_value(), title="Layer_0_weights",
    #                           show=False, write=True)

    logger.info('Best validation score of %f %% obtained at iteration %i, ' %
                (best_validation_loss * 100., best_iter + 1))
    print >> sys.stderr, ('The code for file %s ran for %.2fm' %
                          (os.path.split(__file__)[1],
                           (end_time - start_time) / 60.))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # evaluate_conv()
    evaluate_conv(n_epochs=10, batch_size=1)
