"""
Functions for testing conv net, training and testing on just one image
Version that works with Conv that uses randomization on beginning
"""
import os
import sys
import time
import logging

import numpy as np
import visualize

import theano
import theano.tensor as T

from helpers.data_helper import shared_dataset
from helpers.build_net import build_net, build_net1
from helpers.weight_updates import gradient_updates_rms, gradient_updates_SGD
from helpers.eval import eval_model
from preprocessing.perturb_dataset import change_train_set
from preprocessing.transform_out import resize_marked_image
from util import try_pickle_load

logger = logging.getLogger(__name__)

ReLU = lambda x: T.maximum(x, 0)
lRelU = lambda x: T.maxium(x, 1.0/3.0*x)  # leaky ReLU


def evaluate_conv(path, n_epochs, batch_size):
    """ Evaluates Farabet-like conv network

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer
    """
    ################
    # LOADING DATA #
    ################
    logger.info("... loading data")
    logger.debug("Theano.config.floatX is %s" % theano.config.floatX)

    x_train_allscales = try_pickle_load(path + 'x_train.bin')
    x_train = x_train_allscales[0]  # use only first scale (for now :))
    y_train = try_pickle_load(path + 'y_train.bin')
    x_test_allscales = try_pickle_load(path + 'x_test.bin')
    x_test = x_test_allscales[0]
    y_test = try_pickle_load(path + 'y_test.bin')

    image_shape = (x_train.shape[-2], x_train.shape[-1])
    logger.info("Image shape is %s", image_shape)

    logger.info('Train set has %d images' %
                x_train.shape[0])
    logger.info('Input train set has shape of %s ',
                x_train.shape)
    logger.info('Test set has %d images' %
                x_test.shape[0])
    logger.info('Input test set has shape of %s ',
                x_test.shape)

    # compute number of minibatches for training, validation and testing
    n_train_batches = x_train.shape[0] // batch_size
    n_test_batches = x_test.shape[0] // batch_size
    logger.info('Batch size %d' % (batch_size))

    logger.info("Number of train batches %d" % n_train_batches)
    logger.info("Number of test batches %d" % n_test_batches)

    logger.info("... building network")

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # input is presented as (batch, channel, x, y)
    x = T.tensor4('x')
    # matrix row - batch index, column label of pixel
    # every column is a list of pixel labels (image matrix reshaped to list)
    y = T.imatrix('y')

    # create all layers
    '''
    layers, out_shape = build_net(x, y, batch_size, classes=24,
                                  image_shape=image_shape,
                                  nkerns=[16, 64, 256],
                                  sparse=True)

    '''
    layers, out_shape = build_net1(x, y, batch_size, classes=24,
                                   image_shape=image_shape,
                                   nkerns=[32, 128, 256, 800],
                                   sparse=True,
                                   activation=ReLU, bias=0.001)
    logger.info("Image out shape is %s", out_shape)

    # last layer, log reg
    log_reg_layer = layers[0]
    y_flat = y.flatten(1)

    y_train_shape = (y_train.shape[0], out_shape[0], out_shape[1])
    y_test_shape = (y_test.shape[0], out_shape[0], out_shape[1])

    # resize marked images to out_size of the network
    y_test_downscaled = np.empty(y_test_shape)
    for i in xrange(y_test.shape[0]):
        y_test_downscaled[i] = resize_marked_image(y_test[i], out_shape)

    x_train_shared, y_train_shared = \
        shared_dataset((np.zeros_like(x_train),
                        np.zeros(y_train_shape)))
    x_test_shared, y_test_shared = \
        shared_dataset((x_test,
                        y_test_downscaled))

    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    y_train_shared_i32 = T.cast(y_train_shared, 'int32')
    y_test_shared_i32 = T.cast(y_test_shared, 'int32')

    ###############
    # BUILD MODEL #
    ###############
    logger.info("... building model")

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        (log_reg_layer.errors(y_flat),
         log_reg_layer.negative_log_likelihood(y_flat)),
        givens={
            x: x_test_shared[index * batch_size: (index + 1) * batch_size],
            y: y_test_shared_i32[index * batch_size: (index + 1) * batch_size]
        }
    )

    predict_image = theano.function(
        [index],
        log_reg_layer.y_pred.reshape(out_shape),
        givens={x: x_test_shared[index:index+1]}
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
    cost = log_reg_layer.negative_log_likelihood(y_flat)\
        + 10**-5 * T.sum([T.sum(w ** 2) for w in weights])

    # train_model is a function that updates the model parameters
    train_model = theano.function(
        [index],
        cost,
        # updates=gradient_updates_rms(cost, params, 0.0001, 0.8),
        updates=gradient_updates_SGD(cost, params, 0.001, 0.999),
        givens={
            x: x_train_shared[index * batch_size: (index + 1) * batch_size],
            y: y_train_shared_i32[index * batch_size: (index + 1) * batch_size]
        }
    )
    pre_fn = lambda: change_train_set(
        x_train_shared, x_train,
        y_train_shared, y_train,
        out_shape)

    ###############
    # TRAIN MODEL #
    ###############
    logger.info("... training model")
    start_time = time.clock()
    best_validation_loss, best_iter, best_params = eval_model(
        n_epochs, train_model, test_model, n_train_batches, n_test_batches,
        layers, pre_fn)
    end_time = time.clock()

    # load the best params
    for params, layer in zip(best_params, layers):
        layer.set_weights(params)

    '''
    #   write to file images classified with best params
    n_trainset = x_test_shared.get_value(borrow=True).shape[0]
    [visualize.show_out_image(predict_image(i), title="image" + str(i),
                              show=False, write=True)
     for i in xrange(n_trainset)]
    '''

    #   write to file visualization of the best filters
    # visualize.visualize_array(layer0.W.get_value(), title="Layer_0_weights",
    #                           show=False, write=True)

    logger.info('Best validation score of %f %% obtained at iteration %i, ' %
                (best_validation_loss * 100., best_iter + 1))
    print >> sys.stderr, ('The code for file %s ran for %.2fm' %
                          (os.path.split(__file__)[1],
                           (end_time - start_time) / 60.))


if __name__ == '__main__':
    """
    Examples of usage:
    python train_borko.py
    """
    logging.basicConfig(level=logging.INFO)

    # create a file handler
    handler = logging.FileHandler('output.log')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s  %(message)s')
    handler.setFormatter(formatter)

    # add the handler to the root logger
    logging.getLogger('').addHandler(handler)

    # evaluate_conv()
    evaluate_conv('./data/MSRC/theano_datasets/', n_epochs=300, batch_size=1)
