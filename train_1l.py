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
from helpers.build_net import build_net, build_net2, extend_net1
from helpers.weight_updates import gradient_updates_rms, gradient_updates_SGD
from helpers.eval import eval_model
from preprocessing.perturb_dataset import change_train_set
from preprocessing.transform_out import resize_marked_image
from util import try_pickle_load
from helpers.load_conf import load_config
from helpers.load_conf import convert_to_function_params

logger = logging.getLogger(__name__)

ReLU = lambda x: T.maximum(x, 0)
lReLU = lambda x: T.maximum(x, 1.0/3.0*x)  # leaky ReLU


def build_weight_updates(configuration, cost, params):
    """
    configuration: dictionary
        'training' part of network configuration
    """
    update_modes = {}
    update_modes['rms'] = gradient_updates_rms
    update_modes['sgd'] = gradient_updates_SGD

    p = convert_to_function_params(configuration['optimization-params'])
    p['cost'] = cost
    p['params'] = params
    return update_modes[configuration['optimization']](**p)


def evaluate_conv(conf, net_weights=None):
    """ Evaluates Farabet-like conv network

    conf: dictionary
        network configuration
    """
    ################
    # LOADING DATA #
    ################
    logger.info("... loading data")
    logger.debug("Theano.config.floatX is %s" % theano.config.floatX)

    path = conf['data']['location']
    batch_size = conf['evaluation']['batch-size']
    assert(type(batch_size) is int)
    logger.info('Batch size %d' % (batch_size))

    try:
        x_train_allscales = try_pickle_load(path + 'x_train.bin')
        x_train = x_train_allscales[0]  # first scale
        y_train = try_pickle_load(path + 'y_train.bin')
        x_test_allscales = try_pickle_load(path + 'x_test.bin')
        x_test = x_test_allscales[0]
        y_test = try_pickle_load(path + 'y_test.bin')
    except IOError:
        logger.error("Unable to load Theano dataset from %s", path)
        exit(1)

    n_classes = int(max(y_train.max(), y_test.max()) + 1)
    logger.info("Dataset has %d classes", n_classes)

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
    layers, out_shape = build_net(x, y, batch_size, classes=NCLASSES.
                                  image_shape=image_shape,
                                  nkerns=[16, 64, 256],
                                  sparse=True)

    '''
    layers, out_shape = build_net2(x, y, batch_size, classes=n_classes,
                                   image_shape=image_shape,
                                   nkerns=[32, 128, 256, 256],
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
        [log_reg_layer.errors(y_flat),
         log_reg_layer.negative_log_likelihood(y_flat)] +
        list(log_reg_layer.accurate_pixels_class(y_flat)),
        givens={
            x: x_test_shared[index * batch_size: (index + 1) * batch_size],
            y: y_test_shared_i32[index * batch_size: (index + 1) * batch_size]
        }
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
    update_params = build_weight_updates(conf['training'], cost, params)
    train_model = theano.function(
        [index],
        cost,
        updates=update_params.updates,
        givens={
            x: x_train_shared[index * batch_size: (index + 1) * batch_size],
            y: y_train_shared_i32[index * batch_size: (index + 1) * batch_size]
        }
    )
    pre_fn = lambda: change_train_set(
        x_train_shared, x_train,
        y_train_shared, y_train,
        out_shape)

    # set loaded weights
    if net_weights is not None:
        try:
            for net_weight, layer in zip(net_weights, layers):
                layer.set_weights(net_weight)
            logger.info("Loaded net weights from file.")
            net_weights = None
        except:
            logger.error("Uncompatible network to load weights in")

    ###############
    # TRAIN MODEL #
    ###############
    logger.info("... training model")
    start_time = time.clock()
    best_validation_loss, best_iter, best_params = eval_model(
        conf['training'], train_model, test_model,
        n_train_batches, n_test_batches,
        layers, pre_fn, update_params)
    end_time = time.clock()

    logger.info('Best validation score of %f %% obtained at iteration %i, ' %
                (best_validation_loss * 100., best_iter + 1))
    print >> sys.stderr, ('The code for file %s ran for %.2fm' %
                          (os.path.split(__file__)[1],
                           (end_time - start_time) / 60.))

    logger.info('Starting second step, with Dropout hidden layers')
    layers, new_layers = extend_net1(
        layers, NCLASSES,
        nkerns=[1000], activation=ReLU, bias=0.001)

    # create a function to compute the mistakes that are made by the model
    test_model2 = theano.function(
        [index],
        [layers[0].errors(y_flat),
         layers[0].bayesian_nll(y_flat)] +
        list(layers[0].accurate_pixels_class(y_flat)),
        givens={
            x: x_test_shared[index * batch_size: (index + 1) * batch_size],
            y: y_test_shared_i32[index * batch_size: (index + 1) * batch_size]
        }
    )

    # create a list of all model parameters to be fit by gradient descent
    params2 = [p for l in new_layers for p in l.params]
    # list of Ws through all layers
    weights2 = [l.params[0] for l in new_layers]

    assert(len(weights2) == len(params2)/2)

    # the cost we minimize during training is the NLL of the model
    #  and L2 regularization (lamda * L2-norm)
    # L2-norm is sum of squared params (using only W, not b)
    #  params has Ws on even locations
    cost2 = layers[0].negative_log_likelihood(y_flat)\
        + 10**-3 * T.sum([T.sum(w ** 2) for w in weights2])

    # train_model is a function that updates the model parameters
    update_params2 = build_weight_updates(conf['training2'], cost2, params2)
    train_model2 = theano.function(
        [index],
        cost2,
        updates=update_params2.updates,
        givens={
            x: x_train_shared[index * batch_size: (index + 1) * batch_size],
            y: y_train_shared_i32[index * batch_size: (index + 1) * batch_size]
        }
    )

    # try to load weights in second stage
    if net_weights is not None:
        try:
            for net_weight, layer in zip(net_weights, layers):
                layer.set_weights(net_weight)
            logger.info("Loaded net weights from file.")
            net_weights = None
        except:
            logger.error("Uncompatible network to load weights in")

    # evaluate model2
    start_time = time.clock()
    best_validation_loss, best_iter, best_params = eval_model(
        conf['training2'], train_model2, test_model2,
        n_train_batches, n_test_batches,
        layers, pre_fn, update_params2)
    end_time = time.clock()

    logger.info('Best validation score of %f %% obtained at iteration %i, ' %
                (best_validation_loss * 100., best_iter + 1))
    print >> sys.stderr, ('The code for file %s ran for %.2fm' %
                          (os.path.split(__file__)[1],
                           (end_time - start_time) / 60.))


if __name__ == '__main__':
    """
    Examples of usage:
    python train_2step.py network.conf

    python train_2step.py network.conf network-12-34.bin
        trains network starting with weights in network-*.bin file
    """
    logging.basicConfig(level=logging.INFO)

    # create a file handler
    handler = logging.FileHandler('output.log')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s  %(message)s')
    handler.setFormatter(formatter)

    # add the handler to the root logger
    logging.getLogger('').addHandler(handler)

    argc = len(sys.argv)
    if argc == 3:
        net_config_path = sys.argv[1]
        params = try_pickle_load(sys.argv[2])
        if params is None:
            exit(1)
    elif argc == 2:
        net_config_path = sys.argv[1]
        params = None
    else:
        logger.error("Too few arguments")
        exit(1)

    conf = load_config(net_config_path)
    if conf is None:
        exit(1)

    #   run evaluation
    evaluate_conv(conf, net_weights=params)
