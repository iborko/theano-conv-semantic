"""
Functions for testing conv net, training and testing on just one image
Version that works with Conv that uses randomization on beginning
"""
from os.path import join
import sys
import time
import logging

import numpy as np

import theano
import theano.tensor as T

from helpers.data_helper import shared_dataset
from helpers.build_multiscale import get_net_builder
from helpers.build_multiscale import extend_net_w1l_drop
# from preprocessing.transform_out import resize_marked_image
from util import try_pickle_load
from helpers.load_conf import load_config

logger = logging.getLogger(__name__)


def lReLU(x): return T.maximum(x, 1./5*x)  # leaky ReLU

#   TODO refactor
# OUT_PATH = './data/kitti/probs/'
OUT_PATH = './data/iccv09Data/probs/'


def set_layers_training_mode(layers, mode):
    """
    Sets training mode in layers which support traning_mode
    """
    for i, layer in enumerate(layers):
        if 'training_mode' in layer.__dict__:
            layer.training_mode.set_value(mode)


def validate(conf, net_weights):

    logger.info("... loading data")

    path = conf['data']['location']
    batch_size = 1

    logfiles_path = path + 'samples_' + conf['run-dataset'] + '.log'
    logger.info('Log files path %s', logfiles_path)

    try:
        data_x_allscales = try_pickle_load(
            path + 'x_' + conf['run-dataset'] + '.bin')
        data_x = data_x_allscales[0]  # first scale
        data_y = try_pickle_load(
            path + 'y_' + conf['run-dataset'] + '.bin')
    except IOError:
        logger.error("Unable to load Theano dataset from %s", path)
        exit(1)

    y_valid = try_pickle_load(path + 'y_validation.bin')
    print path + 'y_validation.bin'
    n_classes = int(max(data_y.max(), y_valid.max()) + 1)
    logger.info("Dataset has %d classes", n_classes)
    del y_valid

    image_shape = (data_x.shape[-2], data_x.shape[-1])
    logger.info("Image shape is %s", image_shape)

    logger.info('Dataset has %d images' %
                data_x.shape[0])
    logger.info('Input data has shape of %s ',
                data_x.shape)

    # compute number of minibatches
    n_batches = data_x.shape[0] // batch_size

    logger.info("Number of train batches %d" % n_batches)

    logger.info("... building network")

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # input is presented as (batch, channel, x, y)
    x0 = T.tensor4('x')
    x2 = T.tensor4('x')
    x4 = T.tensor4('x')
    # matrix row - batch index, column label of pixel
    # every column is a list of pixel labels (image matrix reshaped to list)
    y = T.imatrix('y')

    # create all layers
    builder_name = conf['network']['builder-name']
    layers, out_shape, conv_out = get_net_builder(builder_name)(
        x0, x2, x4, y, batch_size, classes=n_classes,
        image_shape=image_shape,
        nkerns=conf['network']['layers'][:3],
        seed=conf['network']['seed'],
        activation=lReLU, bias=0.001,
        sparse=False)
    logger.info("Image out shape is %s", out_shape)

    y_shape = (data_y.shape[0], out_shape[0], out_shape[1])

    # resize marked images to out_size of the network
    # dummy objecy - y is not used because there is no training
    y_downscaled = np.empty(y_shape)

    x_shared, y_shared = \
        shared_dataset((data_x,
                        y_downscaled))
    x2_shared = theano.shared(data_x_allscales[1], borrow=True)
    x4_shared = theano.shared(data_x_allscales[2], borrow=True)

    ###############
    # BUILD MODEL #
    ###############
    logger.info("... building model")

    layers, new_layers = extend_net_w1l_drop(
        conv_out, conf['network']['layers'][-2] * 3, layers, n_classes,
        nkerns=conf['network']['layers'][-1:],
        seed=conf['network']['seed'],
        activation=lReLU, bias=0.001)

    eval_model = theano.function(
        [index],
        [layers[0].p_y_given_x],
        givens={
            x0: x_shared[index * batch_size: (index + 1) * batch_size],
            x2: x2_shared[index * batch_size: (index + 1) * batch_size],
            x4: x4_shared[index * batch_size: (index + 1) * batch_size]
        }
    )

    # try to load weights
    try:
        if net_weights is not None:
            for net_weight, layer in zip(net_weights, layers):
                layer.set_weights(net_weight)
            logger.info("Loaded net weights from file.")
            net_weights = None
    except:
        logger.error("Uncompatible network to load weights in")
        exit(1)

    set_layers_training_mode(layers, 0)

    logger.info("---> Results - no postprocessing")
    start_time = time.clock()
    #   if you want outputs resized to (48, 152, 12)
    # shp = (out_shape[0], out_shape[1], n_classes)
    # evaluation = [eval_model(i)[0].reshape(shp)
    #               for i in xrange(n_batches)]
    evaluation = [eval_model(i)[0]
                  for i in xrange(n_batches)]
    end_time = time.clock()
    logger.info("Evaluated %d images in %.2f seconds",
                n_batches, end_time - start_time)
    logger.info("Image shape %s", evaluation[0].shape)

    #   load image names
    with open(logfiles_path, 'r') as f:
        fnames = f.read().split('\n')
    for ind, img in enumerate(evaluation):
        with open(join(OUT_PATH, fnames[ind]+".txt"), 'w') as f:
            f.writelines('\n'.join(' '.join(str(x) for x in y) for y in img))


if __name__ == '__main__':
    """
    Examples of usage:
    python probs.py train.conf network-12-34.bin [train/validation/test]
        validates network
    """
    logging.basicConfig(level=logging.INFO)

    argc = len(sys.argv)
    if argc == 4:
        net_config_path = sys.argv[1]
        params = try_pickle_load(sys.argv[2])
        dataset = sys.argv[3]
        if dataset not in ['train', 'validation', 'test']:
            print "Wrong dataset type: train/validation/test"
            exit(1)
        if params is None:
            exit(1)
    else:
        logger.error("Too few arguments")
        exit(1)

    conf = load_config(net_config_path)
    if conf is None:
        exit(1)

    conf['run-dataset'] = dataset

    #   run evaluation
    validate(conf, params)
