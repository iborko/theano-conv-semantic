"""
Functions for testing conv net, training and testing on just one image
Version that works with Conv that uses randomization on beginning
"""
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
from preprocessing.transform_in import resize
from preprocessing.file_helper import open_image
from util import try_pickle_load
from helpers.load_conf import load_config
from helpers.eval import calc_class_accuracy
from scipy.ndimage import zoom
from postprocessing.superpixel import segment
import matplotlib.pyplot as plt
import matplotlib.colors
from sklearn.grid_search import ParameterGrid

logger = logging.getLogger(__name__)

ReLU = lambda x: T.maximum(x, 0)
lReLU = lambda x: T.maximum(x, 1./5*x)  # leaky ReLU

IMGS = "./data/iccv09Data/images/"
SHAPE = (240, 320)
NET_OUT_SHAPE = (60, 80)
IMGS_TO_SHOW = 8


def set_layers_training_mode(layers, mode):
    """
    Sets training mode in layers which support traning_mode
    """
    for i, layer in enumerate(layers):
        if 'training_mode' in layer.__dict__:
            # logger.info('Found layer with trainig mode %d, setting to %d',
            #              i, mode)
            layer.training_mode.set_value(mode)


def find_best_superpixel_params(func):
    params_grid = {
        'sigma': [0.3, 0.5, 0.8],
        'k': [100, 200, 300, 400],
        'min_size': [50, 100, 200, 300]
    }
    best_params = None
    best_result = 0.0
    for params in ParameterGrid(params_grid):
        result = func(params)
        if result > best_result:
            best_result = result
            best_params = params

    return best_params


def oversegment(orig_img, marked, sigma, k, min_size):
    segmented = segment(orig_img, sigma, k, min_size)
    # crop to real size
    segmented = segmented[:SHAPE[0], :SHAPE[1]]
    """
    # DEBUG
    plt.subplot(2, 1, 1)
    plt.axis('off')
    plt.imshow(orig_img)
    plt.subplot(2, 1, 2)
    plt.axis('off')
    plt.imshow(segmented, cmap=plt.get_cmap('Paired'))
    plt.show()
    plt.draw()
    """
    new_mark = np.copy(marked)
    for i in np.unique(segmented):
        curr_marks = marked[segmented == i]
        most_freq = np.bincount(curr_marks).argmax()
        new_mark[segmented == i] = most_freq

    return new_mark


def get_stats(results, y, n_classes, dont_care_classes,
              fnames_path, dataset_type, show=False, log=True,
              postproc=None, postproc_params=None):
    assert(len(results) == y.shape[0])
    fnames = []
    with open(fnames_path, 'r') as f:
        fnames = f.read().split('\n')

    care_classes = np.ones((n_classes), dtype='int8')
    care_classes[dont_care_classes] = 0

    correct = np.zeros((n_classes), dtype='int32')
    total = np.zeros((n_classes), dtype='int32')
    for ind, img in enumerate(results):
        curr_y = y[ind]
        img_up = zoom(img, 4, order=0)
        img_old = img_up
        assert(img_up.shape[0] == SHAPE[0])

        images_path = IMGS
        orig_img = open_image(images_path, fnames[ind] + '.jpg')
        orig_img = resize(orig_img, SHAPE)
        # apply postprocessing
        if postproc is not None:
            img_up = postproc(orig_img, img_up, **postproc_params)

        if show and ind < IMGS_TO_SHOW:
            bounds = np.linspace(0, 30, 31)
            # cmap = plt.get_cmap('Paired')
            cmap = plt.get_cmap('gist_ncar')
            norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
            plt.subplot(2, 2, 1)
            plt.axis('off')
            plt.imshow(orig_img)
            plt.subplot(2, 2, 2)
            plt.axis('off')
            plt.imshow(curr_y, cmap=cmap, norm=norm)
            plt.subplot(2, 2, 3)
            plt.axis('off')
            plt.imshow(img_old, cmap=cmap, norm=norm)
            plt.subplot(2, 2, 4)
            plt.axis('off')
            plt.imshow(img_up, cmap=cmap, norm=norm)
            plt.show()
            plt.draw()

        # count correct pixels
        for i in range(n_classes):
            if np.any(curr_y == i):
                correct[i] += np.sum(np.equal(img_up[curr_y == i],
                                              curr_y[curr_y == i]))
                total[i] += np.sum(curr_y == i)

    care_correct = correct * care_classes
    care_total = total * care_classes
    class_accuracy = calc_class_accuracy(care_correct, care_total)

    total_acc = (np.sum(care_correct, dtype='float32') /
                 np.sum(care_total)) * 100.
    if log:
        logger.info('total pixel accuracy %f %%', total_acc)
        logger.info('mean class accuracy: %f %%',
                    class_accuracy * 100.)
        logger.info('per class accuracies: %s',
                    correct.astype('float32') / total)
    return total_acc


def validate(conf, net_weights):

    logger.info("... loading data")
    logger.debug("Theano.config.floatX is %s" % theano.config.floatX)

    path = conf['data']['location']
    batch_size = 1
    assert(type(batch_size) is int)
    logger.info('Batch size %d' % (batch_size))

    try:
        x_train_allscales = try_pickle_load(
            path + 'x_' + conf['run-dataset'] + '.bin')
        x_train = x_train_allscales[0]  # first scale
        y_train = try_pickle_load(
            path + 'y_' + conf['run-dataset'] + '.bin')
    except IOError:
        logger.error("Unable to load Theano dataset from %s", path)
        exit(1)

    y_valid = try_pickle_load(path + 'y_validation.bin')
    print path + 'y_validation.bin'
    n_classes = int(max(y_train.max(), y_valid.max()) + 1)
    logger.info("Dataset has %d classes", n_classes)

    image_shape = (x_train.shape[-2], x_train.shape[-1])
    logger.info("Image shape is %s", image_shape)

    logger.info('Dataset has %d images' %
                x_train.shape[0])
    logger.info('Input data has shape of %s ',
                x_train.shape)

    # compute number of minibatches
    n_train_batches = x_train.shape[0] // batch_size

    logger.info("Number of train batches %d" % n_train_batches)

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

    y_train_shape = (y_train.shape[0], out_shape[0], out_shape[1])

    # resize marked images to out_size of the network
    y_train_downscaled = np.empty(y_train_shape)
    # for i in xrange(y_train.shape[0]):
    #     y_train_downscaled[i] = resize_marked_image(y_train[i], out_shape)

    x_train_shared, y_train_shared = \
        shared_dataset((x_train,
                        y_train_downscaled))
    x2_train_shared = theano.shared(x_train_allscales[1], borrow=True)
    x4_train_shared = theano.shared(x_train_allscales[2], borrow=True)

    ###############
    # BUILD MODEL #
    ###############
    logger.info("... building model")

    layers, new_layers = extend_net_w1l_drop(
        conv_out, conf['network']['layers'][-2] * 3, layers, n_classes,
        nkerns=conf['network']['layers'][-1:],
        seed=conf['network']['seed'],
        activation=lReLU, bias=0.001)

    test_model = theano.function(
        [index],
        [layers[0].y_pred],
        givens={
            x0: x_train_shared[index * batch_size: (index + 1) * batch_size],
            x2: x2_train_shared[index * batch_size: (index + 1) * batch_size],
            x4: x4_train_shared[index * batch_size: (index + 1) * batch_size]
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
    validation = [test_model(i)[0].reshape(NET_OUT_SHAPE)
                  for i in xrange(n_train_batches)]
    end_time = time.clock()
    logfiles_path = conf['data']['location'] +\
        'samples_' + conf['run-dataset'] + '.log'
    logger.info("Validated %d images in %.2f seconds",
                n_train_batches, end_time - start_time)
    get_stats(validation, y_train, layers[0].n_classes,
              conf['data']['dont-care-classes'], logfiles_path,
              conf['run-dataset'])

    logger.info("---> Results - superpixels")
    stats_func = lambda p: get_stats(
        validation, y_train, layers[0].n_classes,
        conf['data']['dont-care-classes'], logfiles_path,
        conf['run-dataset'], postproc=oversegment, postproc_params=p,
        show=False, log=False)
    start_time = time.clock()
    best_params = find_best_superpixel_params(stats_func)
    end_time = time.clock()
    logger.info("Done in %.2f seconds", end_time - start_time)
    logger.info("Best params are %s", best_params)

    #   run one more time with params, log output this time
    get_stats(
        validation, y_train, layers[0].n_classes,
        conf['data']['dont-care-classes'], logfiles_path,
        conf['run-dataset'], postproc=oversegment, postproc_params=best_params,
        show=False)


if __name__ == '__main__':
    """
    Examples of usage:
    python validate.py network.conf network-12-34.bin [train/validation/test]
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
