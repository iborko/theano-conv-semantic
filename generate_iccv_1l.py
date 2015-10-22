"""
Generates input and output numpy data for both train and test set, given
path to dataset.
Data is pickled in OUT_PATH folder, in .bin files.
"""
import theano
import numpy as np
import logging
import random
import sys
import pylab
import os
import multiprocessing as mp

# from preprocessing.transform_in import yuv
from preprocessing.transform_in import rgb_mean
from preprocessing.transform_out import process_iccv
from dataset.loader_iccv import load_dataset
from helpers.load_conf import load_config
from util import try_pickle_dump

logger = logging.getLogger(__name__)

requested_shape = (240, 320)


def save_result_img(result_list, result):
    i, layers = result
    for j, layer in enumerate(layers):
        result_list[j][i, :, :, :] = layer


def gen_layers_for_image(i, img):
    # img = yuv(img, requested_shape)
    img = rgb_mean(img, requested_shape)
    return i, [img]


def generate_x(samples, n_layers, gen_func):
    """
    Generates list of processed images in a form of a 2d numpy array.

    samples: list
        list of Sample objects
    n_layers: int
        number of layers
    gen_func: function
        function for generating layers (of images)

    Returns: list of numpy arrays
    """
    # list of numpy array, every for one pyramid layer
    x_list = []
    for i in range(n_layers):
        layer_shp = (len(samples), 3,
                     requested_shape[0] / (2**i), requested_shape[1] / (2**i))
        logger.info("Layer %d has shape %s", i, layer_shp)
        x_list.append(np.zeros(layer_shp, dtype=theano.config.floatX))

    cpu_count = mp.cpu_count()
    pool = mp.Pool(cpu_count)
    logger.info("Cpu count %d", cpu_count)

    def result_func(result): save_result_img(x_list, result)

    for i, sample in enumerate(samples):
        pool.apply_async(gen_func, args=(i, sample.image,),
                         callback=result_func)
    pool.close()
    pool.join()

    return x_list


def save_result_segm(result_list, result):
    """ Callback function, called in main process, saves result """
    i, img = result
    result_list[i] = img


def mark_image(i, img, requested_shape):
    logger.info("Marking image %d", i)
    layers = process_iccv(img, requested_shape)
    return i, layers


def generate_targets(samples):
    """
    Generates array of segmented images.

    samples: list
        list of Sample objects

    returns: np.array
        array of class ordinal numbers
    """
    y_shape = (len(samples), requested_shape[0], requested_shape[1])

    y = np.zeros(y_shape, dtype='int8')

    logger.info("Segmented images new shape %s", y.shape)

    # pool = mp.Pool(mp.cpu_count())
    logger.info("Cpu count %d", mp.cpu_count())

    def result_func(result): save_result_segm(y, result)

    for i, sample in enumerate(samples):
        result_func(mark_image(i, sample.marked_image, requested_shape))
    '''
        pool.apply_async(mark_image,
                         args=(i, sample.marked_image,
                               requested_shape,),
                         callback=result_func)
    pool.close()
    pool.join()
    '''

    return y


def split_samples(samples, test_size):
    n = len(samples)
    n_test = int(test_size * n)
    n_train = n - n_test
    logger.info("Splitting dataset, train/test/total, %d/%d/%d"
                % (n_train, n_test, n))

    train_samples = []
    test_samples = []

    for i in xrange(n):
        if i < n_train:
            train_samples.append(samples[i])
        else:
            test_samples.append(samples[i])

    assert(len(train_samples) == n_train)
    assert(len(test_samples) == n_test)

    return train_samples, test_samples


def write_samples_log(samples, outpath):
    with open(outpath, 'w') as f:
        f.writelines("\n".join([s.name for s in samples]))


def main(conf, gen_func, n_layers, show=False):
    """
    conf: dictionary
        configuration dictionary, from json file
    gen_func: function
        function used for generating inputs to network
    n_layers: int
        number of layers of laplacian pyramid used as input
    show: bool
        if true, few parsed images will be shown as a result
    """
    logger.info("... loading data")
    logger.debug("Theano.config.floatX is %s" % theano.config.floatX)

    #   samples is list of Sample objects
    dataset_path = conf['training']['dataset-folder']
    samples = load_dataset(dataset_path)
    samples = list(samples)

    if 'data-subset' in conf['training']:
        #   use only subset of data
        data_to_use = conf['training']['data-subset']
        logger.info("Using only subset of %d samples", data_to_use)
        samples = samples[:data_to_use]

    random.seed(conf['training']['shuffle-seed'])
    random.shuffle(samples)

    out_folder = conf['training']['out-folder']

    #   if test data defined
    if 'test-percent' in conf['training']:
        logger.info("Found test configuration, generating test data")
        test_size = float(conf['training']['test-percent']) / 100.0
        samples, test_samples = split_samples(samples, test_size)

        write_samples_log(test_samples,
                          os.path.join(out_folder, "samples_test.log"))
        x_test = generate_x(test_samples, n_layers, gen_func)
        y_test = generate_targets(test_samples)

        try_pickle_dump(x_test, os.path.join(out_folder, "x_test.bin"))
        try_pickle_dump(y_test, os.path.join(out_folder, "y_test.bin"))
    else:
        logger.info("No test set configuration present")

    validation_size = float(conf['training']['validation-percent']) / 100.0
    train_samples, validation_samples = split_samples(samples, validation_size)
    del samples

    write_samples_log(train_samples,
                      os.path.join(out_folder, "samples_train.log"))
    write_samples_log(validation_samples,
                      os.path.join(out_folder, "samples_validation.log"))

    x_train = generate_x(train_samples, n_layers, gen_func)
    x_validation = generate_x(validation_samples, n_layers, gen_func)
    y_train = generate_targets(train_samples)
    y_validation = generate_targets(validation_samples)
    del train_samples
    del validation_samples

    try_pickle_dump(x_train, os.path.join(out_folder, "x_train.bin"))
    try_pickle_dump(x_validation, os.path.join(out_folder, "x_validation.bin"))
    try_pickle_dump(y_train, os.path.join(out_folder, "y_train.bin"))
    try_pickle_dump(y_validation, os.path.join(out_folder, "y_validation.bin"))

    if show:
        #   show few parsed samples from train set
        n_imgs = 5
        for j in xrange(n_imgs):
            pylab.subplot(2, n_imgs, 0 * n_imgs + j + 1)
            pylab.axis('off')
            pylab.imshow(x_train[0][j, 0, :, :])  # rgb
        for j in xrange(n_imgs):
            pylab.subplot(2, n_imgs, 1 * n_imgs + j + 1)
            pylab.gray()
            pylab.axis('off')
            pylab.imshow(y_train[j, :, :])
        pylab.show()


if __name__ == "__main__":
    '''
    python generate_iccv_1l.py gen.conf [show]
    '''
    logging.basicConfig(level=logging.INFO)

    show = False
    argc = len(sys.argv)
    if argc == 2:
        conf_path = sys.argv[1]
    elif argc == 3:
        conf_path = sys.argv[1]
        if sys.argv[2] == "show":
            show = True
        else:
            print "Wrong arguments"
            exit(1)
    else:
        print "Too few arguments"
        exit(1)

    conf = load_config(conf_path)
    if conf is None:
        exit(1)

    main(conf, gen_layers_for_image, n_layers=1, show=show)
