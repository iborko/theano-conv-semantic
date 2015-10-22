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
import multiprocessing as mp

from preprocessing.transform_in import yuv_laplacian_norm
from preprocessing.transform_out import process_out
from dataset.loader_msrc import load_dataset
from preprocessing.class_counter import ClassCounter
from util import try_pickle_dump

logger = logging.getLogger(__name__)

DATASET_PATH = './data/MSRC/'
OUT_PATH = './data/MSRC/theano_datasets/'
requested_shape = (216, 320)
n_layers = 1


def save_result_img(result_list, result):
    i, layers = result
    for j, layer in enumerate(layers):
        result_list[j][i, :, :, :] = layer


def gen_layers_for_image(i, img):
    layers = yuv_laplacian_norm(img, requested_shape, n_layers)
    return i, layers


def generate_x(samples):
    """
    Generates list of processed images in a form of a 2d numpy array.

    samples: list
        list of Sample objects

    Returns: list of numpy arrays
    """
    x_shape = (len(samples), 3, requested_shape[0], requested_shape[1])

    # list of numpy array, every for one pyramid layer
    x_list = []
    for i in range(n_layers):
        x_list.append(np.zeros(x_shape, dtype=theano.config.floatX))
    logger.info("Input data new shape %s", x_list[0].shape)

    cpu_count = mp.cpu_count()
    pool = mp.Pool(cpu_count)
    logger.info("Cpu count %d", cpu_count)

    result_func = lambda result: save_result_img(x_list, result)

    for i, sample in enumerate(samples):
        pool.apply_async(gen_layers_for_image, args=(i, sample.image,),
                         callback=result_func)
    pool.close()
    pool.join()

    return x_list


def save_result_segm(result_list, result):
    """ Callback function, called in main process, saves result """
    i, img = result
    result_list[i] = img


def mark_image(i, img, cc, requested_shape):
    logger.info("Marking image %d", i)
    layers = process_out(img, cc, requested_shape)
    return i, layers


def generate_targets(samples, class_counter):
    """
    Generates array of segmented images.

    samples: list
        list of Sample objects
    class_counter: ClassCounter object
        object used for generating class markings (class ordinal numbers)

    returns: np.array
        array of class ordinal numbers
    """
    y_shape = (len(samples), requested_shape[0], requested_shape[1])

    y = np.zeros(y_shape, dtype='int8')

    logger.info("Segmented images new shape %s", y.shape)

    pool = mp.Pool(mp.cpu_count())
    logger.info("Cpu count %d", mp.cpu_count())

    result_func = lambda result: save_result_segm(y, result)

    for i, sample in enumerate(samples):
        # result_func(mark_image(i, sample.marked_image,
        #                        class_counter, requested_shape))
        pool.apply_async(mark_image,
                         args=(i, sample.marked_image,
                               class_counter, requested_shape,),
                         callback=result_func)
    pool.close()
    pool.join()

    return y


def split_samples(samples, test_size=0.1):
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


def main(show=False):
    logger.info("... loading data")
    logger.debug("Theano.config.floatX is %s" % theano.config.floatX)

    #   samples is list of Sample objects
    samples = load_dataset(DATASET_PATH)
    samples = list(samples)

    #   use only subset of data TODO remove this
    # DATA_TO_USE = 60
    # samples = samples[:DATA_TO_USE]

    random.seed(23455)
    random.shuffle(samples)

    train_samples, test_samples = split_samples(samples, 0.1)
    del samples

    cc = ClassCounter()

    x_train = generate_x(train_samples)
    x_test = generate_x(test_samples)
    y_train = generate_targets(train_samples, cc)
    y_test = generate_targets(test_samples, cc)
    del train_samples
    del test_samples

    cc.log_stats()

    try_pickle_dump(x_train, OUT_PATH + "x_train.bin")
    try_pickle_dump(x_test, OUT_PATH + "x_test.bin")
    try_pickle_dump(y_train, OUT_PATH + "y_train.bin")
    try_pickle_dump(y_test, OUT_PATH + "y_test.bin")

    # print x_train[0][0, 0, 80:90, 80:90]
    # print x_test[0][0, 0, 80:90, 80:90]

    if show:
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
    python generate_datasets.py [show]
    '''
    logging.basicConfig(level=logging.INFO)

    show = False
    argc = len(sys.argv)
    if argc > 1 and sys.argv[1] != "show":
        print "Wrong arguments"
        exit(1)
    if argc > 1 and sys.argv[1] == "show":
        show = True

    main(show)
