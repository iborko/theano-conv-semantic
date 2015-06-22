import cPickle as pickle
import theano
import numpy
import logging

logger = logging.getLogger(__name__)


def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    logger.info("Reshaping data_y, old shape is %s", data_y.shape)
    data_y = data_y.reshape((data_y.shape[0], -1))
    logger.info("Reshaping data_y, new shape is %s", data_y.shape)

    shared_x = theano.shared(numpy.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    return shared_x, shared_y


def load_data(x_path, y_path):
    """ Load pickled data. X are data, Y are results """

    with open(x_path, 'rb') as f:
        x_data = pickle.load(f)

    with open(y_path, 'rb') as f:
        y_data = pickle.load(f)

    return x_data, y_data


def calc_class_freqs(y):
    """
    Calculate class frequencies, create theano shared variable and
    save it into.

    :type y: numpy.array
    :param y: array of marked images

    Returns: theano shared variable whose length is number of classes
    """
    n_classes = y.max() + 1
    sums = numpy.bincount(y.reshape((-1)))
    ii = numpy.nonzero(sums)[0]
    total = sums.sum()

    freqs = numpy.zeros((n_classes), dtype=theano.config.floatX)
    for i, num in zip(ii, sums[ii]):
        freqs[i] = 1. * num / total

    logger.info("Total of %d items", total)
    logger.info("Class frequencies\n %s", freqs)

    freqs_shared = theano.shared(freqs, borrow=True)
    return freqs_shared


def build_care_classes(n_classes, data_conf):
    """
    Builds array of 0 and 1, where 0 marks class we don't care about:
    like unknown segmentation or added black pixels around image.

    :type n_classes: int
    :param n_classes: number of classes in dataset

    :type data_conf: dictionary
    :param data_conf: dicionary of data configurations, has to have a
    'dont-care-classes' section, where don't care classes are listed
    """
    dont_care_classes_str = 'dont-care-classes'
    assert(type(data_conf) is dict)
    assert(dont_care_classes_str in data_conf)

    care_vals = numpy.ones((n_classes), dtype='int8')
    care_vals[data_conf[dont_care_classes_str]] = 0
    care = theano.shared(care_vals, borrow=True)

    logger.info("Care classes indices\n %s", care.get_value())

    return care
