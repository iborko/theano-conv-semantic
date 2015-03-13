import cPickle as pickle
import theano
import theano.tensor as T
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
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return shared_x, T.cast(shared_y, 'int32')


def load_data(x_path, y_path):
    """ Load pickled data. X are data, Y are results """

    with open(x_path, 'rb') as f:
        x_data = pickle.load(f)

    with open(y_path, 'rb') as f:
        y_data = pickle.load(f)

    return x_data, y_data
