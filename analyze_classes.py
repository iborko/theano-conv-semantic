import sys
import numpy as np
import logging
from util import try_pickle_load

logger = logging.getLogger(__name__)


def analyze_classes(path):

    try:
        y_train = try_pickle_load(path + 'y_train.bin')
        y_test = try_pickle_load(path + 'y_test.bin')
    except IOError:
        logger.error("Unable to load Theano dataset from %s", path)
        exit(1)

    n_classes = int(max(y_train.max(), y_test.max()) + 1)
    logger.info("Dataset has %d classes", n_classes)

    image_shape = (y_train.shape[-2], y_train.shape[-1])
    logger.info("Image shape is %s", image_shape)

    y = np.concatenate((y_train, y_test), axis=0).astype('int8').reshape((-1))
    print y.shape
    yy = np.bincount(y)
    ii = np.nonzero(yy)[0]
    counts = np.vstack((ii, yy[ii])).T
    print "Counts\n", counts

    total = counts[:, 1].sum(dtype='float32')
    print "Total %10.0f" % total

    percents = counts[:, 1] / total
    print "Percents\n", "\n".join([("%7.2f" % (x * 100.)) for x in percents])


if __name__ == "__main__":
    """
    Usage:
        python analyze_classes.py data/iccv09Data/theano_datasets/
    """

    argc = len(sys.argv)

    if (argc == 2):
        analyze_classes(sys.argv[1])
    else:
        print "Wrong arguments"
        exit(1)
