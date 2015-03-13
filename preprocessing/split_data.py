"""
Spliting datasets into train and test dataset.
Input data has to pickled, output is pickled aswell.
"""

import sys
import os
import numpy as np
from util import try_pickle_load, try_pickle_dump

TEST_SIZE = 0.1


def main(x_path, y_path):
    x = try_pickle_load(x_path)
    y = try_pickle_load(y_path)
    print "Shape of loaded x data is", x.shape
    print "Shape of loaded y data is", y.shape
    assert(x.shape[0] == y.shape[0])

    test_size = int(x.shape[0] * TEST_SIZE)
    train_size = x.shape[0] - test_size
    assert(train_size + test_size == x.shape[0])
    print "Train size", train_size
    print "Test size", test_size

    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)

    train_ind = indices[:train_size]
    test_ind = indices[train_size:]

    train_set_x = x[train_ind]
    test_set_x = x[test_ind]
    train_set_y = y[train_ind]
    test_set_y = y[test_ind]

    folder_name = os.path.split(x_path)[0]
    print "Folder to save", folder_name

    try_pickle_dump(train_set_x, os.path.join(folder_name, "x_train.bin"))
    try_pickle_dump(test_set_x, os.path.join(folder_name, "x_test.bin"))
    try_pickle_dump(train_set_y, os.path.join(folder_name, "y_train.bin"))
    try_pickle_dump(test_set_y, os.path.join(folder_name, "y_test.bin"))

    print "Done"


if __name__ == "__main__":
    """
    Usage:
        python split_data.py <net_input> <net_output>
    Example:
        python split_data.py data/data_x.bin data/data_y.bin
    """
    argc = len(sys.argv)
    if argc == 3:
        x_path = sys.argv[1]
        y_path = sys.argv[2]
        main(x_path, y_path)
    else:
        print "Wrong arguments"
