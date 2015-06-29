"""
Plots cost on train and validation set.

Usage:
    python plot_cost.py output.log
"""
import matplotlib.pyplot as plt
import numpy as np
import sys

if len(sys.argv) < 2:
    print 'Not enough input args. Filename required!'
    sys.exit(0)


def parse_log(fname):
    train = []
    val = []
    val_error = []

    with open(fname) as f:
        for line in f:
            if 'training cost' in line:
                train.append(float(line.split('cost')[-1].strip()))

            if 'validation cost' in line:
                val.append(float(line.split('cost:')[-1].strip()))

            if 'validation error' in line and 'best' not in line:
                val_error.append(
                    float(line.split('validation error')[-1].split()[-2]))

    train = np.vstack(train)
    val = np.vstack(val)
    val_error = np.vstack(val_error)

    return (train, val, val_error)


def main(path):

    train, val, val_error = parse_log(path)

    fig, ax1 = plt.subplots()

    ax1.plot(train, label='training cost')
    ax1.plot(val, label='validation cost')
    ax1.plot(np.ones(len(val)) * np.min(val), 'r--')
    ax1.plot([np.argmin(val)], [np.min(val)], 'ro')
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('cost')
    ax1.legend(loc=1)

    ax2 = ax1.twinx()
    ax2.plot(val_error, 'r', label='validation error')
    ax2.plot(np.ones(len(val_error)) * np.min(val_error), 'r--')
    ax2.plot([np.argmin(val_error)], [np.min(val_error)], 'ro')
    ax2.set_ylabel('validation error', color='r')
    for tl in ax2.get_yticklabels():
            tl.set_color('r')
    ax2.legend(loc=2)
    plt.show()


if __name__ == "__main__":
    """
    Usage: python plot_cost.py output.log
    """
    argc = len(sys.argv)
    if (argc != 1):
        main(sys.argv[1])
    else:
        print "Wrong arguments"
        exit(1)
